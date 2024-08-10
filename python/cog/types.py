import base64
import io
import mimetypes
import os
import pathlib
import shutil
import tempfile
import urllib.parse
import urllib.request
import urllib.response
from types import TracebackType
from typing import (
    IO,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import pydantic
import requests
from typing_extensions import NotRequired, TypedDict

if pydantic.__version__.startswith("1."):
    PYDANTIC_V2 = False
else:
    PYDANTIC_V2 = True


FILENAME_ILLEGAL_CHARS = set("\u0000/")

# Linux allows files up to 255 bytes long. We enforce a slightly shorter
# filename so that there's room for prefixes added by
# tempfile.NamedTemporaryFile, etc.
FILENAME_MAX_LENGTH = 200


class CogConfig(TypedDict):  # pylint: disable=too-many-ancestors
    build: "CogBuildConfig"
    image: NotRequired[str]
    predict: NotRequired[str]
    train: NotRequired[str]


class CogBuildConfig(TypedDict, total=False):  # pylint: disable=too-many-ancestors
    cuda: Optional[str]
    gpu: Optional[bool]
    python_packages: Optional[List[str]]
    system_packages: Optional[List[str]]
    python_requirements: Optional[str]
    python_version: Optional[str]
    run: Optional[Union[List[str], List[Dict[str, Any]]]]


def Input(  # pylint: disable=invalid-name, too-many-arguments
    default: Any = ...,
    description: str = None,
    ge: float = None,
    le: float = None,
    min_length: int = None,
    max_length: int = None,
    regex: str = None,
    choices: List[Union[str, int]] = None,
) -> Any:
    """Input is similar to pydantic.Field, but doesn't require a default value to be the first argument."""
    field_kwargs = {
        "default": default,
        "description": description,
        "ge": ge,
        "le": le,
        "min_length": min_length,
        "max_length": max_length,
    }

    if PYDANTIC_V2:
        field_kwargs["pattern"] = regex
        field_kwargs["json_schema_extra"] = {"enum": choices}
    else:
        field_kwargs["regex"] = regex
        field_kwargs["enum"] = choices
    return pydantic.Field(**field_kwargs)


Item = TypeVar("Item")


class ConcatenateIterator(Iterator[Item]):  # pylint: disable=abstract-method
    @classmethod
    def validate(cls, value: Iterator[Any]) -> Iterator[Any]:
        return value

    if PYDANTIC_V2:
        from pydantic import (  # pylint: disable=import-outside-toplevel
            GetCoreSchemaHandler,
        )
        from pydantic.json_schema import (  # pylint: disable=import-outside-toplevel
            JsonSchemaValue,
        )
        from pydantic_core import (  # pylint: disable=import-outside-toplevel
            CoreSchema,
        )

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source: Type[Any],  # pylint: disable=unused-argument
            handler: "pydantic.GetCoreSchemaHandler",  # pylint: disable=unused-argument
        ) -> "CoreSchema":
            from pydantic_core import (  # pylint: disable=import-outside-toplevel
                core_schema,
            )

            return core_schema.union_schema(
                [
                    core_schema.is_instance_schema(Iterator),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ]
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: "CoreSchema", handler: "pydantic.GetJsonSchemaHandler"
        ) -> "JsonSchemaValue":  # type: ignore # noqa: F821
            json_schema = handler(core_schema)
            json_schema.pop("allOf", None)
            json_schema.update(
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-cog-array-type": "iterator",
                    "x-cog-array-display": "concatenate",
                }
            )
            return json_schema
    else:

        @classmethod
        def __get_validators__(cls) -> Iterator[Any]:
            yield cls.validate

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            """Defines what this type should be in openapi.json"""
            field_schema.pop("allOf", None)
            field_schema.update(
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "x-cog-array-type": "iterator",
                    "x-cog-array-display": "concatenate",
                }
            )


class Secret(pydantic.SecretStr):
    if PYDANTIC_V2:
        from pydantic.json_schema import (  # pylint: disable=import-outside-toplevel
            JsonSchemaValue,
        )
        from pydantic_core import CoreSchema  # pylint: disable=import-outside-toplevel

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: CoreSchema, handler: Any
        ) -> JsonSchemaValue:
            json_schema = handler(core_schema)
            cls._update_schema(json_schema)
            return json_schema
    else:

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            """Defines what this type should be in openapi.json"""
            cls._update_schema(field_schema)

    @classmethod
    def _update_schema(cls, schema: Dict[str, Any]) -> None:
        schema.update(
            {
                "type": "string",
                "format": "password",
                "x-cog-secret": True,
            }
        )


if PYDANTIC_V2:
    from pydantic import TypeAdapter
    from pydantic_core.core_schema import (  # pylint: disable=import-outside-toplevel
        CoreSchema,
    )

    class IOBaseMeta(type(io.IOBase)):
        @classmethod
        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, io.IOBase)

    class File(metaclass=IOBaseMeta):  # type: ignore
        url: Optional[str]
        content_type: Optional[str]
        name: Optional[str]
        size: Optional[int]
        _data: Optional[bytes] = None
        _closed: bool = False

        def __init__(
            self,
            url: Optional[str] = None,
            content_type: Optional[str] = None,
            file_name: Optional[str] = None,
        ) -> None:
            self.url = url

            if url and url.startswith("data:"):
                try:
                    header, encoded = url.split(",", 1)
                    encoding = "base64" if ";base64" in header else None
                    self._data = (
                        base64.b64decode(encoded)
                        if encoding == "base64"
                        else urllib.parse.unquote_to_bytes(encoded)
                    )

                    if content_type is None:
                        mime_type = header.split(";")[0].split(":", 1)
                        content_type = (
                            mime_type[1] if len(mime_type) > 1 else "text/plain"
                        )
                except (ValueError, IndexError):
                    raise ValueError("Invalid data URL format") from None

            else:
                parsed_url = urllib.parse.urlparse(url)
                if parsed_url:
                    path = urllib.parse.unquote(parsed_url.path)
                    if content_type is None and path:
                        content_type = mimetypes.guess_type(path.split("/")[-1])[0]
                    if file_name is None and path:
                        file_name = os.path.basename(path)

            self.content_type = content_type or "application/octet-stream"
            self.file_name = file_name

        @property
        def data(self) -> bytes:
            if self._data is None:
                self._data = self._load_data()
            return self._data

        def _load_data(self) -> bytes:
            if self.url:
                resp = requests.get(self.url, stream=True, timeout=None)
                resp.raise_for_status()
                resp.raw.decode_content = True
                return resp.raw.read()
            raise ValueError("No data or URL provided")

        def __str__(self) -> str:
            return self.url or ""

        def __enter__(self) -> "File":
            return self  # type: ignore

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> None:
            self.close()

        def readable(self) -> bool:
            return not self._closed

        def seekable(self) -> bool:
            return False

        def writable(self) -> bool:
            return False

        def open(self, mode: str = "rb") -> "File":
            if mode not in ("r", "rb"):
                raise ValueError(
                    f"Unsupported mode: {mode}. Only 'r' and 'rb' are allowed."
                )
            if self._closed:
                raise ValueError("I/O operation on closed file")
            return self  # type: ignore

        def close(self) -> None:
            self._closed = True

        def read(self, size: int = -1) -> bytes:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            return self.data[:size] if size > 0 else self.data

        def __fspath__(self) -> str:
            if self.url and self.url.startswith(("http://", "https://", "file://")):
                # For URLs, we need to download the file and return a local path
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=self.file_name
                ) as temp_file:
                    temp_file.write(self.data)
                    return temp_file.name
            elif self.url and self.url.startswith("data:"):
                # For data URLs, we need to decode and save to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=self.file_name
                ) as temp_file:
                    temp_file.write(self.data)
                    return temp_file.name
            else:
                raise ValueError("Cannot convert File to filesystem path")

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Type[Any],  # pylint: disable=unused-argument
            handler: "pydantic.GetCoreSchemaHandler",  # pylint: disable=unused-argument
        ) -> "CoreSchema":
            from pydantic_core import (  # pylint: disable=import-outside-toplevel
                SchemaSerializer,
                core_schema,
            )

            schema = core_schema.json_or_python_schema(
                json_schema=core_schema.str_schema(),
                python_schema=core_schema.union_schema(
                    [
                        core_schema.no_info_plain_validator_function(cls.validate),
                        core_schema.is_instance_schema(cls),
                    ]
                ),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    cls.serialize,
                    return_schema=core_schema.str_schema(),
                    when_used="json",
                ),
            )

            cls.__pydantic_serializer__ = SchemaSerializer(schema)

            return schema

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: "CoreSchema", handler: "pydantic.GetJsonSchemaHandler"
        ) -> "JsonSchemaValue":  # type: ignore # noqa: F821
            json_schema = handler(core_schema)
            json_schema.update(type="string", format="uri")
            return json_schema

        @classmethod
        def validate(cls, value: Any) -> "File":
            if isinstance(value, str):
                parsed_url = urllib.parse.urlparse(value)
                if parsed_url.scheme not in ["http", "https", "data"]:
                    raise ValueError("value must be a valid URL")
                return cls(url=parsed_url.geturl())  # type: ignore
            if isinstance(value, io.IOBase):  # type: ignore
                if isinstance(value, io.FileIO):
                    return cls(url=f"file://{value.name}")  # type: ignore
                elif isinstance(value, io.BytesIO):
                    data = value.read()
                    encoded = base64.b64encode(data).decode()
                    return cls(url=f"data:application/octet-stream;base64,{encoded}")  # type: ignore
                elif isinstance(value, io.StringIO):
                    data = value.read().encode("utf-8")
                    encoded = base64.b64encode(data).decode()
                    return cls(url=f"data:text/plain;base64,{encoded}")  # type: ignore
            raise ValueError(f"Unsupported file type: {type(value)}")

        @classmethod
        def serialize(cls, value: "File") -> str:
            if hasattr(value, "url") and value.url:
                return value.url  # type: ignore
            if hasattr(value, "data") and value.data:
                # If there's no URL but we have data, create a data URL
                encoded = base64.b64encode(value.data).decode()
                mime_type = value.content_type or "application/octet-stream"
                return f"data:{mime_type};base64,{encoded}"
            else:
                raise ValueError("File has no URL or data to serialize")

        def __iter__(self) -> Iterator[bytes]:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as temp_file:
                temp_file.write(self.data)
                temp_file.flush()
                temp_file.seek(0)
                yield from temp_file
            os.unlink(temp_file.name)

    # https://github.com/pydantic/pydantic/issues/7779#issuecomment-1775629521
    _ = TypeAdapter(File)

    class PosixPathMeta(type(pathlib.PosixPath)):
        @classmethod
        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, pathlib.PosixPath)

    class Path(metaclass=PosixPathMeta):  # type: ignore
        def __init__(
            self,
            path: Union[str, pathlib.Path],
            content_type: Optional[str] = None,
        ) -> None:
            self._path = pathlib.Path(path).absolute()
            self.url = f"file://{self._path}"
            self.content_type = content_type or mimetypes.guess_type(self._path)[0]
            self.file_name = self._path.name
            self._file_handle: Optional[IO[Any]] = None
            self._position: int = 0
            self._mode: str = "rb+"
            self._closed: bool = False

        def _load_data(self) -> bytes:
            with open(self._path, "rb") as f:
                return f.read()

        def __fspath__(self) -> str:
            return str(self._path)

        def open(self, mode: str = "rb+", encoding: Optional[str] = None) -> "Path":
            if mode not in ("r", "rb", "r+", "rb+", "w", "wb", "w+", "wb+"):
                raise ValueError(
                    f"Unsupported mode: {mode}. Only 'r', 'rb', 'r+', 'rb+', 'w', 'wb', 'w+', 'wb+' are allowed."
                )
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if self._file_handle is None:
                self._file_handle = open(self._path, mode, encoding=encoding)  # pylint: disable=consider-using-with
                self._mode = mode
            return self  # type: ignore

        def close(self) -> None:
            if self._file_handle:
                self._file_handle.close()
            self._file_handle = None
            self._position = 0
            self._closed = True

        def read(self, size: int = -1) -> Union[str, bytes]:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if self._file_handle is None:
                self.open()
            assert self._file_handle is not None
            data = self._file_handle.read(size)
            self._position = self._file_handle.tell()
            return data

        def write(self, data: Union[str, bytes]) -> int:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if self._file_handle is None:
                self.open("wb+")
            assert self._file_handle is not None
            if isinstance(data, str):
                data = data.encode()
            bytes_written = self._file_handle.write(data)
            self._position = self._file_handle.tell()
            return bytes_written

        def seekable(self) -> bool:
            return True

        def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if self._file_handle is None:
                self.open()
            assert self._file_handle is not None
            position = self._file_handle.seek(offset, whence)
            self._position = position
            return position

        def tell(self) -> int:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            return self._position

        def writable(self) -> bool:
            return self._mode.endswith("+") or self._mode.startswith("w")

        # Implement pathlib.PosixPath-like methods
        def __truediv__(self, key: Union[str, pathlib.Path]) -> "Path":
            return Path(self._path / key)

        def __rtruediv__(self, key: Union[str, pathlib.Path]) -> "Path":
            return Path(pathlib.Path(key) / self._path)

        @property
        def parent(self) -> "Path":
            return Path(self._path.parent)

        @property
        def name(self) -> str:
            return self._path.name

        @property
        def suffix(self) -> str:
            return self._path.suffix

        @property
        def stem(self) -> str:
            return self._path.stem

        def is_file(self) -> bool:
            return self._path.is_file()

        def is_dir(self) -> bool:
            return self._path.is_dir()

        def exists(self) -> bool:
            return self._path.exists()

        def glob(self, pattern: str) -> Iterator["Path"]:
            return (Path(p) for p in self._path.glob(pattern))

        def rglob(self, pattern: str) -> Iterator["Path"]:
            return (Path(p) for p in self._path.rglob(pattern))

        def relative_to(self, other: Union[str, pathlib.Path]) -> "Path":
            return Path(self._path.relative_to(other))

        def resolve(self) -> "Path":
            return Path(self._path.resolve())

        def absolute(self) -> "Path":
            return Path(self._path.absolute())

        @classmethod
        def cwd(cls) -> "Path":
            return cls(pathlib.Path.cwd())  # type: ignore

        @classmethod
        def home(cls) -> "Path":
            return cls(pathlib.Path.home())  # type: ignore

        def __str__(self) -> str:
            return str(self._path)

        def __eq__(self, other: object) -> bool:
            if isinstance(other, (Path, pathlib.Path, str)):
                return self._path == pathlib.Path(other)
            return NotImplemented

        def __lt__(self, other: Union["Path", pathlib.Path, str]) -> bool:
            if isinstance(other, (Path, pathlib.Path, str)):  # type: ignore
                return self._path < pathlib.Path(other)
            return NotImplemented

        def __le__(self, other: Union["Path", pathlib.Path, str]) -> bool:
            if isinstance(other, (Path, pathlib.Path, str)):  # type: ignore
                return self._path <= pathlib.Path(other)
            return NotImplemented

        def __gt__(self, other: Union["Path", pathlib.Path, str]) -> bool:
            if isinstance(other, (Path, pathlib.Path, str)):  # type: ignore
                return self._path > pathlib.Path(other)
            return NotImplemented

        def __ge__(self, other: Union["Path", pathlib.Path, str]) -> bool:
            if isinstance(other, (Path, pathlib.Path, str)):  # type: ignore
                return self._path >= pathlib.Path(other)
            return NotImplemented

        def __hash__(self) -> int:
            return hash(self._path)

        def __enter__(self) -> "Path":
            return self.open()

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> None:
            self.close()

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Type[Any],  # pylint: disable=unused-argument
            handler: "pydantic.GetCoreSchemaHandler",  # pylint: disable=unused-argument
        ) -> "CoreSchema":
            from pydantic_core import (  # pylint: disable=import-outside-toplevel
                SchemaSerializer,
                core_schema,
            )

            schema = core_schema.json_or_python_schema(
                json_schema=core_schema.str_schema(),
                python_schema=core_schema.union_schema(
                    [
                        core_schema.no_info_plain_validator_function(cls.validate),
                        core_schema.is_instance_schema(cls),
                    ]
                ),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    cls.serialize,
                    return_schema=core_schema.str_schema(),
                    when_used="json",
                ),
            )

            # https://github.com/pydantic/pydantic/issues/7779#issuecomment-1775629521
            cls.__pydantic_serializer__ = SchemaSerializer(schema)

            return schema

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: "CoreSchema", handler: "pydantic.GetJsonSchemaHandler"
        ) -> "JsonSchemaValue":  # type: ignore # noqa: F821
            json_schema = handler(core_schema)

            # FIXME: This matches previous behavior, but it's not correct.
            json_schema.update(type="string", format="uri")

            return json_schema

        @classmethod
        def validate(cls, value: Any) -> "Path":
            if isinstance(value, str):
                if value.startswith(("data:", "http:", "https:")):
                    return File(url=value)  # type: ignore

            if isinstance(value, cls):
                return value  # type: ignore
            if isinstance(value, (str, pathlib.Path)):
                return cls(value)  # type: ignore
            raise ValueError(f"Cannot convert {type(value)} to Path")

        @classmethod
        def serialize(cls, value: "Path") -> str:
            return str(value)  # type: ignore

    # https://github.com/pydantic/pydantic/issues/7779#issuecomment-1775629521
    _ = TypeAdapter(Path)

    class URLPath(Path):  # type: ignore
        def __init__(
            self,
            source: str,
            filename: str,
            fileobj: io.IOBase,
        ) -> None:
            self.source = source
            self.filename = filename
            self.fileobj = fileobj
            self._path: Optional[pathlib.Path] = None
            super().__init__(self.source)

        def _get_temp_path(self) -> pathlib.Path:
            if self._path is None:
                with tempfile.NamedTemporaryFile(
                    suffix=self.filename, delete=False
                ) as temp_file:
                    shutil.copyfileobj(self.fileobj, temp_file)
                    self._path = pathlib.Path(temp_file.name)
            return self._path

        def convert(self) -> Path:
            return Path(self._get_temp_path())  # type: ignore

        def unlink(self, missing_ok: bool = False) -> None:
            if self._path:
                self._path.unlink(missing_ok=missing_ok)

        def __str__(self) -> str:
            return self.source

    class URLFile(File):  # type: ignore
        pass

else:

    class File(io.IOBase):
        """Deprecated: use Path instead."""

        validate_always = True

        @classmethod
        def __get_validators__(cls) -> Iterator[Any]:
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> io.IOBase:
            if isinstance(value, io.IOBase):
                return value

            parsed_url = urllib.parse.urlparse(value)
            if parsed_url.scheme == "data":
                with urllib.request.urlopen(value) as res:  # noqa: S310
                    return io.BytesIO(res.read())
            if parsed_url.scheme in ("http", "https"):
                return URLFile(value)
            raise ValueError(
                f"'{parsed_url.scheme}' is not a valid URL scheme. 'data', 'http', or 'https' is supported."
            )

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            """Defines what this type should be in openapi.json"""
            # https://json-schema.org/understanding-json-schema/reference/string.html#uri-template
            field_schema.update(type="string", format="uri")

        def __iter__(self) -> Iterator[bytes]:
            if self._closed:
                raise ValueError("I/O operation on closed file")
            return iter(self.data)

    class Path(metaclass=PosixPathMeta):  # type: ignore
        validate_always = True

        @classmethod
        def __get_validators__(cls) -> Iterator[Any]:
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> pathlib.Path:
            if isinstance(value, pathlib.Path):
                return value

            return URLPath(
                source=value,
                filename=get_filename(value),
                fileobj=File.validate(value),
            )

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            """Defines what this type should be in openapi.json"""
            # https://json-schema.org/understanding-json-schema/reference/string.html#uri-template
            field_schema.update(type="string", format="uri")

    class URLPath(pathlib.PosixPath):  # pylint: disable=abstract-method
        """
        URLPath is a nasty hack to ensure that we can defer the downloading of a
        URL passed as a path until later in prediction dispatch.

        It subclasses pathlib.PosixPath only so that it can pass isinstance(_,
        pathlib.Path) checks.
        """

        _path: Optional[Path]

        def __init__(self, *, source: str, filename: str, fileobj: io.IOBase) -> None:  # pylint: disable=super-init-not-called
            self.source = source
            self.filename = filename
            self.fileobj = fileobj

            self._path = None

        def convert(self) -> Path:
            if self._path is None:
                dest = tempfile.NamedTemporaryFile(suffix=self.filename, delete=False)  # pylint: disable=consider-using-with
                shutil.copyfileobj(self.fileobj, dest)
                self._path = Path(dest.name)
            return self._path

        def unlink(self, missing_ok: bool = False) -> None:
            if self._path:
                self._path.unlink(missing_ok=missing_ok)

        def __str__(self) -> str:
            # FastAPI's jsonable_encoder will encode subclasses of pathlib.Path by
            # calling str() on them
            return self.source

    class URLFile(io.IOBase):
        """
        URLFile is a proxy object for a :class:`urllib3.response.HTTPResponse`
        object that is created lazily. It's a file-like object constructed from a
        URL that can survive pickling/unpickling.
        """

        __slots__ = ("__target__", "__url__")

        def __init__(self, url: str) -> None:
            object.__setattr__(self, "__url__", url)

        # We provide __getstate__ and __setstate__ explicitly to ensure that the
        # object is always picklable.
        def __getstate__(self) -> Dict[str, Any]:
            return {"url": object.__getattribute__(self, "__url__")}

        def __setstate__(self, state: Dict[str, Any]) -> None:
            object.__setattr__(self, "__url__", state["url"])

        # Proxy getattr/setattr/delattr through to the response object.
        def __setattr__(self, name: str, value: Any) -> None:
            if hasattr(type(self), name):
                object.__setattr__(self, name, value)
            else:
                setattr(self.__wrapped__, name, value)

        def __getattr__(self, name: str) -> Any:
            if name in ("__target__", "__wrapped__", "__url__"):
                raise AttributeError(name)
            return getattr(self.__wrapped__, name)

        def __delattr__(self, name: str) -> None:
            if hasattr(type(self), name):
                object.__delattr__(self, name)
            else:
                delattr(self.__wrapped__, name)

        # Luckily the only dunder method on HTTPResponse is __iter__
        def __iter__(self) -> Iterator[bytes]:
            return iter(self.__wrapped__)

        @property
        def __wrapped__(self) -> Any:
            try:
                return object.__getattribute__(self, "__target__")
            except AttributeError:
                url = object.__getattribute__(self, "__url__")
                resp = requests.get(url, stream=True, timeout=None)
                resp.raise_for_status()
                resp.raw.decode_content = True
                object.__setattr__(self, "__target__", resp.raw)
                return resp.raw

        def __repr__(self) -> str:
            try:
                target = object.__getattribute__(self, "__target__")
            except AttributeError:
                return f"<{type(self).__name__} at 0x{id(self):x} for {object.__getattribute__(self, '__url__')!r}>"

            return f"<{type(self).__name__} at 0x{id(self):x} wrapping {target!r}>"


def get_filename(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme == "data":
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            mime_type = resp.headers.get_content_type()
            extension = mimetypes.guess_extension(mime_type)
            if extension is None:
                return "file"
            return "file" + extension

    basename = os.path.basename(parsed_url.path)
    basename = urllib.parse.unquote_plus(basename)

    # If the filename is too long, we truncate it (appending '~' to denote the
    # truncation) while preserving the file extension.
    # - truncate it
    # - append a tilde
    # - preserve the file extension
    if _len_bytes(basename) > FILENAME_MAX_LENGTH:
        basename = _truncate_filename_bytes(basename, length=FILENAME_MAX_LENGTH)

    for c in FILENAME_ILLEGAL_CHARS:
        basename = basename.replace(c, "_")

    return basename


def _len_bytes(s: str, encoding: str = "utf-8") -> int:
    return len(s.encode(encoding))


def _truncate_filename_bytes(s: str, length: int, encoding: str = "utf-8") -> str:
    """
    Truncate a filename to at most `length` bytes, preserving file extension
    and avoiding text encoding corruption from truncation.
    """
    root, ext = os.path.splitext(s.encode(encoding))
    root = root[: length - len(ext) - 1]
    return root.decode(encoding, "ignore") + "~" + ext.decode(encoding)
