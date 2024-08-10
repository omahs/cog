import base64
import io
import json
import pickle
from unittest.mock import MagicMock, patch

import pytest
from cog.types import PYDANTIC_V2, File, Secret, URLFile, get_filename
from pydantic import TypeAdapter


def test_urlfile_acts_like_response(httpserver):
    httpserver.expect_request("/some/url").respond_with_json(
        {"message": "hello world"}, status=200
    )

    u = URLFile(httpserver.url_for("/some/url"))

    assert isinstance(u, io.IOBase) or PYDANTIC_V2
    assert json.loads(u.read()) == {"message": "hello world"}


def test_urlfile_iterable(httpserver):
    httpserver.expect_request("/some/url").respond_with_data(
        "one\ntwo\nthree\n", status=200
    )

    u = URLFile(httpserver.url_for("/some/url"))
    result = list(u)

    assert result == [b"one\n", b"two\n", b"three\n"]


def test_urlfile_no_request_if_not_used(httpserver):
    # This test would fail if the request were actually made,
    # as we've not set up an expectation for it.
    URLFile(httpserver.url_for("/some/url"))


def test_urlfile_can_be_pickled(httpserver):
    u = URLFile(httpserver.url_for("/some/url"))

    result = pickle.loads(pickle.dumps(u))

    assert isinstance(result, URLFile)


def test_urlfile_can_be_pickled_even_once_loaded(httpserver):
    httpserver.expect_request("/some/url").respond_with_json(
        {"message": "hello world"}, status=200
    )

    u = URLFile(httpserver.url_for("/some/url"))
    u.read()

    result = pickle.loads(pickle.dumps(u))

    assert isinstance(result, URLFile)


@pytest.mark.parametrize(
    "url,filename",
    [
        # Simple URLs
        ("https://example.com/test", "test"),
        ("https://example.com/test.jpg", "test.jpg"),
        (
            "https://example.com/ហ_ត_អ_វ_ប_នជ_ក_រស_គតរបស_ព_រ_យ_ស_ម_នអ_ណ_ចម_ល_Why_Was_The_Death_Of_Jesus_So_Powerful_.m4a",
            "ហ_ត_អ_វ_ប_នជ_ក_រស_គតរបស_ព_រ_យ_ស_ម_នអ_ណ_ចម_ល_Why_Was_The_Death_Of_Jesus_So_Powerful_.m4a",
        ),
        # Data URIs
        (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==",
            "file.png",
        ),
        (
            "data:text/plain,hello world",
            "file.txt",
        ),
        (
            "data:application/data;base64,aGVsbG8gd29ybGQ=",
            "file",
        ),
        # URL-encoded filenames
        (
            "https://example.com/thing+with+spaces.m4a",
            "thing with spaces.m4a",
        ),
        (
            "https://example.com/thing%20with%20spaces.m4a",
            "thing with spaces.m4a",
        ),
        (
            "https://example.com/%E1%9E%A0_%E1%9E%8F_%E1%9E%A2_%E1%9E%9C_%E1%9E%94_%E1%9E%93%E1%9E%87_%E1%9E%80_%E1%9E%9A%E1%9E%9F_%E1%9E%82%E1%9E%8F%E1%9E%9A%E1%9E%94%E1%9E%9F_%E1%9E%96_%E1%9E%9A_%E1%9E%99_%E1%9E%9F_%E1%9E%98_%E1%9E%93%E1%9E%A2_%E1%9E%8E_%E1%9E%85%E1%9E%98_%E1%9E%9B_Why_Was_The_Death_Of_Jesus_So_Powerful_.m4a",
            "ហ_ត_អ_វ_ប_នជ_ក_រស_គតរបស_ព_រ_យ_ស_ម_នអ_ណ_ចម_ល_Why_Was_The_Death_Of_Jesus_So_Powerful_.m4a",
        ),
        # Illegal characters
        ("https://example.com/nulbytes\u0000.wav", "nulbytes_.wav"),
        ("https://example.com/nulbytes%00.wav", "nulbytes_.wav"),
        ("https://example.com/path%2Ftraversal.dat", "path_traversal.dat"),
        # Long filenames
        (
            "https://example.com/some/path/Biden_Trump_sows_chaos_makes_things_worse_U_S_hits_more_than_six_million_COVID_cases_WAPO_Trump_health_advisor_is_pushing_herd_immunity_strategy_despite_warnings_from_Fauci_medical_officials_Biden_says_he_hopes_to_be_able_to_visit_Wisconsin_as_governor_tells_Trump_to_stay_home_.mp3",
            "Biden_Trump_sows_chaos_makes_things_worse_U_S_hits_more_than_six_million_COVID_cases_WAPO_Trump_health_advisor_is_pushing_herd_immunity_strategy_despite_warnings_from_Fauci_medical_officials_Bide~.mp3",
        ),
        (
            "https://coppermerchants.example/complaints/𒀀𒈾𒂍𒀀𒈾𒍢𒅕𒆠𒉈𒈠𒌝𒈠𒈾𒀭𒉌𒈠𒀀𒉡𒌑𒈠𒋫𒀠𒇷𒆪𒆠𒀀𒄠𒋫𒀝𒁉𒄠𒌝𒈠𒀜𒋫𒀀𒈠𒄖𒁀𒊑𒁕𒄠𒆪𒁴𒀀𒈾𒄀𒅖𒀭𒂗𒍪𒀀𒈾𒀜𒁲𒅔𒋫𒀠𒇷𒅅𒈠𒋫𒀝𒁉𒀀𒄠.tablet",
            "𒀀𒈾𒂍𒀀𒈾𒍢𒅕𒆠𒉈𒈠𒌝𒈠𒈾𒀭𒉌𒈠𒀀𒉡𒌑𒈠𒋫𒀠𒇷𒆪𒆠𒀀𒄠𒋫𒀝𒁉𒄠𒌝𒈠𒀜𒋫𒀀𒈠𒄖𒁀𒊑𒁕𒄠𒆪𒁴𒀀𒈾𒄀𒅖~.tablet",
        ),
    ],
)
def test_get_filename(url, filename):
    assert get_filename(url) == filename


def test_secret_type():
    secret_value = "sw0rdf1$h"  # noqa: S105
    secret = Secret(secret_value)

    assert secret.get_secret_value() == secret_value
    assert str(secret) == "**********"


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_creation_with_http_url():
    url = "http://example.com/file.txt"
    file = File(url=url)
    assert file.url == url
    assert file.file_name == "file.txt"
    assert file.content_type == "text/plain"


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_creation_with_https_url():
    url = "https://example.com/image.jpg"
    file = File(url=url)
    assert file.url == url
    assert file.file_name == "image.jpg"
    assert file.content_type == "image/jpeg"


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_creation_with_data_url():
    data = b"Hello, World!"
    encoded = base64.b64encode(data).decode()
    url = f"data:text/plain;base64,{encoded}"
    file = File(url=url)
    assert file.url == url
    assert file.file_name == None
    assert file.content_type == "text/plain"
    assert file.data == data


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_creation_with_custom_content_type():
    url = "https://example.com/file"
    content_type = "application/octet-stream"
    file = File(url=url, content_type=content_type)
    assert file.url == url
    assert file.content_type == content_type


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_creation_with_custom_file_name():
    url = "https://example.com/file"
    file_name = "custom_name.bin"
    file = File(url=url, file_name=file_name)
    assert file.url == url
    assert file.file_name == file_name


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
@patch("requests.get")
def test_file_data_loading(mock_get):
    mock_response = MagicMock()
    mock_response.raw.read.return_value = b"Mocked data"
    mock_get.return_value = mock_response

    file = File(url="https://example.com/file")
    assert file.data == b"Mocked data"


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_str_representation():
    url = "https://example.com/file.txt"
    file = File(url=url)
    assert str(file) == url


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_context_manager():
    url = "https://example.com/file.txt"
    with File(url=url) as file:
        assert not file._closed
    assert file._closed


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_readable():
    file = File(url="https://example.com/file.txt")
    assert file.readable()
    file.close()
    assert not file.readable()


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_seekable():
    file = File(url="https://example.com/file.txt")
    assert not file.seekable()


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_writable():
    file = File(url="https://example.com/file.txt")
    assert not file.writable()


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_open_with_invalid_mode():
    file = File(url="https://example.com/file.txt")
    with pytest.raises(ValueError):
        file.open("w")


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
@patch("tempfile.NamedTemporaryFile")
def test_file_fspath(mock_temp_file, httpserver):
    httpserver.expect_request("/foo.txt").respond_with_data("hello")

    mock_temp = MagicMock()
    mock_temp.name = "/tmp/tempfile"
    mock_temp_file.return_value.__enter__.return_value = mock_temp

    file = File(url=httpserver.url_for("/foo.txt"))
    path = file.__fspath__()

    assert path == "/tmp/tempfile"


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_validation_with_bytes_io():
    adapter = TypeAdapter(File)

    data = b"Hello, World!"
    bytes_io = io.BytesIO(data)

    file = adapter.validate_python(bytes_io)
    assert isinstance(file, File)
    assert file.data == data

    serialized = json.loads(adapter.dump_json(file))
    assert serialized.startswith("data:application/octet-stream;base64,")
    assert base64.b64decode(serialized.split(",")[1]) == data


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_validation_with_string_io():
    adapter = TypeAdapter(File)

    data = "Hello, World!"
    string_io = io.StringIO(data)
    file = adapter.validate_python(string_io)

    assert isinstance(file, File)
    assert file.data == data.encode("utf-8")

    serialized = json.loads(adapter.dump_json(file))
    assert serialized.startswith("data:text/plain;base64,")


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_serialization_with_data():
    adapter = TypeAdapter(File)

    data = b"Hello, World!"
    file = adapter.validate_python(data)

    assert isinstance(file, File)
    assert file.data == data

    serialized = json.loads(adapter.dump_json(file))
    assert serialized.startswith("data:application/octet-stream;base64,")
    assert base64.b64decode(serialized.split(",")[1]) == data


@pytest.mark.skipif(not PYDANTIC_V2, reason="Requires Pydantic v2")
def test_file_validation_with_url(httpserver):
    httpserver.expect_request("/foo.txt").respond_with_data("hello")

    url = httpserver.url_for("/foo.txt")
    adapter = TypeAdapter(File)
    file = adapter.validate_python(url)
    assert isinstance(file, File)
    assert file.url == url

    serialized = json.loads(adapter.dump_json(file))
    assert serialized == url
