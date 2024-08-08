import io
import traceback
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import requests
import structlog
from attrs import define, field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import schema, types
from ..files import put_file_to_signed_endpoint
from ..json import upload_files
from ..predictor import BaseInput
from .eventtypes import Done, Log, PredictionOutput, PredictionOutputType
from .telemetry import current_trace_context
from .useragent import get_user_agent
from .webhook import SKIP_START_EVENT, webhook_caller_filtered
from .worker import Worker, _PublicEventType

log = structlog.get_logger("cog.server.prediction_service")


class BusyError(Exception):
    pass


class FileUploadError(Exception):
    pass


class UnknownPredictionError(Exception):
    pass


@define
class SetupResult:
    started_at: datetime
    completed_at: Optional[datetime] = None
    logs: List[str] = field(factory=list)
    status: Optional[Literal[schema.Status.FAILED, schema.Status.SUCCEEDED]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "logs": "".join(self.logs),
            "status": self.status,
        }


class PredictionService:
    """
    PredictionService manages the state of predictions running through the
    passed worker.
    """

    def __init__(
        self,
        *,
        worker: Worker,
        _setup_event_handler_cls: Optional[Callable[[], "SetupEventHandler"]] = None,
        _prediction_event_handler_cls: Optional[
            Callable[[schema.PredictionRequest], "PredictionEventHandler"]
        ] = None,
    ) -> None:
        self._setup_event_handler_cls = _setup_event_handler_cls or SetupEventHandler
        self._prediction_event_handler_cls = (
            _prediction_event_handler_cls or create_event_handler
        )

        self._setup_event_handler = None
        self._setup_future = None

        self._prediction_event_handler = None
        self._prediction_id = None
        self._predict_future: "Optional[Future[Done]]" = None

        self._worker = worker
        self._worker.subscribe(self._handle_event)

    def setup(self) -> "Future[None]":
        assert self._setup_event_handler is None, "do not call setup twice"
        f = Future()
        self._setup_event_handler = self._setup_event_handler_cls()
        self._setup_future = self._worker.setup()
        self._setup_future.add_done_callback(lambda _: f.set_result(None))
        return f

    @property
    def setup_result(self) -> SetupResult:
        assert (
            self._setup_event_handler is not None
        ), "call setup before accessing setup_result"

        # If we've cleared the future, it's because we've already computed the
        # final setup result.
        if not self._setup_future:
            return self._setup_event_handler.result

        # If setup is still running, return the current state.
        if not self._setup_future.done():
            return self._setup_event_handler.result

        try:
            self._setup_future.result()
        except Exception:  # pylint: disable=broad-exception-caught
            log.error("caught exception while running setup", exc_info=True)
            self._setup_event_handler.append_logs(traceback.format_exc())
            self._setup_event_handler.failed()

        # Clear the future so the result is memoized.
        self._setup_future = None

        return self._setup_event_handler.result

    def predict(
        self,
        prediction: schema.PredictionRequest,
        event_handler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Future[None]":
        busy = self.is_busy()
        if busy:
            raise busy

        event_handler_kwargs = event_handler_kwargs or {}

        # Set up logger context for main thread.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(prediction_id=prediction.id)

        self._prediction_event_handler = self._prediction_event_handler_cls(
            prediction, **event_handler_kwargs
        )
        self._prediction_id = prediction.id

        fut = Future()

        try:
            payload = _prepare_predict_payload(prediction.input)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._predict_future = Future()
            self._predict_future.set_exception(e)
        else:
            self._predict_future = self._worker.predict(payload)

        def _handle_done(f: "Future[Done]") -> None:
            # Propagate predict exceptions to our caller.
            exc = f.exception()
            if exc:
                fut.set_exception(exc)
            else:
                fut.set_result(None)

        self._predict_future.add_done_callback(_handle_done)

        return fut

    @property
    def prediction_response(self) -> schema.PredictionResponse:
        assert (
            self._prediction_event_handler is not None
        ), "predict must be called before accessing prediction_response"

        # If we've cleared the future, it's because we've already computed the
        # final prediction response.
        if not self._predict_future:
            return self._prediction_event_handler.response

        # If predict is still running, return the current state.
        if not self._predict_future.done():
            return self._prediction_event_handler.response

        try:
            self._predict_future.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("caught exception while running predict", exc_info=True)
            self._prediction_event_handler.append_logs(traceback.format_exc())
            self._prediction_event_handler.failed(error=str(e))

        # Clear the future so the result is memoized.
        self._predict_future = None

        return self._prediction_event_handler.response

    def is_busy(self) -> Optional[BusyError]:
        if self._setup_event_handler is None:
            # Setup hasn't been called yet.
            return BusyError("setup has not started")
        if self._setup_future is not None and not self._setup_future.done():
            # Setup is still running.
            return BusyError("setup is not complete")
        if self._predict_future is not None and not self._predict_future.done():
            # Prediction is still running.
            return BusyError("prediction running")
        return None

    def cancel(self, prediction_id: str) -> None:
        if not prediction_id:
            raise ValueError("prediction_id is required")
        if self._prediction_id != prediction_id:
            raise UnknownPredictionError()
        self._worker.cancel()

    def _handle_event(self, event: _PublicEventType) -> None:
        if self._prediction_event_handler:
            if self._prediction_id:
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(
                    prediction_id=self._prediction_id
                )

            self._prediction_event_handler.handle_event(event)
        elif self._setup_event_handler:
            self._setup_event_handler.handle_event(event)
        else:
            raise RuntimeError("received event when no events were expected")


class SetupEventHandler:
    def __init__(self, _clock: Optional[Callable[[], datetime]] = None) -> None:
        self._clock = _clock
        if self._clock is None:
            self._clock = lambda: datetime.now(timezone.utc)

        self._result = SetupResult(started_at=self._clock())

    @property
    def result(self) -> SetupResult:
        return self._result

    def append_logs(self, message: str) -> None:
        self._result.logs.append(message)

    def succeeded(self) -> None:
        assert self._clock
        self._result.completed_at = self._clock()
        self._result.status = schema.Status.SUCCEEDED

    def failed(self) -> None:
        assert self._clock
        self._result.completed_at = self._clock()
        self._result.status = schema.Status.FAILED

    def handle_event(self, event: _PublicEventType) -> None:
        if isinstance(event, Log):
            self.append_logs(event.message)
        elif isinstance(event, Done):
            if event.error:
                self.failed()
            else:
                self.succeeded()
        else:
            log.warn("received unexpected event during setup", data=event)


def create_event_handler(
    prediction: schema.PredictionRequest,
    upload_url: Optional[str] = None,
) -> "PredictionEventHandler":
    response = schema.PredictionResponse(**prediction.dict())

    webhook = prediction.webhook
    events_filter = (
        prediction.webhook_events_filter or schema.WebhookEvent.default_events()
    )

    webhook_sender = None
    if webhook is not None:
        webhook_sender = webhook_caller_filtered(webhook, set(events_filter))

    file_uploader = None
    if upload_url is not None:
        file_uploader = generate_file_uploader(upload_url, prediction_id=prediction.id)

    event_handler = PredictionEventHandler(
        response, webhook_sender=webhook_sender, file_uploader=file_uploader
    )

    return event_handler


def generate_file_uploader(
    upload_url: str, prediction_id: Optional[str]
) -> Callable[[Any], Any]:
    client = _make_file_upload_http_client()

    def file_uploader(output: Any) -> Any:
        def upload_file(fh: io.IOBase) -> str:
            return put_file_to_signed_endpoint(
                fh, endpoint=upload_url, prediction_id=prediction_id, client=client
            )

        return upload_files(output, upload_file=upload_file)

    return file_uploader


class PredictionEventHandler:
    def __init__(
        self,
        p: schema.PredictionResponse,
        webhook_sender: Optional[Callable[[Any, schema.WebhookEvent], None]] = None,
        file_uploader: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        log.info("starting prediction")
        self.p = p
        self.p.status = schema.Status.PROCESSING
        self._output_type_multi = None
        self.p.output = None
        self.p.logs = ""
        self.p.started_at = datetime.now(tz=timezone.utc)

        self._webhook_sender = webhook_sender
        self._file_uploader = file_uploader

        # HACK: don't send an initial webhook if we're trying to optimize for
        # latency (this guarantees that the first output webhook won't be
        # throttled.)
        if not SKIP_START_EVENT:
            self._send_webhook(schema.WebhookEvent.START)

    @property
    def response(self) -> schema.PredictionResponse:
        return self.p

    def set_output_type(self, *, multi: bool) -> None:
        assert (
            self._output_type_multi is None
        ), "Predictor unexpectedly returned multiple output types"
        assert (
            self.p.output is None
        ), "Predictor unexpectedly returned output type after output"

        if multi:
            self.p.output = []

        self._output_type_multi = multi

    def append_output(self, output: Any) -> None:
        assert (
            self._output_type_multi is not None
        ), "Predictor unexpectedly returned output before output type"

        uploaded_output = self._upload_files(output)
        if self._output_type_multi:
            self.p.output.append(uploaded_output)
            self._send_webhook(schema.WebhookEvent.OUTPUT)
        else:
            self.p.output = uploaded_output
            # We don't send a webhook for compatibility with the behaviour of
            # redis_queue. In future we can consider whether it makes sense to send
            # one here.

    def append_logs(self, logs: str) -> None:
        assert self.p.logs is not None
        self.p.logs += logs
        self._send_webhook(schema.WebhookEvent.LOGS)

    def succeeded(self) -> None:
        log.info("prediction succeeded")
        self.p.status = schema.Status.SUCCEEDED
        self._set_completed_at()
        # These have been set already: this is to convince the typechecker of
        # that...
        assert self.p.completed_at is not None
        assert self.p.started_at is not None
        self.p.metrics = {
            "predict_time": (self.p.completed_at - self.p.started_at).total_seconds()
        }
        self._send_webhook(schema.WebhookEvent.COMPLETED)

    def failed(self, error: str) -> None:
        log.info("prediction failed", error=error)
        self.p.status = schema.Status.FAILED
        self.p.error = error
        self._set_completed_at()
        self._send_webhook(schema.WebhookEvent.COMPLETED)

    def canceled(self) -> None:
        log.info("prediction canceled")
        self.p.status = schema.Status.CANCELED
        self._set_completed_at()
        self._send_webhook(schema.WebhookEvent.COMPLETED)

    def handle_event(self, event: _PublicEventType) -> None:
        if isinstance(event, Log):
            self.append_logs(event.message)
        elif isinstance(event, PredictionOutputType):
            self.set_output_type(multi=event.multi)
        elif isinstance(event, PredictionOutput):
            self.append_output(event.payload)
        elif isinstance(event, Done):  # pyright: ignore reportUnnecessaryIsinstance
            if event.canceled:
                self.canceled()
            elif event.error:
                self.failed(error=str(event.error_detail))
            else:
                self.succeeded()
        else:  # shouldn't happen, exhausted the type
            log.warn("received unexpected event during predict", data=event)

    def _set_completed_at(self) -> None:
        self.p.completed_at = datetime.now(tz=timezone.utc)

    def _send_webhook(self, event: schema.WebhookEvent) -> None:
        if self._webhook_sender is not None:
            self._webhook_sender(self.response, event)

    def _upload_files(self, output: Any) -> Any:
        if self._file_uploader is None:
            return output

        try:
            # TODO: clean up output files
            return self._file_uploader(output)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # If something goes wrong uploading a file, it's irrecoverable.
            # The re-raised exception will be caught and cause the prediction
            # to be failed, with a useful error message.
            raise FileUploadError("Got error trying to upload output files") from error


def _prepare_predict_payload(
    prediction_input: Union[BaseInput, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(prediction_input, BaseInput):
        input_dict = prediction_input.dict()
    else:
        input_dict = prediction_input.copy()

    for k, v in input_dict.items():
        # Check if v is an instance of URLPath
        if isinstance(v, types.URLPath):
            input_dict[k] = v.convert()
        # Check if v is a list of URLPath instances
        elif isinstance(v, list) and all(isinstance(item, types.URLPath) for item in v):
            input_dict[k] = [item.convert() for item in v]

    return input_dict


def _make_file_upload_http_client() -> requests.Session:
    session = requests.Session()
    session.headers["user-agent"] = (
        get_user_agent() + " " + str(session.headers["user-agent"])
    )

    ctx = current_trace_context() or {}
    for key, value in ctx.items():
        session.headers[key] = str(value)

    adapter = HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["PUT"],
        ),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
