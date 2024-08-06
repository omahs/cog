import traceback
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import structlog

from .. import schema
from .eventtypes import Done, Log
from .runner import (
    PredictionEventHandler,
    SetupResult,
    UnknownPredictionError,
    _prepare_predict_payload,
    create_event_handler,
)
from .worker import Worker, _PublicEventType

log = structlog.get_logger("cog.server.prediction_service")


class BusyError(Exception):
    pass


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
            Callable[[schema.PredictionRequest], PredictionEventHandler]
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

        f = Future()

        try:
            payload = _prepare_predict_payload(prediction.input)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._predict_future = Future()
            self._predict_future.set_exception(e)
        else:
            self._predict_future = self._worker.predict(payload)

        self._predict_future.add_done_callback(lambda _: f.set_result(None))

        return f

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


# TODO: Move PredictionEventHandler in here.


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
