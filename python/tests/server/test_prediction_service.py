import os
from concurrent.futures import Future
from datetime import datetime, timezone
from unittest import mock

import pytest

from cog.schema import PredictionRequest, PredictionResponse, Status
from cog.server.eventtypes import Done, Log
from cog.server.prediction_service import (
    BusyError,
    PredictionService,
    SetupEventHandler,
    UnknownPredictionError,
)
from cog.server.runner import SetupResult
from cog.server.worker import Worker


def _fixture_path(name):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, f"fixtures/{name}.py") + ":Predictor"


class FakeClock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


tick = mock.sentinel.tick


class FakeSetupEventHandler:
    def __init__(self):
        self.logs = []
        self.status = Status.STARTING

    @property
    def result(self):
        return SetupResult(
            started_at=datetime(2024, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
            status=self.status,
            logs=self.logs,
        )

    def append_logs(self, message):
        self.logs.append(message)

    def failed(self):
        self.status = Status.FAILED

    def handle_event(self, event):
        if isinstance(event, Log):
            self.append_logs(event.message)
        if isinstance(event, Done):
            if event.error:
                self.status = Status.FAILED
            else:
                self.status = Status.SUCCEEDED


class FakePredictionEventHandler:
    def __init__(self, prediction):
        self.input = prediction.input
        self.status = Status.PROCESSING
        self.error = None
        self.logs = []

    @property
    def response(self):
        return PredictionResponse(
            input=self.input,
            output="Hello, world!",
            logs="".join(self.logs),
            error=self.error,
            created_at=datetime(2024, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2024, 8, 7, 12, 1, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 8, 7, 12, 2, 0, tzinfo=timezone.utc),
            status=self.status,
        )

    def append_logs(self, message):
        self.logs.append(message)

    def failed(self, error=None):
        self.error = error
        self.status = Status.FAILED

    def handle_event(self, event):
        if isinstance(event, Log):
            self.logs.append(event.message)
        elif isinstance(event, Done):
            if event.canceled:
                self.status = Status.CANCELED
            elif event.error:
                self.status = Status.FAILED
                self.error = event.error_detail
            else:
                self.status = Status.SUCCEEDED


class FakeWorker:
    def __init__(self):
        self.subscribers = []
        self.last_prediction_payload = None

        self._setup_future = None
        self._predict_future = None

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)

    def setup(self):
        assert self._setup_future is None
        self._setup_future = Future()
        return self._setup_future

    def run_setup(self, events):
        for event in events:
            if isinstance(event, Exception):
                self._setup_future.set_exception(event)
                return
            for subscriber in self.subscribers:
                subscriber(event)
            if isinstance(event, Done):
                self._setup_future.set_result(event)

    def predict(self, payload):
        assert self._predict_future is None or self._predict_future.done()
        self.last_prediction_payload = payload
        self._predict_future = Future()
        return self._predict_future

    def run_predict(self, events):
        for event in events:
            if isinstance(event, Exception):
                self._predict_future.set_exception(event)
                return
            for subscriber in self.subscribers:
                subscriber(event)
            if isinstance(event, Done):
                self._predict_future.set_result(event)

    def cancel(self):
        done = Done(canceled=True)
        for subscriber in self.subscribers:
            subscriber(done)
        self._predict_future.set_result(done)


def test_prediction_service_setup_success():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    assert s.setup_result.status == Status.STARTING

    w.run_setup([Log(message="Setting up...", source="stdout")])
    assert s.setup_result.logs == ["Setting up..."]
    assert s.setup_result.status == Status.STARTING

    w.run_setup([Done()])
    assert s.setup_result.status == Status.SUCCEEDED


def test_prediction_service_setup_failure():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    assert s.setup_result.status == Status.STARTING

    w.run_setup([Done(error=True)])
    assert s.setup_result.status == Status.FAILED


def test_prediction_service_setup_exception():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    assert s.setup_result.status == Status.STARTING

    w.run_setup([RuntimeError("kaboom!")])
    assert s.setup_result.status == Status.FAILED
    assert s.setup_result.logs[0].startswith("Traceback")
    assert s.setup_result.logs[0].endswith("kaboom!\n")


def test_prediction_service_predict_success():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(input={"text": "giraffes"}))
    assert w.last_prediction_payload == {"text": "giraffes"}
    assert s.prediction_response.input == {"text": "giraffes"}
    assert s.prediction_response.status == Status.PROCESSING

    w.run_predict([Log(message="Predicting...", source="stdout")])
    assert s.prediction_response.logs == "Predicting..."

    w.run_predict([Done()])
    assert s.prediction_response.status == Status.SUCCEEDED


def test_prediction_service_predict_failure():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(input={"text": "giraffes"}))
    assert w.last_prediction_payload == {"text": "giraffes"}
    assert s.prediction_response.input == {"text": "giraffes"}
    assert s.prediction_response.status == Status.PROCESSING

    w.run_predict([Done(error=True, error_detail="ErrNeckTooLong")])
    assert s.prediction_response.status == Status.FAILED
    assert s.prediction_response.error == "ErrNeckTooLong"


def test_prediction_service_predict_exception():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(input={"text": "giraffes"}))
    assert w.last_prediction_payload == {"text": "giraffes"}
    assert s.prediction_response.input == {"text": "giraffes"}
    assert s.prediction_response.status == Status.PROCESSING

    w.run_predict(
        [
            Log(message="counting shards\n", source="stdout"),
            Log(message="reticulating splines\n", source="stdout"),
            ValueError("splines not reticulable"),
        ]
    )
    assert s.prediction_response.logs.startswith(
        "counting shards\nreticulating splines\n"
    )
    assert "Traceback" in s.prediction_response.logs
    assert "ValueError: splines not reticulable" in s.prediction_response.logs
    assert s.prediction_response.status == Status.FAILED
    assert s.prediction_response.error == "splines not reticulable"


def test_prediction_service_predict_before_setup():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    with pytest.raises(BusyError):
        s.predict(PredictionRequest(input={"text": "giraffes"}))


def test_prediction_service_predict_before_setup_completes():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()

    with pytest.raises(BusyError):
        s.predict(PredictionRequest(input={"text": "giraffes"}))


def test_prediction_service_predict_before_predict_completes():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(input={"text": "giraffes"}))

    with pytest.raises(BusyError):
        s.predict(PredictionRequest(input={"text": "giraffes"}))


def test_prediction_service_predict_after_predict_completes():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(input={"text": "giraffes"}))
    w.run_predict([Done()])

    s.predict(PredictionRequest(input={"text": "elephants"}))
    w.run_predict([Done()])

    assert w.last_prediction_payload == {"text": "elephants"}


def test_prediction_service_is_busy():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    assert s.is_busy()

    s.setup()
    assert s.is_busy()

    w.run_setup([Done()])
    assert not s.is_busy()

    s.predict(PredictionRequest(input={"text": "elephants"}))
    assert s.is_busy()

    w.run_predict([Done()])
    assert not s.is_busy()


def test_prediction_service_predict_cancelation():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(id="abcd1234", input={"text": "giraffes"}))

    with pytest.raises(ValueError):
        s.cancel(None)
    with pytest.raises(ValueError):
        s.cancel("")
    with pytest.raises(UnknownPredictionError):
        s.cancel("wxyz5678")

    w.run_predict([Log(message="Predicting...", source="stdout")])
    assert s.prediction_response.status == Status.PROCESSING

    s.cancel("abcd1234")
    assert s.prediction_response.status == Status.CANCELED


def test_prediction_service_predict_cancelation_multiple_predictions():
    w = FakeWorker()
    s = PredictionService(
        worker=w,
        _setup_event_handler_cls=FakeSetupEventHandler,
        _prediction_event_handler_cls=FakePredictionEventHandler,
    )

    s.setup()
    w.run_setup([Done()])

    s.predict(PredictionRequest(id="abcd1234", input={"text": "giraffes"}))
    w.run_predict([Done()])

    s.predict(PredictionRequest(id="defg6789", input={"text": "elephants"}))
    with pytest.raises(UnknownPredictionError):
        s.cancel("abcd1234")

    s.cancel("defg6789")
    assert s.prediction_response.status == Status.CANCELED


def test_prediction_service_setup_e2e():
    w = Worker(predictor_ref=_fixture_path("sleep"))
    s = PredictionService(worker=w)

    try:
        s.setup().result(timeout=5)
    finally:
        w.shutdown()

    result = s.setup_result

    assert result.status == Status.SUCCEEDED
    assert result.logs == []
    assert isinstance(result.started_at, datetime)
    assert isinstance(result.completed_at, datetime)


def test_prediction_service_predict_e2e():
    w = Worker(predictor_ref=_fixture_path("sleep"))
    s = PredictionService(worker=w)

    try:
        s.setup().result(timeout=5)
        s.predict(PredictionRequest(input={"sleep": 0.1})).result(timeout=1)
    finally:
        w.shutdown()

    response = s.prediction_response

    assert response.output == "done in 0.1 seconds"
    assert response.status == "succeeded"
    assert response.error is None
    assert response.logs == "starting\n"
    assert isinstance(response.started_at, datetime)
    assert isinstance(response.completed_at, datetime)


@pytest.mark.parametrize(
    "log,result",
    [
        (
            [],
            SetupResult(started_at=1),
        ),
        (
            [tick, Done()],
            SetupResult(started_at=1, completed_at=2, status=Status.SUCCEEDED),
        ),
        (
            [
                tick,
                Log("running 1\n", source="stdout"),
                Log("running 2\n", source="stdout"),
                Done(),
            ],
            SetupResult(
                started_at=1,
                completed_at=2,
                logs=["running 1\n", "running 2\n"],
                status=Status.SUCCEEDED,
            ),
        ),
        (
            [
                tick,
                tick,
                Done(error=True, error_detail="kaboom!"),
            ],
            SetupResult(
                started_at=1,
                completed_at=3,
                status=Status.FAILED,
            ),
        ),
    ],
)
def test_setup_event_handler(log, result):
    c = FakeClock(t=1)
    h = SetupEventHandler(_clock=c)

    for event in log:
        if event == tick:
            c.t += 1
        else:
            h.handle_event(event)

    assert h.result == result
