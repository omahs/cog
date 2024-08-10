import requests
import pytest
from pytest_httpserver import HTTPServer
from cog.schema import PredictionResponse, Status, WebhookEvent
from cog.server.webhook import webhook_caller, webhook_caller_filtered


def test_webhook_caller_basic(httpserver: HTTPServer):
    httpserver.expect_request(
        "/webhook/123",
        method="POST",
        json={
            "status": Status.PROCESSING,
            "output": {"animal": "giraffe"},
            "input": {},
        },
    ).respond_with_data(status=200)

    c = webhook_caller(httpserver.url_for("/webhook/123"))

    payload = {
        "status": Status.PROCESSING,
        "output": {"animal": "giraffe"},
        "input": {},
    }
    response = PredictionResponse(**payload)

    c(response)


def test_webhook_caller_non_terminal_does_not_retry(httpserver: HTTPServer):
    httpserver.expect_request(
        "/webhook/123",
        method="POST",
        json={
            "status": Status.PROCESSING,
            "output": {"animal": "giraffe"},
            "input": {},
        },
    ).respond_with_data(status=429)

    c = webhook_caller(httpserver.url_for("/webhook/123"))

    payload = {
        "status": Status.PROCESSING,
        "output": {"animal": "giraffe"},
        "input": {},
    }
    response = PredictionResponse(**payload)

    c(response)


def test_webhook_caller_terminal_retries(httpserver: HTTPServer):
    payload = {"status": Status.SUCCEEDED, "output": {"animal": "giraffe"}, "input": {}}

    httpserver.expect_ordered_request(
        "/webhook/123",
        method="POST",
        json=payload,
    ).respond_with_data(status=429)

    httpserver.expect_ordered_request(
        "/webhook/123",
        method="POST",
        json=payload,
    ).respond_with_data(status=429)

    httpserver.expect_ordered_request(
        "/webhook/123",
        method="POST",
        json=payload,
    ).respond_with_data(status=200)

    c = webhook_caller(httpserver.url_for("/webhook/123"))
    response = PredictionResponse(**payload)

    c(response)


def test_webhook_includes_user_agent(httpserver: HTTPServer):
    httpserver.expect_request(
        "/webhook/123",
        method="POST",
        json={
            "status": Status.PROCESSING,
            "output": {"animal": "giraffe"},
            "input": {},
        },
    ).respond_with_data(status=200)

    c = webhook_caller(httpserver.url_for("/webhook/123"))

    payload = {
        "status": Status.PROCESSING,
        "output": {"animal": "giraffe"},
        "input": {},
    }
    response = PredictionResponse(**payload)

    c(response)

    assert len(httpserver.log) == 1
    user_agent = httpserver.log[0].headers["user-agent"]
    assert user_agent.startswith("cog-worker/")


def test_webhook_caller_filtered_basic(httpserver: HTTPServer):
    events = WebhookEvent.default_events()

    httpserver.expect_request(
        "/webhook/123",
        method="POST",
        json={
            "status": Status.PROCESSING,
            "animal": "giraffe",
            "input": {},
        },
    ).respond_with_data(status=200)

    c = webhook_caller_filtered(httpserver.url_for("/webhook/123"), events)

    payload = {"status": Status.PROCESSING, "animal": "giraffe", "input": {}}
    response = PredictionResponse(**payload)

    c(response, WebhookEvent.LOGS)


def test_webhook_caller_filtered_omits_filtered_events(httpserver: HTTPServer):
    events = {WebhookEvent.COMPLETED}
    c = webhook_caller_filtered(httpserver.url_for("/webhook/123"), events)

    payload = {
        "status": Status.PROCESSING,
        "output": {"animal": "giraffe"},
        "input": {},
    }
    response = PredictionResponse(**payload)

    c(response, WebhookEvent.LOGS)

    # Assert that no request was made
    assert len(httpserver.log) == 0


def test_webhook_caller_connection_errors(httpserver: HTTPServer):
    httpserver.expect_request(
        "/webhook/123",
        method="POST",
    ).respond_with_data(status=200)

    payload = {
        "status": Status.PROCESSING,
        "output": {"animal": "giraffe"},
        "input": {},
    }
    response = PredictionResponse(**payload)

    c = webhook_caller(httpserver.url_for("/webhook/123"))
    # this should not raise an error
    c(response)
