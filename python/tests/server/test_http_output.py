import base64
import io

import pytest
from pytest_httpserver import HTTPServer
from cog.types import PYDANTIC_V2
from responses.matchers import multipart_matcher

from .conftest import uses_predictor, uses_predictor_with_client_options


@uses_predictor("output_wrong_type")
def test_return_wrong_type(client):
    resp = client.post("/predictions")
    assert resp.status_code == 500


@uses_predictor("output_file")
def test_output_file(client, match):
    res = client.post("/predictions")
    assert res.status_code == 200
    assert res.json() == match(
        {
            "status": "succeeded",
            "output": "data:application/octet-stream;base64,aGVsbG8=",  # hello
        }
    )


@uses_predictor("output_file_named")
def test_output_file_to_http(client, match, httpserver: HTTPServer):
    httpserver.expect_request(
        "/upload/foo.txt", method="PUT", data={"file": ("foo.txt", b"hello")}
    ).respond_with_data(status=201)

    upload_url = httpserver.url_for("/upload/")

    res = client.post("/predictions", json={"output_file_prefix": upload_url})
    assert res.json() == match(
        {
            "status": "succeeded",
            "output": f"{upload_url}foo.txt",
        }
    )
    assert res.status_code == 200

    # Verify that the request was received by the server
    assert httpserver.log
    assert httpserver.log[0].data["file"][0] == "foo.txt"
    assert httpserver.log[0].data["file"][1] == b"hello"


@uses_predictor_with_client_options("output_file_named", upload_url="https://dontuseme")
def test_output_file_to_http_with_upload_url_specified(client, match, httpserver: HTTPServer):
    httpserver.expect_request(
        "/upload/foo.txt",
        method="PUT",
        data={"file": ("foo.txt", b"hello")},
    ).respond_with_data(status=201)

    res = client.post(
        "/predictions", json={"output_file_prefix": httpserver.url_for("/upload/")}
    )
    assert res.json() == match(
        {
            "status": "succeeded",
            "output": f"{httpserver.url_for('/upload/')}foo.txt",
        }
    )
    assert res.status_code == 200


@uses_predictor("output_path_image")
def test_output_path(client):
    res = client.post("/predictions")
    assert res.status_code == 200
    header, b64data = res.json()["output"].split(",", 1)
    # need both image/bmp and image/x-ms-bmp until https://bugs.python.org/issue44211 is fixed
    assert header in ["data:image/bmp;base64", "data:image/x-ms-bmp;base64"]
    assert len(base64.b64decode(b64data)) == 195894


@uses_predictor("output_path_text")
def test_output_path_to_http(client, match, httpserver: HTTPServer):
    httpserver.expect_request(
        "/upload/file.txt",
        method="PUT",
        data={"file": ("file.txt", b"hello")},
    ).respond_with_data(status=201)

    res = client.post(
        "/predictions", json={"output_file_prefix": httpserver.url_for("/upload/")}
    )
    assert res.json() == match(
        {
            "status": "succeeded",
            "output": f"{httpserver.url_for('/upload/')}file.txt",
        }
    )
    assert res.status_code == 200


@uses_predictor("output_complex")
def test_complex_output(client, match):
    resp = client.post("/predictions")

    assert resp.status_code == 200

    json = resp.json()
    assert (
        json["output"]["file"] == "data:application/octet-stream;base64,aGVsbG8="
        or json["output"]["file"] == "data:text/plain;base64,aGVsbG8="
    )
    assert json["output"]["text"] == "hello"
    assert json["status"] == "succeeded"


@uses_predictor("output_iterator_complex")
def test_iterator_of_list_of_complex_output(client, match):
    resp = client.post("/predictions")
    assert resp.json() == match(
        {
            "output": [[{"text": "hello"}]],
            "status": "succeeded",
        }
    )
    assert resp.status_code == 200


if not PYDANTIC_V2:

    @uses_predictor("output_numpy")
    def test_json_output_numpy(client, match):
        resp = client.post("/predictions")
        assert resp.status_code == 200
        assert resp.json() == match({"output": 1.0, "status": "succeeded"})
