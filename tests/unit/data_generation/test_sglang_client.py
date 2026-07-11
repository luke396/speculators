import json

import httpx
import pytest
import torch

from speculators.data_generation import sglang_client

_VALID_RESPONSE = {
    "meta_info": {
        "hidden_states": [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    }
}


class _Response:
    def __init__(self, payload=None):
        self.payload = _VALID_RESPONSE if payload is None else payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _install_sync_http(monkeypatch, respond):
    class FakeHTTPClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, *, json):
            return respond(url, self.timeout, json)

    monkeypatch.setattr(sglang_client.httpx, "Client", FakeHTTPClient)


def _generate(**kwargs):
    return sglang_client.generate_hidden_states(
        "http://127.0.0.1:8000/v1/",
        {"input_ids": [10, 11]},
        **kwargs,
    )


def test_generate_hidden_states_uses_sglang_native_generate(monkeypatch):
    calls = []

    def respond(url, timeout, payload):
        calls.append((url, timeout, payload))
        return _Response()

    _install_sync_http(monkeypatch, respond)

    result = _generate(timeout=7)

    assert calls == [
        (
            "http://127.0.0.1:8000/generate",
            7,
            {
                "input_ids": [10, 11],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_hidden_states": True,
                "log_metrics": False,
            },
        )
    ]
    assert torch.equal(result["token_ids"], torch.tensor([10, 11]))
    assert torch.equal(
        result["hidden_states"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


@pytest.mark.parametrize(
    ("response_payload", "message"),
    [
        ({}, "missing meta_info"),
        ({"meta_info": {}}, "missing hidden_states"),
        ({"meta_info": {"hidden_states": []}}, "hidden_states is empty"),
        ({"meta_info": {"hidden_states": [[]]}}, "hidden_states is empty"),
        (
            {"meta_info": {"hidden_states": [[1.0, 2.0]]}},
            r"shape \[seq_len, hidden_size\]",
        ),
        (
            {"meta_info": {"hidden_states": [[[1.0, 2.0]]]}},
            "sequence length does not match prompt tokens",
        ),
        (
            {"meta_info": {"hidden_states": [[[1.0, 2.0], [float("nan"), 4.0]]]}},
            "hidden_states contains NaN",
        ),
    ],
    ids=["meta", "field", "chunks", "tensor", "rank", "length", "nan"],
)
def test_generate_hidden_states_rejects_malformed_wire_payload(
    monkeypatch, response_payload, message
):
    _install_sync_http(
        monkeypatch,
        lambda _url, _timeout, _payload: _Response(response_payload),
    )

    with pytest.raises(sglang_client.InvalidResponseError, match=message):
        _generate()


def test_generate_hidden_states_retries_transport_errors(monkeypatch):
    attempts = 0

    def respond(url, _timeout, _payload):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError(
                "endpoint unavailable",
                request=httpx.Request("POST", url),
            )
        return _Response()

    _install_sync_http(monkeypatch, respond)
    monkeypatch.setattr(sglang_client, "RETRY_BACKOFF_BASE", 0)

    result = _generate(max_retries=2)

    assert attempts == 3
    assert torch.equal(result["token_ids"], torch.tensor([10, 11]))


def test_generate_hidden_states_does_not_retry_http_status_errors(monkeypatch):
    attempts = 0

    def respond(url, _timeout, _payload):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=httpx.Request("POST", url))

    _install_sync_http(monkeypatch, respond)

    with pytest.raises(httpx.HTTPStatusError):
        _generate(max_retries=3)

    assert attempts == 1


def test_generate_hidden_states_does_not_retry_invalid_json(monkeypatch):
    attempts = 0

    class InvalidJSONResponse(_Response):
        def json(self):
            raise json.JSONDecodeError("invalid JSON", "", 0)

    def respond(_url, _timeout, _payload):
        nonlocal attempts
        attempts += 1
        return InvalidJSONResponse()

    _install_sync_http(monkeypatch, respond)

    with pytest.raises(json.JSONDecodeError):
        _generate(max_retries=3)

    assert attempts == 1
