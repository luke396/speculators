import time
from typing import Any, TypedDict

import httpx
import torch

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)

RETRY_BACKOFF_BASE = 2


class InvalidResponseError(Exception):
    pass


GeneratedHiddenStates = dict[str, torch.Tensor]


class ClientItem(TypedDict):
    input_ids: list[int]


def _generate_url(endpoint: str) -> str:
    base_url = endpoint.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return f"{base_url}/generate"


def _generate_payload(token_ids: list[int]) -> dict[str, Any]:
    return {
        "input_ids": token_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_hidden_states": True,
        "log_metrics": False,
    }


def _parse_hidden_states(
    payload: dict[str, Any], token_ids: list[int]
) -> GeneratedHiddenStates:
    meta_info = payload.get("meta_info")
    if meta_info is None:
        raise InvalidResponseError("Response missing meta_info")
    hidden_states_chunks = meta_info.get("hidden_states")
    if hidden_states_chunks is None:
        raise InvalidResponseError("Response meta_info missing hidden_states")
    if not hidden_states_chunks:
        raise InvalidResponseError("Response hidden_states is empty")
    hidden_states = torch.as_tensor(hidden_states_chunks[0])
    if hidden_states.numel() == 0:
        raise InvalidResponseError("Response hidden_states is empty")
    if hidden_states.dim() != 2:  # noqa: PLR2004
        raise InvalidResponseError(
            "Response hidden_states must have shape [seq_len, hidden_size]"
        )
    if hidden_states.shape[0] != len(token_ids):
        raise InvalidResponseError(
            "Response hidden_states sequence length does not match prompt tokens "
            f"(got {hidden_states.shape[0]}, expected {len(token_ids)})"
        )
    if hidden_states.isnan().any():
        raise InvalidResponseError("Response hidden_states contains NaN")

    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "hidden_states": hidden_states,
    }


def generate_hidden_states(
    endpoint: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> GeneratedHiddenStates:
    """Request inline prompt hidden states from SGLang's native endpoint.

    A one-token decode makes the first returned hidden-state chunk the full
    prompt prefill states needed by training.
    """
    token_ids = client_item["input_ids"]

    payload: dict[str, Any]
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as http_client:
                response = http_client.post(
                    _generate_url(endpoint),
                    json=_generate_payload(token_ids),
                )
                response.raise_for_status()
                payload = response.json()
            break
        except httpx.TransportError:
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

    return _parse_hidden_states(payload, token_ids)
