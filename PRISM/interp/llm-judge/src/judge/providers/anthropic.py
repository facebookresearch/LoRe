"""Anthropic (Claude) judge provider. SDK imported lazily so tests never need it."""

from __future__ import annotations

import os

from . import JudgeResponse
from ._retry import call_with_retry

_MAX_TOKENS = 1024


class AnthropicProvider:
    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _client_or_raise(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, temperature: float | None) -> JudgeResponse:
        try:
            client = self._client_or_raise()
            kwargs = dict(
                model=self._model_id,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            msg = call_with_retry(lambda: client.messages.create(**kwargs))
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )
            return JudgeResponse(text=text)
        except Exception as exc:  # noqa: BLE001 -- any failure becomes an api_error row
            return JudgeResponse(text="", error=f"{type(exc).__name__}: {exc}")
