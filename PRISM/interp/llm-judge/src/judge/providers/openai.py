"""OpenAI (ChatGPT) judge provider. SDK imported lazily so tests never need it."""

from __future__ import annotations

import os

from . import JudgeResponse
from ._retry import call_with_retry


class OpenAIProvider:
    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _client_or_raise(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, temperature: float | None) -> JudgeResponse:
        try:
            client = self._client_or_raise()
            kwargs = dict(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = call_with_retry(lambda: client.chat.completions.create(**kwargs))
            return JudgeResponse(text=resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 -- any failure becomes an api_error row
            return JudgeResponse(text="", error=f"{type(exc).__name__}: {exc}")
