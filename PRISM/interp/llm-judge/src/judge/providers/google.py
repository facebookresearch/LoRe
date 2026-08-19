"""Google (Gemini) judge provider. Uses the google-genai SDK, imported lazily."""

from __future__ import annotations

import os

from . import JudgeResponse
from ._retry import call_with_retry


class GoogleProvider:
    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _client_or_raise(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, temperature: float | None) -> JudgeResponse:
        try:
            client = self._client_or_raise()
            from google.genai import types

            config_kwargs: dict = {"system_instruction": system}
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            resp = call_with_retry(
                lambda: client.models.generate_content(
                    model=self._model_id,
                    contents=user,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            )
            return JudgeResponse(text=resp.text or "")
        except Exception as exc:  # noqa: BLE001 -- any failure becomes an api_error row
            return JudgeResponse(text="", error=f"{type(exc).__name__}: {exc}")
