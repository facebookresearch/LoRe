"""Provider layer: a narrow seam between the runner and the vendor SDKs.

Everything above this line (runner, cache, writer) sees only the ``Judge`` protocol
and the ``JudgeResponse`` dataclass. Real providers are imported lazily inside
``get_provider`` so that running the test suite -- or a ``--dry-run`` -- never requires
the vendor SDKs or an API key to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JudgeResponse:
    """One raw completion from a judge model.

    ``text`` is the model's verbatim output (parsing happens downstream). ``error`` is
    set iff the API call itself failed; in that case ``text`` is empty.
    """

    text: str
    error: str | None = None


class Judge(Protocol):
    """Minimal contract a provider must satisfy to be usable by the runner."""

    @property
    def model_id(self) -> str: ...

    def complete(self, system: str, user: str, temperature: float | None) -> JudgeResponse: ...


def get_provider(provider: str, model_id: str) -> Judge:
    """Construct a provider by name. Imports the SDK-backed module lazily.

    ``provider`` is the family's backend id ("anthropic" | "google" | "openai"), not
    the CLI family alias. ``fake`` is always available for tests and offline use.
    """
    if provider == "fake":
        from .fake import FakeProvider

        return FakeProvider(model_id=model_id)
    if provider == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(model_id=model_id)
    if provider == "google":
        from .google import GoogleProvider

        return GoogleProvider(model_id=model_id)
    if provider == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(model_id=model_id)
    raise ValueError(f"unknown provider: {provider!r}")
