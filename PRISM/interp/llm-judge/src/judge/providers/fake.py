"""Offline judge for tests and dry runs. No network, fully deterministic.

By default it returns a well-formed JSON object voting "tie" on every requested
concept (which maps to 0.5, the "equivalent" score), so the parsing and ordering
paths can be exercised without an API key. Tests can inject scripted responses or a
per-call delay to provoke out-of-order completion under concurrency.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

from . import JudgeResponse

_KEY_RE = re.compile(r"^\s*-\s*([a-z_][a-z0-9_]*)\s+\(", re.MULTILINE)


def _keys_from_prompt(user: str) -> list[str]:
    """Recover the concept keys from the rendered ``<concepts>`` block.

    The prompt renders each concept as ``- <key> (<label>): <definition>``; we pull the
    keys back out so the default response covers exactly what was asked, with no shared
    state between the prompt builder and this fake.
    """
    block = user
    if "<concepts>" in user and "</concepts>" in user:
        block = user.split("<concepts>", 1)[1].split("</concepts>", 1)[0]
    return _KEY_RE.findall(block)


class FakeProvider:
    """A ``Judge`` that answers offline.

    Parameters
    ----------
    model_id:
        Reported verbatim as ``model_id``.
    responder:
        Optional ``(system, user, temperature) -> str`` returning the raw text to
        emit. When omitted, returns a valid all-"tie" object over the prompt's keys.
    delay:
        Seconds to sleep before returning, to force out-of-order completion in
        concurrency tests.
    error:
        If set, every call returns a ``JudgeResponse`` carrying this error.
    """

    def __init__(
        self,
        model_id: str = "fake-1",
        responder: Callable[[str, str, float], str] | None = None,
        delay: float = 0.0,
        error: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._responder = responder
        self._delay = delay
        self._error = error

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, system: str, user: str, temperature: float) -> JudgeResponse:
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            return JudgeResponse(text="", error=self._error)
        if self._responder is not None:
            return JudgeResponse(text=self._responder(system, user, temperature))
        keys = _keys_from_prompt(user)
        return JudgeResponse(text=json.dumps({k: "tie" for k in keys}))
