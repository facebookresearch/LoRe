"""Retry transient API failures with exponential backoff + jitter.

Judge APIs return transient errors under load -- HTTP 429 (rate limit) and 5xx
("experiencing high demand", "temporarily unavailable"). These are explicitly
"try again later", so we retry them a few times before giving up. Non-transient
errors (bad key, bad request, unknown model) are re-raised immediately -- retrying
them just wastes time and quota.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = (
    "high demand",
    "unavailable",
    "overloaded",
    "rate limit",
    "temporarily",
    "try again",
    "timeout",
    "timed out",
)


def is_transient(exc: Exception) -> bool:
    """True if ``exc`` looks like a load/availability blip worth retrying."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` and retry it on transient errors, backing off between tries.

    ``sleep`` is injectable so tests can run without real delays. On the final
    attempt (or any non-transient error) the exception propagates unchanged.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- classify, then re-raise or retry
            if attempt >= max_attempts or not is_transient(exc):
                raise
            backoff = min(max_delay, base_delay * 2 ** (attempt - 1))
            sleep(backoff * (0.5 + random.random()))  # full jitter around the backoff
    raise AssertionError("unreachable")  # pragma: no cover
