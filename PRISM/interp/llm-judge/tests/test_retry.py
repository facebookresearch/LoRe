"""Retry/backoff tests. Offline; sleep is stubbed so there is no real delay."""

from __future__ import annotations

import pytest

from judge.providers._retry import call_with_retry, is_transient


class _Err(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


def test_is_transient_by_status_code():
    assert is_transient(_Err("boom", status_code=503))
    assert is_transient(_Err("boom", status_code=429))
    assert not is_transient(_Err("bad key", status_code=401))
    assert not is_transient(_Err("bad request", status_code=400))


def test_is_transient_by_message():
    assert is_transient(Exception("This model is experiencing high demand"))
    assert is_transient(Exception("503 UNAVAILABLE"))
    assert not is_transient(Exception("invalid x-api-key"))


def test_retries_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Err("high demand", status_code=503)
        return "done"

    out = call_with_retry(fn, max_attempts=5, sleep=lambda _: None)
    assert out == "done"
    assert calls["n"] == 3


def test_non_transient_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Err("invalid x-api-key", status_code=401)

    with pytest.raises(_Err):
        call_with_retry(fn, max_attempts=5, sleep=lambda _: None)
    assert calls["n"] == 1  # gave up immediately


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Err("high demand", status_code=503)

    with pytest.raises(_Err):
        call_with_retry(fn, max_attempts=4, sleep=lambda _: None)
    assert calls["n"] == 4
