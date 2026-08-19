"""Cache-key and on-disk cache tests. Offline."""

from __future__ import annotations

from judge.cache import Cache, compute_key


def _kw(**over):
    base = dict(
        model_id="m",
        temperature=0.0,
        concept_set_version="v1",
        order="forward",
        prompt="p",
        answer_a="a",
        answer_b="b",
    )
    base.update(over)
    return base


def test_key_is_deterministic():
    assert compute_key(**_kw()) == compute_key(**_kw())


def test_key_is_hex_sha256():
    k = compute_key(**_kw())
    assert len(k) == 64
    int(k, 16)  # raises if not hex


def test_each_input_field_changes_key():
    base = compute_key(**_kw())
    for field, value in [
        ("model_id", "other"),
        ("temperature", 0.7),
        ("concept_set_version", "v2"),
        ("order", "reverse"),
        ("prompt", "q"),
        ("answer_a", "x"),
        ("answer_b", "y"),
    ]:
        assert compute_key(**_kw(**{field: value})) != base, field


def test_roundtrip_put_get(tmp_path):
    c = Cache(tmp_path)
    c.put("k1", "raw output", None)
    hit = c.get("k1")
    assert hit is not None
    assert hit.raw_text == "raw output"
    assert hit.error is None


def test_miss_returns_none(tmp_path):
    assert Cache(tmp_path).get("nope") is None


def test_disabled_cache_is_noop(tmp_path):
    c = Cache(tmp_path, enabled=False)
    c.put("k", "x", None)
    assert c.get("k") is None


def test_api_error_is_not_cached(tmp_path):
    c = Cache(tmp_path)
    c.put("k", "", "boom")
    assert c.get("k") is None
