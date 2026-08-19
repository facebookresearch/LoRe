"""FakeProvider tests. Offline, no network."""

from __future__ import annotations

from judge.concepts import Concept
from judge.parsing import Status, parse_scores
from judge.prompt import build_prompt
from judge.providers import get_provider
from judge.providers.fake import FakeProvider


def _prompt():
    concepts = [
        Concept(key="helpfulness", label="Helpfulness", definition="how useful"),
        Concept(key="fluency", label="Fluency", definition="how smooth"),
    ]
    return build_prompt("q", "a", "b", concepts)


def test_get_provider_returns_fake():
    p = get_provider("fake", "fake-1")
    assert isinstance(p, FakeProvider)
    assert p.model_id == "fake-1"


def test_default_response_is_parseable_all_half():
    p = _prompt()
    resp = FakeProvider().complete(p.system, p.user, 0.0)
    assert resp.error is None
    parsed = parse_scores(resp.text, p.concept_keys)
    assert parsed.status is Status.OK
    assert parsed.scores == {"helpfulness": 0.5, "fluency": 0.5}


def test_scripted_responder_is_used():
    p = _prompt()
    fake = FakeProvider(responder=lambda s, u, t: '{"helpfulness": "A", "fluency": "B"}')
    resp = fake.complete(p.system, p.user, 0.0)
    parsed = parse_scores(resp.text, p.concept_keys)
    assert parsed.scores == {"helpfulness": 0.0, "fluency": 1.0}


def test_error_is_surfaced():
    resp = FakeProvider(error="boom").complete("s", "u", 0.0)
    assert resp.error == "boom"
    assert resp.text == ""


def test_unknown_provider_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope", "x")
