"""Prompt-builder tests. Pure string construction, no API calls."""

from __future__ import annotations

from judge.concepts import Concept
from judge.prompt import build_prompt


def _concepts():
    return [
        Concept(key="helpfulness", label="Helpfulness", definition="how useful"),
        Concept(key="fluency", label="Fluency", definition="how smooth"),
    ]


def test_build_prompt_returns_keys_in_order():
    p = build_prompt("q", "a", "b", _concepts())
    assert p.concept_keys == ("helpfulness", "fluency")


def test_answers_and_prompt_are_embedded():
    p = build_prompt("What is 2+2?", "four", "IV", _concepts())
    assert "What is 2+2?" in p.user
    assert "<answer_a>\nfour\n</answer_a>" in p.user
    assert "<answer_b>\nIV\n</answer_b>" in p.user


def test_every_concept_key_and_definition_rendered():
    p = build_prompt("q", "a", "b", _concepts())
    for c in _concepts():
        assert c.key in p.user
        assert c.definition in p.user


def test_output_format_lists_all_keys_and_example():
    p = build_prompt("q", "a", "b", _concepts())
    assert "helpfulness, fluency" in p.user
    assert '{"helpfulness": "tie", "fluency": "tie"}' in p.user


def test_verdict_is_directional_a_b_tie():
    p = build_prompt("q", "a", "b", _concepts())
    assert '"A"   = Answer A exhibits it more' in p.user
    assert '"B"   = Answer B exhibits it more' in p.user
    assert '"tie" = neither answer clearly exhibits it more' in p.user


def test_system_prompt_is_stable():
    p1 = build_prompt("q", "a", "b", _concepts())
    p2 = build_prompt("other", "x", "y", _concepts())
    assert p1.system == p2.system


def test_single_concept_prompt():
    p = build_prompt("q", "a", "b", [_concepts()[0]])
    assert p.concept_keys == ("helpfulness",)
    assert '{"helpfulness": "tie"}' in p.user
