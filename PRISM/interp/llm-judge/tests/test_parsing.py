"""Defensive-parsing tests. All offline, no API calls.

Verdicts are discrete tokens -- "A", "B", "tie" -- mapped to 0.0 / 1.0 / 0.5.
"""

from __future__ import annotations

import json

import pytest

from judge.parsing import Status, parse_scores

KEYS = ("helpfulness", "fluency")


def test_clean_object():
    r = parse_scores('{"helpfulness": "A", "fluency": "B"}', KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.0, "fluency": 1.0}
    assert r.detail == ""


def test_tie_maps_to_half():
    r = parse_scores('{"helpfulness": "tie", "fluency": "tie"}', KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.5, "fluency": 0.5}


def test_markdown_fenced_object_recovered():
    raw = '```json\n{"helpfulness": "A", "fluency": "B"}\n```'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.0, "fluency": 1.0}


def test_leading_prose_then_object_recovered():
    raw = 'Here are my verdicts:\n{"helpfulness": "B", "fluency": "A"}'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 1.0, "fluency": 0.0}


def test_tokens_are_case_and_whitespace_insensitive():
    r = parse_scores('{"helpfulness": "TIE", "fluency": " a "}', KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.5, "fluency": 0.0}


def test_brace_in_string_value_does_not_break_extraction():
    raw = 'note {weird}\n{"helpfulness": "A", "fluency": "B", "junk": "a } b"}'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert "ignored extra keys: junk" in r.detail


def test_reasoning_then_verdict_object_is_used():
    raw = (
        "helpfulness: A is more direct.\nfluency: B is choppy.\n"
        'Here is a stray {"note": "ignore me"} fragment.\n'
        '{"helpfulness": "A", "fluency": "B"}'
    )
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.0, "fluency": 1.0}


def test_last_complete_object_wins_over_earlier_complete_object():
    raw = '{"helpfulness": "A", "fluency": "A"}\n...revised...\n{"helpfulness": "B", "fluency": "tie"}'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 1.0, "fluency": 0.5}


def test_missing_key_is_validation_error():
    r = parse_scores('{"helpfulness": "A"}', KEYS)
    assert r.status is Status.VALIDATION_ERROR
    assert "fluency" in r.detail


def test_unknown_token_is_validation_error():
    r = parse_scores('{"helpfulness": "C", "fluency": "A"}', KEYS)
    assert r.status is Status.VALIDATION_ERROR
    assert r.scores is None


def test_numeric_value_is_validation_error():
    r = parse_scores('{"helpfulness": 0.5, "fluency": "A"}', KEYS)
    assert r.status is Status.VALIDATION_ERROR
    assert r.scores is None


def test_freeform_string_value_is_validation_error():
    r = parse_scores('{"helpfulness": "high", "fluency": "A"}', KEYS)
    assert r.status is Status.VALIDATION_ERROR


def test_boolean_value_rejected():
    r = parse_scores('{"helpfulness": true, "fluency": "A"}', KEYS)
    assert r.status is Status.VALIDATION_ERROR


def test_no_json_at_all_is_parse_error():
    r = parse_scores("I scored them in my head, trust me.", KEYS)
    assert r.status is Status.PARSE_ERROR
    assert r.scores is None


def test_empty_string_is_parse_error():
    r = parse_scores("", KEYS)
    assert r.status is Status.PARSE_ERROR


def test_refusal_detected():
    r = parse_scores("I'm unable to compare these responses.", KEYS)
    assert r.status is Status.REFUSAL


def test_refusal_marker_ignored_when_valid_json_present():
    raw = "I won't editorialize. {\"helpfulness\": \"A\", \"fluency\": \"B\"}"
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK


def test_extra_keys_ignored_with_note():
    raw = json.dumps({"helpfulness": "A", "fluency": "B", "safety": "tie"})
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert "safety" in r.detail
