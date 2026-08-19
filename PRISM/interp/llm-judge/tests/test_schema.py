"""Input validation and id-assignment tests. All offline, no API calls."""

from __future__ import annotations

import json

import pytest

from judge.schema import InputItem, InputValidationError, load_items, parse_items


def _item(**over):
    base = {"prompt": "p", "answer_a": "a", "answer_b": "b"}
    base.update(over)
    return base


def test_valid_minimal_item():
    items = parse_items([_item()])
    assert items == [InputItem(id="item_000000", prompt="p", answer_a="a", answer_b="b", extra={})]


def test_id_auto_assignment_is_positional():
    items = parse_items([_item(), _item(), _item()])
    assert [it.id for it in items] == ["item_000000", "item_000001", "item_000002"]


def test_explicit_id_preserved():
    items = parse_items([_item(id="keep-me")])
    assert items[0].id == "keep-me"


def test_duplicate_explicit_id_is_error():
    with pytest.raises(InputValidationError, match="duplicate id 'dup'"):
        parse_items([_item(id="dup"), _item(id="dup")])


@pytest.mark.parametrize("field", ["prompt", "answer_a", "answer_b"])
def test_missing_required_field(field):
    raw = _item()
    del raw[field]
    with pytest.raises(InputValidationError, match=f"missing required field '{field}'"):
        parse_items([raw])


@pytest.mark.parametrize("field", ["prompt", "answer_a", "answer_b"])
def test_empty_string_field(field):
    with pytest.raises(InputValidationError, match="non-empty string"):
        parse_items([_item(**{field: "   "})])


def test_non_string_field():
    with pytest.raises(InputValidationError, match="must be a string"):
        parse_items([_item(answer_a=123)])


def test_extra_keys_preserved_not_rejected():
    items = parse_items([_item(concept="helpfulness", note="x")])
    assert items[0].extra == {"concept": "helpfulness", "note": "x"}


def test_top_level_must_be_array():
    with pytest.raises(InputValidationError, match="must be an array"):
        parse_items({"prompt": "p"})


def test_empty_array_is_error():
    with pytest.raises(InputValidationError, match="no items"):
        parse_items([])


def test_item_not_object():
    with pytest.raises(InputValidationError, match="item 0: expected a JSON object"):
        parse_items(["just a string"])


def test_empty_id_is_error():
    with pytest.raises(InputValidationError, match="'id' must be a non-empty string"):
        parse_items([_item(id="  ")])


def test_load_items_from_file(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([_item(), _item(id="second")]), encoding="utf-8")
    items = load_items(p)
    assert [it.id for it in items] == ["item_000000", "second"]


def test_load_items_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputValidationError, match="not valid JSON"):
        load_items(p)
