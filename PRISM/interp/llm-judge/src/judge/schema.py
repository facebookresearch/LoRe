"""Input schema and validation for the pairwise LLM-judge harness.

The entire input file is validated up front (`load_items`) so we never fire a
single API call against a malformed file. Every failure names the offending item
index and gives a clear reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REQUIRED_STR_FIELDS = ("prompt", "answer_a", "answer_b")
_KNOWN_FIELDS = ("id", "prompt", "answer_a", "answer_b")


class InputValidationError(ValueError):
    """Raised when the input file is malformed. Message names the offending index."""


@dataclass(frozen=True)
class InputItem:
    """One validated input row. `extra` preserves any unknown keys for the sidecar."""

    id: str
    prompt: str
    answer_a: str
    answer_b: str
    extra: dict[str, Any] = field(default_factory=dict)


def _validate_one(index: int, raw: Any) -> tuple[str | None, str, str, str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise InputValidationError(
            f"item {index}: expected a JSON object, got {type(raw).__name__}"
        )

    for key in _REQUIRED_STR_FIELDS:
        if key not in raw:
            raise InputValidationError(f"item {index}: missing required field '{key}'")
        value = raw[key]
        if not isinstance(value, str):
            raise InputValidationError(
                f"item {index}: field '{key}' must be a string, got {type(value).__name__}"
            )
        if value.strip() == "":
            raise InputValidationError(f"item {index}: field '{key}' must be a non-empty string")

    explicit_id = raw.get("id")
    if explicit_id is not None and (not isinstance(explicit_id, str) or explicit_id.strip() == ""):
        raise InputValidationError(
            f"item {index}: 'id' must be a non-empty string when present"
        )

    extra = {k: v for k, v in raw.items() if k not in _KNOWN_FIELDS}
    return explicit_id, raw["prompt"], raw["answer_a"], raw["answer_b"], extra


def parse_items(data: Any) -> list[InputItem]:
    """Validate an already-parsed JSON value into a list of InputItem."""
    if not isinstance(data, list):
        raise InputValidationError(
            f"top-level JSON must be an array of items, got {type(data).__name__}"
        )
    if len(data) == 0:
        raise InputValidationError("input file contains no items")

    items: list[InputItem] = []
    seen_ids: dict[str, int] = {}
    for index, raw in enumerate(data):
        explicit_id, prompt, answer_a, answer_b, extra = _validate_one(index, raw)
        item_id = explicit_id if explicit_id is not None else f"item_{index:06d}"
        if item_id in seen_ids:
            what = "duplicate id" if explicit_id is not None else "auto-assigned id collides"
            raise InputValidationError(
                f"item {index}: {what} '{item_id}' (first seen at item {seen_ids[item_id]})"
            )
        seen_ids[item_id] = index
        items.append(
            InputItem(id=item_id, prompt=prompt, answer_a=answer_a, answer_b=answer_b, extra=extra)
        )
    return items


def load_items(path: str | Path) -> list[InputItem]:
    """Read and fully validate an input JSON file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise InputValidationError(f"cannot read input file '{path}': {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"input file '{path}' is not valid JSON: {exc}") from exc
    return parse_items(data)
