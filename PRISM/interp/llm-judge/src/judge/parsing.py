"""Defensive parsing of a judge model's raw text into per-concept scores.

Models are told to emit a bare JSON object mapping each concept to a discrete verdict
-- "A", "B", or "tie" -- but they misbehave: markdown fences, leading prose, a refusal,
a trailing comma. We try hard to recover a valid object without ever *guessing* a
verdict -- if a concept is missing or its value is not one of the three tokens we fail
the parse rather than invent one.

The verdict is discrete on purpose: a numeric 0-to-1 scale invited the model to read the
endpoints as an intensity ("1.0 = very confident") rather than a direction ("1.0 = B"),
which flipped scores under position swap. A/B/tie has no magnitude reading. We map
A -> 0.0, B -> 1.0, tie -> 0.5 so downstream folding/averaging is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    OK = "ok"
    PARSE_ERROR = "parse_error"          # could not recover a JSON object at all
    VALIDATION_ERROR = "validation_error"  # got JSON, but keys/values are wrong
    REFUSAL = "refusal"                  # model declined to answer
    API_ERROR = "api_error"              # the API call itself failed (set by the runner)


@dataclass(frozen=True)
class ParseResult:
    status: Status
    scores: dict[str, float] | None  # populated iff status is OK
    detail: str                      # human-readable reason when not OK; "" when OK


# Substrings that, in the ABSENCE of any recoverable JSON, signal a refusal rather
# than a mangled answer. Kept deliberately small and specific to avoid false hits.
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
)

# The three allowed verdict tokens, mapped to the directional score the rest of the
# pipeline expects. Matched case-insensitively after stripping surrounding whitespace.
_VERDICT_MAP = {"a": 0.0, "b": 1.0, "tie": 0.5}


def _iter_json_spans(text: str):
    """Yield every balanced ``{...}`` span in ``text``, left to right.

    Brace-counting that respects JSON string literals and escapes, so a ``}`` inside
    a quoted value does not close the object early. Yielding all top-level spans (not
    just the first) lets the caller skip an earlier ``{...}`` that is not valid JSON.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escaped = False
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    break
            j += 1
        i = j + 1


def _iter_objects(raw_text: str):
    """Yield every JSON dict recoverable from ``raw_text``, left to right.

    The whole string is tried first (the clean case), then each balanced ``{...}``
    span. When the judge reasons before answering, its reply holds prose and possibly
    stray braces followed by the real verdict object, so callers pick among these
    candidates rather than trusting the first hit.
    """
    stripped = raw_text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            yield obj
    except json.JSONDecodeError:
        pass

    for span in _iter_json_spans(stripped):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_scores(raw_text: str, concept_keys: tuple[str, ...]) -> ParseResult:
    """Parse ``raw_text`` into a score for every key in ``concept_keys``.

    Success requires exactly the expected keys to be present, each mapped to a verdict
    token -- "A", "B", or "tie" (case-insensitive) -- which becomes 0.0 / 1.0 / 0.5.
    Extra keys are ignored (with a note); missing keys or values that are not one of the
    three tokens are validation errors. If no JSON object can be recovered, an
    empty/refusal-looking response maps to REFUSAL, otherwise PARSE_ERROR.

    The judge reasons before it answers, so the reply can contain several ``{...}``
    fragments. We take the LAST object that carries every expected key as the verdict;
    if none does, the last object seen drives a precise missing-keys error.
    """
    candidates = list(_iter_objects(raw_text))
    if not candidates:
        lowered = raw_text.lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return ParseResult(Status.REFUSAL, None, "model refused to answer")
        return ParseResult(Status.PARSE_ERROR, None, "no JSON object found in response")

    obj = None
    for cand in candidates:
        if all(k in cand for k in concept_keys):
            obj = cand  # keep going: the verdict is the last complete object
    if obj is None:
        obj = candidates[-1]

    missing = [k for k in concept_keys if k not in obj]
    if missing:
        return ParseResult(
            Status.VALIDATION_ERROR, None, f"missing keys: {', '.join(missing)}"
        )

    scores: dict[str, float] = {}
    for k in concept_keys:
        v = obj[k]
        if not isinstance(v, str):
            return ParseResult(
                Status.VALIDATION_ERROR, None, f"verdict for '{k}' is not a string: {v!r}"
            )
        token = v.strip().lower()
        if token not in _VERDICT_MAP:
            return ParseResult(
                Status.VALIDATION_ERROR,
                None,
                f"verdict for '{k}' is not one of A/B/tie: {v!r}",
            )
        scores[k] = _VERDICT_MAP[token]

    extra = [k for k in obj if k not in concept_keys]
    detail = f"ignored extra keys: {', '.join(extra)}" if extra else ""
    return ParseResult(Status.OK, scores, detail)
