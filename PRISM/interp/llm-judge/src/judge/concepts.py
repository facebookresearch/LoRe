"""Load and validate the concept configuration (config/concepts.py)."""

from __future__ import annotations

from config.concepts import CONCEPT_SET_VERSION, CONCEPTS, Concept

__all__ = [
    "Concept",
    "CONCEPT_SET_VERSION",
    "ConceptConfigError",
    "validate_concepts",
    "load_concepts",
    "concept_keys",
]


class ConceptConfigError(ValueError):
    """Raised when config/concepts.py is invalid."""


def validate_concepts(concepts: list[Concept]) -> list[Concept]:
    """Enforce: non-empty list, unique valid-identifier keys, non-empty label/definition."""
    if not concepts:
        raise ConceptConfigError("CONCEPTS is empty; define at least one concept")

    seen: set[str] = set()
    for i, c in enumerate(concepts):
        if not isinstance(c.key, str) or not c.key.isidentifier():
            raise ConceptConfigError(
                f"concept {i}: key {c.key!r} is not a valid Python identifier"
            )
        if c.key in seen:
            raise ConceptConfigError(f"concept {i}: duplicate key {c.key!r}")
        seen.add(c.key)
        if not c.label or not c.label.strip():
            raise ConceptConfigError(f"concept {i} ({c.key!r}): label must be non-empty")
        if not c.definition or not c.definition.strip():
            raise ConceptConfigError(f"concept {i} ({c.key!r}): definition must be non-empty")
    return concepts


def load_concepts() -> list[Concept]:
    """Return the validated CONCEPTS from config, raising ConceptConfigError if invalid."""
    return validate_concepts(list(CONCEPTS))


def concept_keys() -> list[str]:
    """Ordered list of concept keys (== CSV concept columns)."""
    return [c.key for c in load_concepts()]
