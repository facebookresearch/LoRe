"""Concept-config validation tests. All offline, no API calls."""

from __future__ import annotations

import pytest

from judge.concepts import (
    CONCEPT_SET_VERSION,
    ConceptConfigError,
    Concept,
    concept_keys,
    load_concepts,
    validate_concepts,
)


def _c(key="helpfulness", label="Helpfulness", definition="def"):
    return Concept(key=key, label=label, definition=definition)


def test_shipped_config_is_valid():
    concepts = load_concepts()
    assert len(concepts) == 11
    assert concept_keys()[0] == "helpfulness"
    assert "verbosity" not in concept_keys()
    assert CONCEPT_SET_VERSION == "v3"


def test_empty_list_rejected():
    with pytest.raises(ConceptConfigError, match="empty"):
        validate_concepts([])


def test_duplicate_key_rejected():
    with pytest.raises(ConceptConfigError, match="duplicate key"):
        validate_concepts([_c(key="dup"), _c(key="dup")])


def test_non_identifier_key_rejected():
    with pytest.raises(ConceptConfigError, match="not a valid Python identifier"):
        validate_concepts([_c(key="not a key")])


def test_empty_definition_rejected():
    with pytest.raises(ConceptConfigError, match="definition must be non-empty"):
        validate_concepts([_c(definition="   ")])


def test_empty_label_rejected():
    with pytest.raises(ConceptConfigError, match="label must be non-empty"):
        validate_concepts([_c(label="")])
