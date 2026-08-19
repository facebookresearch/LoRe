"""Writer tests: file naming, input-ordered rows, NaN cells, JSONL audit record."""

from __future__ import annotations

import csv
import json

from judge.cache import Cache
from judge.concepts import Concept
from judge.providers.fake import FakeProvider
from judge.runner import run_model
from judge.schema import InputItem
from judge.writer import output_paths, write_results

CONCEPTS = [
    Concept(key="helpfulness", label="Helpfulness", definition="how useful"),
    Concept(key="fluency", label="Fluency", definition="how smooth"),
]


def _results(provider, items):
    return run_model(
        provider,
        items,
        CONCEPTS,
        temperature=0.0,
        concept_set_version="v1",
        cache=Cache("", enabled=False),
    )


def _items(n):
    return [InputItem(id=f"id{i}", prompt=f"p{i}", answer_a="A", answer_b="B") for i in range(n)]


def test_output_paths_naming():
    paths = output_paths("out", "contrastive_pairs_sample", "claude")
    assert paths.judgments.name == "contrastive_pairs_sample__claude__judgments.csv"
    assert paths.disagreement.name == "contrastive_pairs_sample__claude__disagreement.csv"
    assert paths.raw.name == "contrastive_pairs_sample__claude__raw.jsonl"


def test_disagreement_csv_shape_and_order(tmp_path):
    items = _items(3)
    results = _results(FakeProvider(), items)
    paths = write_results(
        tmp_path, "stem", "claude", results, CONCEPTS,
        model_id="fake-1", temperature=0.0, concept_set_version="v1",
    )
    rows = list(csv.reader(paths.disagreement.open(encoding="utf-8")))
    assert rows[0] == ["id", "helpfulness", "fluency"]  # id-keyed, no answer text
    assert [r[0] for r in rows[1:]] == ["id0", "id1", "id2"]
    assert rows[1][1] == "0.0000"  # consistent fake -> no position disagreement


def test_judgments_csv_shape_and_order(tmp_path):
    items = _items(3)
    results = _results(FakeProvider(), items)
    paths = write_results(
        tmp_path, "stem", "claude", results, CONCEPTS,
        model_id="fake-1", temperature=0.0, concept_set_version="v1",
    )
    rows = list(csv.reader(paths.judgments.open(encoding="utf-8")))
    assert rows[0] == ["id", "prompt", "answer_a", "answer_b", "helpfulness", "fluency"]
    assert [r[0] for r in rows[1:]] == ["id0", "id1", "id2"]
    assert rows[1][1:4] == ["p0", "A", "B"]  # prompt + both answers are shown
    assert rows[1][4] == "0.5000"  # default fake -> 0.5 both passes


def test_nan_becomes_empty_cell(tmp_path):
    items = _items(1)
    results = _results(FakeProvider(error="boom"), items)  # API error -> NaN
    paths = write_results(
        tmp_path, "stem", "claude", results, CONCEPTS,
        model_id="fake-1", temperature=0.0, concept_set_version="v1",
    )
    rows = list(csv.reader(paths.judgments.open(encoding="utf-8")))
    assert rows[1] == ["id0", "p0", "A", "B", "", ""]


def test_raw_jsonl_has_one_record_per_item_with_audit_fields(tmp_path):
    items = _items(2)
    results = _results(FakeProvider(), items)
    paths = write_results(
        tmp_path, "stem", "gemini", results, CONCEPTS,
        model_id="gem-1", temperature=0.3, concept_set_version="v1",
    )
    lines = paths.raw.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["id"] == "id0"
    assert rec["family"] == "gemini"
    assert rec["model_id"] == "gem-1"
    assert rec["forward"]["status"] == "ok"
    assert rec["reverse"]["order"] == "reverse"
    assert set(rec["final"]) == {"helpfulness", "fluency"}
    assert "raw_text" in rec["forward"]
