"""Write a model's results to the three per-run artifacts.

For input stem ``S`` and family alias ``F`` (claude | gemini | chatgpt):
  * ``S__F__judgments.csv``    -- the deliverable table: one self-describing row per
    input item (prompt + both answers), then one column per concept holding the final
    position-debiased score.
  * ``S__F__disagreement.csv`` -- diagnostic grid keyed by item id: |forward - reverse|
    per concept, i.e. how position-sensitive the judge was. High = distrust that cell.
  * ``S__F__raw.jsonl``        -- full audit record: both raw completions, both parses,
    and the folded scores.

All three are written in input order. A NaN score (a pass that failed to parse) is an
empty cell in the CSVs; the JSONL preserves the raw text so you can see exactly why.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .concepts import Concept
from .runner import ItemResult


@dataclass(frozen=True)
class OutputPaths:
    judgments: Path
    disagreement: Path
    raw: Path


def output_paths(output_dir: str | Path, input_stem: str, family: str) -> OutputPaths:
    base = Path(output_dir)
    prefix = f"{input_stem}__{family}"
    return OutputPaths(
        judgments=base / f"{prefix}__judgments.csv",
        disagreement=base / f"{prefix}__disagreement.csv",
        raw=base / f"{prefix}__raw.jsonl",
    )


def _fmt(x: float) -> str:
    return "" if math.isnan(x) else f"{x:.4f}"


def _write_judgments_csv(
    path: Path, results: list[ItemResult], concepts: list[Concept]
) -> None:
    keys = [c.key for c in concepts]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "prompt", "answer_a", "answer_b", *keys])
        for r in results:
            writer.writerow(
                [
                    r.item.id,
                    r.item.prompt,
                    r.item.answer_a,
                    r.item.answer_b,
                    *[_fmt(r.final[k]) for k in keys],
                ]
            )


def _write_disagreement_csv(
    path: Path, results: list[ItemResult], concepts: list[Concept]
) -> None:
    keys = [c.key for c in concepts]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", *keys])
        for r in results:
            writer.writerow([r.item.id, *[_fmt(r.disagreement[k]) for k in keys]])


def _pass_record(pr) -> dict:
    return {
        "order": pr.order,
        "status": pr.status.value,
        "cached": pr.cached,
        "detail": pr.detail,
        "scores": pr.scores,
        "raw_text": pr.raw_text,
    }


def _write_raw_jsonl(
    path: Path,
    results: list[ItemResult],
    concepts: list[Concept],
    *,
    model_id: str,
    family: str,
    temperature: float,
    concept_set_version: str,
) -> None:
    keys = [c.key for c in concepts]
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            record = {
                "id": r.item.id,
                "family": family,
                "model_id": model_id,
                "temperature": temperature,
                "concept_set_version": concept_set_version,
                "prompt": r.item.prompt,
                "answer_a": r.item.answer_a,
                "answer_b": r.item.answer_b,
                "extra": r.item.extra,
                "forward": _pass_record(r.forward),
                "reverse": _pass_record(r.reverse),
                "final": {k: None if math.isnan(r.final[k]) else r.final[k] for k in keys},
                "disagreement": {
                    k: None if math.isnan(r.disagreement[k]) else r.disagreement[k] for k in keys
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_results(
    output_dir: str | Path,
    input_stem: str,
    family: str,
    results: list[ItemResult],
    concepts: list[Concept],
    *,
    model_id: str,
    temperature: float,
    concept_set_version: str,
) -> OutputPaths:
    """Write all three artifacts for one model run and return their paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, input_stem, family)
    _write_judgments_csv(paths.judgments, results, concepts)
    _write_disagreement_csv(paths.disagreement, results, concepts)
    _write_raw_jsonl(
        paths.raw,
        results,
        concepts,
        model_id=model_id,
        family=family,
        temperature=temperature,
        concept_set_version=concept_set_version,
    )
    return paths
