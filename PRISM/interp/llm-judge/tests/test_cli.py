"""CLI tests: model resolution and the offline --dry-run path (no keys, no SDKs)."""

from __future__ import annotations

import pytest

from config.models import UNSET_MODEL_ID
from judge import cli


def test_resolve_rejects_unknown_alias():
    with pytest.raises(SystemExit, match="unknown model alias"):
        cli._resolve_models(["nope"], allow_placeholder=False)


def test_resolve_rejects_placeholder_in_real_run(monkeypatch):
    monkeypatch.setitem(
        cli.JUDGE_MODELS, "claude", {"provider": "anthropic", "model_id": UNSET_MODEL_ID}
    )
    with pytest.raises(SystemExit, match="unset model id"):
        cli._resolve_models(["claude"], allow_placeholder=False)


def test_resolve_allows_placeholder_in_dry_run():
    resolved = cli._resolve_models(["gemini"], allow_placeholder=True)
    assert resolved[0][0] == "gemini"


def test_dry_run_prints_prompt_and_plan(capsys, tmp_path):
    inp = tmp_path / "sample.json"
    inp.write_text(
        '[{"id": "one", "prompt": "hi", "answer_a": "AA", "answer_b": "BB"}]',
        encoding="utf-8",
    )
    rc = cli.main([str(inp), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out
    assert "sample__claude__judgments.csv" in out
    assert "<answer_a>\nAA\n</answer_a>" in out  # rendered prompt for the first item
    assert "set version v3" in out
