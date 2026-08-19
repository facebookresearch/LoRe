"""Command-line entry point: ``python -m judge INPUT.json [options]``.

Runs every selected judge family over the input in both position orders and writes
three artifacts per family. ``--dry-run`` renders the prompt for the first item and
prints the run plan without touching any API or key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.models import DEFAULT_MODELS, JUDGE_MODELS, UNSET_MODEL_ID

from .cache import Cache
from .concepts import CONCEPT_SET_VERSION, load_concepts
from .parsing import Status
from .prompt import build_prompt
from .providers import get_provider
from .runner import run_model
from .schema import load_items
from .writer import output_paths, write_results

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CACHE_DIR = ".cache"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m judge", description=__doc__)
    p.add_argument("input", type=Path, help="input JSON file (array of prompt/answer pairs)")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        metavar="ALIAS",
        help=f"family aliases to run (default: {' '.join(DEFAULT_MODELS)})",
    )
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="render the prompt for the first item and print the plan; no API calls",
    )
    return p.parse_args(argv)


def _resolve_models(
    aliases: list[str], *, allow_placeholder: bool
) -> list[tuple[str, str, str, float | None]]:
    """Map aliases to (alias, provider, model_id, temperature), rejecting unknown ones."""
    resolved: list[tuple[str, str, str, float | None]] = []
    for alias in aliases:
        spec = JUDGE_MODELS.get(alias)
        if spec is None:
            raise SystemExit(
                f"unknown model alias {alias!r}; known: {', '.join(JUDGE_MODELS)}"
            )
        if spec["model_id"] == UNSET_MODEL_ID and not allow_placeholder:
            raise SystemExit(
                f"model alias {alias!r} has an unset model id in config/models.py; "
                "set a concrete model id before a real run"
            )
        resolved.append((alias, spec["provider"], spec["model_id"], spec.get("temperature")))
    return resolved


def _dry_run(args: argparse.Namespace) -> int:
    concepts = load_concepts()
    items = load_items(args.input)
    models = _resolve_models(args.models, allow_placeholder=True)
    stem = args.input.stem

    first = items[0]
    prompt = build_prompt(first.prompt, first.answer_a, first.answer_b, concepts)

    print("=== DRY RUN (no API calls) ===")
    print(f"input:            {args.input}  ({len(items)} items)")
    print(f"concepts:         {len(concepts)} (set version {CONCEPT_SET_VERSION})")
    print(f"cache:            {'disabled' if args.no_cache else args.cache_dir}")
    print("families & outputs:")
    for alias, provider, model_id, alias_temp in models:
        shown = model_id if model_id != UNSET_MODEL_ID else f"{UNSET_MODEL_ID} (must set before real run)"
        temp_shown = "omitted" if alias_temp is None else alias_temp
        paths = output_paths(args.output_dir, stem, alias)
        print(f"  - {alias} [{provider}: {shown}]  temperature={temp_shown}")
        print(f"      {paths.judgments.name}")
        print(f"      {paths.disagreement.name}")
        print(f"      {paths.raw.name}")
    print()
    print(f"--- rendered prompt for item '{first.id}' (forward order) ---")
    print("[SYSTEM]")
    print(prompt.system)
    print()
    print("[USER]")
    print(prompt.user)
    return 0


def _summarize(results) -> str:
    ok = sum(1 for r in results if r.forward.status is Status.OK and r.reverse.status is Status.OK)
    failed = len(results) - ok
    return f"{ok} ok, {failed} with a failed pass"


def _real_run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv()

    concepts = load_concepts()
    items = load_items(args.input)
    models = _resolve_models(args.models, allow_placeholder=False)
    stem = args.input.stem
    cache = Cache(args.cache_dir, enabled=not args.no_cache)

    for alias, provider_name, model_id, alias_temp in models:
        temperature = alias_temp
        provider = get_provider(provider_name, model_id)
        temp_shown = "omitted" if temperature is None else temperature
        print(
            f"[{alias}] judging {len(items)} items with {model_id} "
            f"(temperature={temp_shown}) ...",
            flush=True,
        )
        results = run_model(
            provider,
            items,
            concepts,
            temperature=temperature,
            concept_set_version=CONCEPT_SET_VERSION,
            cache=cache,
            max_workers=args.max_workers,
        )
        paths = write_results(
            args.output_dir,
            stem,
            alias,
            results,
            concepts,
            model_id=model_id,
            temperature=temperature,
            concept_set_version=CONCEPT_SET_VERSION,
        )
        print(f"[{alias}] {_summarize(results)}")
        print(f"[{alias}] wrote {paths.judgments}")
        print(f"[{alias}] wrote {paths.disagreement}")
        print(f"[{alias}] wrote {paths.raw}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        return _dry_run(args)
    return _real_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
