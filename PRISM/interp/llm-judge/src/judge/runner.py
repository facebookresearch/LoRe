"""Orchestrates a single judge model over all items, both position orders.

Position-bias control: every item is judged twice.
  * forward -- answer_a in slot "Answer A", answer_b in slot "Answer B"
  * reverse -- answers swapped

The judge's score is directional in [0, 1] where 0 means the slot-A answer exhibits
the concept more and 1 means slot-B does. To put both passes in the same frame ("how
much does the ORIGINAL answer_b exhibit it"):

    fwd_b   = s_forward            # slot-B is answer_b already
    rev_b   = 1 - s_reverse        # slot-B is answer_a, so flip
    final        = (fwd_b + rev_b) / 2
    disagreement = abs(fwd_b - rev_b)

If either pass fails to yield a valid score, final and disagreement are NaN for every
concept -- we never average a real score against a guess.

Calls run concurrently, but results are reassembled in input order, so the output is
deterministic regardless of which completion lands first.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import cache as cache_mod
from .concepts import Concept
from .parsing import Status, parse_scores
from .prompt import build_prompt
from .providers import Judge
from .schema import InputItem

FORWARD = "forward"
REVERSE = "reverse"


@dataclass(frozen=True)
class PassResult:
    order: str
    status: Status
    raw_text: str
    scores: dict[str, float] | None  # slot-directional scores as returned by the judge
    detail: str
    cached: bool


@dataclass(frozen=True)
class ItemResult:
    item: InputItem
    forward: PassResult
    reverse: PassResult
    final: dict[str, float] = field(default_factory=dict)         # concept -> score or NaN
    disagreement: dict[str, float] = field(default_factory=dict)  # concept -> value or NaN


def _run_pass(
    provider: Judge,
    item: InputItem,
    order: str,
    concepts: list[Concept],
    temperature: float | None,
    concept_set_version: str,
    cache: cache_mod.Cache,
) -> PassResult:
    if order == FORWARD:
        slot_a, slot_b = item.answer_a, item.answer_b
    else:
        slot_a, slot_b = item.answer_b, item.answer_a

    keys = tuple(c.key for c in concepts)
    key = cache_mod.compute_key(
        model_id=provider.model_id,
        temperature=temperature,
        concept_set_version=concept_set_version,
        order=order,
        prompt=item.prompt,
        answer_a=item.answer_a,
        answer_b=item.answer_b,
    )

    hit = cache.get(key)
    if hit is not None:
        cached = True
        raw_text, error = hit.raw_text, hit.error
    else:
        cached = False
        prompt = build_prompt(item.prompt, slot_a, slot_b, concepts)
        resp = provider.complete(prompt.system, prompt.user, temperature)
        raw_text, error = resp.text, resp.error
        cache.put(key, raw_text, error)

    if error is not None:
        return PassResult(order, Status.API_ERROR, raw_text, None, error, cached)

    parsed = parse_scores(raw_text, keys)
    return PassResult(order, parsed.status, raw_text, parsed.scores, parsed.detail, cached)


def _combine(
    forward: PassResult, reverse: PassResult, concepts: list[Concept]
) -> tuple[dict[str, float], dict[str, float]]:
    nan = math.nan
    if forward.status is not Status.OK or reverse.status is not Status.OK:
        final = {c.key: nan for c in concepts}
        return final, dict(final)

    final: dict[str, float] = {}
    disagreement: dict[str, float] = {}
    for c in concepts:
        fwd_b = forward.scores[c.key]
        rev_b = 1.0 - reverse.scores[c.key]
        final[c.key] = (fwd_b + rev_b) / 2.0
        disagreement[c.key] = abs(fwd_b - rev_b)
    return final, disagreement


def run_model(
    provider: Judge,
    items: list[InputItem],
    concepts: list[Concept],
    *,
    temperature: float | None,
    concept_set_version: str,
    cache: cache_mod.Cache,
    max_workers: int = 8,
) -> list[ItemResult]:
    """Judge every item with ``provider`` in both orders; return input-ordered results."""
    tasks = [(idx, order) for idx in range(len(items)) for order in (FORWARD, REVERSE)]

    def work(task: tuple[int, str]) -> tuple[int, str, PassResult]:
        idx, order = task
        pr = _run_pass(
            provider, items[idx], order, concepts, temperature, concept_set_version, cache
        )
        return idx, order, pr

    passes: dict[int, dict[str, PassResult]] = {i: {} for i in range(len(items))}
    workers = max(1, min(max_workers, len(tasks))) if tasks else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, order, pr in pool.map(work, tasks):
            passes[idx][order] = pr

    results: list[ItemResult] = []
    for idx in range(len(items)):  # input order, independent of completion order
        fwd = passes[idx][FORWARD]
        rev = passes[idx][REVERSE]
        final, disagreement = _combine(fwd, rev, concepts)
        results.append(ItemResult(items[idx], fwd, rev, final, disagreement))
    return results
