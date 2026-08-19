"""Judge prompt construction. Pure functions only -- no hidden state -- so the
prompt is unit-testable and the cache key over its inputs is meaningful.

The runner owns position swapping; this module always labels the two answers
"Answer A" (slot 1) and "Answer B" (slot 2) and never knows a swap happened.
"""

from __future__ import annotations

from dataclasses import dataclass

from .concepts import Concept

SYSTEM = (
    'You are an impartial evaluator comparing two AI assistant responses -- '
    '"Answer A" and "Answer B" -- to the same user prompt.\n\n'
    "Your job: for each concept you are given, decide WHICH answer exhibits MORE of that "
    'concept. Your verdict is one of exactly three choices -- "A", "B", or "tie" -- naming '
    "the answer that shows more of the concept, or a genuine tie when neither clearly does. "
    "This is a directional comparison between the two answers, not a quality rating of "
    "either answer on its own.\n\n"
    "Reason before you decide. First compare the two answers concept by concept, in a few "
    "short sentences, deciding which one exhibits more of each and why. Only after that "
    "reasoning do you commit to the verdicts.\n\n"
    "Judge every concept INDEPENDENTLY. Do not let an overall preference for one answer pull "
    "your verdicts on unrelated concepts in the same direction. Answer A may exhibit more of "
    "one concept while Answer B exhibits more of another.\n\n"
    "Judge the content only. The same pair may be shown to you in either order, so do NOT let "
    "the position of an answer, its length, or any names or self-identification in the text "
    "influence your verdicts. Be as objective as possible.\n\n"
    "End your reply with a single JSON object as the very last thing -- each concept key "
    'mapped to its verdict string ("A", "B", or "tie"). Put your reasoning before it, and '
    "write nothing after it."
)


@dataclass(frozen=True)
class JudgePrompt:
    system: str
    user: str
    concept_keys: tuple[str, ...]


def _render_concepts(concepts: list[Concept]) -> str:
    return "\n".join(f"- {c.key} ({c.label}): {c.definition}" for c in concepts)


def build_prompt(prompt: str, answer_a: str, answer_b: str, concepts: list[Concept]) -> JudgePrompt:
    """Build the judge prompt for one item over the given concepts.

    Pass all concepts for a joint call, or a single-element list for --per-concept.
    """
    keys = tuple(c.key for c in concepts)
    example = "{" + ", ".join(f'"{k}": "tie"' for k in keys) + "}"

    user = (
        "<user_prompt>\n"
        f"{prompt}\n"
        "</user_prompt>\n\n"
        "<answer_a>\n"
        f"{answer_a}\n"
        "</answer_a>\n\n"
        "<answer_b>\n"
        f"{answer_b}\n"
        "</answer_b>\n\n"
        "<concepts>\n"
        f"{_render_concepts(concepts)}\n"
        "</concepts>\n\n"
        "<verdict>\n"
        "For each concept, choose exactly one of three verdicts naming which answer exhibits "
        "MORE of that concept:\n"
        '  "A"   = Answer A exhibits it more\n'
        '  "B"   = Answer B exhibits it more\n'
        '  "tie" = neither answer clearly exhibits it more\n'
        "Use \"tie\" only for a genuine standoff, not as a way to avoid deciding. This is "
        "about WHICH answer, A or B, shows more of the concept -- not about how good either "
        "answer is.\n"
        "</verdict>\n\n"
        "<output_format>\n"
        "First, write a brief concept-by-concept comparison -- one short line per concept is "
        "plenty. Then, as the VERY LAST thing in your reply, output a single JSON object (no "
        "markdown code fences, and nothing after it) containing every one of these keys, each "
        'mapped to its verdict string ("A", "B", or "tie"):\n'
        f"{', '.join(keys)}\n"
        "Example of the required JSON shape (verdicts are illustrative only):\n"
        f"{example}\n"
        "</output_format>"
    )
    return JudgePrompt(system=SYSTEM, user=user, concept_keys=keys)
