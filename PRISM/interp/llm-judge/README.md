# Pairwise LLM-judge harness

Given a list of `(prompt, answer_a, answer_b)` triples, this tool asks one or more
judge models, for each of several **concepts**, *which* answer exhibits that concept
more — a discrete verdict of `A`, `B`, or `tie`. It is built for auditability and
reproducibility: every judgment is position-debiased, cached by content, and written
out with the raw model text so you can trace any number back to its source.

## What it does

For each item and each judge family, the harness makes **two** calls:

- **forward** — `answer_a` shown as "Answer A", `answer_b` as "Answer B"
- **reverse** — the two answers swapped

The judge returns, per concept, a verdict — `A`, `B`, or `tie` — which we map to `0`,
`1`, and `0.5`: `0` = the slot-A answer exhibits the concept more, `1` = the slot-B
answer does, `0.5` = a genuine tie. A discrete verdict is deliberate: a continuous
`0`-to-`1` number let the judge read an endpoint as an intensity ("very confident")
rather than a direction ("Answer B"), which flipped scores under the swap. We fold the
two passes into the frame "how much does the original `answer_b` exhibit it":

```
fwd_b = s_forward           # slot-B is answer_b
rev_b = 1 - s_reverse       # slot-B is answer_a, so flip
final        = (fwd_b + rev_b) / 2      # position-debiased score
disagreement = |fwd_b - rev_b|          # how position-sensitive the judge was
```

If either pass fails to produce a valid score (bad JSON, an unrecognized verdict,
refusal, API error), `final` and `disagreement` are `NaN` for every concept on that
item — we never average a real score against a guess.

Because verdicts are discrete, `disagreement` takes only three values: `0` (both
orders agreed), `0.5` (a tie one way and a pick the other — mild), and `1.0` (the
judge named the same slot both ways — a real position flip worth inspecting).

## Install

Python 3.11+.

```
pip install -e .
```

This pulls in `anthropic`, `google-genai`, `openai`, `python-dotenv`, and `pytest`.

## Configure

**Concepts** — edit `config/concepts.py`. Each `Concept` becomes one column in the
CSVs; its `definition` is shown verbatim to the judge. Bump `CONCEPT_SET_VERSION`
whenever you change the set or any definition — it is part of the cache key, so
bumping it invalidates stale cached judgments.

**Models** — edit `config/models.py`. The three families ship as:

| alias   | provider  | model_id          |
| ------- | --------- | ----------------- |
| claude  | anthropic | `claude-opus-4-8` |
| gemini  | google    | `gemini-3.5-flash`  |
| chatgpt | openai    | `gpt-5.6-sol`     |

These are each family's current flagship (comparable tier to Opus 4.8). To change one,
drop in another concrete model id; a real run refuses any alias left at the
`UNSET_MODEL_ID` sentinel (a guard against firing calls at an unfilled placeholder).
Where to find the exact strings:

- Anthropic: <https://docs.anthropic.com/en/docs/about-claude/models>
- Google Gemini: <https://ai.google.dev/gemini-api/docs/models>
- OpenAI: <https://platform.openai.com/docs/models>

A real run refuses any alias still set to `UNSET_MODEL_ID`; `--dry-run` allows it.

**API keys** — copy `.env.example` to `.env` (gitignored) and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...
```

- Anthropic key: <https://console.anthropic.com/settings/keys>
- Google (Gemini) key: <https://aistudio.google.com/apikey> (free tier available)
- OpenAI key: <https://platform.openai.com/api-keys> (requires billing/credits)

Keys are only needed for a real run. `--dry-run` needs none.

## Run

Preview the prompt and plan without calling any API:

```
python -m judge data/contrastive_pairs_sample.json --dry-run
```

Run all three families (Claude, Gemini, ChatGPT), one forward+reverse round each —
this is the default when `--models` is omitted:

```
python -m judge data/contrastive_pairs_sample.json
```

Run a single family, or any subset, with `--models`:

```
python -m judge data/contrastive_pairs_sample.json --models claude
python -m judge data/contrastive_pairs_sample.json --models claude gemini
```

The aliases are `claude`, `gemini`, and `chatgpt` (defined in `config/models.py`). You
only need API keys for the families you actually run.

Useful flags:

- `--models claude gemini` — subset of families to run (default: all three)
- `--output-dir outputs` — where artifacts go (default `outputs/`)
- `--cache-dir .cache` / `--no-cache` — control the response cache
- `--max-workers 8` — concurrency across items/orders

Sampling temperature is not a CLI flag; it is set per family in `config/models.py`
(`claude` and `chatgpt` omit it because those models reject the parameter, `gemini`
pins `0.0`).

## Output

For input stem `S` and family `F`, each run writes three files to `--output-dir`, all in
input order:

- `S__F__judgments.csv` — **the deliverable table.** One row per input item; leading
  columns `id, prompt, answer_a, answer_b` so each row is self-describing, then one
  column per concept holding the `final` position-debiased score (empty cell = `NaN`).
- `S__F__disagreement.csv` — **diagnostic.** Same rows keyed by `id`, one column per
  concept holding `|forward - reverse|` — how position-sensitive the judge was. A high
  value means the judge flipped its answer with the swap, so distrust that cell.
- `S__F__raw.jsonl` — one JSON record per item: both raw completions, both parses,
  statuses, cache flags, and the folded scores. This is the audit trail.

### How to read a cell

A `judgments.csv` cell is the `final` score for one item × one concept, on the same
directional `[0, 1]` scale: **`0` = `answer_a` exhibits the concept more, `1` =
`answer_b`, `0.5` = tie.** In the shipped sample `answer_a` is the "high" response and
`answer_b` the "low" one, so for concepts the pairs actually differ on you should see
values well below `0.5` (the high answer winning). Before trusting a cell, glance at the
same cell in `disagreement.csv`: a value near `0` means both position-orderings agreed;
a high value means the judge flipped when the answers were swapped, so treat that
`final` as unreliable.

### Where outputs land, and the committed example

`outputs/` is **gitignored** — your own runs land there and are not committed, so the
folder shows up empty for teammates who clone the repo. To let others see a real run
without spending any API credits, a full three-family run over
`data/contrastive_pairs_sample.json` (Claude, Gemini, ChatGPT — 11 items × 11 concepts,
all passes clean) is committed under `example_run_output/`. It holds the same nine
files (`judgments`, `disagreement`, `raw.jsonl` per family) a fresh run produces. It is
a static snapshot for reference only; the tool never reads from or writes to it.

## Caching — read this

Responses are cached on disk (`.cache/` by default), keyed by a sha256 over the
inputs that determine a completion:

```
(model_id, temperature, concept_set_version, order, prompt, answer_a, answer_b)
```

**Sharp edge:** the key does **not** include the rendered prompt text. If you edit the
wording in `src/judge/prompt.py`, the cache will happily reuse answers produced by the
*old* wording. Changing what the judge is actually asked (concepts or their
definitions) is captured only through `CONCEPT_SET_VERSION`. So after editing prompt
wording or concept definitions, either **bump `CONCEPT_SET_VERSION`** or **clear the
cache** (`rm -rf .cache` / `--no-cache`). API errors are never cached.

## Tests

All tests are offline (no keys, no SDK calls); the provider layer is exercised through
a `FakeProvider`.

```
pytest
```

## Building an input file

Input is a JSON array of objects with `prompt`, `answer_a`, `answer_b`, and an
optional `id` (auto-assigned positionally if omitted). Unknown keys are preserved in
the JSONL sidecar but never shown to the judge. `data/contrastive_pairs_sample.json`
is a worked example: the first item of each of the 11 concepts pulled from
`contrastive_pairs.json`, with the high response mapped to `answer_a`, the low to
`answer_b`, and all concept labels stripped so they can't bias the judge.
```
