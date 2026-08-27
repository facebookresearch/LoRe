# SAE interpretability check: LoRe v2 shared population direction

Branch `sae-jumprelu`, commit `f546252` (repo RachelFreedman/LoRe-BasisInterp). PRISM only — no Community Alignment or MultiPref data is used anywhere in this analysis.

## What was tested

LoRe v2 (`LoReV2`) learns a single signed population weight `wbar` and per-user deltas, with reward `x @ V @ (wbar + delta_u)`. There is no alpha head-anchor; it is replaced by a reward-space ridge (`lam_pop*||V wbar||^2 + lam_d*mean_u||V delta_u||^2`). The shared population direction is

    v_pop = unit(V @ wbar)

The question: is `v_pop` a distinct learned preference axis, or is it just the base Skywork reward head? And is the residual (`v_pop` with its head component removed) a real signal or noise? This is read through two interpretable bases — an 11-concept library and a 16,384-feature TopK SAE.

## Findings

`v_pop` is not the reward head.

- **Nearly orthogonal to the head.** `cos(v_pop, head) = 0.010` (exp2). The head loads above the 0.032 random-direction floor on all 11 concepts; `v_pop` clears it on only 4, and barely (fluency 0.074, confidence 0.055, helpfulness 0.047, diversity 0.046).
- **Cosine understates behavioral overlap.** cos is 0.010, but the two directions' *scores* across the 15,754 held-out pairs correlate `0.580` (exp2, `score_corr_vpop_head`); residual vs head correlate `0.543`. Near-orthogonal as vectors, not unrelated in how they rank pairs — both project onto shared embedding structure. The accuracy gap is the cleaner separation.
- **Different SAE features.** Top-50 decoder-feature Jaccard: `v_pop` vs head `0.042`, residual vs head `0.031`, residual vs `v_pop` `0.961` (exp3).
- **More predictive than the head.** Held-out pair accuracy (1029 users, 15,754 pairs): head `0.5902`, `v_pop` `0.6285`, residual `0.6261` (exp2). `v_pop` beats the head by `0.038`. This is an accuracy result, not only an interpretability one. (The reviewer notes this mirrors the +0.030 gain over base measured separately on Community Alignment; that CA figure is from other work, not this run.)
- **The residual is `v_pop`.** `||residual|| / ||v_pop|| = 1.00`, because `v_pop` was already orthogonal to the head, so removing the head component changes almost nothing. Residual accuracy `0.6261` ≈ `v_pop` `0.6285`, and they overlap `0.961` in feature space. It is not a separate personalization axis.

Controls and counters:

- **Shuffled-label control (exp1).** Flip each pair's label with prob 0.5 and refit. With no anchor, the ridge shrinks `||wbar||` from `0.379` to `0.012` — a collapsed direction with no consistent orientation. Its sign is therefore arbitrary: held-out accuracy reads `0.4593` as oriented and `0.5407` flipped, i.e. at chance either way. What marks it as a null fit is the `||wbar||` collapse and its near-zero correlation with the real `v_pop` (`cos -0.044`), not the accuracy. The real `v_pop`'s `0.6285` depends on the real labels.
- **Single-direction counter (exp4).** Raw mean-diff aligns with the concepts (factuality 0.886, values 0.876, …) and partly with the head (`cos 0.428`) but predicts only `0.5909`. The best logistic single direction, oriented by its own fit, generalizes to `0.5760` (`cos to head -0.004`) — at the mean-diff/head ceiling, not above it. No single fixed direction reaches `v_pop`'s `0.6285`.
- **Positive control (exp5).** A cleanly injected fluency signal (`cos to head -0.041`) is recovered at `cos 0.9996`, staying off the head (`cos -0.040`). The pipeline can represent a non-head direction, so the real-data result is not a fitter that always returns the head.

**Verdict.** Under v2 the shared population direction is a distinct learned axis, not the base reward head: near-orthogonal (`cos 0.010`, score corr `0.580`), 4% top-feature overlap, more predictive out of sample (`0.6285` vs `0.5902`), and not well described by the 11 named concepts. This is the opposite of the vanilla model, where the earlier write-up reported `cos 1.0` — there the `alpha=1e4` anchor forced `v_pop` onto the head. v2 has no such anchor.

### Note on an earlier sign-flip bug (fixed in this commit)

An earlier draft reported below-chance numbers (exp4 `0.4059`, exp1 `0.4593` cited as anti-signal). Both came from reorienting a fitted direction to the head via `if w@head < 0: w = -w`. When a direction is near-orthogonal to the head (`cos ≈ 0`), that flip fixes its sign on a coin toss and the pair-accuracy scorer reports `1 - acc`. exp4's `0.4059` was the mirror of the true `0.5760`. Fixed: the head reorientation is removed from `exp4_data_direction.py` (the fit keeps its own data orientation; the fit is seeded), and exp1's collapsed direction is reported as at-chance up to arbitrary sign. No LoRe / trainer code was changed.

## Where the input files live

All paths are relative to the repo root (`RACHEL_LORE/LoRe-BasisInterp/`).

| input | path | notes |
| :-- | :-- | :-- |
| v2 basis (V, wbar, delta) | `PRISM/basis_v2.pt` | key `PART2_K10_seed42_v2`; committed. Fit by `PRISM/fit_v2_basis.py --K 10 --seed 42`. |
| concept library (11 vectors) | `data/prism/concept_vectors.pt` | committed; regenerated by `PRISM/compute_concept_vectors.py`. |
| PRISM train embeddings | `PRISM/data/prism/train_embeddings.pkl` | ~324 MB, gitignored — reviewer-supplied v2 embeddings. Regenerate via `PRISM/generate-prism-embeddings.py`. |
| PRISM test embeddings | `PRISM/data/prism/test_embeddings.pkl` | ~327 MB, gitignored — same. |
| base reward head | loaded via `PRISM/rm_head_utils.load_reward_head()` | Skywork-Reward-Llama-3.1-8B-v0.2 `score.weight`. |
| TopK SAE (D3) checkpoint | `sae/checkpoints/d3/model.pt` | ~537 MB, gitignored (not committed). Basis-independent decomposition of Skywork space; used only by exp3. |

Embeddings and the SAE checkpoint are intentionally not committed (size). They must be present locally to rerun exp1–exp5; a fresh clone has only the small committed inputs (`basis_v2.pt`, `concept_vectors.pt`).

## Where the code and artifacts live

Scripts sit at the top of `sae/experiments/`; shared primitives are in `sae/experiments/common.py`. Results and saved directions are written under that folder.

| experiment | script | result files (`sae/experiments/results/`) | artifact (`sae/experiments/artifacts/`) |
| :-- | :-- | :-- | :-- |
| exp2 — concept alignment (main) | `exp2_residual_main.py` | `exp2/summary.json`, `exp2/concepts.json`, `exp2/accuracy.json` | `exp2_directions.pt` |
| exp3 — SAE feature overlap | `exp3_feature_space.py` | `exp3/summary.json`, `exp3/top_features.json`, `exp3/alignment.pt` | — |
| exp1 — shuffled control | `exp1_shuffled_control.py` | `exp1/summary.json`, `exp1/concepts.json` | `exp1_shuffled_vpop.pt` |
| exp4 — single-direction counter | `exp4_data_direction.py` | `exp4/summary.json` | — |
| exp5 — positive control | `exp5_positive_control.py` | `exp5/summary.json` | `exp5_synth_vpop.pt` |

Basis fit: `PRISM/fit_v2_basis.py` → `PRISM/basis_v2.pt`.

Every headline number above comes from these committed JSONs. Pinned example:
`https://github.com/RachelFreedman/LoRe-BasisInterp/blob/f546252/sae/experiments/results/exp2/summary.json`

## How to reproduce

```
python PRISM/fit_v2_basis.py --K 10 --seed 42      # fit the v2 basis -> PRISM/basis_v2.pt
cd sae/experiments
python exp2_residual_main.py --device cpu
python exp3_feature_space.py --device cpu          # needs sae/checkpoints/d3/model.pt
python exp1_shuffled_control.py --device cpu
python exp4_data_direction.py --device cpu
python exp5_positive_control.py --target fluency --device cpu
```
