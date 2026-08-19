# Red-team: LoReV2 — Community Alignment (Part 1) + MultiPref (Part 2)

Ifesi — branch `prism-dataset-analysis`

**Target claim (Mohamed's).** LoReV2 beats the base Skywork RM by ~+0.03, and that gain is a single shared population direction (`wbar`) — the per-user personalization deltas add nothing. "Individuals don't matter much here." Part 1 challenges this within Community Alignment; Part 2 re-tests it on an independent dataset (MultiPref) and inspects the learned matrix directly.

# Part 1 — Community Alignment

All runs: LoReV2, rank 8, 5 seeds, leak-free turn-level split, 200 users (median 180 pairs/user). Producing script: `PRISM/community_alignment_lore.py`. Full run log: `results/community_alignment/redteam_sweep.log`.

## What was done

- **Experiment 1 — Reproduction.** Re-ran his exact config (lr 1e-4, lam_pop 0.01, lam_d 10) over ranks 1/5/8/10/20.
- **Experiment 2 — lam_d sweep.** Swept the delta ridge penalty across 0, 1, 2, 4, 6, 8, 10, 14, 16 at rank 8. Tests whether his lam_d=10 was just a setting that happened to nudge so many user weights to zero that manully zeroing them out had no effect on accurcacy.
- **Experiment 3 — High-volume-users cut.** Restricted to users with ≥200 (n=141) and ≥250 (n=37 of 200) pairs each. Testing suspision that maybe individuals matter but 180 pairs/user is too little to learn them. The ≥250 cut leaves only 37 users, which is why its error bar widens.

## Results

- **Reproduction matched.** test_acc − base_rm = **+0.0333 ± 0.0020** at rank 8 (his +0.030). `personal − global = −0.0003`; zeroing deltas (`wbar_only`) ≈ test_acc. Faithful.
- **lam_d doesn't matter.** test_acc flat at ~0.670 across all 9 values. Delta contribution `test_acc − wbar_only` stays within noise everywhere — **including lam_d=0** (ridge fully off): −0.0012 ± 0.0024. → My "cherry-picked lam_d" hypothesis is **disproved**; his choice is on a flat surface but harmless.
- **Data volume doesn't matter up to ~400 pairs/user.** `personal − global`: base −0.0003, min_pairs=200 **−0.0007 ± 0.0042** (n=141), min_pairs=250 **+0.0023 ± 0.0061** (n=37, std > mean). The source `pairs.json` was built with `--max_pairs_per_user 400`, so the richest users are truncated at that ceiling. → "Not enough data" escape hatch **closed up to that ceiling**.
- **Small note .** `personal − other_user` stays positive (+0.013 to +0.030) throughout — user identity isn't literally random, but that signal never converts into beating the shared direction.

**Verdict:** on Community Alignment, the target claim survived reproduction, a lam_d sweep, and a data-volume cut. No within-dataset experiment left that plausibly overturns it. Next challenge would be an independent dataset (MultiPref).

## Artifacts

| Experiment | Result files | Script |
|---|---|---|
| 1. Reproduction | `results/community_alignment/repro_lamd10.csv` | `PRISM/community_alignment_lore.py` |
| 2. lam_d sweep | `results/community_alignment/lamd_{0,1,2,4,6,8,10,14,16}.csv` | `PRISM/community_alignment_lore.py` (`--lam_d`) |
| 3. High-volume cut | `results/community_alignment/highvol_min{200,250}.csv` | `PRISM/community_alignment_lore.py` (`--min_pairs`) |
| Full run log (all 3) | `results/community_alignment/redteam_sweep.log` | — |

Key CSV columns: `test_acc`, `wbar_only` (deltas zeroed), `base_rm`/`global`/`personal`/`other_user` (reference directions), `min_abs_basis_cos` (delta collapse).

# Part 2 — MultiPref (independent dataset)

**Why this dataset.** `allenai/multipref` is a different kind of data and structure: 10,461 single-turn prompts, each judged by ~4 different people (2 are experts and the other 2 are regular people), two answers from different models, a 5-point better/worse label (ties dropped). It's a second, independent test of whether individuals matter. After dropping ties, 179 users have ≥50 pairs each (median 122, max 639). Kept fully separate: data goes in `data/multipref/`, results in `results/multipref/` — nothing here touches the Community Alignment results.

Same setup as Part 1: LoReV2, rank 8, 5 seeds, leak-free split, lr 1e-4, lam_pop 0.01, lam_d 10, 29,674 pairs.

## What was done

One script runs everything: `PRISM/multipref/run.sh` builds the pairs, embeds them with Skywork, then trains LoReV2 and saves the learned matrix. Anyone can run it — it uses a local copy of the data if there is one, otherwise downloads it. On top of that, the same experiments from Part 1:

- **Experiment 1 — Reproduction.** Same accuracy checks as Part 1: does LoReV2 beat the base model, and does the per-user part add anything.
- **Experiment 2 — lam_d sweep.** Swept the per-user penalty across 0, 1, 2, 4, 6, 8, 10, 14, 16 at rank 8. Same question as Part 1 — does penalizing the per-user part harder change accuracy.
- **Experiment 3 — High-volume-users cut.** Kept only users with ≥200 and ≥250 pairs each (56 and 28 users, median 200 and 235). Tests whether the flat result is just too few pairs per user.
- **Extra — look at the matrix directly.** Loaded the saved matrix (no retraining) to see if it points one way, and whether the per-user parts are all basically the same thing.

## Results

- **Same accuracy result as Part 1.** LoReV2 beats the base model by **+0.0384 ± 0.0059**. The per-user part adds nothing over the shared direction (`personal − global = −0.0031 ± 0.0033`). Users beat a random other user by a hair (`+0.0064 ± 0.0028`), but that never turns into beating the shared direction.
- **lam_d doesn't matter (same as Part 1).** Accuracy is flat at ~0.688–0.693 across all nine values. Zeroing the per-user part costs nothing anywhere, including at lam_d=0 with the penalty fully off (`test_acc − wbar_only = −0.0018 ± 0.0037`). `personal − global` sits at −0.0031 the whole way.
- **Data volume doesn't matter (same as Part 1).** `personal − global` stays ~0 and slightly negative even for the richest users: ≥200 pairs **−0.0022 ± 0.0031**, ≥250 pairs **−0.0014 ± 0.0026**. Those users still beat the base model (+0.037 and +0.041), they just get no gain from their own per-user part. So "not enough pairs per user" isn't the explanation.
- **Everyone ends up pointing the same way.** Each user's direction lines up almost exactly with the one shared direction (overlap **0.9999–1.0000** for all 179 users). The personal part is tiny — about **1%** of the size of the shared part.
- **The per-user parts are basically one thing.** The table of per-user offsets is almost a single line: one piece holds **83–96%** of it in 4 of 5 runs. So it's one small tweak turned up or down per user, not 179 different tastes. (One run spreads more, but its offsets are the smallest of all — just noise.)
- **One caveat — the shared direction isn't the same from run to run.** The 8-column basis isn't itself small, and the shared direction changes between random seeds (overlap 0.28–0.71 as a vector, 0.07–0.18 as a subspace — low). Each run finds a different direction that works just as well. So don't read too much into any single learned direction. This doesn't bring back personalization, though — in every run the per-user part is still tiny and basically one line.

**Verdict:** on MultiPref, a different dataset with many judges, the result is the same, and it held up under the same lam_d sweep and high-volume cut as Part 1. The gain is one shared direction; the per-user part barely does anything, and if anything there's less personal signal here than in Community Alignment. One thing to keep in mind: the shared direction isn't stable across runs, so be careful before calling any single direction meaningful.

## Artifacts

| Item | Path | Produced by |
|---|---|---|
| Preference pairs | `data/multipref/pairs.json` (gitignored) | `PRISM/multipref/multipref_prep.py` |
| Skywork embeddings | `data/multipref/embeddings.pt` (gitignored) | `PRISM/embed_community_alignment.py` |
| Reproduction CSV | `results/multipref/repro_lamd10.csv` | `PRISM/community_alignment_lore.py` |
| lam_d sweep CSVs | `results/multipref/lamd_{0,1,2,4,6,8,10,14,16}.csv` | `PRISM/community_alignment_lore.py` (`--lam_d`) |
| High-volume cut CSVs | `results/multipref/highvol_min{200,250}.csv` | `PRISM/community_alignment_lore.py` (`--min_pairs`) |
| **Learned matrix** (V, wbar, per-user delta, wbar_dir; 5 seeds) | `results/multipref/lorev2_matrix_lamd10.pt` | `PRISM/community_alignment_lore.py --save_model` |
| Run logs | `results/multipref/{savepass,exp23}.log` | — |

Scripts: `PRISM/multipref/run.sh` (orchestrates all three steps) · `PRISM/multipref/multipref_prep.py` (dataset → pairs) · `PRISM/community_alignment_lore.py` (trains LoReV2, writes the CSV and, with `--save_model`, the matrix) · **`PRISM/multipref/analyze_matrix.py`** (loads the saved matrix and prints every collapse metric above — reproduces this section's numbers).

Reproduce the analysis: `python PRISM/multipref/analyze_matrix.py --matrix results/multipref/lorev2_matrix_lamd10.pt`
