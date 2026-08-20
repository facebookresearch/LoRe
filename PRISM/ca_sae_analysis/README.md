# CA SAE interpretability check: LoRe v2 shared population direction

Community Alignment (CA) analogue of the PRISM analysis in `PRISM/sae_analysis.md`.
Same question — is the LoRe v2 shared population direction `v_pop = unit(V @ wbar)` a
distinct learned preference axis, or just the base Skywork reward head? — but on CA,
where each user has ~180 preference pairs (median) instead of PRISM's ~15. CA sits well
above the ~50-pairs/user phase transition where LoRe can actually recover a planted axis,
so it is the stronger test.

Everything for this analysis lives in this folder: code, artifacts, results, summaries.

## What is fitted vs. what is analyzed here

The v2 basis itself was fit upstream by `PRISM/community_alignment_lore.py --model v2
--save_model`, producing `results/community_alignment/ca_v_pop.pt`. These scripts do NOT
refit the main basis; they load the committed `CA_K10_seed42_lam.01_.01` fit (K=10,
lam_pop=lam_d=0.01, seed=42, lr=1e-4, iters=5000, min_pairs=50, test/val_frac=0.2, 200
users) and analyze its `v_pop`. Only exp1 and exp5 run their own LoReV2 fits (a shuffled
control and a synthetic positive control), using the same hyperparameters.

The held-out split is reproduced exactly: `ca_common.build_user_diffs` is a verbatim copy
of the trainer's split (leak-free by conversation/turn) driven by the fit's own seed, so
the test pairs scored here are the ones the fit never saw. `uids` are asserted to match
the stored fit order.

## Experiments

| experiment | script | needs a LoReV2 fit? | result files |
| :-- | :-- | :-- | :-- |
| exp2 — concept alignment + held-out accuracy (main) | `exp2_residual_main.py` | no | `results/exp2/{summary,concepts,accuracy}.json`, `artifacts/exp2_directions.pt` |
| exp4 — single-direction counter | `exp4_data_direction.py` | logistic single-direction only | `results/exp4/summary.json` |
| exp1 — shuffled-label control | `exp1_shuffled_control.py` | yes (shuffled) | `results/exp1/{summary,concepts}.json`, `artifacts/exp1_shuffled_vpop.pt` |
| exp5 — positive control | `exp5_positive_control.py` | yes (synthetic) | `results/exp5/summary.json`, `artifacts/exp5_synth_vpop.pt` |

exp3 (SAE feature-space overlap) runs `exp3_feature_space.py` against the retrained D3
TopK SAE (`D3_16384_k256_string-vs-string-v1.pt`, 16,384 features, k=256). That checkpoint
is ~512 MB and gitignored — download it separately and drop it in this folder. The SAE
decodes the shared Skywork space, so one script reports both CA and PRISM v_pop; top
features are ranked by |decoder alignment| so direction sign is irrelevant. Results:
`results/exp3/{summary.json, ca_top_features.json, ca_alignment.pt, prism_*}`.

exp5 also writes `results/exp5/summary_lr1e-2_convergence_check.json`: the CA fit's own
lr (1e-4) is tuned low for stability, so the injected direction is only partially
recovered (cos ~0.29 to target) though already off the head — the verdict still holds.
At the standard v2 lr (1e-2) recovery is much fuller (cos ~0.80), confirming the partial
number is under-convergence, not a pipeline limitation. `summary.json` is the canonical
matched-hyperparameter run.

## Design notes

- `v_pop` is used in its natural (data) orientation from the stored fit, NOT re-oriented
  to the head. Re-orienting a near-orthogonal direction to the head flips its sign on a
  coin toss and makes pair accuracy report `1 - acc` — the sign-flip bug the reviewer
  caught in the PRISM exp4. The logistic fit in exp4 is likewise kept in its own data
  orientation.
- The pure-vector scoring primitives (concept cosines, null threshold, pair accuracy,
  score correlation, residual) are imported from `sae/experiments/common.py`, so the
  sign-flip fix and the scoring logic have a single source of truth.
- Concept vectors (`data/prism/concept_vectors.pt`) and the base head are Skywork-space
  and dataset-independent, so they are shared with the PRISM analysis unchanged.

## Inputs (repo-relative)

| input | path | notes |
| :-- | :-- | :-- |
| CA v2 basis (V, wbar, delta, v_pop_unit) | `results/community_alignment/ca_v_pop.pt` | key `CA_K10_seed42_lam.01_.01` |
| CA pairs | `data/community_alignment/pairs.json` | ~80 MB |
| CA embeddings | `data/community_alignment/embeddings.pt` | ~795 MB, `{"emb":[N,4096],"keys":[...]}` |
| concept library (11 vectors) | `data/prism/concept_vectors.pt` | Skywork-space, shared with PRISM |
| base reward head | via `PRISM/rm_head_utils.load_reward_head()` | Skywork-Reward-Llama-3.1-8B-v0.2 |

## Reproduce

```
cd PRISM/ca_sae_analysis
python exp2_residual_main.py --device cpu
python exp4_data_direction.py --device cpu
python exp1_shuffled_control.py --device cpu
python exp5_positive_control.py --target fluency --device cpu
python exp3_feature_space.py --device cpu   # needs D3_16384_k256_string-vs-string-v1.pt here
```
