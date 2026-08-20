# SAE Interpretability Check: LoRe v2 Shared Population Direction (Community Alignment)

Owner: Ifesi   Red-teamers: Prerana, Mohamed

Where the work lives: folder `PRISM/ca_sae_analysis/` (self-contained: code, artifacts, results, summaries). Run locally on the reviewer-supplied CA embeddings; not yet committed. The v2 basis analysed here is not refit by these scripts — it is loaded from `results/community_alignment/ca_v_pop.pt`, key `CA_K10_seed42_lam.01_.01`, fit upstream by `PRISM/community_alignment_lore.py --model v2 --save_model`. The per-experiment file map is in Section 3.

This is the Community Alignment (CA) companion to the PRISM check. Same question, same five experiments, same primitives (imported from `sae/experiments/common.py`, so the sign-flip fix is inherited). The one reason CA matters on its own: synthetic_recovery found LoRe needs ~50+ pairs/user before it can recover a planted axis. PRISM sits below that (~15 pairs/user); CA sits well above it (~180). CA is the stronger test, and it is the dataset the paper's accuracy numbers are actually on — so if the PRISM story survives here, it is describing the object the results section describes, not a low-data artifact. **[This doc directly answers Mohamed's PRISM-review scope point — see Section 6.]**

## 1. Pre-registration

Testing: What LoRe v2's shared population direction (v_pop = unit(V @ wbar)) actually is on CA, read through two interpretable bases: the 11-concept library and the SAE decoder features. Is it a distinct learned preference axis, or the base Skywork reward head? And is the residual (v_pop with its head component removed) interpretable signal, or noise?

v2 (LoReV2) has no alpha head-anchor. The anchor was replaced by a reward-space ridge (lam_pop·||V wbar||^2 + lam_d·||V delta_u||^2), and the weights are signed (wbar + delta_u), not softmax simplex. So the shared direction is unit(V @ wbar).

Basis: `CA_K10_seed42_lam.01_.01` — K=10, seed=42, lam_pop=lam_d=0.01, lr=1e-4, iters=5000, min_pairs=50, test_frac=val_frac=0.2. 200 users, median 180 pairs/user, 27,036 train / 9,010 held-out pairs. The held-out split is the fit's own (leak-free by conversation/turn, seed 42), reproduced verbatim so the test pairs scored here are the ones the fit never saw. Concept noise floor tau = 0.032. v_pop is used in its natural (data) orientation, not re-oriented to the head.

How: three experiments on the CA basis plus two counter-experiments.

  - exp2 (main, concept alignment): cosine of head, v_pop, residual against each of the 11 concept directions; held-out pair accuracy, score correlation to the head, and ||residual||/||v_pop|| alongside.
  - exp3 (SAE feature space): project head, v_pop, residual onto the 16,384 SAE decoder features; compare each direction's top-50 features by Jaccard overlap. Run on the retrained D3 TopK SAE (`D3_16384_k256_string-vs-string-v1.pt`, 16,384 features, k=256).
  - exp1 (control): sign-flip each pair's label with prob 0.5, refit LoRe v2 with the same hyperparameters, re-run the concept read.
  - exp4 (counter): does any single direction (raw mean-diff, best logistic fit) carry the preference signal v_pop has?
  - exp5 (counter, positive control): inject a known non-head direction (fluency) and refit; can the v2 pipeline recover a non-head signal at all?

Outcomes: If v_pop is nearly orthogonal to the head, picks out different SAE features, and predicts held-out pairs at least as well as the head, the shared direction is its own thing. If v_pop's concepts and top features match the head and the residual is noise, then the shared direction is the reward head.

## 2. Result

v_pop is not the reward head. cos(v_pop, head) = 0.035 (exp2, results/exp2/summary.json). The head loads heavily across the concept library; v_pop loads on almost none of it.

| concept | head | v_pop | residual (head removed) |
| :-- | :--: | :--: | :--: |
| helpfulness | +0.493 | +0.037 | +0.020 |
| factuality | +0.458 | +0.018 | +0.002 |
| values | +0.445 | +0.023 | +0.007 |
| safety | +0.423 | +0.024 | +0.009 |
| diversity | +0.409 | +0.038 | +0.024 |
| formatting | +0.362 | +0.025 | +0.012 |
| confidence | +0.252 | +0.047 | +0.038 |
| fluency | −0.041 | +0.046 | +0.047 |
| creativity | −0.383 | −0.010 | +0.004 |
| repetition | −0.393 | −0.019 | −0.006 |
| sycophancy | −0.401 | −0.022 | −0.008 |

The head crosses the 0.032 floor on all 11 concepts. v_pop crosses it on 4 (confidence 0.047, fluency 0.046, diversity 0.038, helpfulness 0.037), and only barely — the same four concepts, in almost the same order, as on PRISM. So v_pop is a real direction the 11-concept library does not describe well.

Cosine understates how similarly the two directions score responses. cos(v_pop, head) = 0.035, but their scores across the 9,010 held-out pairs correlate 0.722 (exp2, score_corr_vpop_head). Near-orthogonal as vectors, closely related in how they rank pairs; both project onto shared embedding structure. (The residual, having had the head projected out, correlates only 0.251 with the head — lower than on PRISM.) The held-out accuracy gap is the cleaner separation.

In SAE feature space the same conclusion holds — v_pop and the head share almost none of their top-50 features, while the residual overlaps v_pop almost entirely (exp3, results/exp3/summary.json, on the retrained D3 SAE):

| pair | top-50 Jaccard |
| :-- | :--: |
| v_pop vs head | 0.010 (1/50 features shared) |
| residual vs head | 0.010 (1/50) |
| residual vs v_pop | 0.961 (49/50) |

v_pop and the head fire essentially disjoint dictionary features (1 of the top 50 in common), so v_pop is not the head in feature space either. Projecting the head component out of v_pop changes almost nothing — the residual keeps 49 of v_pop's top 50 features — confirming from the SAE side that v_pop was already nearly orthogonal to the head.

Held-out pair accuracy (results/exp2/accuracy.json, 200 users, 9,010 pairs): head 0.6364, v_pop 0.6797, residual 0.6513. v_pop predicts held-out preferences better than the head, by 0.043 — an accuracy result, not only an interpretability one, and on the dataset the paper's numbers are on.

The residual is not noise and not a separate direction either. ||residual||/||v_pop|| = 1.00, and cos(v_pop, head) = 0.035, so removing the head component from v_pop changes almost nothing (it was already nearly orthogonal). The residual scores 0.6513, close to v_pop's 0.6797. The residual is essentially v_pop.

The control (exp1, results/exp1/summary.json) shows the signal comes from the labels. Sign-flipping the labels flattens the data term; the refit lands on a direction uncorrelated with the real v_pop (cos −0.013), with zero significant concepts, whose held-out accuracy is 0.4891 — at chance. Note a real difference from PRISM: here ||wbar|| does not collapse to zero (0.908 → 0.752). With ~180 pairs/user the ridge still finds a sizeable wbar, but it carries no held-out signal, no concept alignment, and is orthogonal to the real direction. So on CA the discriminators are the at-chance accuracy, the zero concepts, and the −0.013 correlation with the real v_pop — not a ||wbar|| collapse. Same "it's a null" conclusion, read off different quantities.

The counter-experiments check the mechanism.

  - exp4 (results/exp4/summary.json): the raw mean-diff direction aligns partly with the head (cos 0.451) and predicts 0.6378, about the same as the head (0.6364). The best logistic single direction, fit to maximise the pairwise margin, generalises to 0.6494 (cos to head 0.037) — the best of the single directions, but still below v_pop's 0.6797. No single fixed direction tested reaches v_pop, and v_pop is the one that does not align with the concept library.
  - exp5 (results/exp5/summary.json): a cleanly injected fluency signal (cos to head −0.041) is recovered off the head (cos to head −0.003), verdict recovered_target. At the CA fit's own hyperparameters (lr=1e-4, tuned low for stability) recovery is partial (cos 0.29 to the target); at the standard v2 lr (1e-2) it recovers far more fully (cos 0.80, results/exp5/summary_lr1e-2_convergence_check.json). Either way the fit lands off the head, so the pipeline can represent a non-head direction — the partial number is under-convergence at the low lr, not a pipeline limit.

Verdict: Read through both the concept library and the SAE decoder features, the shared population direction under v2 on CA is a distinct learned axis, not the base reward head. It is nearly orthogonal to the head (cos 0.035; score correlation 0.722), shares only 1 of its top 50 SAE features with the head, is not well described by the 11 named concepts (the same four it touches on PRISM), and predicts held-out CA pairs better than the head (0.6797 vs 0.6364). The residual is v_pop itself. This replicates the PRISM v2 finding on ~12× more data per user and, being CA-native, wins on the data it was fit on — resolving the scope concern from the PRISM review (Section 6).

## 3. Testing info

Folder: `PRISM/ca_sae_analysis/` (run locally; not yet committed)   Shared primitives: imported from `sae/experiments/common.py`   CA adapter: `ca_common.py` (loads the stored basis, reproduces the fit's leak-free split)   Basis: `results/community_alignment/ca_v_pop.pt` key `CA_K10_seed42_lam.01_.01`, fit by `PRISM/community_alignment_lore.py --model v2 --save_model`.

File map:

| experiment | script | result files (under results/) | artifact (under artifacts/) |
| :-- | :-- | :-- | :-- |
| exp2 — concept alignment (main) | exp2_residual_main.py | exp2/summary.json, exp2/concepts.json, exp2/accuracy.json | exp2_directions.pt |
| exp3 — SAE feature overlap | exp3_feature_space.py | exp3/summary.json, exp3/ca_top_features.json, exp3/ca_alignment.pt | — |
| exp1 — shuffled control | exp1_shuffled_control.py | exp1/summary.json, exp1/concepts.json | exp1_shuffled_vpop.pt |
| exp4 — single-direction counter | exp4_data_direction.py | exp4/summary.json | — |
| exp5 — positive control | exp5_positive_control.py | exp5/summary.json, exp5/summary_lr1e-2_convergence_check.json | exp5_synth_vpop.pt |

exp3 uses the retrained D3 TopK SAE checkpoint (`D3_16384_k256_string-vs-string-v1.pt`, 16,384 features, k=256, in this folder). The SAE decodes the Skywork embedding space, not any one dataset, so no CA-specific SAE is needed — `exp3_feature_space.py` (a self-contained port of `sae/experiments/exp3_feature_space.py` against `ca_common`) feeds it the CA v_pop/head/residual vectors. Top features are ranked by |decoder alignment|, so direction sign does not matter here.

Commands:

```
cd PRISM/ca_sae_analysis
python exp2_residual_main.py --device cpu
python exp4_data_direction.py --device cpu
python exp1_shuffled_control.py --device cpu
python exp5_positive_control.py --target fluency --device cpu
python exp3_feature_space.py --device cpu   # needs D3_16384_k256_string-vs-string-v1.pt in this folder
```

## 4. Alternative explanations, ruled out

v_pop is distinct from the head only because the fit is noisy: ruled out by exp2 and exp4. v_pop predicts held-out CA pairs at 0.6797, above the head's 0.6364 and above every single data direction tested (mean-diff 0.6378, best logistic 0.6494). A noisy fit would not beat the head out of sample on ~180 pairs/user.

v_pop's predictive power comes from the fitter, not the preferences: ruled out by exp1. Sign-flipping the labels produces a direction uncorrelated with the real v_pop (cos −0.013), with zero significant concepts, at chance out of sample (0.4891). The 0.6797 depends on the real labels.

The v2 pipeline can only ever return the head: ruled out by exp5. An injected fluency direction is recovered off the head (cos to head −0.003; cos to target 0.29 at the matched lr, 0.80 at the standard lr). The pipeline can represent a non-head direction.

## 5. What could still prove this wrong

v_pop predicts held-out pairs (0.6797) but does not align with any of the 11 named concepts above 0.047, and exp3 shows it fires almost entirely different SAE features from the head without pinning it to a nameable feature theme. So it is a real, predictive direction that neither the concept library nor the raw SAE-overlap read explains in human terms — it is proven distinct, not yet positively named. A larger/different concept set, or reading the top-activating responses of v_pop's own SAE features, might describe it; it may also be capturing something outside the current probes (format, length, topic).

The exp1 null presents differently than on PRISM: ||wbar|| does not collapse (0.908 → 0.752). The discriminators that mark it as a null are the at-chance accuracy (0.4891), the zero significant concepts, and the −0.013 correlation with the real v_pop — not a norm collapse. If a future reviewer expects the PRISM-style ||wbar|| → 0 signature, it is worth stating up front that at CA's data volume the null is read off accuracy/concepts/orthogonality instead.

exp5 recovery is only partial (cos 0.29) at the CA fit's own lr; it takes the standard v2 lr to reach cos 0.80. The positive control still passes (off-head both times), but the low-lr number should be reported with the convergence caveat, not as a clean recovery.

Only one basis was fit (K=10, seed=42, lam_pop=lam_d=0.01). No sweep over K, seed, or ridge strengths. Note the paper's tuned CA config is lam_pop=0.01, lam_d=10 (a second config, `CA_K8_seed42_tuned`, is present in ca_v_pop.pt but not analysed here); Mohamed reports cos(v_pop, head) moves between 0.035 and 0.055 across the two configs, so the exact cosine is config-dependent even though the qualitative finding holds.

## 6. Relation to the PRISM review (Mohamed)

This doc closes the one open scope point from the PRISM review. Mohamed pulled the PRISM basis and scored it on CA:

  - cos(CA v_pop, PRISM v_pop) = 0.078; score correlation on CA pair diffs = 0.713.
  - On CA held-out pairs he measured: CA v_pop 0.6754, head 0.6463, PRISM v_pop 0.6219.

His point: the PRISM-fit v_pop scores below CA's base RM on CA pairs, while the CA-fit direction beats it — "the direction wins on the data it was fit on and loses elsewhere… either interpret the CA direction or say plainly that this is PRISM-specific." This analysis interprets the CA direction directly. The CA-fit v_pop beats CA's base RM (0.6797 vs 0.6364 here; 0.6754 vs 0.6463 in his independent scoring — same ordering), and shows the same interpretability profile as on PRISM: near-orthogonal to the head, off the concept library, residual ≈ v_pop. So the finding is not PRISM-specific; it holds for the direction fit on, and evaluated on, the paper's own dataset.

A formal CA red-team pass (Prerana/Mohamed) is still pending.

---

### To-dos

- Commit the folder (exp3 now in; the 512 MB D3 checkpoint stays gitignored, not committed).
- Optional: run the tuned config `CA_K8_seed42_tuned` (lam_d=10) for the stability point in Section 5.
- Optional: formal CA red-team review; append as Section 6 appendix when it arrives.
