# What does the learned shared reward direction encode?

Mohamed Eldagla — branch `prism-dataset-analysis`

**1. Pre-registration** *(written retroactively — this work predates the workflow. Flagging rather than back-dating.)*

Testing whether the shared population direction learned by LoReV2 on Community Alignment aligns with our 11-concept library, and how it relates to Skywork's pretrained reward head. Strong alignment would give the paper named axes. Weak alignment would mean the direction that beats the base RM is not explained by our concepts, and the concept-library workstream needs different concepts or a different method.

**2. Headline**

The learned shared direction is only weakly concept-aligned — the 84th percentile of a random-direction null, against the pretrained head's 100th — and it ranks real responses far more like that head (r = 0.78) than its near-orthogonal vector cosine (0.04) suggests.

**3. How I tested it**

LoReV2 at rank 8 (lr=1e-4, lam_pop=0.01, lam_d=10), trained on 200 Community Alignment users with the leak-free turn-level split. Scored all 48,385 distinct (prompt, response) items and all 45,057 (chosen − rejected) differences against the 11 concept vectors. 3–5 seeds throughout, standard deviations in the CSVs.

- `PRISM/interpret_wbar.py` → `results/community_alignment/wbar_concepts.csv` (a252e38)
- `PRISM/probe_wbar_direction.py` (d43340b)
- `PRISM/concept_score_alignment.py` → `results/community_alignment/concept_score_align.csv` (657694d, b025d52)

**4. What else could explain it, and how I ruled it out**

*Verbosity.* The direction correlates r = +0.64 with response word count, higher than the head's +0.49. But "prefer the longer response" scores 0.4969 over 44,857 pairs — chance. Length carries no label information here, so it cannot be driving the accuracy.

*Cone geometry.* A random direction already correlates ~0.49 with any concept probe, because the embeddings occupy a narrow cone rather than filling 4096 dimensions. Every figure above is calibrated against a per-concept null of 300–500 random directions.

Two earlier versions of this analysis were wrong and are superseded. I first compared vector cosines, which the 0.04-versus-0.78 gap shows is the wrong test. I then thresholded on the null's 95th percentile alone and read "below threshold" as "no alignment", which confuses non-significance with no effect. Both corrections are in the commit history.

**5. What would prove it wrong**

The library is 11 concepts, PRISM-derived and LLM-generated. "Not aligned with these" is not "uninterpretable" — the relevant axis may simply be absent. Those concepts are also highly correlated with one another, so eleven readings near the 84th percentile are not eleven independent tests and should not be pooled as though they were. This is one dataset, one rank, one hyperparameter setting. And the direction comes from my LoReV2; Rachel's mLoRe reference implementation may produce a different one, which would invalidate the specific numbers even if the methodological point holds.
