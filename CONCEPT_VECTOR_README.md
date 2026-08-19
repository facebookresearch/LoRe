# PRISM Concept Vector Interpretability Pipeline

An automated evaluation pipeline for measuring whether [LoRe](https://arxiv.org/abs/2504.14439)'s learned low-rank reward model bases naturally align with human-interpretable concepts.

This work builds on top of the [LoRe-BasisInterp](https://github.com/RachelFreedman/LoRe-BasisInterp) codebase by Meta FAIR.

---

## Research Question

> *Do the basis vectors learned by LoRe's low-rank reward model correspond to distinct, human-interpretable concepts (e.g., helpfulness, formatting, factuality)?*

We answer this by constructing **concept vectors** — pure mathematical directions in the reward model's hidden space — and measuring their alignment with the learned bases using cosine similarity and causal ablation.

---

## Methodology

### 1. Concept Library
We define 11 concepts (7 from the PRISM dataset, 4 from the Reward-Lens paper) representing human-interpretable traits:

| # | Concept | Source |
|---|---------|--------|
| 1 | Helpfulness | PRISM |
| 2 | Fluency | PRISM |
| 3 | Factuality | PRISM |
| 4 | Safety | PRISM |
| 5 | Diversity | PRISM |
| 6 | Creativity | PRISM |
| 7 | Values | PRISM |
| 8 | Confidence | Reward-Lens |
| 9 | Formatting | Reward-Lens |
| 10 | Sycophancy | Reward-Lens |
| 11 | Repetition | Reward-Lens |

### 2. Contrastive Pair Generation
For each concept, we use an LLM (Claude Sonnet 4 via AWS Bedrock) to generate **50 contrastive prompt–response pairs**: one response exhibiting a *high* manifestation of the concept and one exhibiting a *low* manifestation, yielding **550 total pairs**.

### 3. Concept Vector Extraction
The contrastive pairs are passed through the reward model (`Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`) on cloud GPUs (Modal A10G) to extract 4096-dimensional hidden state embeddings. The concept vector is the mean difference:

```
C_concept = E[h_high − h_low]
```

### 4. Alignment Analysis
We compute cosine similarity between each learned basis and each concept vector, using a null distribution (95th percentile of random vector similarities) as a statistical significance threshold.

### 5. Causal Intervention
We ablate (zero out) individual bases and measure the resulting drop in concept-specific reward signal power to test for causal localization vs. entanglement.

---

## Project Structure

```
PRISM/
├── concept_library.py            # Dictionary of 11 concepts with high/low prompts
├── generate_contrastive_pairs.py # LLM-based contrastive pair generation (Bedrock)
├── compute_concept_vectors.py    # Compute concept vectors from embeddings
├── concept_basis_alignment.py    # Alignment analysis with null distribution
├── compare_ranks.py              # Cross-rank interpretability comparison
├── causal_intervention.py        # Causal ablation analysis
├── train_basis.py                # Original Meta FAIR training script
├── prepare.py                    # Data preparation
└── eval_rb2.py                   # RewardBench2 evaluation

modal_compute_vectors.py          # Modal cloud GPU runner for vector extraction

data/prism/
├── contrastive_pairs.json        # 550 generated contrastive pairs
└── concept_vectors.pt            # 11 computed concept vectors (4096-dim each)

checkpoints/checkpoints/
└── PRISM_V_lore_K_{K}_alpha_{α}.pt  # Learned basis matrices

results/
├── K_{K}/                        # Per-checkpoint alignment reports & heatmaps
└── rank_comparison/              # Cross-rank comparison plots
```

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

You also need:
- **AWS Bedrock** access (for Claude Sonnet 4 contrastive pair generation)
- **Modal** account (for cloud GPU concept vector computation)

### Step-by-step

```bash
# 1. Generate contrastive pairs (uses Claude Sonnet 4 via Bedrock)
python PRISM/generate_contrastive_pairs.py

# 2. Compute concept vectors on cloud GPU
modal run modal_compute_vectors.py

# 3. Run alignment analysis for a single checkpoint
python PRISM/concept_basis_alignment.py

# 4. Compare interpretability across all ranks
python PRISM/compare_ranks.py

# 5. Run causal intervention (ablation study)
python PRISM/causal_intervention.py
```

---

## Key Findings

### Rank Comparison
| Rank (K) | Learned Bases | Significant Bases | Interpretability Score | Concepts Covered |
|----------|--------------|-------------------|----------------------|-----------------|
| 1 | 1 | 1 | 100.0% | 4 / 11 |
| 5 | 3 | 3 | 100.0% | 4 / 11 |
| 10 | 4 | 4 | 100.0% | 4 / 11 |
| 15 | 5 | 5 | 100.0% | 4 / 11 |
| 20 | 7 | 7 | 100.0% | 4 / 11 |
| 25 | 4 | 4 | 100.0% | 4 / 11 |
| 50 | 6 | 5 | 83.33% | 4 / 11 |

### Rank-1 Collapse Discovery
Causal ablation and inter-basis similarity analysis revealed that **all checkpoints suffered from Rank-1 collapse** — every basis within each checkpoint is a near-identical copy of a single vector (cosine similarity ≥ 0.9996).

**Root cause:** The sparsity penalty `α = 10,000` (hardcoded in `train_basis.py`) was too aggressive, forcing the optimizer to collapse all bases into a single direction.

---

## References

- **LoRe Paper:** [Personalizing LLMs via Low-Rank Reward Modeling](https://arxiv.org/abs/2504.14439)
- **Reward Model:** [Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2)
- **PRISM Dataset:** [HuggingFace](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)

---

## License
CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file.
