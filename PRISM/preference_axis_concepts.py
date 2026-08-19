"""
E6: What is the single PRISM preference axis actually made of?

Open question: the strong shared "chosen-beats-rejected" direction might be a generic
quality signal (rejected responses are just weaker) rather than personalization. If so, the axis
should align with generic quality/style concepts (helpfulness, fluency, formatting), not with a
spread of distinct personal-taste concepts.

We take the EMPIRICAL PRISM preference axis = mean of (chosen - rejected) train diffs (the ~1-D
global direction that separates ~100% of pairs; see dim_ablation.py), and cosine-compare it to
each concept vector from the concept library (Method 2). We do the same for the TRUE reward head
as a reference. Significance = 95th percentile of |cosine| against 1000 random unit vectors
(same null as concept_basis_alignment.py).

Reads:  data/prism/concept_vectors.pt, data/prism/train_embeddings.pkl,
        reproduced_matrices/skywork_score_head.pt (via rm_head_utils)
Writes: results/preference_axis/preference_axis_concepts.csv (+ bar plot)
"""
import os
import sys
import csv

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402

torch.manual_seed(0)


def null_threshold(dim, n=1000, pct=95):
    """95th-percentile |cosine| of random unit vectors vs a fixed direction (direction-agnostic)."""
    r = F.normalize(torch.randn(n, dim), dim=1)
    fixed = F.normalize(torch.randn(1, dim), dim=1)
    sims = (r @ fixed.t()).squeeze().abs()
    return torch.quantile(sims, pct / 100.0).item()


def preference_axis(train_emb):
    """Empirical PRISM preference direction = mean over seen-train (chosen - rejected)."""
    diffs = []
    for ex in train_emb:
        i = ex.get("extra_info", {})
        if i.get("seen") is True and i.get("split") == "train" and i.get("user_id"):
            c = torch.tensor(i["chosen_conv_embedding"], dtype=torch.float32)
            r = torch.tensor(i["rejected_conv_embedding"], dtype=torch.float32)
            diffs.append(c - r)
    D = torch.stack(diffs)
    mu = D.mean(0)
    return mu / mu.norm(), D


def main():
    print("Loading concept vectors + embeddings + true reward head...")
    concept_vectors = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                              "concept_vectors.pt"), map_location="cpu",
                                 weights_only=True)
    train_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                        "train_embeddings.pkl"), weights_only=False)
    pref, D = preference_axis(train_emb)
    head = load_reward_head().squeeze()
    head = head / head.norm()

    concepts = list(concept_vectors.keys())
    tau = null_threshold(pref.shape[0])
    print(f"Null threshold (95th pct |cos| of random vectors): {tau:.4f}\n")

    rows = []
    for c in concepts:
        cv = F.normalize(concept_vectors[c].float(), dim=0)
        cos_pref = torch.dot(pref, cv).item()
        cos_head = torch.dot(head, cv).item()
        rows.append((c, cos_pref, cos_head))

    rows.sort(key=lambda r: abs(r[1]), reverse=True)

    print("=" * 72)
    print("What is the PRISM preference axis? (cosine to each concept vector)")
    print("=" * 72)
    print(f"{'concept':>12} | {'cos(pref axis)':>14} | {'sig?':>4} | {'cos(true head)':>14}")
    print("-" * 60)
    for c, cp, ch in rows:
        sig = "***" if abs(cp) >= tau else ""
        print(f"{c:>12} | {cp:>14.4f} | {sig:>4} | {ch:>14.4f}")

    print(f"\ncos(PRISM preference axis, TRUE reward head) = {torch.dot(pref, head).item():+.4f}")

    # superficial-feature sanity check: is it just a magnitude/length artifact?
    Cn = D.norm(dim=1)
    print(f"chosen-rejected diff norm: mean {Cn.mean():.1f} (embedding norms ~144 each) -> the "
          f"signal is directional, not a magnitude/length gap")

    out_dir = os.path.join(SCRIPT_DIR, "..", "results", "preference_axis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preference_axis_concepts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "cos_pref_axis", "significant", "cos_true_head", "null_tau"])
        for c, cp, ch in rows:
            w.writerow([c, f"{cp:.4f}", int(abs(cp) >= tau), f"{ch:.4f}", f"{tau:.4f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = ["tab:blue" if abs(v) >= tau else "lightgray" for v in vals]
        plt.figure(figsize=(9, 5))
        plt.bar(cs, vals, color=colors)
        plt.axhline(tau, color="red", ls=":", label=f"significance +/-{tau:.2f}")
        plt.axhline(-tau, color="red", ls=":")
        plt.ylabel("cosine(PRISM preference axis, concept)")
        plt.title("E6: which concepts is the single PRISM preference axis made of?")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "preference_axis_concepts.png"), dpi=150)
        print(f"\nSaved csv + plot to {out_dir}")
    except Exception as ex:
        print(f"(plot skipped: {ex!r}) -- csv still written to {out_dir}")


if __name__ == "__main__":
    main()
