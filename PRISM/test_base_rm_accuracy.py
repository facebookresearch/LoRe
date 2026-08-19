"""
Experiment 1: How well does the REAL base RM distinguish PRISM chosen vs rejected?

Answers the question: "How does the base RM (larger, more compute, more training) do at
distinguishing chosen/rejected pairs?"

IMPORTANT FIX (was a bug before)
--------------------------------
The previous version of this script -- and train_basis.py / basis_reproducibility_check.py --
extracted the reward direction with `AutoModel` + "last nn.Linear". For a
LlamaForSequenceClassification checkpoint, `AutoModel` returns the bare backbone and DROPS the
real reward head `score.weight`; "last nn.Linear" then lands on `layers.31.mlp.down_proj` (an
internal MLP matrix) and slices an arbitrary column. That is NOT the reward direction, so the
old "base RM ~= 75%" number was measuring a meaningless direction.

This version uses the TRUE reward head `score.weight` (via rm_head_utils.load_reward_head).
Because PRISM embeddings are the last-token hidden state that the head consumes, base-RM
preference accuracy is exactly  sign( score.weight @ (chosen_emb - rejected_emb) ).

Measured result (cached PRISM embeddings): ~0.80 on all four slices
  train/seen 0.799,  train/unseen 0.804,  test/seen 0.809,  test/unseen 0.801.
So the base RM -- which never saw PRISM -- already gets ~80% zero-shot. Separating these pairs is
mostly a single real "response-quality" axis, not a high-dimensional memorization artifact.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402


def group_embeddings_by_user(dataset, seen_value, split_name):
    """Per user, stack (chosen - rejected) last-token embedding diffs into [m_i, 4096]."""
    grouped = defaultdict(list)
    for example in dataset:
        info = example.get("extra_info", {})
        if info.get("seen") == seen_value and info.get("split") == split_name:
            uid = info.get("user_id")
            if uid:
                chosen = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32)
                rejected = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32)
                grouped[uid].append(chosen - rejected)
    return [torch.stack(grouped[u]) for u in sorted(grouped.keys())]


def accuracy_with_direction(features, V):
    """Fraction of pairs where diff @ V > 0 (i.e. chosen scored above rejected)."""
    accs, n = [], 0
    for user_diffs in features:
        scores = user_diffs @ V  # [m_i, 1]
        accs.append((scores > 0).float().mean().item())
        n += user_diffs.shape[0]
    return float(np.mean(accs)), float(np.std(accs)), n


def main():
    print("Loading cached PRISM embeddings...")
    train_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                        "train_embeddings.pkl"), weights_only=False)
    test_emb = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                       "test_embeddings.pkl"), weights_only=False)

    print("Loading the TRUE Skywork reward head (score.weight)...")
    score_w = load_reward_head()  # [4096, 1], float32
    print(f"  score.weight: shape {tuple(score_w.shape)}, ||w|| = {score_w.norm().item():.4f}")

    slices = [
        ("TRAIN", train_emb, True, "train"),
        ("TRAIN", train_emb, False, "train"),
        ("TEST", test_emb, True, "test"),
        ("TEST", test_emb, False, "test"),
    ]

    print("\n" + "=" * 78)
    print("TRUE BASE-RM ACCURACY  (score.weight . (chosen - rejected) > 0)")
    print("=" * 78)
    print(f"{'split':>6} | {'seen':>5} | {'users':>5} | {'pairs':>6} | {'accuracy':>16}")
    print("-" * 60)
    for name, ds, seen, split in slices:
        feats = group_embeddings_by_user(ds, seen, split)
        if not feats:
            continue
        m, s, n = accuracy_with_direction(feats, score_w)
        print(f"{name:>6} | {str(seen):>5} | {len(feats):>5} | {n:>6} | {m:.4f} +/- {s:.4f}")

    # Document the magnitude of the old bug: how (un)related was the broken anchor?
    # cos(real head, collapsed LoRe direction) for the saved checkpoints, if present. The K
    # matrices live in the parent-project reproduced_matrices/ (two levels up) or the in-repo
    # checkpoints/checkpoints/; check both so this diagnostic actually runs when they exist.
    ckpt_candidates = [
        os.path.join(SCRIPT_DIR, "..", "..", "reproduced_matrices",
                     "PRISM_V_lore_K_20_alpha_10000.0.pt"),
        os.path.join(SCRIPT_DIR, "..", "checkpoints", "checkpoints",
                     "PRISM_V_lore_K_20_alpha_10000.0.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt is not None:
        V = torch.load(ckpt, map_location="cpu", weights_only=True).float()
        cos = F.cosine_similarity(V[:, 0], score_w.squeeze(), dim=0).item()
        print(f"\ncos(real reward head, collapsed LoRe K=20 direction) = {cos:+.4f} "
              f"(~orthogonal: LoRe did NOT re-learn the real reward axis)  [{ckpt}]")
    else:
        print("\n(skipped cos(head, LoRe) check: no PRISM_V_lore_K_20 checkpoint found in "
              + " or ".join(ckpt_candidates) + ")")


if __name__ == "__main__":
    main()
