"""
Retrain LoRe with the CORRECTED anchor (real Skywork reward head) and test two predictions:

  (a) With the correct anchor and weaker regularization, the learned K=1 direction should move
      toward the empirical mean-diff separator and accuracy toward ~100%. As alpha grows, it
      should be pulled toward the true reward head and accuracy toward the base-RM ~80%.
  (b) The basis should STILL collapse to rank-1, because the PRISM signal is ~1-D (see
      dim_ablation.py): with K>1 every column converges to (near) the same direction.

This is a REDUCED-BUDGET diagnostic (CPU, ~1500 iters, converges to NLL~0 well before that at
alpha=0), not a full 20k-iter production sweep -- enough to test the predictions, not to ship
checkpoints. Uses seen-train to fit and seen-test to evaluate.
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
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import LoRe_regularized  # noqa: E402
from rm_head_utils import load_reward_head      # noqa: E402

ITERS = 1500
LR = 0.5
SEED = 42


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def group(ds, seen, split):
    g = defaultdict(list)
    for ex in ds:
        i = ex.get("extra_info", {})
        if i.get("seen") == seen and i.get("split") == split and i.get("user_id"):
            c = torch.tensor(i["chosen_conv_embedding"], dtype=torch.float32)
            r = torch.tensor(i["rejected_conv_embedding"], dtype=torch.float32)
            g[i["user_id"]].append(c - r)
    return [torch.stack(g[u]) for u in sorted(g)]


def eval_acc(W, V, feats):
    accs = []
    for i, X in enumerate(feats):
        s = X @ (V @ W[i])
        accs.append((s > 0).float().mean().item())
    return float(np.mean(accs))


def collapse_metrics(V_full):
    """min pairwise |cosine| among columns and s2/s1 (both ~1 / ~0 => rank-1 collapse)."""
    Vn = F.normalize(V_full, dim=0)
    cos = Vn.t() @ Vn
    K = V_full.shape[1]
    if K == 1:
        return 1.0, 0.0
    off = cos[~torch.eye(K, dtype=torch.bool)]
    s = torch.linalg.svdvals(V_full)
    return off.abs().min().item(), (s[1] / s[0]).item()


def main():
    print("Loading embeddings + anchor (real reward head)...")
    tr = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism", "train_embeddings.pkl"),
                    weights_only=False)
    te = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism", "test_embeddings.pkl"),
                    weights_only=False)
    train_seen, test_seen = group(tr, True, "train"), group(te, True, "test")
    head = load_reward_head().reshape(-1, 1)                       # [4096,1] correct anchor
    head_u = F.normalize(head.squeeze(), dim=0)

    D = torch.cat(train_seen, 0)
    pref = F.normalize(D.mean(0), dim=0)                           # empirical mean-diff separator
    base_acc = eval_acc([torch.ones(1)] * len(test_seen),
                        head, test_seen)                           # true-head baseline
    print(f"Users {len(train_seen)} | base-RM (true head) seen-test acc = {base_acc:.4f}")
    print(f"cos(mean-diff separator, true head) = {torch.dot(pref, head_u).item():+.4f}\n")

    rows = []

    print("=== Prediction (a): K=1, sweep alpha with the CORRECT anchor ===")
    print(f"{'alpha':>8} | {'test_acc':>8} | {'cos(dir,mean-diff)':>18} | {'cos(dir,true head)':>18}")
    print("-" * 62)
    for alpha in [0.0, 100.0, 1000.0, 10000.0]:
        set_seed(SEED)
        am = LoRe_regularized(head, alpha, len(train_seen), 4096, 1, ITERS, LR)
        Wk, Vk = am.train(train_seen)
        v = F.normalize(am.V.detach().squeeze(), dim=0)
        acc = eval_acc(Wk, Vk.detach(), test_seen)
        cmd, chd = torch.dot(v, pref).item(), torch.dot(v, head_u).item()
        print(f"{alpha:>8.0f} | {acc:>8.4f} | {cmd:>+18.4f} | {chd:>+18.4f}")
        rows.append(["K=1", alpha, acc, cmd, chd, 1, 1.0, 0.0])

    print("\n=== Prediction (b): K=5, does collapse persist with the correct anchor? ===")
    print(f"{'alpha':>8} | {'test_acc':>8} | {'bases_kept':>10} | {'min|cos| cols':>13} | {'s2/s1':>8}")
    print("-" * 62)
    for alpha in [0.0, 1000.0]:
        set_seed(SEED)
        am = LoRe_regularized(head, alpha, len(train_seen), 4096, 5, ITERS, LR)
        Wk, Vk = am.train(train_seen)
        acc = eval_acc(Wk, Vk.detach(), test_seen)
        mincos, s2s1 = collapse_metrics(am.V.detach())
        print(f"{alpha:>8.0f} | {acc:>8.4f} | {Vk.shape[1]:>10} | {mincos:>13.4f} | {s2s1:>8.4f}")
        rows.append(["K=5", alpha, acc, float("nan"), float("nan"), Vk.shape[1], mincos, s2s1])

    out_dir = os.path.join(SCRIPT_DIR, "..", "results", "correct_anchor")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrain_correct_anchor.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "alpha", "test_acc", "cos_meandiff", "cos_truehead",
                    "bases_kept", "min_abs_cos_cols", "s2_over_s1"])
        w.writerows(rows)
    print(f"\nSaved {out_dir}/retrain_correct_anchor.csv "
          f"(ITERS={ITERS}, LR={LR}, seed={SEED}, reduced-budget CPU diagnostic)")


if __name__ == "__main__":
    main()
