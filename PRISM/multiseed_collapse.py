"""
Multi-seed K=5 collapse check: is basis collapse driven by the data alone, or the data PLUS the
shared-anchor regularizer? For each seed we train K=5 with the correct anchor at alpha=0 (no
regularization) and alpha=1000, and measure how collapsed the learned bases are.

Metrics per run (on the FULL [4096,5] basis matrix, all columns):
  min_abs_cos  = smallest |cosine| between any two basis columns (1.0 => identical directions)
  s2_over_s1   = 2nd/1st singular value          (0.0 => effectively rank-1)
  bases_kept   = columns surviving the softmax-weight pruning
  test_acc     = seen-test accuracy

Expected (to be confirmed): alpha=0 -> highly correlated but NOT identical (partial collapse);
alpha=1000 -> hard rank-1 collapse. If that holds across seeds, the regularizer is a real
co-driver of the collapse, not prompt-uniqueness alone.
"""
import os
import sys
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized  # noqa: E402
from rm_head_utils import load_reward_head  # noqa: E402

ITERS, LR, K = 1500, 0.5, 5
SEEDS = [0, 1, 2, 42]
ALPHAS = [0.0, 1000.0]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


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
    # feats live on CPU; V/W may be on GPU after training -> bring everything to CPU to eval.
    V = V.detach().cpu()
    W = W.detach().cpu()
    return float(np.mean([(X.cpu() @ (V @ W[i]) > 0).float().mean().item()
                          for i, X in enumerate(feats)]))


def collapse_metrics(V_full):
    Vn = F.normalize(V_full, dim=0)
    cos = Vn.t() @ Vn
    k = V_full.shape[1]
    off = cos[~torch.eye(k, dtype=torch.bool)]
    s = torch.linalg.svdvals(V_full)
    return off.abs().min().item(), (s[1] / s[0]).item()


def main():
    print("Loading embeddings + correct anchor...", flush=True)
    tr = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism", "train_embeddings.pkl"),
                    weights_only=False)
    te = torch.load(os.path.join(SCRIPT_DIR, "..", "data", "prism", "test_embeddings.pkl"),
                    weights_only=False)
    train_seen, test_seen = group(tr, True, "train"), group(te, True, "test")
    head = load_reward_head().reshape(-1, 1)

    out_dir = os.path.join(SCRIPT_DIR, "..", "results", "correct_anchor")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "multiseed_collapse.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["K", "alpha", "seed", "iters", "bases_kept", "min_abs_cos_cols",
                "s2_over_s1", "test_acc"])
    f.flush()

    print(f"\n{'K':>2} | {'alpha':>7} | {'seed':>4} | {'kept':>4} | {'min|cos|':>9} | "
          f"{'s2/s1':>8} | {'test_acc':>8}", flush=True)
    print("-" * 60, flush=True)
    rows = defaultdict(list)
    for alpha in ALPHAS:
        for seed in SEEDS:
            set_seed(seed)
            m = LoRe_regularized(head, alpha, len(train_seen), 4096, K, ITERS, LR)
            Wk, Vk = m.train(train_seen)
            mincos, s2s1 = collapse_metrics(m.V.detach())
            ta = eval_acc(Wk, Vk.detach(), test_seen)
            print(f"{K:>2} | {alpha:>7.0f} | {seed:>4} | {Vk.shape[1]:>4} | {mincos:>9.4f} | "
                  f"{s2s1:>8.4f} | {ta:>8.4f}", flush=True)
            w.writerow([K, alpha, seed, ITERS, Vk.shape[1], f"{mincos:.4f}",
                        f"{s2s1:.4f}", f"{ta:.4f}"])
            f.flush()
            rows[alpha].append((mincos, s2s1))
    f.close()

    print("\n=== summary (mean +/- std across seeds) ===", flush=True)
    for alpha in ALPHAS:
        mc = np.array([r[0] for r in rows[alpha]])
        ss = np.array([r[1] for r in rows[alpha]])
        print(f"alpha={alpha:>7.0f}: min|cos| {mc.mean():.4f}+/-{mc.std():.4f} | "
              f"s2/s1 {ss.mean():.4f}+/-{ss.std():.4f}", flush=True)
    print(f"\nSaved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
