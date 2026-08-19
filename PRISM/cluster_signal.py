"""
Is there learnable preference signal at the GROUP level, even if not per-individual?

Per-user failed (per_user_signal.py): a user's own direction predicts their held-out pairs worse
than a single global direction. The obvious objection is sparsity -- ~15 pairs/user is too few to
fit an individual. This script tests one level up: cluster users by their preference direction,
fit ONE direction per cluster on the cluster's POOLED train pairs (many more pairs), and check
whether a user's own-cluster direction beats the global direction on their held-out pairs.

Read-out:
  * cluster >> global (and grows with #clusters up to a point) -> real group-level structure;
    personalization is learnable at the cluster level -> a concrete path (few group-shaped bases).
  * cluster ~= global for all K -> no separable structure even after pooling; closes the sparsity
    objection to the per-user null.

Controls, all evaluated per user on held-out TEST pairs:
  global        : one shared mean-diff direction (the bar to beat)
  own_cluster   : direction of the cluster this user was assigned to (from TRAIN directions)
  random_cluster: a random OTHER cluster's direction (does assignment matter, or is any cluster
                  direction just ~the global axis?)

CPU-only. Uses per-user TRAIN mean-diff directions for clustering so test pairs never leak.
"""
import os
import sys
import random

import numpy as np
import torch
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402

MIN_TRAIN, MIN_TEST = 5, 3
KS = [2, 3, 5, 10, 20, 50]
SEED = 0


def diffs_by_user(dataset, seen_value, split_name):
    g = defaultdict(list)
    for ex in dataset:
        i = ex.get("extra_info", {})
        if i.get("seen") == seen_value and i.get("split") == split_name and i.get("user_id"):
            c = torch.tensor(i["chosen_conv_embedding"], dtype=torch.float32)
            r = torch.tensor(i["rejected_conv_embedding"], dtype=torch.float32)
            g[i["user_id"]].append(c - r)
    return {u: torch.stack(v) for u, v in g.items()}


def unit(v):
    return v / (v.norm() + 1e-8)


def acc(D, direction):
    return (D @ direction > 0).float().mean().item()


def kmeans_cosine(X, k, iters=50, seed=0):
    """Spherical k-means (cosine) on unit rows X:[n,d]. Returns labels:[n], centroids:[k,d] (unit)."""
    g = torch.Generator().manual_seed(seed)
    Xn = torch.nn.functional.normalize(X, dim=1)
    cent = Xn[torch.randperm(Xn.shape[0], generator=g)[:k]].clone()
    labels = torch.zeros(Xn.shape[0], dtype=torch.long)
    for _ in range(iters):
        sim = Xn @ cent.t()                       # [n,k] cosine
        new = sim.argmax(1)
        if torch.equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                cent[j] = unit(Xn[m].mean(0))
            else:  # empty cluster: reseed to a random point
                cent[j] = Xn[torch.randint(0, Xn.shape[0], (1,), generator=g)].squeeze(0)
    return labels, cent


def main():
    import argparse
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(SCRIPT_DIR, "..", "data", "prism")
    ap.add_argument("--train", default=os.path.join(default_dir, "train_embeddings.pkl"))
    ap.add_argument("--test", default=os.path.join(default_dir, "test_embeddings.pkl"))
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Loading embeddings:\n  train: {args.train}\n  test:  {args.test}")
    tr = torch.load(args.train, weights_only=False)
    te = torch.load(args.test, weights_only=False)
    train_u = diffs_by_user(tr, True, "train")
    test_u = diffs_by_user(te, True, "test")

    users = [u for u in train_u
             if u in test_u and train_u[u].shape[0] >= MIN_TRAIN and test_u[u].shape[0] >= MIN_TEST]
    print(f"{len(users)} qualifying seen users")

    # Per-user TRAIN direction (used ONLY for clustering) + pooled train diffs per user.
    user_dir = {u: unit(train_u[u].mean(0)) for u in users}
    X = torch.stack([user_dir[u] for u in users])          # [n_users, 4096], unit rows
    global_dir = unit(torch.cat([train_u[u] for u in users], 0).mean(0))
    head = unit(load_reward_head().reshape(-1))

    # Baselines that don't depend on K
    g_acc = np.mean([acc(test_u[u], global_dir) for u in users])
    h_acc = np.mean([acc(test_u[u], head) for u in users])
    print(f"\nglobal_meandiff : {g_acc:.4f}   (bar to beat)")
    print(f"true_head       : {h_acc:.4f}\n")

    print(f"{'#clusters':>9} | {'own_cluster':>11} | {'random_cluster':>14} | "
          f"{'own - global':>12} | {'cluster sizes (min/med/max)':>28}")
    print("-" * 92)
    rows = []
    for k in KS:
        labels, _ = kmeans_cosine(X, k, seed=SEED)
        # cluster direction = unit mean of POOLED train diffs of the cluster's users (train only)
        cdir = {}
        for j in range(k):
            members = [users[i] for i in range(len(users)) if labels[i] == j]
            if not members:
                continue
            pooled = torch.cat([train_u[u] for u in members], 0)
            cdir[j] = unit(pooled.mean(0))
        own, rnd = [], []
        for i, u in enumerate(users):
            j = int(labels[i])
            own.append(acc(test_u[u], cdir[j]))
            others = [x for x in cdir if x != j]
            rnd.append(acc(test_u[u], cdir[random.choice(others)]) if others else float("nan"))
        sizes = np.bincount(labels.numpy(), minlength=k)
        own_m = float(np.mean(own)); rnd_m = float(np.nanmean(rnd))
        print(f"{k:>9} | {own_m:>11.4f} | {rnd_m:>14.4f} | {own_m - g_acc:>+12.4f} | "
              f"{sizes.min():>6}/{int(np.median(sizes)):>3}/{sizes.max():<5}")
        rows.append((k, own_m, rnd_m, own_m - g_acc))

    best = max(rows, key=lambda r: r[1])
    print(f"\nBest: K={best[0]} own_cluster={best[1]:.4f} (own-global {best[3]:+.4f})")
    print("Read-out: if own_cluster clearly beats global and exceeds random_cluster -> real "
          "group-level signal. If own ~= global ~= random for all K -> no separable structure "
          "even after pooling (per-user null is not just a sparsity artifact).")


if __name__ == "__main__":
    main()
