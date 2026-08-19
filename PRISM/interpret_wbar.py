"""
What does the learned shared reward direction actually encode?

This is the original interpretability question, asked about the object that turned out to exist.
LoRe's per-user bases carry no recoverable signal, but LoReV2's population direction (V @ wbar)
beats Skywork's own reward head by ~+0.030 on held-out Community Alignment pairs, in-sample and on
users it never trained on. Nobody has looked at what it measures.

Three directions are scored against the 11 concept vectors:
  * wbar        : V @ wbar, the learned population reward direction
  * base_head   : Skywork's true score.weight, the pretrained reward head it beats
  * residual    : the component of wbar ORTHOGONAL to base_head, i.e. specifically what the
                  learned direction adds. This is the interesting one -- if wbar simply
                  rediscovered the pretrained head, the residual would align with nothing.

Significance uses the same convention as concept_basis_alignment.py: the 95th percentile of
cosine similarities between the concept vector and random vectors in the same space.

CPU-only.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2                                             # noqa: E402
from community_alignment_lore import build_user_diffs, unit          # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402


def null_threshold(dim, n=2000, pct=95, gen=None):
    """95th percentile of |cos| between a fixed vector and random ones -- direction-independent,
    since cosine against isotropic noise does not depend on which vector you fix."""
    R = torch.randn(n, dim, generator=gen)
    v = torch.randn(dim, generator=gen)
    c = (F.normalize(R, dim=1) @ unit(v)).abs()
    return float(torch.quantile(c, pct / 100.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--concept_vectors", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                              "concept_vectors.pt"))
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "wbar_concepts.csv"))
    args = ap.parse_args()

    cvs = torch.load(args.concept_vectors, weights_only=False)
    concepts = list(cvs)
    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    head = unit(load_reward_head().reshape(-1).float())

    tau = null_threshold(4096, gen=torch.Generator().manual_seed(0))
    print(f"{len(concepts)} concepts | null threshold |cos| = {tau:.4f} (95th pct vs random)\n")

    per_seed = {"wbar": [], "base_head": [], "residual": []}
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                                 split_by_turn=True, val_frac=args.val_frac)
        uids = sorted(users)
        tr = [users[u][0] for u in uids]
        va = [users[u][1] for u in uids]
        m = LoReV2(len(tr), 4096, args.rank, lam_pop=args.lam_pop, lam_d=args.lam_d,
                   num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        m.train(tr, val=va)

        wbar = unit((m.V.detach() @ m.wbar.detach()).float())
        # sign is arbitrary in the factorization; orient so it agrees with the pretrained head,
        # otherwise the concept cosines flip sign between seeds for no meaningful reason
        if float(wbar @ head) < 0:
            wbar = -wbar
        resid = unit(wbar - float(wbar @ head) * head)

        for name, d in (("wbar", wbar), ("base_head", head), ("residual", resid)):
            per_seed[name].append({c: float(d @ unit(cvs[c].float())) for c in concepts})
        print(f"seed {seed}: cos(wbar, base_head) = {float(wbar @ head):+.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "concept", "cos_mean", "cos_std",
                                          "null_tau", "significant"])
        w.writeheader()
        for name in per_seed:
            for c in concepts:
                v = np.array([s[c] for s in per_seed[name]])
                w.writerow({"direction": name, "concept": c, "cos_mean": round(v.mean(), 4),
                            "cos_std": round(v.std(), 4), "null_tau": round(tau, 4),
                            "significant": bool(abs(v.mean()) > tau)})

    print(f"\ncosine with concept vectors, mean over {len(args.seeds)} seeds "
          f"(* = |cos| exceeds the {tau:.3f} null)\n")
    print(f"{'concept':<14} | {'wbar':>16} | {'base_head':>10} | {'residual':>16}")
    print("-" * 68)
    order = sorted(concepts, key=lambda c: -abs(np.mean([s[c] for s in per_seed['wbar']])))
    for c in order:
        a = np.array([s[c] for s in per_seed["wbar"]])
        b = per_seed["base_head"][0][c]
        r = np.array([s[c] for s in per_seed["residual"]])
        star = lambda x: "*" if abs(x) > tau else " "
        print(f"{c:<14} | {a.mean():+.4f}{star(a.mean())} +/- {a.std():.4f} | "
              f"{b:+.4f}{star(b)} | {r.mean():+.4f}{star(r.mean())} +/- {r.std():.4f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
