#!/usr/bin/env python3
"""Fit the LoRe v2 basis (LoReV2) on PRISM and save it for the SAE experiments.

PRISM-only. No CA / MultiPref anywhere. Produces the v2 analogue of the vanilla
`basis_matrices.pt`: instead of softmax simplex weights W anchored to the reward
head (alpha=1e4), v2 learns signed weights wbar + delta_u with a reward-space
ridge and NO head anchor. The shared population direction the SAE work attributes
against is therefore

    v_pop = unit( V @ wbar )                     (not unit(V @ mean_users(W)))

Reads:  PRISM/data/prism/{train,test}_embeddings.pkl   (per-user chosen/rejected)
Writes: PRISM/basis_v2.pt   dict keyed by run_key ->
        {V [4096,K], wbar [K], delta [users,K], wbar_dir [4096], uids, lam_pop,
         lam_d, K, seed}
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
# reuse the exact PRISM per-user grouping the experiments use
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "sae", "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import load_user_diffs                     # noqa: E402
from utils import LoReV2, set_seed                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10, help="number of basis vectors")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="ridge on ||V wbar||^2")
    ap.add_argument("--lam_d", type=float, default=0.01, help="ridge on mean_u ||V delta_u||^2")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "basis_v2.pt"))
    ap.add_argument("--run_key", default=None, help="defaults to PART2_K{K}_seed{seed}_v2")
    args = ap.parse_args()

    run_key = args.run_key or f"PART2_K{args.K}_seed{args.seed}_v2"
    set_seed(args.seed)

    # Fit on the seen users' train pairs; early-stop on their held-out (test-split) pairs.
    train = load_user_diffs(split="train", seen=True)
    val = load_user_diffs(split="test", seen=True)
    if len(val) != len(train):
        # user sets/order must line up for LoReV2's per-user val; fall back to train-loss stopping
        print(f"[warn] seen-user count differs train={len(train)} val={len(val)}; "
              f"stopping on train loss instead", file=sys.stderr)
        val = None
    n_pairs = sum(int(x.shape[0]) for x in train)
    print(f"[fit_v2] K={args.K} users={len(train)} train_pairs={n_pairs} "
          f"lam_pop={args.lam_pop} lam_d={args.lam_d} run_key={run_key}")

    model = LoReV2(len(train), 4096, args.K, lam_pop=args.lam_pop, lam_d=args.lam_d,
                   num_iterations=args.iters, learning_rate=args.lr)
    model.train(train, val=val)

    V = model.V.detach().cpu().float()
    wbar = model.wbar.detach().cpu().float()
    delta = model.delta.detach().cpu().float()
    wbar_dir = (V @ wbar).float()

    entry = {"V": V, "wbar": wbar, "delta": delta, "wbar_dir": wbar_dir,
             "uids": list(range(len(train))), "lam_pop": args.lam_pop,
             "lam_d": args.lam_d, "K": args.K, "seed": args.seed}
    store = torch.load(args.out, map_location="cpu") if os.path.exists(args.out) else {}
    store[run_key] = entry
    torch.save(store, args.out)

    vp = wbar_dir / wbar_dir.norm().clamp_min(1e-8)
    print(f"[fit_v2] saved {run_key} -> {args.out}")
    print(f"[fit_v2] ||wbar||={wbar.norm():.3f} ||wbar_dir||={wbar_dir.norm():.3f} "
          f"v_pop[:3]={vp[:3].tolist()}")


if __name__ == "__main__":
    main()
