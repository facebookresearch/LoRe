"""
Few-shot adaptation to UNSEEN users -- the second half of the wbar/delta design.

Train LoReV2 on a set of seen users, then hold V and wbar FIXED and learn only a new delta_u for
each unseen user from a handful of their pairs. The question is whether learning that delta buys
anything over just handing the new user the population reward function.

Reported on unseen users' held-out pairs, so their test prompts were never seen by anything:
  * base_rm      : Skywork's true reward head (untrained bar)
  * wbar_only    : the fitted population direction, NO adaptation at all
  * adapted      : wbar + delta_u learned from `shots` pairs of that user
  * meandiff     : the user's own mean-diff direction from the same shots, as a reference
                   estimator (unshrunk -- adapted should beat it if the ridge is doing its job)

Read-out:
  adapted > wbar_only, growing with shots -> few-shot personalization works
  adapted ~= wbar_only at every budget    -> nothing user-specific to adapt to, and the low-rank
                                             basis is just carrying a shared quality axis

CPU-only once embeddings exist.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2, PersonalizeDelta                           # noqa: E402
from community_alignment_lore import build_user_diffs, unit, acc     # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--unseen_frac", type=float, default=0.3,
                    help="fraction of users held out entirely from training V and wbar")
    ap.add_argument("--shots", type=int, nargs="+", default=[5, 10, 25, 50],
                    help="pairs per unseen user used to fit their delta")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adapt_lr", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--adapt_iters", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "unseen_user.csv"))
    args = ap.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    head = unit(load_reward_head().reshape(-1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["seed", "shots", "n_unseen", "base_rm", "wbar_only",
                                      "adapted", "meandiff", "adapted_minus_wbar"])
    w.writeheader()

    rows = []
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)

        users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                                 split_by_turn=True, val_frac=args.val_frac)
        uids = sorted(users)
        perm = torch.randperm(len(uids), generator=gen).tolist()
        n_unseen = int(round(args.unseen_frac * len(uids)))
        unseen = [uids[i] for i in perm[:n_unseen]]
        seen = [uids[i] for i in perm[n_unseen:]]

        # ---- train the shared model on SEEN users only ----
        tr = [users[u][0] for u in seen]
        va = [users[u][1] for u in seen]
        model = LoReV2(len(tr), 4096, args.rank, lam_pop=args.lam_pop, lam_d=args.lam_d,
                       num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        model.train(tr, val=va)
        V = model.V.detach()
        wbar = model.wbar.detach()
        pop_dir = (V @ wbar).detach()
        if seed == args.seeds[0]:
            print(f"{len(seen)} seen / {len(unseen)} unseen users, rank {args.rank}\n")

        for shots in args.shots:
            # each unseen user contributes `shots` pairs to fit their delta; evaluate on their
            # held-out test pairs, which nothing has touched
            shot_feats, test_feats, keep = [], [], []
            for u in unseen:
                tr_u, te_u = users[u][0], users[u][2]
                if tr_u.shape[0] < shots:
                    continue
                idx = torch.randperm(tr_u.shape[0], generator=gen)[:shots]
                shot_feats.append(tr_u[idx]); test_feats.append(te_u); keep.append(u)

            fitter = PersonalizeDelta(len(shot_feats), args.rank, lam_d=args.lam_d,
                                      num_iterations=args.adapt_iters,
                                      learning_rate=args.adapt_lr)
            W_ad = fitter.train(shot_feats, V, wbar)

            base_a, wbar_a, ad_a, md_a = [], [], [], []
            for i, u in enumerate(keep):
                te = test_feats[i]
                base_a.append(acc(te, head))
                wbar_a.append(acc(te, pop_dir))
                ad_a.append(acc(te, V @ W_ad[i]))
                md_a.append(acc(te, unit(shot_feats[i].mean(0))))
            m = lambda x: float(np.mean(x))
            row = {"seed": seed, "shots": shots, "n_unseen": len(keep),
                   "base_rm": round(m(base_a), 4), "wbar_only": round(m(wbar_a), 4),
                   "adapted": round(m(ad_a), 4), "meandiff": round(m(md_a), 4),
                   "adapted_minus_wbar": round(m(ad_a) - m(wbar_a), 4)}
            rows.append(row); w.writerow(row); f.flush()
            print(f"seed {seed} shots {shots:>3}: base {row['base_rm']:.4f} | "
                  f"wbar {row['wbar_only']:.4f} | adapted {row['adapted']:.4f} | "
                  f"meandiff {row['meandiff']:.4f} | adapted-wbar {row['adapted_minus_wbar']:+.4f}")
    f.close()

    print(f"\n=== mean over {len(args.seeds)} seeds (unseen users, held-out pairs) ===")
    print(f"{'shots':>6} | {'base_rm':>8} | {'wbar_only':>9} | {'adapted':>8} | {'meandiff':>8} | "
          f"{'adapted - wbar':>15}")
    print("-" * 74)
    for shots in args.shots:
        rs = [r for r in rows if r["shots"] == shots]
        if not rs:
            continue
        g = lambda k: np.array([r[k] for r in rs])
        d = g("adapted") - g("wbar_only")
        print(f"{shots:>6} | {g('base_rm').mean():>8.4f} | {g('wbar_only').mean():>9.4f} | "
              f"{g('adapted').mean():>8.4f} | {g('meandiff').mean():>8.4f} | "
              f"{d.mean():>+9.4f} +/- {d.std():.4f}")
    print(f"\nSaved {args.out}")
    print("Read-out: adapted > wbar_only and growing with shots => few-shot personalization works. "
          "Flat => there is nothing user-specific for a new user's delta to capture.")


if __name__ == "__main__":
    main()
