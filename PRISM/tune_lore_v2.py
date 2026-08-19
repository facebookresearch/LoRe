"""
Tune LoReV2's learning rate and regularization strengths on a HELD-OUT VALIDATION split.

Spec: sweep lr through {1e-4, 1e-3, 1e-2, 0.1} and grid lam_pop x lam_d over
{0.001, 0.01, 0.1, 1, 10}, both at a fixed K (8), with everything selected on validation and the
chosen values then reused for every K -- regularization must not depend on K.

Factorization: lr first (at a mid-grid regularization), then the 5x5 lam grid at the chosen lr.
That is 4 + 25 = 29 runs instead of the 100 a full joint sweep would cost. lr and regularization
do interact, so the script finishes with a narrow re-check of lr at the winning lam pair; if that
moves the answer, the factorization was too coarse and it says so.

Selection metric is validation PAIRWISE ACCURACY, not validation loss: accuracy is what every
downstream comparison against base_rm reports, and the ridge makes the loss scale incomparable
across different lam settings.

CPU-only. Reuses the leak-free turn-level split from community_alignment_lore.py.
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
from utils import LoReV2                                             # noqa: E402
from community_alignment_lore import build_user_diffs, unit          # noqa: E402
from synthetic_recovery import eval_acc, collapse_metrics            # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402


def fit_eval(train_feats, val_feats, K, lam_pop, lam_d, lr, iters, init_seed=0):
    """Fit on train, report accuracy on val. Never touches test.

    init_seed is reset before every fit so all grid cells start from the SAME initialization.
    Without it the run-to-run spread at fixed hyperparameters (~0.03 val accuracy) swamps the
    differences between grid cells, and the sweep would just be selecting on init noise.
    """
    torch.manual_seed(init_seed)
    model = LoReV2(len(train_feats), 4096, K, lam_pop=lam_pop, lam_d=lam_d,
                   num_iterations=iters, learning_rate=lr, verbose=False)
    Wk, Vk = model.train(train_feats, val=val_feats)
    return {
        "val_acc": eval_acc(Wk, Vk, val_feats),
        "train_acc": eval_acc(Wk, Vk, train_feats),
        "min_abs_basis_cos": collapse_metrics(model.V.detach().cpu()),
        "stopped_at": model.stopped_at,
        "delta_norm": float(torch.linalg.vector_norm(
            model.delta.detach(), ord=2, dim=1).median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--k_tune", type=int, default=8,
                    help="fixed rank to tune at; the chosen lam values are reused for every K")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 0.1])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "v2_tuning.csv"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}

    users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                             split_by_turn=True, max_pairs=args.max_pairs,
                             val_frac=args.val_frac)
    uids = sorted(users)
    train_feats = [users[u][0] for u in uids]
    val_feats = [users[u][1] for u in uids]
    n_tr = sum(x.shape[0] for x in train_feats)
    n_va = sum(x.shape[0] for x in val_feats)
    print(f"{len(uids)} users | {n_tr} train pairs | {n_va} val pairs | K={args.k_tune}")

    # base_rm on val, as an orientation point (not used for selection)
    head = unit(load_reward_head().reshape(-1))
    base_val = float(np.mean([(v @ head > 0).float().mean().item() for v in val_feats]))
    print(f"base_rm on val: {base_val:.4f}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["phase", "lr", "lam_pop", "lam_d", "val_acc", "train_acc",
                                      "min_abs_basis_cos", "stopped_at", "delta_norm", "base_rm"])
    w.writeheader()

    def record(phase, lr, lp, ld):
        r = fit_eval(train_feats, val_feats, args.k_tune, lp, ld, lr, args.iters,
                     init_seed=args.seed)
        w.writerow({"phase": phase, "lr": lr, "lam_pop": lp, "lam_d": ld, "base_rm": round(base_val, 4),
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
        f.flush()
        return r

    # ---- phase 1: learning rate, at mid-grid regularization ----
    mid = 0.1
    print(f"=== phase 1: lr sweep (lam_pop=lam_d={mid}) ===")
    print(f"{'lr':>8} | {'val_acc':>8} | {'train':>7} | {'min|cos|':>8} | {'stop':>5}")
    print("-" * 48)
    lr_scores = {}
    for lr in args.lrs:
        r = record("lr", lr, mid, mid)
        lr_scores[lr] = r["val_acc"]
        print(f"{lr:>8g} | {r['val_acc']:>8.4f} | {r['train_acc']:>7.4f} | "
              f"{r['min_abs_basis_cos']:>8.4f} | {r['stopped_at']:>5}")
    best_lr = max(lr_scores, key=lr_scores.get)
    print(f"-> best lr = {best_lr:g} (val {lr_scores[best_lr]:.4f})\n")

    # ---- phase 2: lam_pop x lam_d grid at the chosen lr ----
    print(f"=== phase 2: lam_pop x lam_d grid (lr={best_lr:g}) ===")
    print("val accuracy, rows = lam_pop, cols = lam_d")
    hdr = "".join(f"{ld:>9g}" for ld in args.lams)
    print(f"{'':>9}{hdr}")
    grid = {}
    for lp in args.lams:
        row = []
        for ld in args.lams:
            r = record("grid", best_lr, lp, ld)
            grid[(lp, ld)] = r["val_acc"]
            row.append(r["val_acc"])
        print(f"{lp:>9g}" + "".join(f"{v:>9.4f}" for v in row))
    best_lp, best_ld = max(grid, key=grid.get)
    print(f"\n-> best lam_pop={best_lp:g}, lam_d={best_ld:g} (val {grid[(best_lp, best_ld)]:.4f})\n")

    # ---- phase 3: re-check lr at the winning lam pair (is the factorization safe?) ----
    print(f"=== phase 3: lr re-check at lam_pop={best_lp:g}, lam_d={best_ld:g} ===")
    recheck = {}
    for lr in args.lrs:
        r = record("recheck", lr, best_lp, best_ld)
        recheck[lr] = r["val_acc"]
        print(f"{lr:>8g} | {r['val_acc']:>8.4f}")
    final_lr = max(recheck, key=recheck.get)
    f.close()

    print(f"\n{'='*60}")
    print(f"TUNED: lr={final_lr:g}  lam_pop={best_lp:g}  lam_d={best_ld:g}")
    print(f"  val acc {recheck[final_lr]:.4f}  vs base_rm {base_val:.4f} "
          f"({recheck[final_lr]-base_val:+.4f})")
    if final_lr != best_lr:
        print(f"  [warn] lr moved {best_lr:g} -> {final_lr:g} after fixing lam, so lr and "
              f"regularization interact more than the factorization assumed. Treat the grid as "
              f"coarse and consider a joint sweep before trusting fine differences.")
    print(f"Saved {args.out}")
    print("\nThese values are selected on VALIDATION only. Pass them to "
          "community_alignment_lore.py --model v2 for the held-out test comparison.")


if __name__ == "__main__":
    main()
