"""
Positive control: can LoRe recover KNOWN preference axes when the signal definitely exists?

The question being tested: "Construct a synthetic dataset with really clearly defined axes of
variation and tons of data per user, and see if LoRe can pick them up."

This is the experiment that has to come before anything else: PRISM gave a null, but a null is only
interpretable if we know the method CAN succeed when signal is present. Here we plant the ground
truth ourselves, so any failure is the method/optimizer, not the data.

Setup
-----
  * K_true orthonormal ground-truth axes V* in R^d.
  * Each user u gets a weight vector w*_u over those axes (by default: dominated by one axis, which
    is the regime LoRe's softmax-simplex W is designed for).
  * Their true reward direction is  r_u = V* w*_u.
  * For each pair we draw a feature difference vector `diff` (= chosen - rejected) and LABEL it by
    the user's own direction: we orient diff so that <diff, r_u> > 0, flipping a `noise` fraction.
    So the ONLY thing distinguishing users is which axis they care about.

We then train LoRe and measure whether it recovers the planted structure:
  test_acc          : held-out pair accuracy (per user)
  subspace_align    : mean cos of principal angles between span(V_learned) and span(V*)
                      (1.0 = the learned bases span exactly the true axes)
  best_axis_match   : for each true axis, the best |cos| to any learned column (are axes individually
                      recovered, not just the subspace?)
  min_abs_basis_cos : collapse metric (1.0 = all learned columns identical)
  w_recovery        : agreement between learned per-user argmax basis and the true dominant axis,
                      up to a permutation matched greedily (chance = 1/K_true)

Sweeps pairs-per-user, because "tons of data per user" is exactly the condition PRISM lacked
(~15 pairs/user). If LoRe recovers the axes at high pairs/user and degrades as pairs shrink, that
localizes the PRISM failure to data volume rather than the method.

CPU-only, no embeddings needed.
"""
import argparse
import os
import sys
import random
import csv

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized, LoReV2, canonical_variation_axes  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_ground_truth(d, k_true, gen):
    """k_true orthonormal axes in R^d -> [d, k_true]."""
    A = torch.randn(d, k_true, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k_true]


def make_users(n_users, k_true, gen, concentration):
    """Per-user weights over the true axes. concentration=1.0 -> one-hot (pure single-axis user);
    lower values mix in the other axes."""
    dom = torch.randint(0, k_true, (n_users,), generator=gen)
    W = torch.rand(n_users, k_true, generator=gen) * (1.0 - concentration)
    W[torch.arange(n_users), dom] += concentration
    W = W / W.sum(1, keepdim=True)
    return W, dom


def make_pairs(n, r_u, d, gen, noise, signal_scale):
    """Feature diffs labeled by the user's own direction r_u.

    diff = signal_scale * s * r_u_hat + isotropic noise, with s>0 meaning 'chosen preferred'.
    We return diffs already oriented as (chosen - rejected), so a correct model scores diff@r > 0.
    A `noise` fraction of labels is flipped (diff negated) to keep the problem non-separable.
    """
    r_hat = r_u / (r_u.norm() + 1e-8)
    # component along the user's true direction (always positive before flipping)
    mag = torch.rand(n, 1, generator=gen) * signal_scale + 0.05
    # isotropic distractor content orthogonal to nothing in particular (realistic clutter)
    clutter = torch.randn(n, d, generator=gen)
    clutter = clutter - (clutter @ r_hat).unsqueeze(1) * r_hat.unsqueeze(0)  # remove r component
    diffs = mag * r_hat.unsqueeze(0) + clutter
    flip = torch.rand(n, generator=gen) < noise
    diffs[flip] = -diffs[flip]
    return diffs


def collapse_metrics(V):
    Vn = F.normalize(V, dim=0)
    c = Vn.t() @ Vn
    k = V.shape[1]
    if k == 1:
        return 1.0
    off = c[~torch.eye(k, dtype=torch.bool)]
    return off.abs().min().item()


def subspace_alignment(V_learned, V_true):
    """Mean cosine of principal angles between the two column spaces (1.0 = identical span)."""
    Ql, _ = torch.linalg.qr(V_learned)
    Qt, _ = torch.linalg.qr(V_true)
    k = min(Ql.shape[1], Qt.shape[1])
    s = torch.linalg.svdvals(Ql[:, :Ql.shape[1]].t() @ Qt[:, :Qt.shape[1]])
    return s[:k].mean().item()


def best_axis_match(V_learned, V_true):
    """For each true axis, the best |cos| against any learned column; averaged."""
    Ln = F.normalize(V_learned, dim=0)
    Tn = F.normalize(V_true, dim=0)
    sim = (Tn.t() @ Ln).abs()          # [k_true, k_learned]
    return sim.max(dim=1).values.mean().item()


def eval_acc(W, V, feats):
    V = V.detach().cpu(); W = W.detach().cpu()
    return float(np.mean([(X.cpu() @ (V @ W[i]) > 0).float().mean().item()
                          for i, X in enumerate(feats)]))


def w_recovery(W_learned, dom_true, k_true, signed=False):
    """Do users who share a true dominant axis get assigned the same learned basis?
    Greedy-match learned argmax basis -> true axis, then report agreement (chance = 1/k_true).

    signed=True for LoReV2: its weights are signed and centred on zero, so the dominant basis is
    the largest-MAGNITUDE component (a strongly negative weight is just as much "this user cares
    about this axis" as a positive one). A plain argmax would silently pick the least-negative
    entry instead. Pass the delta (individual variation), not wbar + delta.
    """
    W_learned = W_learned.detach().cpu()
    pred = (W_learned.abs() if signed else W_learned).argmax(1).numpy()
    true = dom_true.numpy()
    # build contingency and greedily match
    ks = sorted(set(pred.tolist())); ts = sorted(set(true.tolist()))
    table = np.zeros((len(ts), len(ks)))
    for t_i, t in enumerate(ts):
        for k_i, k in enumerate(ks):
            table[t_i, k_i] = np.sum((true == t) & (pred == k))
    matched, used = 0, set()
    for _ in range(min(len(ts), len(ks))):
        i, j = np.unravel_index(np.argmax(table), table.shape)
        if table[i, j] <= 0:
            break
        matched += table[i, j]
        table[i, :] = -1; table[:, j] = -1
    return matched / len(true)


def reward_dir_match(reward_dirs, dom_true, V_true):
    """Fraction of users whose IMPLIED reward direction points at their true axis (chance 1/k).

    This is the read-out that actually survives the change of parameterization. best_axis_match
    compares learned basis COLUMNS to planted axes, which presumes the columns are individually
    identified -- true for vanilla (the simplex constraint pins them down) but false for v2, whose
    signed weights leave V determined only up to an invertible transform. Asking instead "does the
    model point each user at the right axis?" is invariant to that transform, so it compares the
    two models fairly.
    """
    sim = F.normalize(reward_dirs, dim=1) @ F.normalize(V_true, dim=0)   # [n_users, k_true]
    return (sim.abs().argmax(1) == dom_true).float().mean().item()


def run_one(pairs_per_user, args, seed):
    gen = torch.Generator().manual_seed(seed)
    set_seed(seed)
    d, k_true, n_users = args.dim, args.k_true, args.n_users

    V_true = make_ground_truth(d, k_true, gen)
    Wu, dom = make_users(n_users, k_true, gen, args.concentration)

    train, test = [], []
    for u in range(n_users):
        r_u = V_true @ Wu[u]
        train.append(make_pairs(pairs_per_user, r_u, d, gen, args.noise, args.signal))
        test.append(make_pairs(args.test_pairs, r_u, d, gen, args.noise, args.signal))

    if args.model == "v2":
        model = LoReV2(n_users, d, args.k_fit, lam_pop=args.lam_pop, lam_d=args.lam_d,
                       num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        Wk, Vk = model.train(train)
        V_raw = model.V.detach().cpu()
        # v2's raw basis columns are NOT individually identified: the reward-space ridge
        # constrains the products V@wbar and V@delta_u, so any V -> VR, w -> R^-1 w leaves both
        # the loss and the penalty unchanged. Per-column metrics on V_raw are therefore
        # meaningless for v2. Canonicalize to the principal axes of across-user reward variation
        # first -- those ARE identified, and they are what "areas of variation" means.
        axes, _, _ = canonical_variation_axes(model.V, model.delta, k=args.k_fit)
        V_axes = axes
        W_assign = (model.delta.detach().cpu() @ V_raw.T) @ axes   # user coords in canonical basis
        signed = True
        r_dirs = model.reward_dirs().detach().cpu().T              # [n_users, d]
    else:
        # anchor for the regularizer: the population-mean direction (analogue of v_sft)
        anchor = torch.cat(train, 0).mean(0).reshape(-1, 1)
        model = LoRe_regularized(anchor, args.alpha, n_users, d, args.k_fit,
                                 args.iters, args.lr)
        Wk, Vk = model.train(train)
        # vanilla's simplex weights pin down the columns, so V is already the axis estimate
        V_raw = V_axes = model.V.detach().cpu()
        W_assign, signed = model.W, False
        r_dirs = (Vk.detach().cpu() @ Wk.detach().cpu().T).T       # [n_users, d]

    return {
        "pairs_per_user": pairs_per_user,
        "seed": seed,
        "test_acc": eval_acc(Wk, Vk, test),
        "train_acc": eval_acc(Wk, Vk, train),
        "subspace_align": subspace_alignment(V_axes, V_true),
        "best_axis_match": best_axis_match(V_axes, V_true),
        # collapse is a property of the fitted bases themselves, so always read it off the raw V
        # (canonical axes are orthonormal by construction and would report ~0 trivially)
        "min_abs_basis_cos": collapse_metrics(V_raw),
        "w_recovery": w_recovery(W_assign, dom, k_true, signed=signed),
        "reward_dir_match": reward_dir_match(r_dirs, dom, V_true),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256, help="feature dimension")
    ap.add_argument("--k_true", type=int, default=5, help="number of planted axes")
    ap.add_argument("--k_fit", type=int, default=5, help="rank LoRe is given")
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--pairs", type=int, nargs="+", default=[10, 25, 50, 100, 400, 1000],
                    help="pairs per user to sweep ('tons of data per user' = the large end)")
    ap.add_argument("--test_pairs", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.1, help="fraction of flipped labels")
    ap.add_argument("--signal", type=float, default=1.0, help="scale of the true-direction component")
    ap.add_argument("--concentration", type=float, default=1.0,
                    help="1.0 = each user cares about exactly one axis")
    ap.add_argument("--model", choices=["vanilla", "v2"], default="vanilla",
                    help="vanilla = LoRe_regularized (simplex weights, V_sft anchor, column "
                         "dropping); v2 = LoReV2 (signed wbar+delta weights, reward-space ridge)")
    ap.add_argument("--alpha", type=float, default=0.0, help="vanilla only: V_sft anchor strength")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 only: ridge on ||V wbar||^2")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 only: ridge on ||V delta_u||^2")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 0.5 for vanilla (as published), 1e-2 for v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None,
                    help="default: results/synthetic/synthetic_recovery{,_v2}.csv -- kept separate "
                         "per model so a v2 run never clobbers the committed vanilla baseline")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-2 if args.model == "v2" else 0.5
    if args.out is None:
        suffix = "_v2" if args.model == "v2" else ""
        args.out = os.path.join(SCRIPT_DIR, "..", "results", "synthetic",
                                f"synthetic_recovery{suffix}.csv")

    reg = (f"lam_pop={args.lam_pop}, lam_d={args.lam_d}" if args.model == "v2"
           else f"alpha={args.alpha}")
    print(f"Synthetic positive control [{args.model}]: d={args.dim}, {args.k_true} planted axes, "
          f"{args.n_users} users, k_fit={args.k_fit}, {reg}, lr={args.lr}, "
          f"label noise={args.noise}, concentration={args.concentration}")
    print(f"Chance accuracy = 0.5 ; ceiling given label noise = {1 - args.noise:.2f}")
    print(f"w_recovery chance = {1/args.k_true:.2f}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["pairs_per_user", "seed", "train_acc", "test_acc",
                                      "subspace_align", "best_axis_match",
                                      "min_abs_basis_cos", "w_recovery", "reward_dir_match"])
    w.writeheader()

    print(f"{'pairs/user':>10} | {'test_acc':>8} | {'subspace':>8} | {'axis_match':>10} | "
          f"{'min|cos|':>8} | {'w_recov':>7} | {'user->axis':>10}")
    print("-" * 82)
    for p in args.pairs:
        rows = [run_one(p, args, s) for s in args.seeds]
        w.writerows(rows); f.flush()
        agg = {k: np.mean([r[k] for r in rows]) for k in
               ("test_acc", "subspace_align", "best_axis_match", "min_abs_basis_cos",
                "w_recovery", "reward_dir_match")}
        print(f"{p:>10} | {agg['test_acc']:>8.4f} | {agg['subspace_align']:>8.4f} | "
              f"{agg['best_axis_match']:>10.4f} | {agg['min_abs_basis_cos']:>8.4f} | "
              f"{agg['w_recovery']:>7.4f} | {agg['reward_dir_match']:>10.4f}")
    f.close()

    print(f"\nSaved {args.out}")
    print("Read-out: if subspace_align and axis_match -> ~1.0 with enough pairs/user, LoRe CAN "
          "recover planted axes, and the PRISM null is about the data (signal/volume), not the "
          "method. If they stay low even with 1000 pairs/user and clean axes, the method/optimizer "
          "itself cannot recover known structure.")


if __name__ == "__main__":
    main()
