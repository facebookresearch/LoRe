"""
Can LoRe recover KNOWN, human-interpretable preference axes from REAL Skywork embeddings?

The question being tested: "Construct a synthetic dataset with really clearly defined axes of
variation and tons of data per user, and see if LoRe can pick them up."

synthetic_recovery.py answered the abstract version of this (planted axes in R^d, Gaussian features)
and showed LoRe recovers planted axes above ~50 pairs/user. But it generated feature vectors
directly, so it says nothing about the embedding space. This script runs the same test through the
REAL pipeline: LLM-written responses that vary along named concepts, embedded with Skywork's
last-token hidden state -- the identical representation PRISM used.

Construction
------------
  * Axes = a mutually-distinct subset of the concept library (default avoids the redundant
    "quality cluster": safety~values +0.98, factuality~values +0.89).
  * For each concept C we create TWO user groups: users who prefer high-C and users who prefer low-C.
    Both groups see the SAME pairs and choose oppositely.

    This is the crucial design choice. In PRISM no two users ever disagreed on a prompt, so one
    global direction always sufficed and the bases had no reason to differ. Here disagreement is
    built in, so a single global direction CANNOT satisfy both groups and LoRe is forced to use
    multiple bases. It is also realistic -- formatting, confidence and verbosity are matters of
    taste, not universal quality.

  * Labels come from the GENERATIVE PROCESS, not from projecting onto a concept vector. A user who
    likes high-formatting has chosen = high_response for every formatting pair, full stop. Labelling
    by projection onto concept_vector_C would be circular: it would guarantee linear separability by
    construction. Labelling semantically means recovery depends on whether the embedding space
    ACTUALLY encodes the axis linearly -- which is the empirical question.

Read-out
--------
  * Recovers the axes  -> reward-tuned embeddings do encode interpretable preference axes, and
    PRISM's null was about data volume (~15 pairs/user), not the representation.
  * Fails despite blatant axes and adequate data -> the representation is the bottleneck.

Controls (these decide whether the numbers mean anything)
  1. personal-vs-global: by construction personal MUST beat global here. If global wins, the
     construction is broken and everything downstream is void. This also serves as a positive
     control for per_user_signal.py -- proving that diagnostic can detect personalization when it
     exists, so PRISM's null was not a dead instrument.
  2. collapse at alpha=0: with genuinely conflicting users the bases should NOT collapse.

CPU-only once embeddings exist.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import (LoRe_regularized, LoReV2,                          # noqa: E402
                   canonical_variation_axes)
from synthetic_recovery import (subspace_alignment, best_axis_match,  # noqa: E402
                                collapse_metrics, eval_acc, w_recovery)


def signed_ground_truth(V_true):
    """[F, n_concepts] -> [F, 2*n_concepts] laid out to match group_id = concept*2 + sign_i.

    Users come in disagreeing pairs: sign_i=0 prefers HIGH-C (+c), sign_i=1 prefers LOW-C (-c).
    Matching a user's reward direction has to respect that sign, otherwise a model that gets the
    concept right but the direction backwards would score as correct.
    """
    cols = []
    for i in range(V_true.shape[1]):
        cols.append(V_true[:, i])
        cols.append(-V_true[:, i])
    return torch.stack(cols, dim=1)


def group_dir_match(reward_dirs, group_id, V_signed):
    """Fraction of users whose implied reward direction points at their own (concept, sign) group.

    Sign-sensitive, so no .abs() here -- unlike reward_dir_match in synthetic_recovery.py, where
    the planted axes are unsigned.
    """
    sim = F.normalize(reward_dirs, dim=1) @ F.normalize(V_signed, dim=0)
    return (sim.argmax(1) == group_id).float().mean().item()

# Mutually-distinct axes (greedy selection, max |cos| <= 0.68), avoiding the quality cluster.
DEFAULT_CONCEPTS = ["fluency", "repetition", "confidence", "creativity", "formatting", "diversity"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def unit(v):
    return v / (v.norm() + 1e-8)


def build_users(emb, concepts, users_per_group, test_frac, gen, high_frac=0.5,
                subset_frac=0.7):
    """Create disagreeing users over the concept axes.

    high_frac controls how many of each concept's users prefer HIGH-C (the rest prefer LOW-C).
      * 0.5 (default): perfectly balanced. Cleanest recovery test, but the pooled mean diff is then
        exactly zero, so 'global' is degenerate and beating it is nearly trivial.
      * e.g. 0.75: a genuine majority exists, so a NON-degenerate global direction is available.
        Personal beating global in that setting is a much stronger claim.

    subset_frac gives every user an INDEPENDENT random subset of their concept's train pairs.
    Without it, all users in a group share byte-identical features, so N users are really one user
    replicated N times -- which inflates how easy the problem looks and makes per-user metrics
    (w_recovery, personal-vs-other) far too flattering. With it, users overlap but differ, which is
    how real annotators behave. subset_frac=1.0 restores the old degenerate behaviour.

    Returns (train_feats, test_feats, group_id) where group_id[i] identifies the (concept, sign)
    group of user i -- the ground-truth 'axis' that user cares about.
    """
    train_feats, test_feats, group_id = [], [], []
    total_per_concept = 2 * users_per_group
    n_high = int(round(high_frac * total_per_concept))
    for ci, c in enumerate(concepts):
        high, low = emb[c]["high"], emb[c]["low"]
        n = high.shape[0]
        n_test = max(1, int(round(test_frac * n)))
        perm = torch.randperm(n, generator=gen)
        te_idx, tr_idx = perm[:n_test], perm[n_test:]
        # diff as (chosen - rejected): +1 group prefers high, -1 group prefers low
        d_tr = high[tr_idx] - low[tr_idx]
        d_te = high[te_idx] - low[te_idx]
        n_sub = max(2, int(round(subset_frac * d_tr.shape[0])))
        for sign_i, (sign, count) in enumerate(((+1.0, n_high),
                                                (-1.0, total_per_concept - n_high))):
            for _ in range(count):
                # each user sees their own random subset of the concept's train pairs
                sub = torch.randperm(d_tr.shape[0], generator=gen)[:n_sub]
                train_feats.append(sign * d_tr[sub])
                test_feats.append(sign * d_te)
                group_id.append(ci * 2 + sign_i)
    return train_feats, test_feats, torch.tensor(group_id)


def _acc(scores):
    """Fraction scored positive, counting exact ties as chance.

    Ties matter here: with perfectly balanced +/- user groups the pooled mean diff is EXACTLY zero,
    so the 'global direction' is the zero vector and every score is 0. Treating those as wrong would
    report global=0.0, which reads like a catastrophic failure when the truth is 'no global direction
    exists' -- i.e. chance. Counting ties as 0.5 reports that honestly.
    """
    s = scores.flatten()
    return ((s > 0).float().sum() + 0.5 * (s == 0).float().sum()).item() / max(s.numel(), 1)


def personal_vs_global(train_feats, test_feats, group_id):
    """Control: does a user's own direction beat one shared direction on their held-out pairs?

    On this dataset personal MUST win (users disagree by construction). Mirrors per_user_signal.py
    so the two are directly comparable to the PRISM numbers.
    """
    pooled_mean = torch.cat(train_feats, 0).mean(0)
    global_dir = unit(pooled_mean)
    personal, glob, other = [], [], []
    n = len(train_feats)
    for i in range(n):
        p = unit(train_feats[i].mean(0))
        personal.append(_acc(test_feats[i] @ p))
        glob.append(_acc(test_feats[i] @ global_dir))
        # control: a random user from a DIFFERENT group
        others = [j for j in range(n) if group_id[j] != group_id[i]]
        j = random.choice(others) if others else i
        other.append(_acc(test_feats[i] @ unit(train_feats[j].mean(0))))
    return (float(np.mean(personal)), float(np.mean(glob)), float(np.mean(other)),
            pooled_mean.norm().item())


def ground_truth_matrix(concept_vectors, concepts):
    """[4096, n_concepts] of the true concept directions we hope to recover."""
    return torch.stack([concept_vectors[c].float() for c in concepts], dim=1)


def null_recovery(V_true, k_fit, n_trials=20):
    """Chance floor for subspace_align / best_axis_match using RANDOM bases of the same shape.

    Necessary for honest reading: with k_fit > n_concepts these metrics are well above 0 even for
    random matrices, so the raw value alone does not indicate recovery -- it has to beat this floor.
    """
    d = V_true.shape[0]
    subs, axis = [], []
    for _ in range(n_trials):
        R = torch.randn(d, k_fit)
        subs.append(subspace_alignment(R, V_true))
        axis.append(best_axis_match(R, V_true))
    return float(np.mean(subs)), float(np.mean(axis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                  "contrastive_pair_embeddings.pt"))
    ap.add_argument("--concept_vectors", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                              "concept_vectors.pt"))
    ap.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    ap.add_argument("--users_per_group", type=int, default=20,
                    help="users preferring high-C, and the same number preferring low-C, per concept")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--subset_frac", type=float, default=0.7,
                    help="fraction of a concept's train pairs each user sees (own random subset). "
                         "1.0 makes all users in a group identical -- the degenerate case")
    ap.add_argument("--high_frac", type=float, default=0.5,
                    help="fraction of each concept's users preferring HIGH-C. 0.5 = balanced "
                         "(global direction degenerate); >0.5 gives a real global to beat")
    ap.add_argument("--k_fit", type=int, default=None,
                    help="rank given to LoRe (default: 2 x #concepts, to allow +/- directions)")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 0.5 for vanilla (as published), 1e-2 for v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", choices=["vanilla", "v2"], default="vanilla",
                    help="vanilla = LoRe_regularized; v2 = LoReV2 (signed wbar+delta weights, "
                         "reward-space ridge)")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 only: ridge on ||V wbar||^2")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 only: ridge on ||V delta_u||^2")
    ap.add_argument("--out", default=None,
                    help="default: results/synthetic/concept_users{,_v2}.csv -- kept separate per "
                         "model so a v2 run never clobbers the committed vanilla baseline")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-2 if args.model == "v2" else 0.5
    if args.out is None:
        suffix = "_v2" if args.model == "v2" else ""
        args.out = os.path.join(SCRIPT_DIR, "..", "results", "synthetic",
                                f"concept_users{suffix}.csv")

    emb = torch.load(args.emb, weights_only=False)
    cvs = torch.load(args.concept_vectors, weights_only=False)
    concepts = [c for c in args.concepts if c in emb and c in cvs]
    missing = [c for c in args.concepts if c not in concepts]
    if missing:
        print(f"[warn] skipping concepts absent from the embeddings/vectors: {missing}")
    k_fit = args.k_fit or 2 * len(concepts)

    V_true = ground_truth_matrix(cvs, concepts)
    V_signed = signed_ground_truth(V_true)
    n_pairs = emb[concepts[0]]["high"].shape[0]
    n_groups = 2 * len(concepts)
    print(f"Concepts ({len(concepts)}): {concepts}")
    print(f"{n_pairs} pairs/concept -> ~{int(n_pairs*(1-args.test_frac))} train pairs per user; "
          f"{args.users_per_group} users per (concept, sign) group; "
          f"{n_groups*args.users_per_group} users; k_fit={k_fit}, alpha={args.alpha}; "
          f"high_frac={args.high_frac}, subset_frac={args.subset_frac}")
    print(f"w_recovery chance = 1/{n_groups} = {1/n_groups:.3f}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["seed", "train_acc", "test_acc", "subspace_align",
                                      "best_axis_match", "min_abs_basis_cos", "w_recovery",
                                      "group_dir_match", "personal", "global", "other_group",
                                      "global_dir_norm"])
    w.writeheader()

    print(f"{'seed':>4} | {'test_acc':>8} | {'subspace':>8} | {'axis':>6} | {'min|cos|':>8} | "
          f"{'w_recov':>7} | {'usr->grp':>8} | {'personal':>8} | {'global':>7} | {'other':>7}")
    print("-" * 97)
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        train_feats, test_feats, group_id = build_users(
            emb, concepts, args.users_per_group, args.test_frac, gen, args.high_frac,
            args.subset_frac)

        if args.model == "v2":
            model = LoReV2(len(train_feats), V_true.shape[0], k_fit, lam_pop=args.lam_pop,
                           lam_d=args.lam_d, num_iterations=args.iters,
                           learning_rate=args.lr, verbose=False)
            Wk, Vk = model.train(train_feats)
            V_raw = model.V.detach().cpu()
            # signed weights leave V identified only up to an invertible transform -- canonicalize
            # to the (varimax-rotated) principal axes of user variation before any per-axis metric
            V_axes, _, _ = canonical_variation_axes(model.V, model.delta, k=k_fit)
            W_assign, signed = (model.delta.detach().cpu() @ V_raw.T) @ V_axes, True
            r_dirs = model.reward_dirs().detach().cpu().T
        else:
            anchor = torch.cat(train_feats, 0).mean(0).reshape(-1, 1)
            model = LoRe_regularized(anchor, args.alpha, len(train_feats), V_true.shape[0],
                                     k_fit, args.iters, args.lr)
            Wk, Vk = model.train(train_feats)
            V_raw = V_axes = model.V.detach().cpu()
            W_assign, signed = model.W, False
            r_dirs = (Vk.detach().cpu() @ Wk.detach().cpu().T).T

        pers, glob, oth, gnorm = personal_vs_global(train_feats, test_feats, group_id)
        row = {
            "seed": seed,
            "train_acc": eval_acc(Wk, Vk, train_feats),
            "test_acc": eval_acc(Wk, Vk, test_feats),
            "subspace_align": subspace_alignment(V_axes, V_true),
            "best_axis_match": best_axis_match(V_axes, V_true),
            "min_abs_basis_cos": collapse_metrics(V_raw),
            "w_recovery": w_recovery(W_assign, group_id, n_groups, signed=signed),
            "group_dir_match": group_dir_match(r_dirs, group_id, V_signed),
            "personal": pers, "global": glob, "other_group": oth,
            "global_dir_norm": gnorm,
        }
        rows.append(row); w.writerow(row); f.flush()
        print(f"{seed:>4} | {row['test_acc']:>8.4f} | {row['subspace_align']:>8.4f} | "
              f"{row['best_axis_match']:>6.4f} | {row['min_abs_basis_cos']:>8.4f} | "
              f"{row['w_recovery']:>7.4f} | {row['group_dir_match']:>8.4f} | "
              f"{pers:>8.4f} | {glob:>7.4f} | {oth:>7.4f}")
    f.close()

    m = lambda k: float(np.mean([r[k] for r in rows]))
    null_sub, null_axis = null_recovery(V_true, k_fit)
    print("\n=== mean across seeds ===")
    print(f"  LoRe test acc        : {m('test_acc'):.4f}   (train {m('train_acc'):.4f})")
    print(f"  subspace -> true axes: {m('subspace_align'):.4f}   (random-basis floor {null_sub:.4f})")
    print(f"  per-axis best match  : {m('best_axis_match'):.4f}   (random-basis floor {null_axis:.4f})")
    print(f"  basis collapse min|cos|: {m('min_abs_basis_cos'):.4f}  (1.0 = fully collapsed)")
    print(f"  w_recovery           : {m('w_recovery'):.4f}   (chance {1/n_groups:.3f})")
    print("\n=== control: personal vs global (cf. PRISM: personal 0.566 < global 0.591) ===")
    print(f"  personal   {m('personal'):.4f}")
    print(f"  global     {m('global'):.4f}")
    print(f"  other group {m('other_group'):.4f}")
    print(f"  (pooled mean-diff norm {m('global_dir_norm'):.3g}: ~0 means the +/- groups cancel\n   exactly, i.e. NO global direction exists -- global is at chance by construction)")
    if m("personal") <= m("global"):
        print("\n  [!] personal did NOT beat global. Users are supposed to disagree by construction, "
              "so this means the construction (or the embedding space) failed -- treat the recovery "
              "numbers above as void until this is explained.")
    else:
        print("\n  personal > global, as the construction requires. This also positively controls "
              "per_user_signal.py: it CAN detect personalization when it exists, so the PRISM null "
              "was a real null, not a dead instrument.")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
