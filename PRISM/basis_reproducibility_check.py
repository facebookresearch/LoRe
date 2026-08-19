#!/usr/bin/env python3
"""
LoRe-PRISM basis reproducibility sanity check (two-part).

PURPOSE
-------
Confirm that everyone on the team learns the SAME basis reward functions from
the PRISM preference data. Each teammate runs this script and compares the
`basis_fp` column for the agreed configuration.

The script runs TWO parts in a single invocation:

  PART 1 - RANK SWEEP  (vary K, hold the seed fixed)
      Sweeps the number of basis functions K (the rank of V). Shows how
      generalization accuracy changes with rank. Each row uses a different K,
      so the learned bases are EXPECTED to differ between rows -- this part is
      about the rank/accuracy trade-off, not about matching teammates.

  PART 2 - SEED VARIANCE / TEAM REPRODUCIBILITY  (hold K fixed, vary the seed)
      The original LoRe code does NOT lock the random seed, so the basis
      initialization -- and therefore the learned bases -- randomizes on every
      run. We lock it here (see set_seed). Seeds 0,1,2 measure how much the
      seed actually moves the result; seed 42 is the TEAM-AGREED value. Compare
      your seed=42 row's basis_fp against your teammates' seed=42 row to confirm
      you all converge to the same bases.

HYPERPARAMETERS (documented so the script is self-explanatory)
--------------------------------------------------------------
  K            number of basis functions = rank of the basis matrix V. The
               central LoRe knob. (PART 1 sweeps it; PART 2 fixes it.)
  seed         RNG seed controlling basis initialization during training. MUST
               match across teammates to reproduce the same bases. Team value = 42.
  alpha        strength of regularization pulling the bases toward the backbone's
               single SFT reward direction (V_sft). Default 1e4 (repo default).
  iters        optimization steps for the alternating W / V solve. Default 20000.
  lr           learning rate for that solve. Default 0.5.
  data split   fixed by prepare.py at seed=123. Do NOT change it -- it keeps the
               train/test user split identical for everyone on the team.

  bases_kept   how many of the K basis functions survive pruning: a basis is
               dropped if EVERY user gives it <1% weight (softmax(W) max < 1e-2,
               see utils.py). This is informational -- the model often collapses
               onto fewer "effective" bases than K. It can differ between
               seeds/machines, which is exactly why it must NOT be the thing you
               compare.

  basis_fp     = ||V||_F over the FULL [4096, K] basis matrix (ALL K columns,
               BEFORE pruning) -- so it includes sub-threshold bases and is
               comparable across teammates even when bases_kept differs. It is
               invariant to column permutation and sign, so it survives the
               harmless reordering of bases between runs while still flagging a
               genuinely different fit. Same GPU + same seed -> matches tightly;
               different GPUs -> agree to ~3-4 decimals (CUDA float
               nondeterminism), NOT bitwise. So compare approximately. The full
               matrices themselves are also saved (see OUTPUT) for direct
               inspection.

PREREQUISITE (one-time, slow): the cached embeddings must already exist at
    data/prism/train_embeddings.pkl
    data/prism/test_embeddings.pkl
produced by:  python prepare.py  &&  python generate-prism-embeddings.py

OUTPUT
------
Prints a summary table for each part, THEN a PER-CONFIG BASIS DETAIL section
(one fingerprint row per individual basis vector -- screenshot this to share),
AND writes:
  - basis_reproducibility_results.csv : the summary rows (gitignored)
  - basis_detail.csv                  : the eyeball view -- one row PER BASIS
                                        vector per run (norm, cos to V_sft,
                                        checksum, max user weight, kept). Same
                                        numbers as the BASIS DETAIL tables, so
                                        you can diff/sort in a spreadsheet instead
                                        of squinting at a screenshot (gitignored).
  - basis_matrices.pt                 : a self-contained dict per run, keyed by run
                                        (e.g. PART2_K10_seed42). Each entry holds
                                        the FULL [4096, K] basis matrix V, the full
                                        [num_users, K] user weights W, and all the
                                        summary metadata -- so every number in the
                                        printout (and the per-basis detail) is
                                        derivable from this file alone. Load and
                                        compare the actual basis vectors directly,
                                        including sub-threshold ones.
Both are gitignored; copy them to your local machine to share / compare.

The 4096-dim basis vectors are too long to print in full, so the PER-CONFIG
BASIS DETAIL table prints, for each of the K bases, a set of order-/sign-aware
fingerprints (norm, alignment to V_sft, checksum, max user weight, kept?). Two
runs that learned the same bases produce the same set of rows (up to ordering);
the raw vectors themselves live in basis_matrices.pt for exact comparison.

USAGE (run from inside the PRISM/ directory)
--------------------------------------------
    python basis_reproducibility_check.py
    python basis_reproducibility_check.py --rank-values 1,5,10,20,50
    python basis_reproducibility_check.py --seed-values 0,1,2,42 --fixed-K 10
    python basis_reproducibility_check.py --out my_results.csv
"""
import os
import sys
import csv
import argparse
import random
import numpy as np
import torch
from collections import defaultdict

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Make utils.py importable (same trick train_basis.py uses)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import LoRe_regularized  # noqa: E402

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_embeddings_by_user(dataset, seen_value, split_name):
    """Per user, stack the (chosen - rejected) embedding diffs into [m_i, 4096]."""
    grouped = defaultdict(list)
    for ex in dataset:
        info = ex.get("extra_info", {})
        if info.get("seen") == seen_value and info.get("split") == split_name:
            uid = info.get("user_id")
            if uid:
                chosen = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                rejected = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                grouped[uid].append(chosen - rejected)
    return [torch.stack(grouped[u]) for u in sorted(grouped.keys())]


def get_reference_direction():
    """The single global reward direction = Skywork's TRUE reward head score.weight.

    FIXED: the old code used AutoModel + "last nn.Linear", which returns the bare backbone
    (score.weight dropped) and grabs layers.31.mlp.down_proj -- an arbitrary MLP column, NOT the
    reward direction. rm_head_utils.load_reward_head() returns the real score.weight [4096, 1]
    (via AutoModelForSequenceClassification, or a lightweight ranged fetch)."""
    sys.path.append(SCRIPT_DIR)
    from rm_head_utils import load_reward_head
    return load_reward_head(device=device)


def accuracy(W, V, features):
    """Device-safe pairwise accuracy. A pair is correct when the personalized
    reward of (chosen - rejected) is positive, i.e. chosen scored higher."""
    accs = []
    for i, X in enumerate(features):
        X = X.to(V.device, dtype=V.dtype)
        scores = X @ (V @ W[i])  # [m_i]
        accs.append((scores > 0).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


def run_one(part, label, K, seed, V_final, train_seen, test_seen, args):
    """Train one basis set under a given (K, seed). Returns (row, V_full) where
    V_full is the FULL [4096, K] basis matrix BEFORE pruning -- that full matrix
    is the thing teammates should compare, since pruning can keep a different
    number of columns on different machines/seeds."""
    print("\n" + "=" * 64)
    print(f"[{part}] {label}  (K={K}, seed={seed}, alpha={args.alpha}, "
          f"iters={args.iters}, lr={args.lr})")
    print("=" * 64)
    set_seed(seed)  # lock RNG BEFORE the solve so the basis init is reproducible
    # Build the solver directly (instead of solve_regularized_simplex) so we can
    # grab the FULL basis matrix am.V (all K columns). solve_regularized_simplex
    # would only hand back the pruned columns, hiding the sub-threshold bases.
    am = LoRe_regularized(V_final, args.alpha, len(train_seen), 4096, K,
                          args.iters, args.lr)
    W_kept, V_kept = am.train(train_seen)            # pruned: what eval uses
    V_full = am.V.detach()                           # [4096, K] every basis column
    W_full = torch.softmax(am.W, dim=1).detach()     # [num_users, K] full weights

    # Accuracy uses the pruned (canonical) bases; the dropped near-zero-weight
    # columns add ~nothing, so this matches the repo's reported metric.
    train_acc, _ = accuracy(W_kept, V_kept, train_seen)
    test_acc, test_std = accuracy(W_kept, V_kept, test_seen)
    row = {
        "part": part, "run": label, "K": K, "seed": seed,
        "alpha": args.alpha, "iters": args.iters, "lr": args.lr,
        "bases_kept": V_kept.shape[1],                        # survived pruning (info)
        "basis_fp": float(torch.linalg.norm(V_full).item()),  # ||V||_F over ALL K cols
        "train_acc": train_acc, "test_acc": test_acc, "test_std": test_std,
    }
    return row, V_full.cpu(), W_full.cpu()


def print_table(title, rows):
    print("\n\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)
    header = (f"{'run':>9} | {'K':>3} | {'seed':>4} | {'bases_kept':>10} | "
              f"{'basis_fp':>10} | {'train_acc':>9} | {'test_acc':>9} | {'test_std':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:>9} | {r['K']:>3} | {r['seed']:>4} | {r['bases_kept']:>10} | "
              f"{r['basis_fp']:>10.4f} | {r['train_acc']:>9.4f} | "
              f"{r['test_acc']:>9.4f} | {r['test_std']:>8.4f}")


def print_basis_detail(key, m, vsft):
    """Pretty-print one fingerprint row per individual basis vector for a run.

    The full 4096-dim vectors can't be printed legibly, so for each basis column
    V[:, i] we print order-/sign-aware fingerprints that two identical fits will
    reproduce (up to column ordering):
      norm         ||V[:, i]||         -- magnitude of the basis
      cos(V_sft)   cosine to the backbone reward direction -- where it points
      checksum     sum of the entries  -- catches a different vector w/ same norm
      max_user_wt  max over users of softmax(W)[:, i] -- the pruning criterion
      kept         yes if max_user_wt >= 1e-2 (survived pruning), else no
    Rows are sorted by norm (descending) so the ordering is canonical and two
    teammates' tables line up row-for-row regardless of internal column order.
    The raw vectors live in basis_matrices.pt for exact element-wise comparison.
    """
    V = m["V"]                      # [4096, K]
    W = m["W"]                      # [num_users, K]
    vsft = vsft.reshape(-1).to(V.dtype)
    vsft_unit = vsft / (torch.linalg.norm(vsft) + 1e-12)
    K = V.shape[1]
    rows = []
    for i in range(K):
        col = V[:, i]
        norm = float(torch.linalg.norm(col).item())
        cos = float((col @ vsft_unit / (norm + 1e-12)).item())
        checksum = float(col.sum().item())
        max_wt = float(W[:, i].max().item())
        rows.append((norm, cos, checksum, max_wt, max_wt >= 1e-2))
    rows.sort(key=lambda r: r[0], reverse=True)

    print("\n\n" + "=" * 78)
    print(f"  BASIS DETAIL: {key}   "
          f"(K={K}, bases_kept={m['bases_kept']}, ||V||_F={m['basis_fp']:.4f})")
    print("=" * 78)
    header = (f"{'basis':>5} | {'norm':>12} | {'cos(V_sft)':>11} | "
              f"{'checksum':>13} | {'max_user_wt':>11} | {'kept':>4}")
    print(header)
    print("-" * len(header))
    detail_rows = []
    for idx, (norm, cos, checksum, max_wt, kept) in enumerate(rows):
        print(f"{idx:>5} | {norm:>12.5f} | {cos:>11.5f} | {checksum:>13.5f} | "
              f"{max_wt:>11.5f} | {'yes' if kept else 'no':>4}")
        detail_rows.append({
            "run_key": key, "part": m["part"], "K": K, "seed": m["seed"],
            "basis": idx, "norm": norm, "cos_vsft": cos, "checksum": checksum,
            "max_user_wt": max_wt, "kept": "yes" if kept else "no",
        })
    return detail_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-part LoRe-PRISM basis reproducibility check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # PART 1 -- rank sweep
    p.add_argument("--rank-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[5, 10, 20],
                   help="PART 1: comma-separated K values to sweep (seed held fixed)")
    p.add_argument("--rank-seed", type=int, default=42,
                   help="PART 1: seed held fixed while sweeping K")
    # PART 2 -- seed variance / team reproducibility
    p.add_argument("--seed-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[0, 1, 2, 42],
                   help="PART 2: comma-separated seeds to sweep (42 = team-agreed value)")
    p.add_argument("--fixed-K", type=int, default=10,
                   help="PART 2: K held fixed while sweeping the seed")
    # shared hyperparameters (documented in the module docstring)
    p.add_argument("--alpha", type=float, default=1e4, help="regularization strength")
    p.add_argument("--iters", type=int, default=20000, help="optimization steps")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--out", default="basis_reproducibility_results.csv",
                   help="CSV path for results (gitignored; copy to your machine when done)")
    p.add_argument("--matrices-out", default="basis_matrices.pt",
                   help="path to save the FULL [4096,K] basis matrix per run "
                        "(gitignored; copy down to inspect/compare ALL bases, "
                        "including sub-threshold ones)")
    p.add_argument("--detail-out", default="basis_detail.csv",
                   help="CSV path for the per-basis eyeball view (one row per "
                        "basis vector: norm, cos to V_sft, checksum, max user "
                        "weight, kept; gitignored; copy down to share/compare)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Device: {device}")
    print(f"Hyperparameters: alpha={args.alpha} iters={args.iters} lr={args.lr} "
          f"(data split fixed by prepare.py seed=123)")
    print("Loading cached embeddings...")
    train_emb = torch.load("data/prism/train_embeddings.pkl")
    test_emb = torch.load("data/prism/test_embeddings.pkl")
    train_seen = group_embeddings_by_user(train_emb, seen_value=True, split_name="train")
    test_seen = group_embeddings_by_user(test_emb, seen_value=True, split_name="test")
    print(f"Seen users: {len(train_seen)}")

    print("Loading backbone once for reference direction (V_sft)...")
    V_final = get_reference_direction()

    part1, part2 = [], []
    # run-key -> self-contained dict {V[4096,K], W[users,K], **metadata}, so every
    # printed number is derivable from this file alone.
    matrices = {}
    order = []  # preserve run order for the per-config detail printout

    # ---- PART 1: rank sweep (vary K, fixed seed) -- bases differ by design ----
    for K in args.rank_values:
        row, V_full, W_full = run_one("PART1-rank", f"K={K}", K, args.rank_seed,
                                      V_final, train_seen, test_seen, args)
        part1.append(row)
        key = f"PART1_K{K}_seed{args.rank_seed}"
        matrices[key] = {"V": V_full, "W": W_full, **row}
        order.append(key)

    # ---- PART 2: seed variance / team reproducibility (fixed K, vary seed) ----
    for s in args.seed_values:
        row, V_full, W_full = run_one("PART2-seed", f"seed={s}", args.fixed_K, s,
                                      V_final, train_seen, test_seen, args)
        part2.append(row)
        key = f"PART2_K{args.fixed_K}_seed{s}"
        matrices[key] = {"V": V_full, "W": W_full, **row}
        order.append(key)

    print_table(
        f"PART 1  RANK SWEEP  (seed fixed at {args.rank_seed}; bases differ between rows by design)",
        part1)
    print_table(
        f"PART 2  SEED VARIANCE  (K fixed at {args.fixed_K}; compare the seed=42 row to teammates)",
        part2)

    # ---- PER-CONFIG BASIS DETAIL: one fingerprint row per basis vector ----
    # This is the screenshot-and-share view: it surfaces EVERY basis (including
    # sub-threshold ones) so teammates can confirm they learned the same bases.
    print("\n\n" + "#" * 80)
    print("# PER-CONFIG BASIS DETAIL  (one row per basis vector; sorted by norm)")
    print("# Same fit -> same set of rows up to ordering. Raw vectors in the .pt file.")
    print("#" * 80)
    vsft_cpu = V_final.detach().cpu()
    detail_all = []
    for key in order:
        detail_all.extend(print_basis_detail(key, matrices[key], vsft_cpu))

    # ---- save the FULL basis matrices + weights + metadata per run ----
    torch.save(matrices, args.matrices_out)
    print(f"\nSaved {len(matrices)} runs to {args.matrices_out} "
          f"(each: full V[4096,K], full W[users,K], and metadata)")
    print("  load with: m = torch.load('basis_matrices.pt'); "
          "m['PART2_K10_seed42']['V'] -> [4096, K]")

    # ---- write combined CSV (gitignored; copy to your machine to share) ----
    fields = ["part", "run", "K", "seed", "alpha", "iters", "lr",
              "bases_kept", "basis_fp", "train_acc", "test_acc", "test_std"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in part1 + part2:
            row = dict(r)
            for k in ("basis_fp", "train_acc", "test_acc", "test_std"):
                row[k] = f"{row[k]:.4f}"
            writer.writerow(row)
    print(f"\nWrote {len(part1) + len(part2)} rows to {args.out}")

    # ---- write per-basis detail CSV (the eyeball view, one row per basis) ----
    detail_fields = ["run_key", "part", "K", "seed", "basis",
                     "norm", "cos_vsft", "checksum", "max_user_wt", "kept"]
    with open(args.detail_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for r in detail_all:
            row = dict(r)
            for k in ("norm", "cos_vsft", "checksum", "max_user_wt"):
                row[k] = f"{row[k]:.5f}"
            writer.writerow(row)
    print(f"Wrote {len(detail_all)} per-basis rows to {args.detail_out} "
          "(same fingerprints as the BASIS DETAIL tables above)")
    print("basis_fp = ||V||_F (permutation/sign-invariant). Compare the PART 2 "
          "seed=42 row against teammates to confirm matching bases.")


if __name__ == "__main__":
    main()
