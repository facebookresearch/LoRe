"""Experiment 1 -- shuffled-label control (the key experiment).

Question: is the shared reward direction's concept alignment / held-out accuracy
(Experiment 2) real preference signal, or something the LoRe fit would produce from
ANY data?

How the signal lives in the data
--------------------------------
LoRe never sees a label column. Each PRISM pair is folded into one diff vector
    d = chosen_embedding - rejected_embedding
and the fit pushes  d . (V W_u) > 0. So the ORIENTATION of the diff *is* the label:
in the real data every diff already points toward the chosen response.

The control
-----------
For each pair, randomly keep (chosen - rejected) or swap to (rejected - chosen) --
i.e. flip each diff's sign with prob 0.5. That is exactly "randomly relabel which
response was chosen." It kills the consistent preferred direction (each user's mean
diff -> ~0) while leaving every embedding, magnitude, and per-user grouping intact.
Then re-run the SAME solve.

What to expect (mentally tested)
--------------------------------
The real fit uses alpha=1e4, a strong pull of every V column toward the base reward
head (V_sft). Under shuffled labels the data term goes flat, so that regularizer
wins and V -> head, i.e. shuffled v_pop -> head. So the null-fit direction is NOT
"aligned with nothing" -- it is the bare head. The real question is therefore a
three-way comparison:
    head            the base reward model
    shuffled v_pop  what the fitter returns with no preference signal (~ head)
    real v_pop      the actual fit
Does using the real labels pull v_pop AWAY from the null-fit/head toward concept
structure, or does real v_pop just sit on the head too? Run with --alpha 0 for the
pure data-only variant (no head anchor), which isolates whether the DATA alone
carries a shared concept-aligned direction.

Writes:
  artifacts/exp1_shuffled_vpop.pt   the shuffled shared direction [4096] + V, W
  results/exp1/summary.json         head vs real vs shuffled: #sig concepts, acc
  results/exp1/concepts.json        full per-concept cosines for all three
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C


def refit_shared_direction(train_diffs, head_raw, head_unit, K, alpha, iters, lr, seed, device):
    """Sign-shuffle each diff, refit LoRe, return oriented v_pop plus V, W."""
    from utils import LoRe_regularized

    g = torch.Generator().manual_seed(seed)
    neg, pos = torch.tensor(-1.0), torch.tensor(1.0)
    shuffled = []
    for X in train_diffs:
        flip = (torch.rand(X.shape[0], 1, generator=g) < 0.5)
        signs = torch.where(flip, neg, pos)
        shuffled.append((X.float() * signs).to(device))

    torch.manual_seed(seed)
    am = LoRe_regularized(head_raw.to(device), alpha, len(shuffled), 4096, K, iters, lr)
    am.train(shuffled)
    V_full = am.V.detach().cpu().float()
    W_full = torch.softmax(am.W, dim=1).detach().cpu().float()
    return C.shared_direction(V_full, W_full, head_unit), V_full, W_full


def report(name, direction, concepts, tau, test_diffs, head_unit, ref_vpop=None):
    cos = C.concept_cosines(direction, concepts)
    sig = dict(sorted(((c, v) for c, v in cos.items() if abs(v) > tau),
                      key=lambda kv: -abs(kv[1])))
    acc = C.direction_pair_accuracy(direction, test_diffs)
    out = {
        "name": name,
        "n_significant_concepts": len(sig),
        "significant_concepts": sig,
        "held_out_accuracy": acc,
        "cos_to_head": float(direction @ head_unit),
    }
    if ref_vpop is not None:
        out["cos_to_real_vpop"] = float(direction @ ref_vpop)
    return cos, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1e4,
                    help="head-anchor strength; 1e4 matches the real fit, 0 = data-only variant")
    ap.add_argument("--iters", type=int, default=20000, help="LoRe solve steps (match the real fit)")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp1")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)
    print(f"[exp1] device={device} alpha={args.alpha} iters={args.iters} seed={args.seed}")

    head_unit = C.load_head()
    from rm_head_utils import load_reward_head
    head_raw = load_reward_head().reshape(-1, 1).float()   # [4096,1] SFT anchor V_sft

    V, W = C.load_basis()
    v_real = C.shared_direction(V, W, head_unit)
    K = V.shape[1]

    concepts = C.load_concepts()
    tau = C.null_threshold()
    print(f"[exp1] significance threshold |cos|>{tau:.4f}")

    train_diffs = C.load_user_diffs(split="train", seen=True)
    test_diffs = C.load_user_diffs(split="test", seen=True)
    print(f"[exp1] refitting on {len(train_diffs)} users with signs shuffled...")

    v_shuf, V_shuf, W_shuf = refit_shared_direction(
        train_diffs, head_raw, head_unit, K, args.alpha, args.iters, args.lr, args.seed, device
    )
    torch.save({"v_pop_shuffled": v_shuf, "V": V_shuf, "W": W_shuf,
                "seed": args.seed, "alpha": args.alpha},
               os.path.join(C.ARTIFACTS_DIR, "exp1_shuffled_vpop.pt"))

    head_cos, head_rep = report("head", head_unit, concepts, tau, test_diffs, head_unit, v_real)
    real_cos, real_rep = report("real_vpop", v_real, concepts, tau, test_diffs, head_unit, v_real)
    shuf_cos, shuf_rep = report("shuffled_vpop", v_shuf, concepts, tau, test_diffs, head_unit, v_real)

    summary = {
        "significance_threshold": tau,
        "alpha": args.alpha,
        "cos_real_vpop_head": float(v_real @ head_unit),
        "cos_shuffled_head": float(v_shuf @ head_unit),
        "cos_shuffled_real": float(v_shuf @ v_real),
        "head": head_rep,
        "real_vpop": real_rep,
        "shuffled_vpop": shuf_rep,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "concepts.json"), "w") as f:
        json.dump({"head": head_cos, "real_vpop": real_cos, "shuffled_vpop": shuf_cos}, f, indent=2)

    for r in (head_rep, real_rep, shuf_rep):
        print(f"[exp1] {r['name']:14s} sig={r['n_significant_concepts']:2d} "
              f"acc={r['held_out_accuracy']['overall_pair_acc']:.4f} "
              f"cos(head)={r['cos_to_head']:+.4f}")
    print(f"[exp1] cos(shuffled,head)={float(v_shuf @ head_unit):+.4f}  "
          f"cos(shuffled,real)={float(v_shuf @ v_real):+.4f}")
    print(f"[exp1] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
