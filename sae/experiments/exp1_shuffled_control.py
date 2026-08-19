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
v2 (LoReV2) has NO head anchor -- the vanilla alpha*(V-V_sft)^2 term is gone,
replaced by a reward-space ridge lam_pop*||V wbar||^2. Under shuffled labels the
data (NLL) term goes flat, so nothing pushes wbar in any direction and the ridge
simply shrinks V@wbar toward ZERO -- not toward the head. So the null-fit
direction is a near-degenerate low-norm vector with no consistent orientation.
The comparison is:
    head            the base reward model
    shuffled v_pop  what the fitter returns with no preference signal (~ noise)
    real v_pop      the actual v2 fit
Does using the real labels produce a shared direction with concept structure and
above-chance held-out accuracy that the shuffled fit does not? If real v_pop is
concept-aligned / predictive while shuffled v_pop is not, the shared direction is
carrying real preference signal, not a fitter+embedding artifact.

Writes:
  artifacts/exp1_shuffled_vpop.pt   the shuffled shared direction [4096] + V, wbar
  results/exp1/summary.json         head vs real vs shuffled: #sig concepts, acc
  results/exp1/concepts.json        full per-concept cosines for all three
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C


def refit_shared_direction(train_diffs, head_unit, K, lam_pop, lam_d, iters, lr, seed, device):
    """Sign-shuffle each diff, refit LoRe v2, return oriented v_pop plus V, wbar."""
    from utils import LoReV2, set_seed

    g = torch.Generator().manual_seed(seed)
    neg, pos = torch.tensor(-1.0), torch.tensor(1.0)
    shuffled = []
    for X in train_diffs:
        flip = (torch.rand(X.shape[0], 1, generator=g) < 0.5)
        signs = torch.where(flip, neg, pos)
        shuffled.append((X.float() * signs).to(device))

    set_seed(seed)
    am = LoReV2(len(shuffled), 4096, K, lam_pop=lam_pop, lam_d=lam_d,
                num_iterations=iters, learning_rate=lr)
    am.train(shuffled)
    V_full = am.V.detach().cpu().float()
    wbar = am.wbar.detach().cpu().float().reshape(1, -1)   # [1, K]; mean == wbar
    return C.shared_direction(V_full, wbar, head_unit), V_full, wbar


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
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 ridge on ||V wbar||^2 (match the real fit)")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 ridge on mean_u ||V delta_u||^2")
    ap.add_argument("--iters", type=int, default=2000, help="LoReV2 solve steps (match the real fit)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp1")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)
    print(f"[exp1] device={device} lam_pop={args.lam_pop} lam_d={args.lam_d} iters={args.iters} seed={args.seed}")

    head_unit = C.load_head()

    V, W = C.load_basis()
    v_real = C.shared_direction(V, W, head_unit)
    K = V.shape[1]

    concepts = C.load_concepts()
    tau = C.null_threshold()
    print(f"[exp1] significance threshold |cos|>{tau:.4f}")

    train_diffs = C.load_user_diffs(split="train", seen=True)
    test_diffs = C.load_user_diffs(split="test", seen=True)
    print(f"[exp1] refitting on {len(train_diffs)} users with signs shuffled...")

    v_shuf, V_shuf, wbar_shuf = refit_shared_direction(
        train_diffs, head_unit, K, args.lam_pop, args.lam_d, args.iters, args.lr, args.seed, device
    )
    torch.save({"v_pop_shuffled": v_shuf, "V": V_shuf, "wbar": wbar_shuf,
                "seed": args.seed, "lam_pop": args.lam_pop, "lam_d": args.lam_d},
               os.path.join(C.ARTIFACTS_DIR, "exp1_shuffled_vpop.pt"))

    head_cos, head_rep = report("head", head_unit, concepts, tau, test_diffs, head_unit, v_real)
    real_cos, real_rep = report("real_vpop", v_real, concepts, tau, test_diffs, head_unit, v_real)
    shuf_cos, shuf_rep = report("shuffled_vpop", v_shuf, concepts, tau, test_diffs, head_unit, v_real)

    summary = {
        "significance_threshold": tau,
        "lam_pop": args.lam_pop,
        "lam_d": args.lam_d,
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
