"""CA Experiment 1 -- shuffled-label control on Community Alignment (the key control).

Question: is v_pop's concept alignment / held-out accuracy (exp2) real preference
signal, or something the LoRe v2 fit would produce from ANY CA data?

How the signal lives in the data: LoRe never sees a label column. Each CA pair is
folded into one diff d = chosen_embedding - rejected_embedding, and the fit pushes
d . (V (wbar + delta_u)) > 0. So the ORIENTATION of each diff *is* the label.

The control: flip each diff's sign with prob 0.5 -- exactly "randomly relabel which
response was chosen." This kills the consistent preferred direction (each user's mean
diff -> ~0) while leaving every embedding, magnitude, and per-user grouping intact.
Then re-run the SAME LoReV2 solve with the SAME hyperparameters (lam_pop, lam_d, lr,
iters) as the committed CA fit.

v2 has no head anchor, so under shuffled labels the data term goes flat and the ridge
lam_pop*||V wbar||^2 simply shrinks V@wbar toward ZERO -- not toward the head. The
null-fit direction is a near-degenerate low-norm vector with arbitrary orientation.
Discriminators are the ||wbar|| collapse and cos(shuffled, real) ~ 0, NOT the accuracy
(whose sign is arbitrary once the direction has collapsed).

Writes:
  artifacts/exp1_shuffled_vpop.pt   the shuffled shared direction [4096] + V, wbar
  results/exp1/summary.json         head vs real vs shuffled: #sig concepts, acc, norms
  results/exp1/concepts.json        full per-concept cosines for all three
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import ca_common as C


def refit_shared_direction(train_diffs, head_unit, K, lam_pop, lam_d, iters, lr, seed, device):
    """Sign-shuffle each diff, refit LoRe v2, return oriented v_pop plus V, wbar, ||wbar||."""
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
                num_iterations=iters, learning_rate=lr, verbose=False)
    am.train(shuffled)
    V_full = am.V.detach().cpu().float()
    wbar = am.wbar.detach().cpu().float().reshape(1, -1)   # [1, K]; mean == wbar
    wbar_norm = float(am.wbar.detach().cpu().float().norm())
    return C.shared_direction(V_full, wbar, head_unit), V_full, wbar, wbar_norm


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
    ap.add_argument("--config", default=C.DEFAULT_CONFIG_KEY)
    ap.add_argument("--lam_pop", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--lam_d", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--iters", type=int, default=None, help="default: match the CA fit")
    ap.add_argument("--lr", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--seed", type=int, default=None, help="default: match the CA fit")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp1")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head_unit = C.load_head()
    V, wbar_real, v_real, cfg = C.load_ca_basis(args.config)
    K = int(cfg["K"])
    # hyperparameters default to the committed CA fit so the control matches the real run
    lam_pop = args.lam_pop if args.lam_pop is not None else float(cfg["lam_pop"])
    lam_d = args.lam_d if args.lam_d is not None else float(cfg["lam_d"])
    iters = args.iters if args.iters is not None else int(cfg["iters"])
    lr = args.lr if args.lr is not None else float(cfg["lr"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    wbar_real_norm = float(wbar_real.norm())
    print(f"[ca-exp1] device={device} K={K} lam_pop={lam_pop} lam_d={lam_d} "
          f"iters={iters} lr={lr} seed={seed}")

    concepts = C.load_concepts()
    tau = C.null_threshold()
    _, train_diffs, _, test_diffs = C.load_ca_diffs(cfg)
    print(f"[ca-exp1] refitting on {len(train_diffs)} users with signs shuffled...")

    v_shuf, V_shuf, wbar_shuf, wbar_shuf_norm = refit_shared_direction(
        train_diffs, head_unit, K, lam_pop, lam_d, iters, lr, seed, device
    )
    torch.save({"v_pop_shuffled": v_shuf, "V": V_shuf, "wbar": wbar_shuf,
                "wbar_norm": wbar_shuf_norm, "seed": seed,
                "lam_pop": lam_pop, "lam_d": lam_d},
               os.path.join(C.ARTIFACTS_DIR, "exp1_shuffled_vpop.pt"))

    head_cos, head_rep = report("head", head_unit, concepts, tau, test_diffs, head_unit, v_real)
    real_cos, real_rep = report("real_vpop", v_real, concepts, tau, test_diffs, head_unit, v_real)
    shuf_cos, shuf_rep = report("shuffled_vpop", v_shuf, concepts, tau, test_diffs, head_unit, v_real)

    summary = {
        "dataset": "community_alignment",
        "config": args.config,
        "significance_threshold": tau,
        "lam_pop": lam_pop, "lam_d": lam_d, "iters": iters, "lr": lr, "seed": seed,
        "wbar_norm_real": wbar_real_norm,
        "wbar_norm_shuffled": wbar_shuf_norm,
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
        print(f"[ca-exp1] {r['name']:14s} sig={r['n_significant_concepts']:2d} "
              f"acc={r['held_out_accuracy']['overall_pair_acc']:.4f} "
              f"cos(head)={r['cos_to_head']:+.4f}")
    print(f"[ca-exp1] ||wbar|| real={wbar_real_norm:.4f} -> shuffled={wbar_shuf_norm:.4f}  "
          f"cos(shuffled,real)={float(v_shuf @ v_real):+.4f}")
    print(f"[ca-exp1] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
