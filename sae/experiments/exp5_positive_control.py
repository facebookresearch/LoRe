"""Experiment 5 -- positive control: does the v2 pipeline have the POWER to recover a
shared direction that is NOT the head?

Even without an anchor, we should confirm the LoReV2 fit + reward-space ridge can
recover a clean shared signal when one exists -- otherwise the real-data result
(v_pop != head) could be a fitter that simply never produces a strong direction.
This checks that directly.

Method: keep the exact per-user pair structure of the real PRISM train data, but
REPLACE every diff's direction with a known target u* (a concept vector chosen to be
nearly orthogonal to the head -- 'fluency', cos ~ -0.04 to head), preserving each
diff's original magnitude. So the synthetic data has a clean, strong shared
preference along u*. Then refit LoRe v2 with the REAL hyperparameters (lam_pop,
lam_d).

Read:
  cos(v_pop_synth, u*)     high  -> the fit recovers a non-head signal -> pipeline has
                                    power -> the real-data result is meaningful
  cos(v_pop_synth, head)   high  -> the fit drifts to the head even given a clean
                                    non-head signal -> our test is uninformative

Writes:
  artifacts/exp5_synth_vpop.pt
  results/exp5/summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fluency", help="concept used as the injected shared direction u*")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 ridge on ||V wbar||^2 (match the real fit)")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 ridge on mean_u ||V delta_u||^2")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp5")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head_unit = C.load_head()

    concepts = C.load_concepts()
    if args.target not in concepts:
        raise KeyError(f"{args.target!r} not a concept; have {list(concepts)}")
    u_star = concepts[args.target]                         # unit [4096]
    cos_ustar_head = float(u_star @ head_unit)
    print(f"[exp5] target={args.target}  cos(u*, head)={cos_ustar_head:+.4f}")

    V, W = C.load_basis()
    K = V.shape[1]

    # Build synthetic train data: keep per-user counts + magnitudes, direction := u*.
    train_diffs = C.load_user_diffs(split="train", seen=True)
    synth = []
    for X in train_diffs:
        mags = X.float().norm(dim=1, keepdim=True)         # [m,1]
        synth.append((mags * u_star.unsqueeze(0)).to(device))

    from utils import LoReV2, set_seed
    set_seed(args.seed)
    am = LoReV2(len(synth), 4096, K, lam_pop=args.lam_pop, lam_d=args.lam_d,
                num_iterations=args.iters, learning_rate=args.lr)
    am.train(synth)
    V_full = am.V.detach().cpu().float()
    wbar = am.wbar.detach().cpu().float().reshape(1, -1)    # [1, K]; mean == wbar
    v_synth = C.shared_direction(V_full, wbar, u_star)      # orient toward the injected target

    torch.save({"v_pop_synth": v_synth, "V": V_full, "wbar": wbar,
                "target": args.target, "lam_pop": args.lam_pop, "lam_d": args.lam_d},
               os.path.join(C.ARTIFACTS_DIR, "exp5_synth_vpop.pt"))

    cos_recover = float(v_synth @ u_star)
    cos_head = float(v_synth @ head_unit)
    # what did the fit actually land on -- head or target?
    verdict = ("recovered_target" if abs(cos_recover) > abs(cos_head)
               else "collapsed_to_head")

    summary = {
        "target": args.target,
        "lam_pop": args.lam_pop,
        "lam_d": args.lam_d,
        "cos_ustar_head": cos_ustar_head,
        "cos_vpop_synth_target": cos_recover,
        "cos_vpop_synth_head": cos_head,
        "verdict": verdict,
        "interpretation": (
            "pipeline recovers a non-head signal -> real-data result is meaningful"
            if verdict == "recovered_target"
            else "fit drifts to head even given a clean signal -> test uninformative"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[exp5] cos(v_pop_synth, u*)={cos_recover:+.4f}  "
          f"cos(v_pop_synth, head)={cos_head:+.4f}  -> {verdict}")
    print(f"[exp5] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
