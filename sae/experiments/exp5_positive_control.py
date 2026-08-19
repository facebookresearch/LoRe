"""Experiment 5 -- positive control: does the pipeline have the POWER to detect a
shared direction that is NOT the head?

The worry after exp1-3: alpha=1e4 pulls every basis column toward the head, so maybe
the fit returns the head no matter what the data says, making our null result
uninformative. This checks that directly.

Method: keep the exact per-user pair structure of the real PRISM train data, but
REPLACE every diff's direction with a known target u* (a concept vector chosen to be
nearly orthogonal to the head -- 'fluency', cos ~ -0.04 to head), preserving each
diff's original magnitude. So the synthetic data has a clean, strong shared
preference along u*. Then refit LoRe with the REAL hyperparameters (alpha=1e4).

Read:
  cos(v_pop_synth, u*)     high  -> the fit recovers a non-head signal -> pipeline has
                                    power -> the real-data null is meaningful (H stands)
  cos(v_pop_synth, head)   high  -> the anchor dominates; the fit ignores even a clean
                                    signal -> our test is uninformative (H unproven)

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
    ap.add_argument("--alpha", type=float, default=1e4, help="head-anchor strength (1e4 = real fit)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp5")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head_unit = C.load_head()
    from rm_head_utils import load_reward_head
    head_raw = load_reward_head().reshape(-1, 1).float()

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

    from utils import LoRe_regularized
    torch.manual_seed(args.seed)
    am = LoRe_regularized(head_raw.to(device), args.alpha, len(synth), 4096, K, args.iters, args.lr)
    am.train(synth)
    V_full = am.V.detach().cpu().float()
    W_full = torch.softmax(am.W, dim=1).detach().cpu().float()
    v_synth = C.shared_direction(V_full, W_full, u_star)   # orient toward the injected target

    torch.save({"v_pop_synth": v_synth, "V": V_full, "W": W_full,
                "target": args.target, "alpha": args.alpha},
               os.path.join(C.ARTIFACTS_DIR, "exp5_synth_vpop.pt"))

    cos_recover = float(v_synth @ u_star)
    cos_head = float(v_synth @ head_unit)
    # what did the fit actually land on -- head or target?
    verdict = ("recovered_target" if abs(cos_recover) > abs(cos_head)
               else "collapsed_to_head")

    summary = {
        "target": args.target,
        "alpha": args.alpha,
        "cos_ustar_head": cos_ustar_head,
        "cos_vpop_synth_target": cos_recover,
        "cos_vpop_synth_head": cos_head,
        "verdict": verdict,
        "interpretation": (
            "pipeline recovers a non-head signal -> real-data null is meaningful"
            if verdict == "recovered_target"
            else "anchor dominates even a clean signal -> test uninformative"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[exp5] cos(v_pop_synth, u*)={cos_recover:+.4f}  "
          f"cos(v_pop_synth, head)={cos_head:+.4f}  -> {verdict}")
    print(f"[exp5] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
