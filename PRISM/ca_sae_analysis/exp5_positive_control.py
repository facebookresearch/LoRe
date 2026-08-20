"""CA Experiment 5 -- positive control: does the v2 pipeline have the POWER to recover a
shared direction that is NOT the head, on Community Alignment data?

Even without an anchor, confirm the LoReV2 fit + reward-space ridge can recover a clean
shared signal when one exists -- otherwise the real-data result (v_pop != head) could be
a fitter that never produces a strong non-head direction.

Method: keep the exact per-user pair structure of the real CA train data, but REPLACE
every diff's direction with a known target u* (a concept vector chosen nearly orthogonal
to the head -- 'fluency', cos ~ -0.04), preserving each diff's original magnitude. Then
refit LoRe v2 with the REAL CA hyperparameters (lam_pop, lam_d, lr, iters).

Read:
  cos(v_pop_synth, u*)     high -> fit recovers a non-head signal -> pipeline has power
  cos(v_pop_synth, head)   high -> fit drifts to head even given a clean signal -> test uninformative

Convergence note: at the CA fit's own hyperparameters (lr=1e-4, iters=5000 -- tuned low
for stability on ~180 pairs/user) recovery is only partial (cos ~0.29 to u*) but already
clearly off the head (cos ~0.00), so the verdict is recovered_target. At the standard v2
lr (1e-2, iters=2000) the fit recovers the injected direction far more fully (cos ~0.80),
confirming the partial number is under-convergence at the low lr, not a pipeline limit.
See results/exp5/summary_lr1e-2_convergence_check.json. The canonical summary.json below
uses the matched CA hyperparameters so the control mirrors the real fit.

Writes:
  artifacts/exp5_synth_vpop.pt
  results/exp5/summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import ca_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG_KEY)
    ap.add_argument("--target", default="fluency", help="concept used as the injected shared direction u*")
    ap.add_argument("--lam_pop", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--lam_d", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--iters", type=int, default=None, help="default: match the CA fit")
    ap.add_argument("--lr", type=float, default=None, help="default: match the CA fit")
    ap.add_argument("--seed", type=int, default=None, help="default: match the CA fit")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = C.resolve_device(args.device)
    out_dir = os.path.join(C.RESULTS_DIR, "exp5")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head_unit = C.load_head()

    concepts = C.load_concepts()
    if args.target not in concepts:
        raise KeyError(f"{args.target!r} not a concept; have {list(concepts)}")
    u_star = concepts[args.target]
    cos_ustar_head = float(u_star @ head_unit)
    print(f"[ca-exp5] target={args.target}  cos(u*, head)={cos_ustar_head:+.4f}")

    V, wbar_real, v_real, cfg = C.load_ca_basis(args.config)
    K = int(cfg["K"])
    lam_pop = args.lam_pop if args.lam_pop is not None else float(cfg["lam_pop"])
    lam_d = args.lam_d if args.lam_d is not None else float(cfg["lam_d"])
    iters = args.iters if args.iters is not None else int(cfg["iters"])
    lr = args.lr if args.lr is not None else float(cfg["lr"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])

    _, train_diffs, _, _ = C.load_ca_diffs(cfg)
    synth = []
    for X in train_diffs:
        mags = X.float().norm(dim=1, keepdim=True)
        synth.append((mags * u_star.unsqueeze(0)).to(device))

    from utils import LoReV2, set_seed
    set_seed(seed)
    am = LoReV2(len(synth), 4096, K, lam_pop=lam_pop, lam_d=lam_d,
                num_iterations=iters, learning_rate=lr, verbose=False)
    am.train(synth)
    V_full = am.V.detach().cpu().float()
    wbar = am.wbar.detach().cpu().float().reshape(1, -1)
    v_synth = C.shared_direction(V_full, wbar, u_star)      # orient toward the injected target

    torch.save({"v_pop_synth": v_synth, "V": V_full, "wbar": wbar,
                "target": args.target, "lam_pop": lam_pop, "lam_d": lam_d},
               os.path.join(C.ARTIFACTS_DIR, "exp5_synth_vpop.pt"))

    cos_recover = float(v_synth @ u_star)
    cos_head = float(v_synth @ head_unit)
    verdict = ("recovered_target" if abs(cos_recover) > abs(cos_head)
               else "collapsed_to_head")

    summary = {
        "dataset": "community_alignment",
        "config": args.config,
        "target": args.target,
        "lam_pop": lam_pop, "lam_d": lam_d, "iters": iters, "lr": lr, "seed": seed,
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

    print(f"[ca-exp5] cos(v_pop_synth, u*)={cos_recover:+.4f}  "
          f"cos(v_pop_synth, head)={cos_head:+.4f}  -> {verdict}")
    print(f"[ca-exp5] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
