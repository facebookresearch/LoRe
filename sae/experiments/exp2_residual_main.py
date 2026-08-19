"""Experiment 2 -- the main result: does the shared direction add anything beyond
the base reward head?

Three directions in the 4096-dim reward-embedding space:
  head     unit base reward head (Skywork score.weight)
  v_pop    the LoRe shared population direction = unit(V @ mean_users(W))
  residual unit(v_pop - (v_pop.head) head)  -- the part of v_pop the head misses

For each we report:
  concept alignment  cosine to each of the 11 concept vectors, and which clear the
                     random-direction null (|cos| > tau)
  held-out accuracy  fraction of PRISM test (chosen - rejected) pairs the direction
                     ranks correctly

If the residual aligns with no concepts above the null AND scores at chance, then
v_pop carries nothing the base head does not already -- the "shared direction" is
just the base reward model. If the residual DOES align/predict, v_pop adds real
structure. Either way the numbers here settle it.

Writes:
  artifacts/exp2_directions.pt   head, v_pop, residual [4096] each
  results/exp2/concepts.json     per-direction cosines + significant subset
  results/exp2/accuracy.json     per-direction held-out accuracy
  results/exp2/summary.json      compact headline numbers
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C


def report_direction(name, direction, concepts, tau, test_diffs, head_unit):
    cos = C.concept_cosines(direction, concepts)
    sig = dict(sorted(
        ((c, v) for c, v in cos.items() if abs(v) > tau),
        key=lambda kv: -abs(kv[1]),
    ))
    acc = C.direction_pair_accuracy(direction, test_diffs)
    return {
        "name": name,
        "cos_to_head": float(direction @ head_unit),
        "concept_cosines": cos,
        "significant_concepts": sig,
        "n_significant": len(sig),
        "held_out_accuracy": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = os.path.join(C.RESULTS_DIR, "exp2")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head = C.load_head()
    V, W = C.load_basis()
    v_pop = C.shared_direction(V, W, head)
    resid = C.residual(v_pop, head)

    concepts = C.load_concepts()
    tau = C.null_threshold()
    test_diffs = C.load_user_diffs(split="test", seen=True)
    print(f"[exp2] tau={tau:.4f}, {len(test_diffs)} held-out users")

    dirs = {"head": head, "v_pop": v_pop, "residual": resid}
    reports = {name: report_direction(name, d, concepts, tau, test_diffs, head)
               for name, d in dirs.items()}

    torch.save({k: v for k, v in dirs.items()},
               os.path.join(C.ARTIFACTS_DIR, "exp2_directions.pt"))

    with open(os.path.join(out_dir, "concepts.json"), "w") as f:
        json.dump({k: {"cosines": r["concept_cosines"],
                       "significant": r["significant_concepts"]}
                   for k, r in reports.items()}, f, indent=2)
    with open(os.path.join(out_dir, "accuracy.json"), "w") as f:
        json.dump({k: r["held_out_accuracy"] for k, r in reports.items()}, f, indent=2)

    summary = {
        "significance_threshold": tau,
        "cos_vpop_head": float(v_pop @ head),
        "directions": {
            k: {
                "n_significant_concepts": r["n_significant"],
                "top_concepts": list(r["significant_concepts"].items())[:5],
                "overall_pair_acc": r["held_out_accuracy"]["overall_pair_acc"],
                "per_user_mean_acc": r["held_out_accuracy"]["per_user_mean_acc"],
                "cos_to_head": r["cos_to_head"],
            }
            for k, r in reports.items()
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for k, r in reports.items():
        print(f"[exp2] {k:9s} sig={r['n_significant']:2d} "
              f"acc={r['held_out_accuracy']['overall_pair_acc']:.4f} "
              f"cos(head)={r['cos_to_head']:+.4f}")
    print(f"[exp2] cos(v_pop, head)={float(v_pop @ head):+.4f}")
    print(f"[exp2] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
