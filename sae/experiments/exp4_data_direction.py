"""Experiment 4 -- is v_pop = head faithful to the DATA, or forced by the anchor?

No LoRe fit at all. Straight from the PRISM pairs, compute the raw mean preference
direction
    u_data = unit( mean_over_all_pairs (chosen - rejected) )
and the per-user mean-of-means variant. Then compare to the base head.

Reads the exp1/2/3 result: v_pop == head to 5 decimals. Two ways that can happen:
  (a) the data's own average preference points along the head -> LoRe returning the
      head is FAITHFUL; there is genuinely nothing else to learn.
  (b) the alpha=1e4 head anchor forces V -> head regardless of the data.
If cos(u_data, head) is high, (a) holds and the null result is real. If u_data
points somewhere else while the fit still returns the head, (b) holds and we must
reinterpret. This settles it with no fitter in the loop.

Writes:
  results/exp4/summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

import common as C


def best_single_direction(train_diffs, iters=4000, lr=0.05, seed=42, device="cpu"):
    """The OPTIMAL single shared direction: fit w to maximize logsigmoid(d.w) over
    all train pairs (every diff is a 'chosen wins' positive). This is the best any
    one direction can do -- a real ceiling, unlike the mean-diff heuristic.

    The fit orients w toward the chosen side by construction, so we do NOT reorient
    it to the head. Reorienting a direction that is near-orthogonal to the head
    (cos ~ 0) flips its sign on a coin toss and makes direction_pair_accuracy report
    1 - acc instead of acc -- the spurious below-chance 0.406 in the earlier run."""
    torch.manual_seed(seed)
    D = torch.cat([X.float() for X in train_diffs], dim=0).to(device)
    w = torch.nn.Parameter(C.unit(torch.randn(D.shape[1])).to(device))
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = -F.logsigmoid(D @ w).mean()
        loss.backward()
        opt.step()
    return C.unit(w.detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fit-iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = C.resolve_device(args.device)

    out_dir = os.path.join(C.RESULTS_DIR, "exp4")
    C.ensure_dirs(out_dir)

    head = C.load_head()
    V, W = C.load_basis()
    v_pop = C.shared_direction(V, W, head)

    concepts = C.load_concepts()
    tau = C.null_threshold()

    def data_dir(diffs):
        all_pairs = torch.cat([X.float() for X in diffs], dim=0)
        u = C.unit(all_pairs.mean(dim=0))
        if float(u @ head) < 0:
            u = -u
        return u, int(all_pairs.shape[0]), float(all_pairs.norm(dim=1).mean())

    train_diffs = C.load_user_diffs(split="train", seen=True)
    test_diffs = C.load_user_diffs(split="test", seen=True)
    u_train, n_train, norm_train = data_dir(train_diffs)   # honest direction, TRAIN only
    u_test, n_test, _ = data_dir(test_diffs)               # in-sample mean-diff direction

    # The OPTIMAL single shared direction (logistic fit on train) -- a true ceiling.
    print(f"[exp4] fitting best single direction ({args.fit_iters} steps)...")
    w_best = best_single_direction(train_diffs, iters=args.fit_iters, seed=args.seed, device=device)

    # Held-out accuracy on TEST: does anything (honest mean, or the BEST fitted single
    # direction) beat the anchor-forced v_pop (= head)?
    acc = {
        "u_data_train_on_test": C.direction_pair_accuracy(u_train, test_diffs),
        "u_data_test_insample": C.direction_pair_accuracy(u_test, test_diffs),
        "best_single_dir_on_test": C.direction_pair_accuracy(w_best, test_diffs),
        "head_on_test": C.direction_pair_accuracy(head, test_diffs),
        "v_pop_on_test": C.direction_pair_accuracy(v_pop, test_diffs),
    }
    cos_train_cs = C.concept_cosines(u_train, concepts)
    summary = {
        "n_train_pairs": n_train, "n_test_pairs": n_test,
        "mean_train_diff_norm": norm_train,
        "cos_u_train_head": float(u_train @ head),
        "cos_u_train_vpop": float(u_train @ v_pop),
        "cos_u_train_u_test": float(u_train @ u_test),
        "cos_u_test_head": float(u_test @ head),
        "cos_best_single_head": float(w_best @ head),
        "cos_best_single_vpop": float(w_best @ v_pop),
        "held_out_accuracy": {k: v["overall_pair_acc"] for k, v in acc.items()},
        "held_out_accuracy_full": acc,
        "u_train_concepts_sig": dict(sorted(
            ((c, v) for c, v in cos_train_cs.items() if abs(v) > tau),
            key=lambda kv: -abs(kv[1]))),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[exp4] cos(u_data_train, head)={summary['cos_u_train_head']:+.4f}  "
          f"cos(u_data_train, v_pop)={summary['cos_u_train_vpop']:+.4f}")
    a = summary["held_out_accuracy"]
    print(f"[exp4] TEST acc: u_data(train)={a['u_data_train_on_test']:.4f}  "
          f"u_data(insample)={a['u_data_test_insample']:.4f}  "
          f"best_single={a['best_single_dir_on_test']:.4f}  "
          f"head={a['head_on_test']:.4f}  v_pop={a['v_pop_on_test']:.4f}")
    print(f"[exp4] cos(best_single, head)={summary['cos_best_single_head']:+.4f}")
    print(f"[exp4] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
