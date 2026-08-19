"""Experiment 3 -- do the shared direction and the base head live on the same SAE
features?

We project head, v_pop, and residual onto the D3 SAE dictionary using the
b_pre-safe decoder attribution (decoder columns are unit-norm and never touch the
pre-encoder bias): alignment = decoder.weight.T @ direction, one score per feature.

Then we ask whether the top features of v_pop and head are the same set (Jaccard /
rank overlap). If they are, the shared direction is expressed by the same dictionary
features as the base reward head -- it is the head, in feature space. The residual's
top features show what (if anything) v_pop uses that the head does not.

Writes:
  results/exp3/alignment.pt      per-feature alignment + cosine [dict, 3] and ids
  results/exp3/top_features.json top-N features per direction + overlaps
  results/exp3/summary.json      headline overlaps
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C
from sae.src.attribution import decoder_basis_alignment


def top_ids(alignment_col, top_n):
    """Indices of the top-N features by |alignment|."""
    return torch.topk(alignment_col.abs(), k=min(top_n, alignment_col.numel())).indices


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = os.path.join(C.RESULTS_DIR, "exp3")
    C.ensure_dirs(out_dir)

    head = C.load_head()
    V, W = C.load_basis()
    v_pop = C.shared_direction(V, W, head)
    resid = C.residual(v_pop, head)

    sae = C.load_sae()
    Wdec = sae.decoder.weight.detach().cpu().float()   # [input_dim, dict_size]
    print(f"[exp3] decoder {tuple(Wdec.shape)}, top_n={args.top_n}")

    names = ["head", "v_pop", "residual"]
    dirs = torch.stack([head, v_pop, resid], dim=1)     # [4096, 3]
    alignment, cosine = decoder_basis_alignment(Wdec, dirs)  # [dict, 3] each

    tops = {n: top_ids(alignment[:, j], args.top_n) for j, n in enumerate(names)}

    torch.save({"alignment": alignment, "cosine": cosine, "names": names,
                "top_ids": {n: tops[n] for n in names}},
               os.path.join(out_dir, "alignment.pt"))

    overlaps = {
        "jaccard_vpop_head": jaccard(tops["v_pop"], tops["head"]),
        "jaccard_residual_head": jaccard(tops["residual"], tops["head"]),
        "jaccard_residual_vpop": jaccard(tops["residual"], tops["v_pop"]),
    }

    top_features = {}
    for j, n in enumerate(names):
        ids = tops[n].tolist()
        top_features[n] = [
            {"feature_id": int(i),
             "alignment": float(alignment[i, j]),
             "cosine": float(cosine[i, j])}
            for i in ids
        ]

    with open(os.path.join(out_dir, "top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "top_n": args.top_n,
            "overlaps": overlaps,
            "cos_vpop_head": float(v_pop @ head),
            "n_features": int(Wdec.shape[1]),
        }, f, indent=2)

    print(f"[exp3] Jaccard(v_pop, head) top-{args.top_n} = {overlaps['jaccard_vpop_head']:.3f}")
    print(f"[exp3] Jaccard(residual, head)              = {overlaps['jaccard_residual_head']:.3f}")
    print(f"[exp3] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
