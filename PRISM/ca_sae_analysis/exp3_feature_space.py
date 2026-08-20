"""Experiment 3 -- do the shared direction and the base head live on the same SAE
features?  Run for BOTH Community Alignment (CA) and PRISM in one pass.

Same method as sae/experiments/exp3_feature_space.py: project head, v_pop, and the
residual onto the D3 TopK SAE dictionary with the b_pre-safe decoder attribution
(alignment = decoder.weight.T @ direction, one score per feature), then ask whether
the top-N features of v_pop and head are the same set (Jaccard). High Jaccard -> the
shared direction is expressed by the same dictionary features as the base reward head
(it *is* the head in feature space). Low Jaccard + a residual that lights up its own
features -> v_pop uses features the head does not.

The D3 SAE decodes the shared Skywork embedding space (dataset-independent), so the
SAME checkpoint is used for CA and PRISM; only v_pop differs (CA fit vs PRISM fit).
Top features are ranked by |alignment|, so direction sign is irrelevant here -- this
sidesteps the sign-flip concern that matters for the pair-accuracy experiments.

Writes (under results/exp3/):
  ca_alignment.pt / prism_alignment.pt         per-feature alignment + cosine [dict,3]
  ca_top_features.json / prism_top_features.json  top-N features per direction
  summary.json                                  headline overlaps for both datasets
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import ca_common as C
from common import load_sae, load_basis          # PRISM sae/experiments/common.py
from sae.src.attribution import decoder_basis_alignment

LOCAL_SAE_CKPT = os.path.join(os.path.dirname(__file__), "D3_16384_k256_string-vs-string-v1.pt")


def top_ids(alignment_col, top_n):
    return torch.topk(alignment_col.abs(), k=min(top_n, alignment_col.numel())).indices


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def analyze(tag, head_unit, v_pop, Wdec, top_n, out_dir):
    """Feature-overlap report for one dataset's v_pop against the shared head."""
    resid = C.residual(v_pop, head_unit)
    names = ["head", "v_pop", "residual"]
    dirs = torch.stack([head_unit, v_pop, resid], dim=1)          # [4096, 3]
    alignment, cosine = decoder_basis_alignment(Wdec, dirs)       # [dict, 3] each
    tops = {n: top_ids(alignment[:, j], top_n) for j, n in enumerate(names)}

    torch.save({"alignment": alignment, "cosine": cosine, "names": names,
                "top_ids": {n: tops[n] for n in names}},
               os.path.join(out_dir, f"{tag}_alignment.pt"))

    top_features = {}
    for j, n in enumerate(names):
        top_features[n] = [
            {"feature_id": int(i), "alignment": float(alignment[i, j]),
             "cosine": float(cosine[i, j])}
            for i in tops[n].tolist()
        ]
    with open(os.path.join(out_dir, f"{tag}_top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    overlaps = {
        "jaccard_vpop_head": jaccard(tops["v_pop"], tops["head"]),
        "jaccard_residual_head": jaccard(tops["residual"], tops["head"]),
        "jaccard_residual_vpop": jaccard(tops["residual"], tops["v_pop"]),
    }
    rep = {
        "top_n": top_n,
        "cos_vpop_head": float(v_pop @ head_unit),
        "overlaps": overlaps,
    }
    print(f"[exp3:{tag}] cos(v_pop,head)={rep['cos_vpop_head']:+.4f}  "
          f"Jaccard(v_pop,head)={overlaps['jaccard_vpop_head']:.3f}  "
          f"Jaccard(resid,head)={overlaps['jaccard_residual_head']:.3f}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--config", default=C.DEFAULT_CONFIG_KEY, help="CA fit config key")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = os.path.join(C.RESULTS_DIR, "exp3")
    C.ensure_dirs(out_dir, C.ARTIFACTS_DIR)

    head_unit = C.load_head()

    sae = load_sae(LOCAL_SAE_CKPT)
    Wdec = sae.decoder.weight.detach().cpu().float()               # [4096, dict_size]
    print(f"[exp3] SAE decoder {tuple(Wdec.shape)}  ckpt={os.path.basename(LOCAL_SAE_CKPT)}")

    # --- CA v_pop (stored natural orientation; sign irrelevant to |alignment|) ---
    _, _, v_ca, _ = C.load_ca_basis(args.config)
    ca_rep = analyze("ca", head_unit, v_ca, Wdec, args.top_n, out_dir)

    # --- PRISM v_pop (LoRe v2 basis, same recipe as the committed PRISM exp3) ----
    V_p, W_p = load_basis()
    v_prism = C.shared_direction(V_p, W_p, head_unit)
    prism_rep = analyze("prism", head_unit, v_prism, Wdec, args.top_n, out_dir)

    summary = {
        "sae_checkpoint": os.path.basename(LOCAL_SAE_CKPT),
        "n_features": int(Wdec.shape[1]),
        "top_n": args.top_n,
        "community_alignment": ca_rep,
        "prism": prism_rep,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp3] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
