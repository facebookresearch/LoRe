"""
Experiment A: Does the PRISM preference direction separate chosen/rejected in OTHER
preference datasets as well as in PRISM?

We showed a single direction separates PRISM, and characterized it only by cosine
to concept vectors (correlational). Cross-dataset transfer turns that into a testable claim:
  * If the direction is a REAL generic-quality axis -> it should transfer, i.e. score chosen>rejected
    on other datasets roughly as well as the true reward head does.
  * If it is a PRISM ARTIFACT -> its accuracy drops off PRISM, and the datasets/subsets where it
    fails tell us what PRISM-specific thing it was keying on.

We score three directions head-to-head on each dataset:
  1. prism_meandiff : mean of (chosen-rejected) PRISM train diffs (the collapsed ~1-D axis)
  2. lore_basis     : column 0 of a trained LoRe V checkpoint (the collapsed basis)
  3. true_head      : Skywork's real score.weight (reference: a general RM that never saw PRISM)

How scoring works (reused from eval_rb2.py): render prompt+response with the chat template, take
the last non-pad token hidden state of the Skywork backbone, and project onto each direction.
Pairwise accuracy = fraction of pairs where chosen scores above rejected.
"""
import argparse
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head  # noqa: E402

MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


# Same templating/collation as eval_rb2.py, inlined so this script is self-contained
# (eval_rb2.py uses 3.10-only `str | None` syntax and can't be imported on 3.9).
def apply_template(tokenizer, prompt, response):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False)
    except Exception:
        eos = getattr(tokenizer, "eos_token", "") or ""
        return f"Human: {prompt}\nAssistant: {response}{eos}"


def collate(tokenizer, texts, max_length):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")


# ---- dataset loaders: normalize each to a list of (prompt, chosen_text, rejected_text) ----
def load_pairs(name, limit=None):
    if name == "rewardbench2":
        ds = load_dataset("allenai/reward-bench-2", split="test")
        pairs = [(e["prompt"], e["chosen"][0], e["rejected"][0]) for e in ds]
    elif name == "ultrafeedback":
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        pairs = [(e["prompt"], e["chosen"][-1]["content"] if isinstance(e["chosen"], list) else e["chosen"],
                  e["rejected"][-1]["content"] if isinstance(e["rejected"], list) else e["rejected"]) for e in ds]
    elif name == "hh_rlhf":
        ds = load_dataset("Anthropic/hh-rlhf", split="test")
        # HH stores full transcripts; use the last assistant turn as the response, prior text as prompt
        def split_hh(txt):
            marker = "\n\nAssistant:"
            i = txt.rfind(marker)
            return (txt[:i].strip(), txt[i + len(marker):].strip()) if i >= 0 else ("", txt)
        pairs = []
        for e in ds:
            pc, rc = split_hh(e["chosen"]), split_hh(e["rejected"])
            pairs.append((pc[0], pc[1], rc[1]))
    else:
        raise ValueError(f"unknown dataset {name}")
    if limit:
        pairs = pairs[:limit]
    return [(p, c, r) for (p, c, r) in pairs if p and c and r]


@torch.inference_mode()
def last_token_hidden(model, tokenizer, texts, device, max_length, batch_size):
    """Return [N, H] last non-pad hidden states for a list of rendered texts."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = collate(tokenizer, texts[i:i + batch_size], max_length).to(device)
        hs = model(**enc).last_hidden_state                       # [b, T, H]
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        h = hs[torch.arange(hs.size(0), device=device), last_idx]  # [b, H]
        out.append(h.float().cpu())
    return torch.cat(out, 0)


def _find_lore_ckpt(lore_ckpt):
    """Resolve the LoRe checkpoint across the layouts it may live in (the K-matrices are in the
    parent-project reproduced_matrices/, not the in-repo one that only holds the head/mean-diff)."""
    candidates = [lore_ckpt] if lore_ckpt else []
    candidates += [
        os.path.join(SCRIPT_DIR, "..", "..", "reproduced_matrices", "PRISM_V_lore_K_20_alpha_10000.0.pt"),
        os.path.join(SCRIPT_DIR, "..", "checkpoints", "checkpoints", "PRISM_V_lore_K_20_alpha_10000.0.pt"),
    ]
    return next((p for p in candidates if p and os.path.exists(p)), None)


def build_heads(lore_ckpt, device):
    heads, names = [], []
    md = torch.load(os.path.join(SCRIPT_DIR, "..", "reproduced_matrices",
                                 "prism_mean_diff_direction.pt"), weights_only=True).reshape(-1)
    heads.append(md); names.append("prism_meandiff")
    ckpt = _find_lore_ckpt(lore_ckpt)
    if ckpt is not None:
        V = torch.load(ckpt, map_location="cpu", weights_only=True).float()
        heads.append(V[:, 0]); names.append("lore_basis")
        print(f"  lore_basis from {ckpt}")
    else:
        print("  [WARN] no LoRe checkpoint found; skipping lore_basis direction "
              "(pass --lore_ckpt to include it)")
    heads.append(load_reward_head().reshape(-1)); names.append("true_head")
    H = torch.stack(heads, dim=1).to(device)                       # [4096, n_heads]
    return H, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rewardbench2", "ultrafeedback", "hh_rlhf"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--lore_ckpt", default=os.path.join(SCRIPT_DIR, "..", "reproduced_matrices",
                                                        "PRISM_V_lore_K_20_alpha_10000.0.pt"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results", "cross_dataset",
                                                  "exp_a_cross_dataset.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Loading Skywork backbone...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # sdpa (memory-efficient attention) instead of eager: eager materializes the full
    # [B, H, T, T] fp32 attention matrix (~4GB at B=8,T=2048) and OOMs the A10G on long inputs.
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(device).eval()
    H, names = build_heads(args.lore_ckpt, device)
    Hcpu = H.cpu()
    print(f"Directions under test: {names}")

    # Write results incrementally per-dataset (and isolate failures) so one dataset OOMing
    # never discards the datasets that already succeeded.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["dataset", "direction", "n_pairs", "accuracy"])
    w.writeheader(); f.flush()
    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            pairs = load_pairs(name, args.limit)
            print(f"{len(pairs)} pairs")
            chosen_txt = [apply_template(tokenizer, p, c) for p, c, r in pairs]
            reject_txt = [apply_template(tokenizer, p, r) for p, c, r in pairs]
            hc = last_token_hidden(model, tokenizer, chosen_txt, device, args.max_length, args.batch_size)
            hr = last_token_hidden(model, tokenizer, reject_txt, device, args.max_length, args.batch_size)
            acc = ((hc @ Hcpu - hr @ Hcpu) > 0).float().mean(0)   # per-direction pairwise accuracy
            for j, dname in enumerate(names):
                print(f"  {dname:16s}: acc = {acc[j].item():.4f}")
                w.writerow({"dataset": name, "direction": dname, "n_pairs": len(pairs),
                            "accuracy": round(acc[j].item(), 4)})
            f.flush()
        except Exception as ex:
            print(f"  [SKIP] {name} failed: {ex!r}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    f.close()
    print(f"\nSaved {args.out}")
    print("Read-out: if prism_meandiff tracks true_head across datasets -> real generic-quality axis;")
    print("if it collapses off PRISM while true_head holds -> PRISM-specific artifact.")


if __name__ == "__main__":
    main()
