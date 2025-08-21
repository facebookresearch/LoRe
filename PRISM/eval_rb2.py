# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RewardBench 2 evaluator supporting:
  (A) Seq-classification reward models (AutoModelForSequenceClassification)
  (B) Causal LM + custom head vector(s): v∈R^H or V∈R^{H×B}

Features
- Loads allenai/reward-bench-2 (prompt, chosen[], rejected[], subset)
- Chat templating; robust last non-pad token pooling
- Per-subset accuracy + Ties strict + weighted approx
- Multi-head evaluation with CSV export and top-k table

Usage examples are in the README section above.

Author: you + ChatGPT
"""

import argparse
import os
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

SUBSETS_V2 = ["Factuality", "Precise IF", "Math", "Safety", "Focus", "Ties"]

# ----------------------------
# CLI
# ----------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="RewardBench 2 evaluator (seq-clf or LM+head)")
    ap.add_argument("--model", default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", help="HF model id or local path")
    # ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="HF model id or local path")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id (defaults to --model)")
    ap.add_argument("--mode", choices=["auto", "seqclf", "lmhead"], default="auto",
                    help="seqclf = classification head; lmhead = causal LM + custom head matrix; auto tries to infer")

    # Head loading (for lmhead mode)
    ap.add_argument("--rm_head", default=None,
                    help="Path to head vector/matrix: .npy or .pt; shape (H,), (H,1), or (H,B)")
    ap.add_argument("--head_bias", default=None,
                    help="Optional bias: scalar, or path to .npy/.pt of shape (), (1,), or (B,)")
    ap.add_argument("--head_key", default=None,
                    help="If --rm_head is a dict checkpoint, key to pick (e.g., 'v_head.weight')")

    # Dataset / eval
    ap.add_argument("--split", default="test", help="Dataset split")
    ap.add_argument("--subset", default=None, choices=SUBSETS_V2 + [s.lower() for s in SUBSETS_V2],
                    help="Evaluate a single subset only")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts (smoke test)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--truncation_side", choices=["left","right"], default=None,
                    help="Override tokenizer.truncation_side")

    # System
    ap.add_argument("--device", default=None, help="cpu | cuda | auto (default: auto)")
    ap.add_argument("--dtype", default="auto", choices=["auto","bfloat16","float16","float32"])
    ap.add_argument("--trust_remote_code", action="store_true")

    # Output
    ap.add_argument("--save_csv", default=None, help="Write per-head summary CSV when using multi-head")
    ap.add_argument("--top_k", type=int, default=10, help="Number of top heads to print (multi-head)")
    ap.add_argument("--verbose_failures", action="store_true")
    return ap

# ----------------------------
# Helpers
# ----------------------------

def pick_device(name: Optional[str]) -> torch.device:
    if name in ("cpu", "cuda"):
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(name: str | None):
    if name == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def apply_template(tokenizer, prompt: str, response: str) -> str:
    """Render a prompt+response pair using the tokenizer's chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    eos = getattr(tokenizer, "eos_token", "") or ""
    return f"Human: {prompt}\nAssistant: {response}{eos}"


def collate(tokenizer, texts: List[str], max_length: int):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.inference_mode()
def score_texts_seqclf(model_sc, tokenizer, texts: List[str], device: torch.device, max_length: int) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    enc = collate(tokenizer, texts, max_length).to(device)
    out = model_sc(**enc)
    logits = out.logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        scores = logits[:, 0]
    elif logits.ndim == 2 and logits.shape[1] == 2:
        scores = logits[:, 1] - logits[:, 0]
    else:
        scores = logits[:, 0]
    return scores.detach().float().cpu().numpy()


@torch.inference_mode()
def score_texts_lm_multihead(base_model, tokenizer, texts: List[str], device: torch.device, max_length: int,
                             head_matrix: torch.Tensor, bias: Optional[torch.Tensor] = None) -> np.ndarray:
    """Return (N,B) scores, where head_matrix is (H,B)."""
    if not texts:
        return np.zeros((0, head_matrix.shape[1]), dtype=np.float32)
    enc = collate(tokenizer, texts, max_length).to(device)
    out = base_model(**enc, output_hidden_states=True)
    h_last = out.hidden_states[-1]  # (N,T,H)
    # last non-pad token index per row
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    batch = torch.arange(h_last.size(0), device=device)
    last_tok = h_last[batch, last_idx]  # (N,H)

    W = head_matrix.to(device=device, dtype=last_tok.dtype)  # (H,B)
    scores = last_tok @ W  # (N,B)

    if bias is not None:
        b = bias.to(device=device, dtype=scores.dtype).view(1, -1)
        scores = scores + b
    return scores.detach().float().cpu().numpy()


def evaluate_point(scores_correct: np.ndarray, scores_incorrect: np.ndarray) -> Dict[str, float]:
    """scores_correct: (C,) or (C,B); scores_incorrect: (R,) or (R,B)
    Returns dict with acc_strict, margin, ties_weighted_approx per head (vectorized for B heads).
    """
    # Convert to 2D: (C,B) and (R,B)
    if scores_correct.ndim == 1:
        scores_correct = scores_correct[:, None]
    if scores_incorrect.ndim == 1:
        scores_incorrect = scores_incorrect[:, None]

    # min over correct; max over incorrect
    min_c = np.min(scores_correct, axis=0)  # (B,)
    max_i = np.max(scores_incorrect, axis=0)  # (B,)

    acc_strict = (min_c > max_i).astype(np.float32)  # (B,)
    margin = (min_c - max_i).astype(np.float32)  # (B,)

    # Spread among correct answers
    spread_c = (np.max(scores_correct, axis=0) - np.min(scores_correct, axis=0)).astype(np.float32)
    ties_bonus = (margin > spread_c).astype(np.float32)
    ties_weighted_approx = 0.5 * acc_strict + 0.5 * ties_bonus

    return {
        "acc_strict": acc_strict,
        "margin": margin,
        "ties_weighted_approx": ties_weighted_approx,
    }


# ----------------------------
# Main
# ----------------------------

def main():
    args = build_argparser().parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    tok_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=args.trust_remote_code)
    if args.truncation_side:
        tokenizer.truncation_side = args.truncation_side

    # Infer mode when auto
    mode = args.mode
    if mode == "auto":
        if args.rm_head is not None:
            mode = "lmhead"
        else:
            mode = "seqclf"

    # Load model(s)
    if mode == "seqclf":
        model_sc = AutoModelForSequenceClassification.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
        ).to(device)
        model_sc.eval()
        base_model = None
        head_matrix = None
        bias_vec = None
    else:
        # lmhead
        # Prefer CausalLM (works for most LLMs); fall back to AutoModel if needed
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code, device_map=None
            ).to(device)
        except Exception:
            base_model = AutoModel.from_pretrained(
                args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
            ).to(device)
        base_model.eval()
        model_sc = None

        # Load head matrix
        if args.rm_head is None:
            raise ValueError("--rm_head is required in lmhead mode (provide (H,) or (H,B) head file)")

        head_matrix = load_head_matrix(args.rm_head, args.head_key)
        # Normalize shapes: (H,), (H,1), (H,B) -> (H,B)
        if head_matrix.ndim == 1:
            head_matrix = head_matrix[:, None]
        elif head_matrix.ndim == 2 and head_matrix.shape[1] == 1:
            pass
        elif head_matrix.ndim == 2 and head_matrix.shape[0] >= 1:
            pass
        else:
            raise ValueError(f"Unsupported head_matrix shape: {tuple(head_matrix.shape)}")

        # Optional bias
        bias_vec = None
        if args.head_bias is not None:
            bias_vec = load_bias(args.head_bias, head_matrix.shape[1])

    # Load dataset
    ds = load_dataset("allenai/reward-bench-2", split=args.split)
    if args.subset:
        target = args.subset
        if target in [s.lower() for s in SUBSETS_V2]:
            target = [s for s in SUBSETS_V2 if s.lower() == target][0]
        ds = ds.filter(lambda ex: ex["subset"] == target)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Accumulators
    if mode == "seqclf":
        totals = {k: 0 for k in SUBSETS_V2}
        corrects = {k: 0 for k in SUBSETS_V2}
        ties_weighted_sum = 0.0
        ties_count = 0
        margins_all = []
    else:
        B = head_matrix.shape[1]
        totals = {k: 0 for k in SUBSETS_V2}
        corrects = {k: np.zeros(B, dtype=np.int64) for k in SUBSETS_V2}
        ties_weighted_sum = np.zeros(B, dtype=np.float64)
        ties_count = np.zeros(B, dtype=np.int64)
        margins_all = []  # store medians across heads if you like; here we only keep overall distribution per example using head 0

    # Eval loop
    for ex in tqdm(ds, desc="Evaluating"):
        prompt = ex["prompt"]
        chosen_list = ex["chosen"]  # list[str]
        rejected_list = ex["rejected"]
        subset = ex["subset"]

        texts = []
        for r in chosen_list:
            texts.append(apply_template(tokenizer, prompt, r))
        for r in rejected_list:
            texts.append(apply_template(tokenizer, prompt, r))

        if mode == "seqclf":
            scores = score_texts_seqclf(model_sc, tokenizer, texts, device, args.max_length)  # (N,)
            C = len(chosen_list)
            sc = scores[:C]
            sr = scores[C:]
            m = evaluate_point(sc, sr)

            totals[subset] += 1
            if subset == "Ties":
                corrects[subset] += int(m["acc_strict"][0] == 1.0)
                ties_weighted_sum += float(m["ties_weighted_approx"][0])
                ties_count += 1
            else:
                corrects[subset] += int(m["acc_strict"][0] == 1.0)
            if not (np.isnan(m["margin"][0])):
                margins_all.append(float(m["margin"][0]))

            if args.verbose_failures and m["acc_strict"][0] == 0.0:
                print_failure(subset, prompt, chosen_list, rejected_list, sc, sr)

        else:
            scores_mat = score_texts_lm_multihead(base_model, tokenizer, texts, device, args.max_length,
                                                  torch.as_tensor(head_matrix),
                                                  torch.as_tensor(bias_vec) if bias_vec is not None else None)
            C = len(chosen_list)
            sc = scores_mat[:C, :]  # (C,B)
            sr = scores_mat[C:, :]  # (R,B)
            m = evaluate_point(sc, sr)  # vectorized per-head

            totals[subset] += 1
            corrects[subset] += m["acc_strict"].astype(np.int64)
            if subset == "Ties":
                ties_weighted_sum += m["ties_weighted_approx"].astype(np.float64)
                ties_count += 1

            # record margin of head 0 for rough distribution (optional)
            if not np.isnan(m["margin"][0]):
                margins_all.append(float(m["margin"][0]))

            if args.verbose_failures and np.any(m["acc_strict"] == 0.0):
                # Print only first failing head for brevity
                k = int(np.where(m["acc_strict"] == 0.0)[0][0])
                print_failure(subset + f" (head {k})", prompt, chosen_list, rejected_list, sc[:, k], sr[:, k])

    # Reporting
    print("\n================= RewardBench 2 Results =================")

    if mode == "seqclf":
        per_subset = {}
        for s in SUBSETS_V2:
            if totals[s] == 0:
                per_subset[s] = float("nan")
                continue
            if s == "Ties":
                acc_strict = corrects[s] / totals[s]
                ties_weighted = (ties_weighted_sum / ties_count) if ties_count > 0 else float("nan")
                per_subset[s] = acc_strict
                print(f"{s:12s} | strict acc: {acc_strict*100:6.2f}% | weighted approx: {ties_weighted*100:6.2f}%  (n={totals[s]})")
            else:
                per_subset[s] = corrects[s] / totals[s]
                print(f"{s:12s} | accuracy: {per_subset[s]*100:6.2f}%  (n={totals[s]})")

        simple_mean_5 = np.nanmean([per_subset[s] for s in ["Factuality","Precise IF","Math","Safety","Focus"]])
        ties_strict = per_subset["Ties"]
        ties_weighted = (ties_weighted_sum / ties_count) if ties_count > 0 else float("nan")
        overall_strict = np.nanmean([simple_mean_5, ties_strict])
        overall_weighted = np.nanmean([simple_mean_5, ties_weighted])
        print("---------------------------------------------------------")
        print(f"Overall (5 categories, no ties):       {simple_mean_5*100:6.2f}%")
        print(f"Ties strict accuracy:                  {ties_strict*100:6.2f}%")
        print(f"Overall STRICT (incl. ties):           {overall_strict*100:6.2f}%")
        print(f"Overall WEIGHTED approx (incl. ties):  {overall_weighted*100:6.2f}%")
        if margins_all:
            arr = np.asarray(margins_all, dtype=np.float32)
            print(f"Median margin (minC - maxR): {np.median(arr):+.3f} | 10/90%: {np.percentile(arr,10):+.3f}/{np.percentile(arr,90):+.3f}")

    else:
        B = head_matrix.shape[1]
        acc_by_subset = {}
        for s in SUBSETS_V2:
            if totals[s] == 0:
                continue
            vec = corrects[s] / max(totals[s], 1)
            acc_by_subset[s] = vec  # (B,)
            if s == "Ties":
                weighted = np.divide(ties_weighted_sum, np.maximum(ties_count, 1), out=np.full(B, np.nan, dtype=np.float64), where=ties_count>0)
                print(f"{s:12s} | mean strict: {vec.mean()*100:6.2f}% | best head #{int(vec.argmax())} = {vec.max()*100:6.2f}% | mean weighted: {np.nanmean(weighted)*100:6.2f}%  (n={totals[s]})")
            else:
                print(f"{s:12s} | mean acc:   {vec.mean()*100:6.2f}% | best head #{int(vec.argmax())} = {vec.max()*100:6.2f}%  (n={totals[s]})")

        # 5-category mean (no ties)
        five = np.vstack([acc_by_subset.get(s, np.full(B, np.nan)) for s in ["Factuality","Precise IF","Math","Safety","Focus"]])
        five_mean = np.nanmean(five, axis=0)  # (B,)

        ties_strict = acc_by_subset.get("Ties", np.full(B, np.nan))
        ties_weighted = np.divide(ties_weighted_sum, np.maximum(ties_count, 1), out=np.full(B, np.nan, dtype=np.float64), where=ties_count>0)

        overall_strict = np.nanmean(np.stack([five_mean, ties_strict]), axis=0)    # (B,)
        overall_weighted = np.nanmean(np.stack([five_mean, ties_weighted]), axis=0)  # (B,)

        # Rank heads by overall weighted
        order = np.argsort(-overall_weighted)
        print("---------------------------------------------------------")
        top_k = min(args.top_k, B)
        for rank, k in enumerate(order[:top_k], 1):
            print(f"#{rank:02d} head {k:4d} | overall(weighted) {overall_weighted[k]*100:6.2f}% | "
                  f"5-cat {five_mean[k]*100:6.2f}% | ties strict {ties_strict[k]*100:6.2f}% approx {ties_weighted[k]*100:6.2f}%")

        if args.save_csv:
            try:
                import pandas as pd
                df = {
                    "head": np.arange(B),
                    "overall_weighted": overall_weighted,
                    "overall_strict": overall_strict,
                    "five_mean": five_mean,
                    "ties_strict": ties_strict,
                    "ties_weighted": ties_weighted,
                }
                for s in ["Factuality","Precise IF","Math","Safety","Focus"]:
                    if s in acc_by_subset:
                        df[f"{s}_acc"] = acc_by_subset[s]
                pd.DataFrame(df).to_csv(args.save_csv, index=False)
                print(f"Saved per-head results to {args.save_csv}")
            except Exception as e:
                print(f"[WARN] Failed to save CSV: {e}")

        if margins_all:
            arr = np.asarray(margins_all, dtype=np.float32)
            print(f"Median margin (head#0) (minC - maxR): {np.median(arr):+.3f} | 10/90%: {np.percentile(arr,10):+.3f}/{np.percentile(arr,90):+.3f}")


# ----------------------------
# IO helpers for heads / bias
# ----------------------------

def load_head_matrix(path: str, key: Optional[str]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".pt", ".bin", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if key is None:
                # pick the first tensor with suitable rank
                tens = None
                for k, v in obj.items():
                    if isinstance(v, torch.Tensor) and v.ndim in (1,2):
                        tens = v
                        break
                if tens is None:
                    raise ValueError("Checkpoint dict did not contain a 1D/2D tensor; pass --head_key")
                arr = tens.detach().cpu().numpy()
            else:
                v = obj[key]
                if not isinstance(v, torch.Tensor):
                    raise ValueError(f"--head_key '{key}' is not a tensor in the checkpoint")
                arr = v.detach().cpu().numpy()
        elif isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        else:
            raise ValueError("Unsupported object in torch checkpoint for --rm_head")
    else:
        raise ValueError("--rm_head must be .npy or .pt/.pth/.bin")

    if arr.ndim not in (1,2):
        raise ValueError(f"Head array must be 1D or 2D, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_bias(spec: str, B: int) -> torch.Tensor:
    # scalar string?
    try:
        val = float(spec)
        return torch.full((B,), float(val), dtype=torch.float32)
    except ValueError:
        pass

    ext = os.path.splitext(spec)[1].lower()
    if ext == ".npy":
        arr = np.load(spec)
        t = torch.as_tensor(arr, dtype=torch.float32)
    elif ext in (".pt", ".bin", ".pth"):
        obj = torch.load(spec, map_location="cpu")
        if isinstance(obj, dict):
            # pick the first 0D/1D tensor
            v = None
            for k, val in obj.items():
                if isinstance(val, torch.Tensor) and val.ndim in (0,1):
                    v = val
                    break
            if v is None:
                raise ValueError("Bias checkpoint dict had no 0D/1D tensor")
            t = v.detach().cpu().to(torch.float32)
        elif isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().to(torch.float32)
        else:
            raise ValueError("Unsupported object in torch checkpoint for --head_bias")
    else:
        raise ValueError("--head_bias must be scalar, .npy, or .pt/.pth/.bin")

    if t.ndim == 0:
        t = t.view(1).repeat(B)
    elif t.ndim == 1 and t.numel() == 1:
        t = t.repeat(B)
    elif t.ndim == 1 and t.numel() == B:
        pass
    else:
        raise ValueError(f"Bias must be scalar or length-B vector; got shape {tuple(t.shape)} for B={B}")
    return t


# ----------------------------
# Pretty failure printing
# ----------------------------

def print_failure(subset: str, prompt: str, chosen: List[str], rejected: List[str], sc, sr):
    print("\n--- FAILURE ---")
    print(f"Subset: {subset}")
    print(f"Prompt: {prompt[:300]}...")
    for j, r in enumerate(chosen):
        s = float(sc[j]) if np.ndim(sc) == 1 else float(sc[j])
        print(f"[C{j}] {s:+.3f} :: {r[:200]}...")
    for j, r in enumerate(rejected):
        s = float(sr[j]) if np.ndim(sr) == 1 else float(sr[j])
        print(f"[R{j}] {s:+.3f} :: {r[:200]}...")


if __name__ == "__main__":
    main()