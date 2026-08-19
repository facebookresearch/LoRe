"""
Embed the contrastive pairs PER PAIR (not just their mean) so synthetic users can be built from them.

compute_concept_vectors.py embeds the same pairs but immediately averages them into one concept
vector per concept, discarding the per-pair embeddings. synthetic_concept_users.py needs the
individual high/low embeddings to form per-pair (chosen - rejected) diffs, so this script keeps them.

Representation is IDENTICAL to compute_concept_vectors.py and to the PRISM pipeline: render
prompt + response with the chat template and take the last-token hidden state of the Skywork
backbone. Keeping this identical is the point -- we are testing whether *that* space encodes the
concept axes linearly.

Output: data/prism/contrastive_pair_embeddings.pt
  {concept: {"high": FloatTensor[N, 4096], "low": FloatTensor[N, 4096]}}

Usage:
  python embed_contrastive_pairs.py --limit 2      # CPU smoke test (shapes only, seconds)
  python embed_contrastive_pairs.py                # full run (GPU; ~1100 forward passes)
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


@torch.no_grad()
def embed_one(model, tokenizer, prompt, response, device):
    """Last-token hidden state of (prompt, response), exactly as compute_concept_vectors.py does."""
    conv = [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]
    toks = tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
    out = model(toks)
    return out.last_hidden_state[0, -1].cpu().to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                    "contrastive_pairs.json"))
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                  "contrastive_pair_embeddings.pt"))
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only this many pairs per concept (smoke test)")
    args = ap.parse_args()

    with open(args.pairs) as f:
        pairs_by_concept = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Loading {MODEL}...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        device_map=("auto" if torch.cuda.is_available() else None),
        attn_implementation="sdpa",   # memory-efficient; eager OOMs on long inputs
    ).eval()
    if not torch.cuda.is_available():
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    out = {}
    for concept, pairs in pairs_by_concept.items():
        if args.limit:
            pairs = pairs[:args.limit]
        if not pairs:
            continue
        highs, lows = [], []
        for p in tqdm(pairs, desc=concept):
            # Guard against the PRISM string-vs-list artifact: both sides must be plain strings,
            # formatted identically. That bug is exactly what made LoRe's PRISM numbers meaningless.
            assert isinstance(p["high_response"], str) and isinstance(p["low_response"], str), \
                f"{concept}: responses must be plain strings, got " \
                f"{type(p['high_response']).__name__}/{type(p['low_response']).__name__}"
            highs.append(embed_one(model, tokenizer, p["prompt"], p["high_response"], device))
            lows.append(embed_one(model, tokenizer, p["prompt"], p["low_response"], device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out[concept] = {"high": torch.stack(highs), "low": torch.stack(lows)}
        print(f"  {concept}: high {tuple(out[concept]['high'].shape)} "
              f"low {tuple(out[concept]['low'].shape)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f"\nSaved {args.out}  ({len(out)} concepts)")


if __name__ == "__main__":
    main()
