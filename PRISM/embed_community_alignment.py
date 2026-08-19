"""
Embed Community Alignment (prompt, response) texts -- batched and de-duplicated.

Why a new embedder rather than reusing generate-prism-embeddings.py: that one runs at batch size 1,
which is ~3.5 it/s. Community Alignment needs tens of thousands of embeddings, so batch-1 would take
many hours. Two optimisations here:

  1. DE-DUPLICATION. Each turn shows 4 responses and yields 3 pairs (preferred vs each other), so the
     preferred response appears in 3 pairs and every response text recurs across pairs. We embed each
     unique (prompt, response) exactly once -- roughly a 2x saving over embedding both sides of every
     pair independently.
  2. BATCHING with padding + sdpa attention, taking the last NON-PAD token per row. This is the same
     representation as the PRISM pipeline (last-token hidden state), just computed many at a time.

Output: a dict {"keys": [str, ...], "emb": FloatTensor[N, 4096]} where keys are the same
(prompt, response) hashes used by community_alignment_lore.py to look embeddings up.

Usage:
  python embed_community_alignment.py --pairs data/community_alignment/pairs_min100.json --limit 64
  python embed_community_alignment.py --pairs data/community_alignment/pairs_min100.json
"""
import argparse
import hashlib
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def text_key(prompt, response):
    """Stable id for a (prompt, response) pair so embeddings can be de-duplicated and looked up."""
    h = hashlib.sha1()
    h.update(prompt.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(response.encode("utf-8", "ignore"))
    return h.hexdigest()


def render(tokenizer, prompt, response):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False)


@torch.inference_mode()
def embed_texts(model, tokenizer, texts, device, batch_size, max_length):
    """[N, H] last non-pad hidden states, computed in batches."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        hs = model(**enc).last_hidden_state                     # [b, T, H]
        # last NON-PAD token per row (tokenizer pads on the right by default here)
        idx = enc["attention_mask"].sum(dim=1) - 1
        out.append(hs[torch.arange(hs.size(0), device=device), idx].float().cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs JSON from community_alignment_prep.py")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "data", "community_alignment",
                                                  "embeddings.pt"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--resume", default=None,
                    help="existing embeddings .pt to reuse; only new texts are embedded")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N unique texts (smoke test)")
    args = ap.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"{len(pairs)} pairs from {len({p['user_id'] for p in pairs})} users")

    # Collect unique (prompt, response) texts across both sides of every pair.
    uniq = {}
    for p in pairs:
        for side in ("chosen", "rejected"):
            k = text_key(p["prompt"], p[side])
            if k not in uniq:
                uniq[k] = (p["prompt"], p[side])
    keys = list(uniq.keys())
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} unique (prompt, response) texts "
          f"({2*len(pairs)/max(len(keys),1):.2f}x saved by de-duplication)")

    # Resume: reuse embeddings already computed for the same texts. Keys are content hashes, so a
    # previous smaller run (e.g. a lower --max_pairs_per_user) covers a subset of this one and only
    # the new texts need the GPU.
    have_keys, have_emb = [], None
    if args.resume and os.path.exists(args.resume):
        prev = torch.load(args.resume, weights_only=False)
        prev_idx = {k: i for i, k in enumerate(prev["keys"])}
        reuse = [k for k in keys if k in prev_idx]
        if reuse:
            have_keys = reuse
            have_emb = prev["emb"][torch.tensor([prev_idx[k] for k in reuse])]
        keys = [k for k in keys if k not in prev_idx]
        print(f"  resuming from {args.resume}: reusing {len(have_keys)}, embedding {len(keys)} new")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Loading {MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:          # Llama tokenizers often ship without one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        device_map=("auto" if torch.cuda.is_available() else None),
        attn_implementation="sdpa",
    ).eval()
    if not torch.cuda.is_available():
        model = model.to(device)

    if keys:
        texts = [render(tokenizer, *uniq[k]) for k in keys]
        emb = embed_texts(model, tokenizer, texts, device, args.batch_size, args.max_length)
    else:
        emb = torch.empty(0, 4096)
    if have_emb is not None:                      # merge reused + newly embedded
        keys = have_keys + keys
        emb = torch.cat([have_emb, emb], 0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"keys": keys, "emb": emb}, args.out)
    print(f"\nSaved {args.out}  emb {tuple(emb.shape)}")


if __name__ == "__main__":
    main()
