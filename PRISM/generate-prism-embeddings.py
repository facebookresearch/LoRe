#!/usr/bin/env python3
#
# Source: https://github.com/facebookresearch/LoRe/blob/main/PRISM/prepare.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import torch
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# Named function to replace lambda for pickle compatibility
def nested_defaultdict():
    return defaultdict(list)

# Optional: convert nested defaultdicts to regular dicts for clean pickling
def recursive_dict(d):
    if isinstance(d, defaultdict):
        return {k: recursive_dict(v) for k, v in d.items()}
    return d

def generate_prism_embeddings(
    dataset,
    model,
    tokenizer,
    device,
    output_path
):
    """
    Generate embeddings for each user in the dataset.
    Structure: chosen_embeddings[user_id][dialog_id] = [embedding_turn_0, ..., embedding_turn_n]

    Alternate:

    embeddings[user_id][dialog_id][turn_nb][chosen/rejected][seen : True or False][train : True or False]

    Later for given user_id (and specifiec chosen/rejected value, seen True or False value) gather all chosen embeddings as a tensor
    """
    embeddings_data = []
    for entry in tqdm(dataset, desc="Generating embeddings"):
        
        user_id = entry["extra_info"]["user_id"]
        dialog_id = entry["extra_info"]["dialog_id"]
        prompt = entry["prompt"]

        chosen_utt = entry["extra_info"].get("chosen_utterance")
        rejected_utt = entry["extra_info"].get("rejected_utterance")
        # Defensive: older prepared data stored rejected as a LIST of strings. Feeding a list as
        # message content makes the chat template render its Python repr ("['a', 'b']"), a
        # formatting artifact that separates chosen/rejected trivially and collapses the LoRe
        # bases. Always embed a single string so both sides are formatted identically.
        if isinstance(rejected_utt, list):
            rejected_utt = rejected_utt[0] if rejected_utt else None
        if isinstance(chosen_utt, list):
            chosen_utt = chosen_utt[0] if chosen_utt else None
        # Skip entries that are not a valid (chosen, rejected) pair: last-turn prompt-only entries
        # (keys absent -> None after the parquet round-trip) and turns whose rejected list was empty.
        if not chosen_utt or not rejected_utt:
            continue
        chosen = [{"content": chosen_utt, "role": "assistant"}]
        rejected = [{"content": rejected_utt, "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected
        
        # Tokenize the current dialog state
        tokenized = tokenizer.apply_chat_template(
            chosen_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]

        entry["extra_info"]["chosen_conv_embedding"] = embedding

        # Tokenize the current dialog state
        tokenized = tokenizer.apply_chat_template(
            rejected_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]

        entry["extra_info"]["rejected_conv_embedding"] = embedding

        embeddings_data.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_data, output_path)
    print(f"✅ Saved embeddings to {output_path}")

    return embeddings_data


if __name__ == "__main__":
    # --- Configuration ---
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    # --- Load model and tokenizer ---
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load datasets ---
    print("📦 Loading datasets...")
    train_dataset = load_dataset("parquet", data_files="data/prism/train.parquet")["train"]
    test_dataset = load_dataset("parquet", data_files="data/prism/test.parquet")["train"]

    # Optional smoke-test limit: EMBED_LIMIT=8 embeds only the first 8 of each split so the
    # full pipeline (prepare -> embed -> save) can be validated in ~seconds instead of ~2 hours.
    embed_limit = os.environ.get("EMBED_LIMIT")
    if embed_limit:
        n = int(embed_limit)
        print(f"[EMBED_LIMIT] embedding only the first {n} examples of each split (smoke test)")
        train_dataset = train_dataset.select(range(min(n, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(n, len(test_dataset))))

    # # --- Generate embeddings ---
    train_embeddings = generate_prism_embeddings(train_dataset, model, tokenizer, device, "data/prism/train_embeddings.pkl")
    test_embeddings = generate_prism_embeddings(test_dataset, model, tokenizer, device, "data/prism/test_embeddings.pkl")
