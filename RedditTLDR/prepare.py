# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
import pandas as pd
import numpy as np
import gc

# Load the dataset
dataset = load_dataset("openai/summarize_from_feedback", 'comparisons')
# Assuming df is your DataFrame
df = pd.DataFrame(dataset['train'])
# df = pd.DataFrame(dataset['validation'])

# Create an empty dictionary to store the results
worker_results = {}
# Iterate over each row in the dataset
for index, row in df.iterrows():
    # Get the worker ID
    worker_id = row['worker']
    
    # Get the text, winning summary, and losing summary
    text = row['info']['post']
    summaries = row['summaries']
    winning_summary = summaries[row['choice']]['text']
    losing_summary = summaries[1 - row['choice']]['text']
    
    # Add the result to the worker's list
    if worker_id not in worker_results:
        worker_results[worker_id] = []
    worker_results[worker_id].append({
        'text': text,
        'winning_summary': winning_summary,
        'losing_summary': losing_summary
    })

# Sort the worker_results dictionary by the number of entries
worker_results = dict(sorted(worker_results.items(), key=lambda item: len(item[1]), reverse=False))
# for i in range(20, len(all_worker_results), 1):
# worker_results = dict(list(all_worker_results.items())[1:2])

from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}
for worker_id, data in worker_results.items():
    # print(worker_id)
    results[worker_id] = []
    # Print the results
    correct_predictions = 0
    total_predictions = 0
    print(f"Worker ID: {worker_id}")
    for entry in data:
        conv_winning = [{"role": "user", "content": entry['text']}, {"role": "assistant", "content": entry['winning_summary']}]
        conv_losing = [{"role": "user", "content": entry['text']}, {"role": "assistant", "content": entry['losing_summary']}]
        inputs_winning = rm_tokenizer.apply_chat_template(conv_winning, return_tensors="pt").to(device)
        inputs_losing = rm_tokenizer.apply_chat_template(conv_losing, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding_winning = rm(inputs_winning).last_hidden_state[0][-1].cpu()
            embedding_losing = rm(inputs_losing).last_hidden_state[0][-1].cpu()

        results[worker_id].append({
            'text': entry['text'],
            'winning_summary': entry['winning_summary'],
            'losing_summary': entry['losing_summary'],
            'embeddings': {
                'winning': [embedding_winning],
                'losing': [embedding_losing]
            }
        })

import pickle
# with open('tldr_embeddings_val.pkl', 'wb') as f:
with open('tldr_embeddings_train.pkl', 'wb') as f:
    pickle.dump(results, f)
    