# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = load_dataset("namkoong-lab/PersonalLLM")
data = pd.DataFrame(dataset["train"])
data_test = pd.DataFrame(dataset["test"])

# Number of prompts
num_prompts = len(data)
dataset = []
# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    # Iterate over each response (1 to 8)
    prompt_responses = []
    for i in range(1,9):
        # Iterate over each model name
        column_name = f"response_{i}"
        conv = [{"role": "user", "content": row['prompt']}, {"role": "assistant", "content": row[column_name]}]
        prompt_responses.append(conv)
    dataset.append(prompt_responses)

import torch

from transformers import AutoModel, AutoTokenizer

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

# Initialize lists to store embeddings and corresponding labels
embeddings = []
rewards = []

# Iterate over each example in the dataset with a progress bar
for example in dataset:
    for i in range(len(example)):
        # print(example[i])
        conv_tokenized = rm_tokenizer.apply_chat_template(example[i], tokenize=True, return_tensors="pt").to(device)

        with torch.no_grad():
            output = rm(conv_tokenized)  # Forward pass through the model
            # print(output)
            # Extract the last hidden state of the last token and move it to CPU
            # rewards.append(output.logits[0][0].item())
            embeddings.append(output.last_hidden_state[0][-1].cpu())
            # print(rewards[-1])
            # print(embeddings[-1])

embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor
# rewards = torch.stack(rewards, dim=0)  # Stack all embeddings into a single tensor

from safetensors.torch import save_file
import os

save_path = os.path.join(
    "embeddings", "/train"
)  # Construct the save directory path
print(save_path)
# Save the embeddings and labels in a safetensors file with shard indexing
save_file(
    {"embeddings": embeddings},
    f"./{save_path}.safetensors",
)