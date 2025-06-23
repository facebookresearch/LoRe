# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from safetensors.torch import load_file
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch as torch
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
import torchinfo

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

# Initialize a variable to store the last linear layer
last_linear_layer = None
# Iterate over the model's modules
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module
# Print the weights and bias of the last linear layer
V_final = last_linear_layer.weight[:,0].to(device).to(torch.float32).reshape(-1, 1)

# Load the dataset
dataset = load_dataset("namkoong-lab/PersonalLLM")
data = pd.DataFrame(dataset["train"])
data_test = pd.DataFrame(dataset["test"])

# Load the embeddings

embeddings = load_file(f"train.safetensors")
embeddings = embeddings['embeddings']

features = []
# Number of prompts
num_prompts = len(data)
for i in range(num_prompts):
    temp = []
    for j in range(8):
        temp.append(embeddings[i * 8 + j])
    features.append(temp)

embeddings = load_file(f"test.safetensors")
embeddings = embeddings['embeddings']

test_features = []
# Number of prompts
num_prompts = len(data_test)
for i in range(num_prompts):
    temp = []
    for j in range(8):
        temp.append(embeddings[i * 8 + j])
    test_features.append(temp)

# Generate features with the off the shelf reward models
# Define the list of model names corresponding to the columns
model_names = [
    "gemma_2b",
    "gemma_7b",
    "mistral_raft",
    "llama3_sfairx",
    "oasst_deberta_v3",
    "beaver_7b",
    "oasst_pythia_7b",
    "oasst_pythia_1b",
    "mistral_ray",
    "mistral_weqweasdas",
]

# Initialize a 3D NumPy array to store the reward tensors for each prompt
# Number of prompts
num_prompts = len(data)
reward_tensor = np.empty((num_prompts, 8, len(model_names)), dtype=object)
# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    # Create an 8x10 array for the current prompt
    prompt_array = np.empty((8, len(model_names)), dtype=object)

    # Iterate over each response (1 to 8)
    for i in range(1, 9):
        # Iterate over each model name
        for j, model_name in enumerate(model_names):
            # Construct the column name for the current response and model
            column_name = f"response_{i}_{model_name}"

            # Assign the value to the appropriate position in the prompt array
            if column_name in row:
                prompt_array[i - 1, j] = row[column_name]
            else:
                prompt_array[i - 1, j] = None  # or handle missing columns as needed

    # Assign the prompt array to the reward tensor
    reward_tensor[index] = prompt_array
# Now, reward_tensor is a 3D NumPy array with shape (num_prompts, 8, 10)

# Initialize a 3D NumPy array to store the reward tensors for each prompt
num_prompts = len(data_test)
reward_tensor_test = np.empty((num_prompts, 8, len(model_names)), dtype=object)
# Iterate over each row in the DataFrame
for index, row in data_test.iterrows():
    # Create an 8x10 array for the current prompt
    prompt_array = np.empty((8, len(model_names)), dtype=object)

    # Iterate over each response (1 to 8)
    for i in range(1, 9):
        # Iterate over each model name
        for j, model_name in enumerate(model_names):
            # Construct the column name for the current response and model
            column_name = f"response_{i}_{model_name}"

            # Assign the value to the appropriate position in the prompt array
            if column_name in row:
                prompt_array[i - 1, j] = row[column_name]
            else:
                prompt_array[i - 1, j] = None  # or handle missing columns as needed

    # Assign the prompt array to the reward tensor
    reward_tensor_test[index] = prompt_array
# Now, reward_tensor is a 3D NumPy array with shape (num_prompts, 8, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare Train Dataset
N = 1000
alpha_val = 0.1
alpha = alpha_val * np.ones(len(model_names))
W = generate_popupulation(alpha, N)
# Users at train time
all_feature_diff = simulate_population(reward_tensor, features, W)
train_features = create_sparse_tensor(all_feature_diff, 0.005)
# Seen users, unseen prompts
all_feature_diff_test = simulate_population(reward_tensor_test, test_features, W)
test_features_sparse = create_sparse_tensor(all_feature_diff_test, 1.0)
# Unseen Users
N_unseen = 500
W_unseen = generate_popupulation(alpha, N_unseen)
# Unseen Users, Few Shot prompts
all_feature_diff_unseen = simulate_population(reward_tensor, features, W_unseen)
train_features_unseen = create_sparse_tensor(all_feature_diff_unseen, 0.001)
# Unseen Users, unseen prompts
all_feature_diff_test_unseen = simulate_population(reward_tensor_test, test_features, W_unseen)
test_features_sparse_unseen = create_sparse_tensor(all_feature_diff_test_unseen, 1.0)


K_list = [0. 1, 2, 3, 4, 5]
# K_list = [0]
        #   15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
alpha_list = [0]

train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std = run(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                       train_features_unseen, test_features_sparse_unseen, N, N_unseen, device)


