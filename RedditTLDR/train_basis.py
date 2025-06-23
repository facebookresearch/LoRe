# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *
import random
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

with open('tldr_embeddings_train.pkl', 'rb') as f:
    worker_results_train = pickle.load(f)

with open('tldr_embeddings_val.pkl', 'rb') as f:
    worker_results_test = pickle.load(f)

# Sort the worker_results dictionary by the number of entries
all_worker_results_train = dict(sorted(worker_results_train.items(), key=lambda item: len(item[1]), reverse=False)[2:])
all_worker_results_test = dict(sorted(worker_results_test.items(), key=lambda item: len(item[1]), reverse=False)[2:])

# Initialize an empty dictionary to store the merged results
random.seed(0)
merged_worker_results = {}
train_workers = set(worker_results_train.keys()) 
test_workers = set(worker_results_test.keys())
common_workers = list(train_workers & test_workers)
random.shuffle(common_workers)
split = int(len(common_workers) / 2)
print(split)
train_workers = common_workers[:split]
test_workers = common_workers[split:]

import random
def random_sample(N, T):
    """
    Randomly sample T numbers between 0 and N - 1 without replacement.
    Args:
        N (int): The upper bound of the range.
        T (int): The number of samples.
    Returns:
        list: A list of T unique random integers between 0 and N - 1.
    """
    all_numbers = set(range(N))
    # samples = random.sample(all_numbers, T)
    samples = random.sample(list(all_numbers), T)
    remaining = list(all_numbers - set(samples))
    return samples, remaining

train_features = []
test_features_sparse = []
train_features_unseen = []
test_features_sparse_unseen = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = 0
# T = 50
N_unseen = 0
# T_unseen = 50
stats = []
for worker_id, data in all_worker_results_train.items():
    if worker_id in train_workers:
        # print(len(data))
        temp = []
        # T = len(data)
        T = min(len(data), 150)
        idx, idx2 = random_sample(len(data), T)
        # idx, idx2 = random_sample(len(data), T)
        for i in idx:
            x = data[i]['embeddings']['winning'][0] - data[i]['embeddings']['losing'][0]
            temp.append(torch.tensor(x, dtype=torch.float32).to(device))
        train_features.append(torch.stack(temp))
        temp2 = []
        for test_data in all_worker_results_test[worker_id]:
            x = test_data['embeddings']['winning'][0] - test_data['embeddings']['losing'][0]
            temp2.append(torch.tensor(x, dtype=torch.float32).to(device))
        stats.append(len(temp2))
        test_features_sparse.append(torch.stack(temp2))
        N += 1
    elif worker_id in test_workers:
        # for worker_id, data in all_worker_results_test.items():
        temp = []
        T_unseen = min(len(data), 50)
        idx, idx2 = random_sample(len(data), T_unseen)
        for i in idx:
            x = data[i]['embeddings']['winning'][0] - data[i]['embeddings']['losing'][0]
            temp.append(torch.tensor(x, dtype=torch.float32).to(device))
        train_features_unseen.append(torch.stack(temp))
        temp2 = []
        for test_data in all_worker_results_test[worker_id]:
            x = test_data['embeddings']['winning'][0] - test_data['embeddings']['losing'][0]
            temp2.append(torch.tensor(x, dtype=torch.float32).to(device))
        stats.append(len(temp2))
        test_features_sparse_unseen.append(torch.stack(temp2))             
        N_unseen += 1 


print(np.mean(stats))
print(np.std(stats))

K_list = [0, 1, 2, 3, 4, 5, 6]
alpha_list = [0]

train_accuracies_joint, seen_user_unseen_prompts_accuracies_joint, few_shot_train_accuracies_few_shot, unseen_user_unseen_prompts_accuracies_few_shot, train_accuracies_joint_std, seen_user_unseen_prompts_accuracies_joint_std, few_shot_train_accuracies_few_shot_std, unseen_user_unseen_prompts_accuracies_few_shot_std = run(K_list, alpha_list, V_final, train_features, test_features_sparse, 
                    train_features_unseen, test_features_sparse_unseen, N, N_unseen, device)
