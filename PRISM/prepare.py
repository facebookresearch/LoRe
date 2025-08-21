#!/usr/bin/env python3
#
# Source: https://github.com/facebookresearch/LoRe/blob/main/PRISM/prepare.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import pprint
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
import requests
import random

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=1)

# Create data/prism directory if it doesn't exist
import os
import time
os.makedirs("data/prism", exist_ok=True)

def is_valid_jsonl(filepath):
    """Check if a file is valid JSONL (not HTML rate limit page)"""
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            # Check if it's HTML (rate limit page) or empty
            if first_line.startswith('<!DOCTYPE html>') or first_line.startswith('<html') or not first_line:
                return False
            # Try to parse as JSON to ensure it's valid JSONL
            import json
            json.loads(first_line)
            return True
    except (json.JSONDecodeError, Exception):
        return False

def download_file_with_retry(url, filepath, max_retries=10, delay=30):
    """Download a file with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filepath} (attempt {attempt + 1})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Write to file
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Validate the downloaded file
            if is_valid_jsonl(filepath):
                print(f"Successfully downloaded and validated {filepath}")
                return True
            else:
                print(f"Downloaded file {filepath} appears to be rate limited or invalid")
                if attempt < max_retries - 1:
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Failed to download valid file after {max_retries} attempts")
                    return False
                    
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"Failed to download after {max_retries} attempts")
                return False
    
    return False

# Download files with retry logic
files_to_download = [
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/survey.jsonl", "data/prism/survey.jsonl"),
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/conversations.jsonl", "data/prism/conversations.jsonl"),
    ("https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/utterances.jsonl", "data/prism/utterances.jsonl")
]

for url, filepath in files_to_download:
    if not is_valid_jsonl(filepath):
        print(f"File {filepath} doesn't exist or is invalid, downloading...")
        success = download_file_with_retry(url, filepath)
        if not success:
            print(f"CRITICAL: Failed to download {filepath}. Exiting.")
            sys.exit(1)
    else:
        print(f"File {filepath} already exists and is valid.")

# should correspond to parameters in training/evaluation
seed=123

import os
import json
import numpy as np
np.random.seed(seed=seed)

from pydantic import BaseModel
from typing import List, Optional, Dict

class Demographics(BaseModel):
    self_description: str
    preference: List[str] = []
    age: str
    gender: str
    education: str
    employment: str
    marital: str
    english_proficiency: str

class UserInfo(BaseModel):
    user_id: str
    dialog_ids: List[str] = []
    demographics: Demographics
    system_string: str

class DataUser(BaseModel):
    data: Dict[str, UserInfo] = {}

class Turn(BaseModel):
    turn_nb: int
    user_utterance: List[str] = []
    chosen_utterance: List[str] = []
    rejected_utterance: List[str] = []

class DialogInfo(BaseModel):
    dialog_id: str
    user_id: str
    turns: List[Optional[Turn]] = []
    total_turn_nb: int
    open_feedback: str = ""

class DataDialog(BaseModel):
    data: Dict[str, DialogInfo] = {}

# reorganize user related data, skip num_completed_conversations==0
data_user = DataUser()

with open ("data/prism/survey.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        if d["num_completed_conversations"] == 0:
            continue
        data_user.data[d["user_id"]] = UserInfo(
            user_id = d["user_id"],
            demographics =  Demographics(
                self_description = d["self_description"],
                preference = [k for k, v in d["order_stated_prefs"].items() if v in [1,2,3]],
                age = d["age"],
                gender = d["gender"],
                education = d["education"],
                employment = d["employment_status"],
                marital = d["marital_status"],
                english_proficiency = d["english_proficiency"]
            ),
            system_string = d["system_string"]
        )
# reorganize dialog related data
data_dialog = DataDialog()

with open ("data/prism/conversations.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        data_user.data[d["user_id"]].dialog_ids.append(d["conversation_id"])
        data_dialog.data[d["conversation_id"]] = DialogInfo(
            dialog_id = d["conversation_id"],
            user_id = d["user_id"],
            total_turn_nb = d["conversation_turns"],
            turns = [None for _ in range(d["conversation_turns"])],
            open_feedback = d["open_feedback"]
        )
        for utterance in d["conversation_history"]:
            # first utterance of a turn
            if data_dialog.data[d["conversation_id"]].turns[utterance["turn"]] is None:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]] = Turn(
                    turn_nb = utterance["turn"]
                )
            # identify role
            if utterance["role"] == "user":
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].user_utterance.append(utterance["content"])
            elif utterance["if_chosen"]:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].chosen_utterance.append(utterance["content"])
            else:
                data_dialog.data[d["conversation_id"]].turns[utterance["turn"]].rejected_utterance.append(utterance["content"])

# convert to dict
data_dialog = data_dialog.dict()["data"]
data_user = data_user.dict()["data"]

# filter out users with no qualified example
dialog_ids = list(data_dialog.keys())
for dialog_id in dialog_ids:
    qualified_num = data_dialog[dialog_id]["total_turn_nb"]
    for turn in data_dialog[dialog_id]["turns"]:
        if (turn['user_utterance'] == [] 
            or turn['chosen_utterance'] == [] 
            or turn['rejected_utterance'] == [] 
        ):
            qualified_num -= 1
    # only delete when the whole dialogue is not qualifed
    if qualified_num == 0:
        print("delete dialogue", dialog_id, "by", data_dialog[dialog_id]["user_id"])
        data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"].remove(dialog_id)
        if data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"] == []:
            print("delete user", data_dialog[dialog_id]["user_id"])
            del data_user[data_dialog[dialog_id]["user_id"]]
        del data_dialog[dialog_id]

# save as json
with open ("data/prism/prism_data_user.json", 'w') as f:
    json.dump(data_user, f, indent=4)

with open ("data/prism/prism_data_dialog.json", 'w') as f:
    json.dump(data_dialog, f, indent=4)

# split users
import numpy as np
np.random.seed(seed=seed)

user_ids = np.array(list(data_user.keys()))
np.random.shuffle(user_ids)

stats = []
for user_id in user_ids:
    stats.append(len(np.array(data_user[user_id]["dialog_ids"])))

print("Avg no. of dialogs: ", np.mean(stats))
print("Std of dialogs: ", np.std(stats))
print("Max no. of dialogs: ", np.max(stats))
print("Min no. of dialogs: ", np.min(stats))


# seen_user_ids_init = user_ids[:int(len(user_ids)*0.9)]
# unseen_user_ids_init = user_ids[int(len(user_ids)*0.9):]
seen_user_ids_init = user_ids[:int(len(user_ids)*0.8)]
unseen_user_ids_init = user_ids[int(len(user_ids)*0.8):]

# split seen users' dialogs into train/test; add unseen to test
train_dialog_ids = np.array([])
test_dialog_ids = np.array([])

seen_user_ids = []
unseen_user_ids = []

for user_id in seen_user_ids_init:
    to_choose_from = np.array(data_user[user_id]["dialog_ids"])
    if len(to_choose_from) > 5:
        seen_user_ids.append(user_id)
        np.random.shuffle(to_choose_from)
        train_dialog_ids = np.concatenate((train_dialog_ids, to_choose_from[:int(len(to_choose_from)*0.5)]))
        test_dialog_ids = np.concatenate((test_dialog_ids, to_choose_from[int(len(to_choose_from)*0.5):]))

for user_id in unseen_user_ids_init:
    to_choose_from = np.array(data_user[user_id]["dialog_ids"])
    if len(to_choose_from) > 5:
        unseen_user_ids.append(user_id)
        np.random.shuffle(to_choose_from)
        train_dialog_ids = np.concatenate((train_dialog_ids, to_choose_from[:int(len(to_choose_from)*0.5)]))
        test_dialog_ids = np.concatenate((test_dialog_ids, to_choose_from[int(len(to_choose_from)*0.5):]))

print(len(seen_user_ids))
print(len(unseen_user_ids))

split_ids = {"train_dialog_ids": list(train_dialog_ids),
             "test_dialog_ids": list(test_dialog_ids),
             "seen_user_ids": {k:i+1 for i, k in enumerate(seen_user_ids)},
             "unseen_user_ids": {k: 0 for k in unseen_user_ids}
            }

# Save split_ids
with open("data/prism/prism_split_ids_50.json", 'w') as f:
    json.dump(split_ids, f, indent=4)

def load_prism_comparisons(
    prism_data_path=None,
):
    with open("data/prism/prism_data_dialog.json", 'r') as f:
        data_dialog = json.load(f)
    with open("data/prism/prism_data_user.json", 'r') as f:
        data_user = json.load(f)
    with open("data/prism/prism_split_ids_50.json", 'r') as f:
        split_ids = json.load(f)

    n_users = len(data_user)

    def preprocess_function(is_train):
        data = []

        if is_train:
            dialog_ids = split_ids["train_dialog_ids"]
        else:
            dialog_ids = split_ids["test_dialog_ids"]

        for dialog_id in dialog_ids:
            user_id = data_dialog[dialog_id]["user_id"]
            full_dialog = []
            for turn in data_dialog[dialog_id]["turns"]:
                assert len(turn['user_utterance']) == 1
                full_dialog.append({
                    "role": "user",
                    "content": turn["user_utterance"][0]
                })

                entry = {
                    'data_source': 'prism',
                    'prompt': copy.deepcopy(full_dialog),
                    'ability': 'alignment',
                    'reward_model': {
                        'style': 'model',
                        'ground_truth': '', # not used
                    },
                    'extra_info': {
                        'split': 'train' if is_train else 'test',
                        'seen': user_id in split_ids["seen_user_ids"],
                        'user_id': user_id,
                        'dialog_id': dialog_id,
                        'turn_nb': turn['turn_nb'],
                        'total_turn_nb': data_dialog[dialog_id]["total_turn_nb"],
                    }
                }

                if turn['turn_nb'] < data_dialog[dialog_id]["total_turn_nb"]:   
                    assert len(turn['chosen_utterance']) > 0

                    if len(turn['chosen_utterance']) > 1:
                        assert all([
                            x == turn['chosen_utterance'][0]
                            for x in turn['chosen_utterance']])

                    # the single resporse string
                    chosen_utterance = turn["chosen_utterance"][0]
                    entry['extra_info']['chosen_utterance'] = chosen_utterance

                    # all rejected response strings
                    rejected_utterance = turn["rejected_utterance"]
                    entry['extra_info']['rejected_utterance'] = rejected_utterance

                data.append(entry)

                if turn['turn_nb'] < data_dialog[dialog_id]["total_turn_nb"]:   
                    full_dialog.append({
                        "role": "assistant",
                        "content": chosen_utterance,
                    })

        return data

    train_dataset = preprocess_function(is_train=True)
    test_dataset = preprocess_function(is_train=False)

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    
    return train_dataset, test_dataset

train_dataset, test_dataset = load_prism_comparisons()

for i in range(10):
    print(f"Train example {i}:")
    pprint.pprint(train_dataset[i])

train_dataset.to_parquet('data/prism/train.parquet')
test_dataset.to_parquet('data/prism/test.parquet')

print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')
