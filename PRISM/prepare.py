# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
import requests

url = "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/survey.jsonl"
response = requests.get(url)
with open("survey.jsonl", 'w') as f:
    f.write(response.text)
url = "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/conversations.jsonl"
response = requests.get(url)
with open("conversations.jsonl", 'w') as f:
    f.write(response.text)
url = "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/resolve/main/utterances.jsonl"
response = requests.get(url)
with open("utterances.jsonl", 'w') as f:
    f.write(response.text)

# should correspond to parameters in training/evaluation
max_text_length = 2300
max_prompt_string_length = 1400
seed=123

import os
import json
import numpy as np
np.random.seed(seed=123)

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

with open ("./survey.jsonl", 'r') as f:
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

with open ("./conversations.jsonl", 'r') as f:
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
            or len(turn["user_utterance"][0]) + len(turn["chosen_utterance"][0]) > max_text_length
        ):
            qualified_num -= 1
        else:
            for rejected in turn["rejected_utterance"]:
                if len(turn["user_utterance"][0]) + len(rejected) > max_text_length:
                    qualified_num -= 1
                    break
    # only delete when the whole dialogue is not qualifed
    if qualified_num == 0:
        print("delete dialogue", dialog_id, "by", data_dialog[dialog_id]["user_id"])
        data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"].remove(dialog_id)
        if data_user[data_dialog[dialog_id]["user_id"]]["dialog_ids"] == []:
            print("delete user", data_dialog[dialog_id]["user_id"])
            del data_user[data_dialog[dialog_id]["user_id"]]
        del data_dialog[dialog_id]

# save as json
with open ("./prism_data_user.json", 'w') as f:
    json.dump(data_user, f, indent=4)

with open ("./prism_data_dialog.json", 'w') as f:
    json.dump(data_dialog, f, indent=4)

# split users
import numpy as np
np.random.seed(seed=123)

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


# for user_id in seen_user_ids_init:
#     to_choose_from = np.array(data_user[user_id]["dialog_ids"])
#     np.random.shuffle(to_choose_from)
#     train_dialog_ids = np.concatenate((train_dialog_ids, to_choose_from[:int(len(to_choose_from)*0.5)]))
#     test_dialog_ids = np.concatenate((test_dialog_ids, to_choose_from[int(len(to_choose_from)*0.5):]))
#     # move users with no dialog in train to unseen, because int(1*0.9)=0
#     if len(to_choose_from) > 1:
#         seen_user_ids.append(user_id)
#     else:
#         unseen_user_ids.append(user_id)

# for user_id in unseen_user_ids_init:
#     test_dialog_ids = np.concatenate((test_dialog_ids, np.array(data_user[user_id]["dialog_ids"])))
#     unseen_user_ids.append(user_id)

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

# save as json, assign our user ids, 0=unseem, 1...=seen
split_ids = {"train_dialog_ids": list(train_dialog_ids),
             "test_dialog_ids": list(test_dialog_ids),
             "seen_user_ids": {k:i+1 for i, k in enumerate(seen_user_ids)},
             "unseen_user_ids": {k: 0 for k in unseen_user_ids}
            }
# with open ("./prism_split_ids.json", 'w') as f:
    # json.dump(split_ids, f, indent=4)
with open ("./prism_split_ids_50.json", 'w') as f:
    json.dump(split_ids, f, indent=4)

def load_prism_comparisons(
    sep="||",
    n_user_tokens=10,
    max_text_length=2400,
    max_prompt_string_length = 1500,  # about 500 tokens, corresponds to max_prompt_length 550
    sanity_check=False,
    prism_data_path=None,
    seed=123,
    add_textual_info=False,
):
    with open("./prism_data_dialog.json", 'r') as f:
        data_dialog = json.load(f)
    with open("./prism_data_user.json", 'r') as f:
        data_user = json.load(f)
    with open("./prism_split_ids_50.json", 'r') as f:
        split_ids = json.load(f)

    n_users = len(data_user)

    def preprocess_function(is_train, add_textual_info=False):
        new_examples = {
            # "prompt": [],
            # "chosen": [],
            # "rejected": [],
        }

        if is_train:
            dialog_ids = split_ids["train_dialog_ids"]
            # user_ids = split_ids["seen_user_ids"]
        else:
            dialog_ids = split_ids["test_dialog_ids"]
            # user_ids = split_ids["seen_user_ids"]
            # user_ids.update(split_ids["unseen_user_ids"])

        for dialog_id in dialog_ids:
            history = ""
            user_id = data_dialog[dialog_id]["user_id"]
            if user_id not in new_examples:
                # new_examples[user_id] = {
                #     "prompt": [],
                #     "chosen": [],
                #     "rejected": [],
                # }
                new_examples[user_id] = {}
            if dialog_id not in new_examples[user_id]:
                new_examples[user_id][dialog_id] = {
                    "prompt": [],
                    "chosen": [],
                    "rejected": [],
                }
            # textual info of the user
            if add_textual_info:
                preference = ", ".join(data_user[data_dialog[dialog_id]["user_id"]]["demographics"]["preference"])
                textual_info = f"preference: {preference}; "
                for k in data_user[data_dialog[dialog_id]["user_id"]]["demographics"]:
                    if k != "preference":
                        textual_info += f"{k}: {data_user[data_dialog[dialog_id]['user_id']]['demographics'][k]}; "
            for turn in data_dialog[dialog_id]["turns"]:
                # add user utterance to history
                history += f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user_utterance'][0]}<|eot_id|>\n"

                # prepare examples, skip empty or too long examples
                if (    turn['user_utterance'] != [] 
                    and turn['chosen_utterance'] != [] 
                    and turn['rejected_utterance'] != [] 
                    and len(turn["user_utterance"][0]) + len(turn["chosen_utterance"][0]) < max_text_length
                ):
                    # build user identifier
                    # user_identifier = f"USER: {user_ids[data_dialog[dialog_id]['user_id']]} " + ("<|end_of_text|>"*n_user_tokens)
                        
                    # build prompt
                    # prompt = user_identifier + sep
                    # if add_textual_info:
                    #     prompt += 'User textual information: ' + textual_info 
                    # # truncate history
                    # max_history_string_length = max_prompt_string_length - len(prompt)
                    # if len(history) > max_history_string_length:
                    #     history = history[-max_history_string_length:]
                    # prompt += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    max_history_string_length = max_prompt_string_length
                    if len(history) > max_history_string_length:
                        history = history[-max_history_string_length:]
                    prompt = history + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                    # append to examples
                    for rejected in turn["rejected_utterance"]:
                        # skip too long examples
                        if len(turn["user_utterance"][0]) + len(rejected) > max_text_length:
                            continue
                        # new_examples["prompt"].append(prompt)
                        # new_examples["chosen"].append(turn['chosen_utterance'][0])
                        # new_examples["rejected"].append(rejected)
                        new_examples[user_id][dialog_id]["prompt"].append(prompt)
                        new_examples[user_id][dialog_id]["chosen"].append(turn['chosen_utterance'][0])
                        new_examples[user_id][dialog_id]["rejected"].append(rejected)
                        # # if train, duplicate 0
                        # if is_train:
                        #     user_identifier_0 = f"USER: {0} " + ("<|end_of_text|>"*n_user_tokens)
                        #     prompt_0 = user_identifier_0 + sep
                        #     if add_textual_info:
                        #         user_identifier_0 += 'User textual information: ' + textual_info 
                        #     prompt_0 += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        #     # new_examples["prompt"].append(prompt_0)
                        #     # new_examples["chosen"].append(turn['chosen_utterance'][0]) 
                        #     # new_examples["rejected"].append(rejected)
                        #     new_examples[user_id]["prompt"].append(prompt_0)
                        #     new_examples[user_id]["chosen"].append(turn['chosen_utterance'][0])
                        #     new_examples[user_id]["rejected"].append(rejected)
                    
                # add the first chosen utterance to history for next turn
                if turn['chosen_utterance'] != []:
                    history += f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn['chosen_utterance'][0]}<|eot_id|>\n"

        return new_examples

    train_dataset = preprocess_function(is_train=True, add_textual_info=add_textual_info)
    test_dataset = preprocess_function(is_train=False, add_textual_info=add_textual_info)
    
    # train_dataset = Dataset.from_dict(train_examples)
    # test_dataset = Dataset.from_dict(test_examples)

    # use a small subset for sanity check
    if sanity_check: 
        train_dataset = train_dataset.select(range(100))
        test_dataset  = test_dataset.select(range(50))
    
    return train_dataset, test_dataset, n_users

train_dataset, test_dataset, n_users = load_prism_comparisons()

# for user_id, dialogs in train_dataset.items():
#     print(f"User ID: {user_id}")
#     for dialog_id, examples in dialogs.items():
#         print(f"Dialog ID: {dialog_id}")
#         for i in range(len(examples["prompt"])):
#             print(f"Prompt: {examples['prompt'][i]}")
#             print(f"Chosen: {examples['chosen'][i]}")
#             print(f"Rejected: {examples['rejected'][i]}")
#             print("--------------------")
#         break  # Only print the first user's data
#     break
    

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

embeddings = {}
for user_id, dialogs in train_dataset.items():
# for user_id, dialogs in test_dataset.items():
    print(f"User ID: {user_id}")
    embeddings[user_id] = {}
    for dialog_id, examples in dialogs.items():
        print(f"Dialog ID: {dialog_id}")
        embeddings[user_id][dialog_id] = {"chosen":[], "rejected":[]}
        for i in range(len(examples["prompt"])):
            # print(f"Prompt: {examples['prompt'][i]}")
            # print(f"Chosen: {examples['chosen'][i]}")
            # print(f"Rejected: {examples['rejected'][i]}")
            # print("--------------------")
            conv = [{"role": "user", "content": examples['prompt'][i]}, {"role": "assistant", "content": examples['chosen'][i]}]
            conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = rm(conv_tokenized)  # Forward pass through the model
                # Extract the last hidden state of the last token and move it to CPU
                # rewards.append(output.logits[0][0].item())
                embeddings[user_id][dialog_id]["chosen"].append(output.last_hidden_state[0][-1].cpu())
            conv = [{"role": "user", "content": examples['prompt'][i]}, {"role": "assistant", "content": examples['rejected'][i]}]
            conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output = rm(conv_tokenized)  # Forward pass through the model
                # Extract the last hidden state of the last token and move it to CPU
                # rewards.append(output.logits[0][0].item())
                embeddings[user_id][dialog_id]["rejected"].append(output.last_hidden_state[0][-1].cpu())
    # break  # Only the first user's data
torch.save(embeddings, "prism_train_embeddings_50.pt")
# torch.save(embeddings, "prism_test_embeddings_50.pt")
