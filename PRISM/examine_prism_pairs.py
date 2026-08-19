"""
Experiment 2: Qualitative Analysis of PRISM Chosen vs. Rejected Responses

Loads the PRISM dataset directly from HuggingFace and examines 
chosen/rejected pairs to understand why accuracy is so high.
"""
import json
from datasets import load_dataset
from collections import defaultdict

def examine_prism_pairs():
    print("Loading PRISM dataset from HuggingFace...")
    dataset = load_dataset("HannahRoseKirk/prism-alignment", "pairwise")
    
    train_data = dataset["train"]
    
    print(f"Total training examples: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")
    print()
    
    # --- Analysis 1: Look at chosen vs rejected response lengths ---
    chosen_lengths = []
    rejected_lengths = []
    
    for i, example in enumerate(train_data):
        chosen = example.get("best_response", "") or example.get("chosen", "") or ""
        rejected = example.get("worst_response", "") or example.get("rejected", "") or ""
        
        if chosen and rejected:
            chosen_lengths.append(len(chosen))
            rejected_lengths.append(len(rejected))
    
    if chosen_lengths:
        avg_chosen = sum(chosen_lengths) / len(chosen_lengths)
        avg_rejected = sum(rejected_lengths) / len(rejected_lengths)
        print(f"=== Length Analysis ===")
        print(f"Average chosen response length:   {avg_chosen:.0f} chars")
        print(f"Average rejected response length: {avg_rejected:.0f} chars")
        print(f"Chosen is {avg_chosen/avg_rejected:.2f}x longer on average")
        print()
    
    # --- Analysis 2: Print sample pairs side by side ---
    print("=" * 80)
    print("=== SAMPLE CHOSEN vs REJECTED PAIRS ===")
    print("=" * 80)
    
    import random
    random.seed(42)
    indices = random.sample(range(len(train_data)), min(10, len(train_data)))
    
    for idx in indices:
        example = train_data[idx]
        print(f"\n--- Example {idx} ---")
        # Print all available keys to understand the structure
        for key in example.keys():
            val = example[key]
            if isinstance(val, str) and len(val) > 0:
                preview = val[:200] + "..." if len(val) > 200 else val
                print(f"  {key}: {preview}")
            elif isinstance(val, (int, float, bool)):
                print(f"  {key}: {val}")
        print()
    
    # --- Analysis 3: Check for personalization signal ---
    # Do multiple users rate the same prompt differently?
    print("=" * 80)
    print("=== PERSONALIZATION ANALYSIS ===")
    print("=" * 80)
    
    prompt_users = defaultdict(set)
    prompt_choices = defaultdict(list)
    
    for example in train_data:
        prompt = example.get("prompt", "") or example.get("instruction", "") or ""
        user_id = example.get("user_id", "") or example.get("participant_id", "")
        chosen = example.get("best_response", "") or example.get("chosen", "")
        
        if prompt and user_id:
            prompt_key = prompt[:100]  # use first 100 chars as key
            prompt_users[prompt_key].add(str(user_id))
            prompt_choices[prompt_key].append({
                "user": str(user_id),
                "chosen_preview": (chosen or "")[:100]
            })
    
    # Find prompts with multiple users
    multi_user_prompts = {k: v for k, v in prompt_users.items() if len(v) > 1}
    print(f"\nTotal unique prompts (first 100 chars): {len(prompt_users)}")
    print(f"Prompts rated by multiple users: {len(multi_user_prompts)}")
    
    if multi_user_prompts:
        print("\nExamples of prompts with multiple users:")
        for prompt_key, users in list(multi_user_prompts.items())[:5]:
            print(f"\n  Prompt: '{prompt_key}...'")
            print(f"  Users: {users}")
            for choice in prompt_choices[prompt_key]:
                print(f"    User {choice['user']} chose: '{choice['chosen_preview']}...'")

if __name__ == "__main__":
    examine_prism_pairs()
