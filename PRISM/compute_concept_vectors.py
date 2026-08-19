import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def compute_concept_vectors(input_file, output_file, model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}. Please run generate_contrastive_pairs.py first.")
        
    with open(input_file, 'r') as f:
        contrastive_pairs = json.load(f)
        
    print(f"Loading model {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    concept_vectors = {}
    
    for concept, pairs in contrastive_pairs.items():
        if len(pairs) == 0:
            continue
            
        print(f"\nComputing vector for '{concept}' ({len(pairs)} pairs)")
        high_embeddings = []
        low_embeddings = []
        
        for pair in tqdm(pairs):
            prompt = [{"content": pair["prompt"], "role": "user"}]
            
            # High response
            high_conv = prompt + [{"content": pair["high_response"], "role": "assistant"}]
            high_tokenized = tokenizer.apply_chat_template(
                high_conv, tokenize=True, return_tensors="pt"
            ).to(device)
            # newer transformers returns a BatchEncoding (has .keys()); older returns a bare tensor
            high_ids = high_tokenized["input_ids"] if hasattr(high_tokenized, "keys") else high_tokenized

            with torch.no_grad():
                high_out = model(high_ids)
                # Ensure it's in float32 for stable difference and mean computations
                high_emb = high_out.last_hidden_state[0, -1].cpu().to(torch.float32)
                high_embeddings.append(high_emb)
                
            # Low response
            low_conv = prompt + [{"content": pair["low_response"], "role": "assistant"}]
            low_tokenized = tokenizer.apply_chat_template(
                low_conv, tokenize=True, return_tensors="pt"
            ).to(device)
            low_ids = low_tokenized["input_ids"] if hasattr(low_tokenized, "keys") else low_tokenized

            with torch.no_grad():
                low_out = model(low_ids)
                low_emb = low_out.last_hidden_state[0, -1].cpu().to(torch.float32)
                low_embeddings.append(low_emb)
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if len(high_embeddings) > 0:
            mu_high = torch.stack(high_embeddings).mean(dim=0)
            mu_low = torch.stack(low_embeddings).mean(dim=0)
            concept_vector = mu_high - mu_low
            concept_vectors[concept] = concept_vector
            print(f"Vector computed for '{concept}' (shape: {concept_vector.shape}, norm: {torch.norm(concept_vector):.4f})")
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(concept_vectors, output_file)
    print(f"\nSaved concept vectors to {output_file}")

if __name__ == "__main__":
    compute_concept_vectors(
        input_file="data/prism/contrastive_pairs.json",
        output_file="data/prism/concept_vectors.pt"
    )
