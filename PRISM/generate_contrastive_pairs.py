import os
import json
import boto3
from tqdm import tqdm
from concept_library import CONCEPT_LIBRARY

# Boto3 client for Bedrock Runtime.
# Auth: standard AWS creds OR a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK (botocore reads it
#       automatically for bedrock-runtime; keep boto3 recent enough to support it).
# Region: AWS_DEFAULT_REGION / AWS_REGION (Bedrock API keys are region-scoped -- must match the key).
# Model:  BEDROCK_MODEL_ID -- IMPORTANT: the inference-profile prefix must match the region
#         (e.g. us.anthropic.* for a us-* key, eu.anthropic.* for an eu-* key), and the model must
#         be enabled in that region.
_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
try:
    bedrock_client = boto3.client('bedrock-runtime', region_name=_REGION)
    print(f"[bedrock] region={_REGION} model={DEFAULT_MODEL_ID}")
except Exception as e:
    print(f"Failed to initialize boto3 client. Make sure AWS credentials are set: {e}")
    bedrock_client = None

def invoke_llm(prompt, model_id=None, max_tokens=4000, temperature=0.7):
    model_id = model_id or DEFAULT_MODEL_ID
    """
    Invokes Amazon Bedrock Converse API.
    By default uses Claude 3.5 Sonnet for high quality generation.
    """
    if bedrock_client is None:
        return None
        
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        )
        return response['output']['message']['content'][0]['text']
    except Exception as e:
        print(f"\nError invoking Bedrock ({model_id}): {e}")
        return None

def generate_batch(concept_name, concept_info, batch_size=5):
    prompt = f"""We are creating a dataset of contrastive pairs to evaluate AI models on the concept of '{concept_name}'.
Concept description: {concept_info['description']}

Please generate {batch_size} diverse, realistic user prompts (questions or requests). 
For EACH prompt, write two responses:
1. Response A (High Concept): {concept_info['high']}
2. Response B (Low Concept): {concept_info['low']}

Ensure the user prompts are varied in topic (e.g., coding, writing, reasoning, general knowledge) and complexity.

Format your output EXACTLY as a JSON list of objects. Do not wrap it in markdown code blocks.
Example format:
[
  {{
    "prompt": "the user's question here",
    "high_response": "response exhibiting the concept",
    "low_response": "response avoiding the concept"
  }}
]
Return ONLY the JSON array, with no additional text.
"""
    response_text = invoke_llm(prompt, temperature=0.8)
    if not response_text:
        return []
        
    # Clean up potential markdown formatting if the model disobeys
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
        
    try:
        data = json.loads(response_text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON for {concept_name}: {e}")
        return []

def judge_pair(concept_name, concept_info, pair):
    """
    Uses a smaller/faster model to judge if the contrast is clean.
    """
    prompt = f"""You are an impartial judge evaluating two AI responses based on the concept of '{concept_name}'.
Concept description: {concept_info['description']}

User Prompt: {pair['prompt']}

Response 1: {pair['high_response']}
Response 2: {pair['low_response']}

Does Response 1 exhibit the concept of '{concept_name}' significantly more than Response 2?
Consider the definitions:
- High: {concept_info['high']}
- Low: {concept_info['low']}

Answer only with a single word: YES or NO.
"""
    # Judge with the same (env-configurable) model as generation.
    response = invoke_llm(prompt, max_tokens=10, temperature=0.0)
    if not response:
        return False
        
    return "YES" in response.strip().upper()

def main():
    target_pairs_per_concept = 50
    batch_size = 5
    
    output_dir = "data/prism"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "contrastive_pairs.json")
    
    # Load existing to resume if interrupted
    all_pairs = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_pairs = json.load(f)
            
    for concept_name, concept_info in CONCEPT_LIBRARY.items():
        if concept_name not in all_pairs:
            all_pairs[concept_name] = []
            
        current_count = len(all_pairs[concept_name])
        if current_count >= target_pairs_per_concept:
            print(f"{concept_name} already has {current_count} pairs. Skipping.")
            continue
            
        print(f"\nGenerating pairs for '{concept_name}' (Current: {current_count}/{target_pairs_per_concept})")
        
        pbar = tqdm(total=target_pairs_per_concept, initial=current_count)
        
        consecutive_failures = 0
        while current_count < target_pairs_per_concept:
            # Generate a batch
            batch = generate_batch(concept_name, concept_info, batch_size=batch_size)
            if not batch:
                consecutive_failures += 1
                if consecutive_failures > 3:
                    print("Too many failures, moving to next concept.")
                    break
                continue
                
            consecutive_failures = 0
            
            # Judge the batch
            for pair in batch:
                if 'prompt' in pair and 'high_response' in pair and 'low_response' in pair:
                    is_clean = judge_pair(concept_name, concept_info, pair)
                    if is_clean:
                        all_pairs[concept_name].append(pair)
                        current_count += 1
                        pbar.update(1)
                        
                        # Save incrementally
                        with open(output_file, 'w') as f:
                            json.dump(all_pairs, f, indent=2)
                            
                        if current_count >= target_pairs_per_concept:
                            break
                            
        pbar.close()

if __name__ == "__main__":
    main()
