"""
Experiment B: causal probe of WHAT the PRISM preference direction tracks.

Motivating suggestion: "Can you manually edit individual rejected responses so that they score more
like chosen responses?" This turns the correlational E6 result into a causal one: if editing ONE
attribute of a rejected response raises its projection onto the direction, the direction causally
tracks that attribute.

For each sampled (prompt, rejected_response) we produce single-attribute edits via an LLM, each
changing ONLY the target attribute:
    +factuality   : fix false claims / add correct, verifiable detail (no other change)
    +helpfulness  : make it directly and fully answer the prompt (no other change)
    +formatting   : add markdown structure/headers/bullets (same content)
    -sycophancy   : add flattery/agreement (E6 says the direction should score this LOWER)
    +length_ctrl  : pad with on-topic filler that adds NO new quality (the artifact control)
Then we embed original and edited via Skywork (last-token hidden state), project onto the direction,
and report the mean score change per edit type.

Read-out:
  * If +factuality / +helpfulness move the score up most and +length_ctrl is ~0 -> the direction
    tracks genuine quality (a real result, not an artifact).
  * If +length_ctrl moves it as much as the quality edits -> it is largely a length artifact
    (the length-artifact concern confirmed).
  * -sycophancy should move it DOWN if the direction really penalizes sycophancy (E6 cross-check).
We report the same deltas for the true reward head, so we can see whether the PRISM direction and
the general RM respond to the same edits.

Requires Skywork (GPU/Modal) + AWS Bedrock (same setup as generate_contrastive_pairs.py).

Usage:  python exp_b_causal_edit.py --n 60
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from rm_head_utils import load_reward_head            # noqa: E402
from generate_contrastive_pairs import invoke_llm     # reuse the Bedrock client  # noqa: E402

MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


# Same templating/collation as eval_rb2.py, inlined so this script is self-contained
# (eval_rb2.py uses 3.10-only `str | None` syntax and can't be imported on 3.9).
def apply_template(tokenizer, prompt, response):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False)
    except Exception:
        eos = getattr(tokenizer, "eos_token", "") or ""
        return f"Human: {prompt}\nAssistant: {response}{eos}"


def collate(tokenizer, texts, max_length):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")

EDITS = {
    "+factuality":  "Revise the response to fix any inaccurate or unverifiable claims and add correct, "
                    "well-grounded factual detail. Change ONLY factual accuracy; keep length, tone, "
                    "and formatting the same. Return only the revised response.",
    "+helpfulness": "Revise the response so it directly and completely answers the user's request. "
                    "Change ONLY how well it addresses the need; keep length, tone, and formatting "
                    "similar. Return only the revised response.",
    "+formatting":  "Reformat the response with markdown structure (headers, bullet points, bold) "
                    "WITHOUT changing the actual content or wording meaningfully. Return only the "
                    "reformatted response.",
    "-sycophancy":  "Revise the response to add effusive praise of the user and agree with everything "
                    "they say, while keeping the substantive content the same. Return only the revised "
                    "response.",
    "+length_ctrl": "Make the response about twice as long by adding on-topic filler and restatement "
                    "that introduces NO new information, insight, or quality. Return only the padded "
                    "response.",
}

# Length-matched variant (--length-matched): change ONE attribute while holding word count ~constant,
# so the score change reflects the attribute rather than how much text the edit added. "neutral" is
# the control -- a length-preserving paraphrase whose Delta should be ~0 (the noise floor of just
# rewriting); the other edits are only trustworthy if neutral stays small.
_LEN_RULE = ("Keep the word count within 10% of the original and keep the same overall length. "
             "Return only the revised response, nothing else.")
EDITS_LM = {
    "neutral":      "Paraphrase the response, preserving its meaning, quality, tone, and formatting. "
                    "Change nothing of substance. " + _LEN_RULE,
    "+factuality":  "Revise to fix inaccurate/unverifiable claims and add correct factual detail, "
                    "changing ONLY factual accuracy (same tone/formatting). " + _LEN_RULE,
    "+helpfulness": "Revise so it directly and completely answers the request, changing ONLY how well "
                    "it addresses the need (same tone/formatting). " + _LEN_RULE,
    "+formatting":  "Add markdown structure (headers, bullet points, bold) by restructuring the "
                    "EXISTING wording -- do NOT add new content or words. " + _LEN_RULE,
    "-sycophancy":  "Add flattery of the user and agreement, but REMOVE an equal amount of other text "
                    "so the length is unchanged; keep the same formatting. " + _LEN_RULE,
}


def sample_prompts(n):
    # Real PRISM (prompt, rejected) pairs, pre-extracted from the embeddings into a small committed
    # JSON. (The HF dataset has no "pairwise" config -- its configs are survey/conversations/
    # utterances/metadata -- so we ship the pairs directly instead of reconstructing them here.)
    import json
    path = os.path.join(SCRIPT_DIR, "..", "data", "prism", "rejected_samples.json")
    with open(path) as f:
        data = json.load(f)
    return [(d["prompt"], d["rejected"]) for d in data[:n]]


def make_edit(rejected, instruction, target_words=None):
    extra = ""
    if target_words:
        lo, hi = int(0.9 * target_words), int(1.1 * target_words)
        extra = (f"\n\nHARD LENGTH CONSTRAINT: the original is {target_words} words. Your revision "
                 f"MUST be between {lo} and {hi} words -- if you add detail, cut elsewhere to "
                 f"compensate. This constraint overrides everything else.")
    return invoke_llm(f"{instruction}{extra}\n\n---\nResponse to revise:\n{rejected}",
                      max_tokens=1500, temperature=0.3)


@torch.inference_mode()
def embed_one(model, tokenizer, prompt, response, device, max_length):
    text = apply_template(tokenizer, prompt, response)
    enc = collate(tokenizer, [text], max_length).to(device)
    hs = model(**enc).last_hidden_state
    idx = enc["attention_mask"].sum(dim=1) - 1
    return hs[0, idx[0]].float().cpu()          # [H]


def dry_run():
    """Validate the Bedrock credential/region/model with ONE edit call, then exit.
    Skips the Skywork load entirely, so it returns in seconds and costs ~nothing."""
    print("[dry-run] making one Bedrock edit call to validate creds/region/model...")
    out = make_edit("The capital of France is Berlin.",
                    EDITS["+factuality"])
    if out:
        print("[dry-run] SUCCESS. Bedrock returned:\n  " + out.strip().replace("\n", "\n  ")[:600])
        print("\n[dry-run] Credentials/region/model work. Safe to run the full experiment.")
        return 0
    print("[dry-run] FAILED: no response from Bedrock. Check that the `aws` secret has "
          "AWS_BEARER_TOKEN_BEDROCK, AWS_DEFAULT_REGION matches the key's region, and "
          "BEDROCK_MODEL_ID is an inference profile enabled in that region (us.* vs eu.*).")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of rejected responses to edit")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true",
                    help="make one Bedrock call to validate creds/region/model, then exit "
                         "(no Skywork load, no GPU work)")
    ap.add_argument("--length-matched", action="store_true",
                    help="use length-preserving single-attribute edits (+ a neutral paraphrase "
                         "control) and log the edited/original word-count ratio")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds to pause after each Bedrock call (avoids ThrottlingException)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(dry_run())

    edits = EDITS_LM if args.length_matched else EDITS
    out = args.out or os.path.join(SCRIPT_DIR, "..", "results", "causal_edit",
                                   "exp_b_length_matched.csv" if args.length_matched
                                   else "exp_b_causal_edit.csv")
    wc = lambda s: len(s.split())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Loading Skywork backbone + directions...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(device).eval()  # memory-efficient
    v_prism = torch.load(os.path.join(SCRIPT_DIR, "..", "reproduced_matrices",
                                      "prism_mean_diff_direction.pt"), weights_only=True).reshape(-1)
    v_head = load_reward_head().reshape(-1)

    def proj(h):
        return {"prism_meandiff": float(h @ v_prism), "true_head": float(h @ v_head)}

    samples = sample_prompts(args.n)
    print(f"{len(samples)} (prompt, rejected) samples")

    # per-edit: score deltas for each direction + edited/original word-count ratio
    deltas = {e: {"prism_meandiff": [], "true_head": [], "len_ratio": []} for e in edits}
    import csv
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["sample", "edit", "direction", "score_orig", "score_edit", "delta", "len_ratio"])

    for si, (prompt, rejected) in enumerate(tqdm(samples)):
        h0 = embed_one(model, tokenizer, prompt, rejected, device, args.max_length)
        s0 = proj(h0)
        for edit, instr in edits.items():
            tw = wc(rejected) if args.length_matched else None
            edited = make_edit(rejected, instr, target_words=tw)
            if args.sleep:
                time.sleep(args.sleep)          # throttle-avoidance pause after each Bedrock call
            if not edited:
                continue
            lr = wc(edited) / max(wc(rejected), 1)
            deltas[edit]["len_ratio"].append(lr)
            h1 = embed_one(model, tokenizer, prompt, edited, device, args.max_length)
            s1 = proj(h1)
            for d in ("prism_meandiff", "true_head"):
                dl = s1[d] - s0[d]
                deltas[edit][d].append(dl)
                w.writerow([si, edit, d, round(s0[d], 4), round(s1[d], 4), round(dl, 4), round(lr, 3)])
        f.flush()
    f.close()

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    print("\n=== mean score change per edit (higher = pushes rejected toward 'chosen') ===")
    print(f"{'edit':>14} | {'prism_meandiff':>16} | {'true_head':>12} | {'len_ratio':>9}")
    print("-" * 62)
    for edit in edits:
        print(f"{edit:>14} | {mean(deltas[edit]['prism_meandiff']):>16.4f} | "
              f"{mean(deltas[edit]['true_head']):>12.4f} | {mean(deltas[edit]['len_ratio']):>9.2f}")
    if args.length_matched:
        print("\nRead-out (length-matched): len_ratio should be ~1.0 for every row (length held). "
              "With length controlled, +factuality/+helpfulness up and -sycophancy down => the "
              "direction tracks the ATTRIBUTE, not text length. Watch neutral (~0 = noise floor) "
              "and +formatting (now isolates markup from length).")
    else:
        print("\nRead-out: +factuality/+helpfulness should raise the score, -sycophancy lower it; "
              "but edits change text length too -- run --length-matched to isolate attribute from length.")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
