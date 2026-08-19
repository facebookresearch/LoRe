"""
Modal runner for Experiment B (causal edit probe).

Applies single-attribute LLM edits to rejected responses (via AWS Bedrock), re-embeds them through
Skywork on an A10G GPU, and measures the score change along the PRISM direction, then syncs the CSV.

Prereqs (two Modal secrets):
  * "huggingface" : HF_TOKEN (for Skywork + the PRISM dataset)      [mirrors modal_compute_vectors.py]
  * "aws"         : AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY with Bedrock access.
                    generate_contrastive_pairs.py pins region us-east-1; override with AWS creds that
                    can call Claude Sonnet in that region (or edit the region in that file).
Run:      modal run modal_exp_b.py            # default 60 rejected responses x 5 edits
          modal run modal_exp_b.py --n 120
"""
import os
import subprocess

import modal

app = modal.App("lore-prism-exp-b-causal-edit")
prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "boto3",            # Bedrock client for the LLM edits
        "tqdm",
    )
    # Exclude the ~500MB local embedding .pkls (and git/cache) from the mount -- neither
    # experiment needs them (datasets come from HF; the probe direction is a tiny committed .pt).
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__"])
)


@app.function(          # CPU only: one Bedrock call to validate creds/region/model, no GPU
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("aws")],
)
def dry_run_exp_b():
    os.chdir("/workspace")
    subprocess.run(["python", "PRISM/exp_b_causal_edit.py", "--dry-run"], check=True)


@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("aws")],
)
def run_exp_b(n: int, length_matched: bool):
    os.chdir("/workspace")
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    cmd = ["python", "PRISM/exp_b_causal_edit.py", "--n", str(n)]
    if length_matched:
        cmd.append("--length-matched")
    subprocess.run(cmd, check=True)

    fname = "exp_b_length_matched.csv" if length_matched else "exp_b_causal_edit.csv"
    with open(f"/workspace/results/causal_edit/{fname}", "rb") as f:
        return f.read(), fname


@app.local_entrypoint()
def main(n: int = 60, dry_run: bool = False, length_matched: bool = False):
    if dry_run:
        print("Validating Bedrock creds/region/model on Modal (CPU, one call)...")
        dry_run_exp_b.remote()
        return
    label = "length-matched " if length_matched else ""
    print(f"Submitting Experiment B ({label}causal edit probe) to Modal...")
    csv_bytes, fname = run_exp_b.remote(n, length_matched)
    os.makedirs("results/causal_edit", exist_ok=True)
    with open(f"results/causal_edit/{fname}", "wb") as f:
        f.write(csv_bytes)
    print(f"Synced results/causal_edit/{fname} back to your machine.")
    print(csv_bytes.decode(errors="replace")[-800:])
