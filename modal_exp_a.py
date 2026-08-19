"""
Modal runner for Experiment A (cross-dataset transfer).

Scores chosen/rejected in RewardBench2 / UltraFeedback / HH-RLHF with the PRISM mean-diff
direction, a LoRe basis, and the true reward head, on an A10G GPU, then syncs the CSV back.

Prereqs: a Modal secret named "huggingface" holding HF_TOKEN (mirrors modal_compute_vectors.py).
Run:      modal run modal_exp_a.py                 # default: all 3 datasets, 500 pairs each
          modal run modal_exp_a.py --limit 1000    # more pairs
          modal run modal_exp_a.py --datasets "rewardbench2 ultrafeedback"
"""
import os
import subprocess

import modal

app = modal.App("lore-prism-exp-a-cross-dataset")
prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "tqdm",
    )
    # Exclude the ~500MB local embedding .pkls (and git/cache) from the mount -- neither
    # experiment needs them (datasets come from HF; the probe direction is a tiny committed .pt).
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__"])
)


@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_exp_a(datasets: str, limit: int):
    os.chdir("/workspace")
    # Reuse the persistent HF cache so Skywork isn't re-downloaded each run.
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    # The in-repo checkpoints/ dir carries the LoRe K=20 basis for the lore_basis direction.
    lore_ckpt = "/workspace/checkpoints/checkpoints/PRISM_V_lore_K_20_alpha_10000.0.pt"
    cmd = ["python", "PRISM/exp_a_cross_dataset.py",
           "--datasets", *datasets.split(),
           "--limit", str(limit)]
    if os.path.exists(lore_ckpt):
        cmd += ["--lore_ckpt", lore_ckpt]
    subprocess.run(cmd, check=True)

    out = "/workspace/results/cross_dataset/exp_a_cross_dataset.csv"
    with open(out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(datasets: str = "rewardbench2 ultrafeedback hh_rlhf", limit: int = 500):
    print("Submitting Experiment A (cross-dataset transfer) to Modal...")
    csv_bytes = run_exp_a.remote(datasets, limit)
    os.makedirs("results/cross_dataset", exist_ok=True)
    with open("results/cross_dataset/exp_a_cross_dataset.csv", "wb") as f:
        f.write(csv_bytes)
    print("Synced results/cross_dataset/exp_a_cross_dataset.csv back to your machine.")
    print(csv_bytes.decode(errors="replace"))
