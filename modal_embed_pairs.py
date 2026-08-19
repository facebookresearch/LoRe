"""
Modal runner: embed the contrastive pairs per-pair on an A10G, then run the concept-user experiment.

Stage A of the synthetic-text positive control -- reuses the 550 already-generated contrastive pairs
(11 concepts x 50), so there is no LLM cost; the only spend is ~1100 forward passes through Skywork.

NOTE on volume persistence: add_local_dir(".") already creates /workspace/data, so the
`if not os.path.exists("data"): os.symlink(...)` pattern used by modal_compute_vectors.py silently
does nothing and results are written to the container's ephemeral disk (this exact bug cost ~2h of
GPU earlier in this project). We therefore copy outputs to the volume EXPLICITLY and commit before
running any analysis.

Run:
  modal run modal_embed_pairs.py --smoke      # 2 pairs/concept, verifies persistence, ~2 min
  modal run modal_embed_pairs.py              # full Stage A
"""
import os
import shutil
import subprocess

import modal

app = modal.App("lore-concept-users")
volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1", "transformers==4.47.0", "accelerate==1.2.1",
                 "datasets==3.2.0", "tqdm", "numpy")
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__",
                           "data/community_alignment/**"])
)

VOL_DIR = "/vol/concept_users"
EMB_NAME = "contrastive_pair_embeddings.pt"


@app.function(image=image, gpu="A10G", timeout=86400,
              volumes={"/vol": volume}, secrets=[modal.Secret.from_name("huggingface")])
def embed_and_run(smoke: bool = False):
    os.chdir("/workspace")
    os.makedirs(VOL_DIR, exist_ok=True)
    os.makedirs("data/prism", exist_ok=True)
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    cmd = ["python", "PRISM/embed_contrastive_pairs.py"]
    if smoke:
        cmd += ["--limit", "2"]
    print("[1/2] embedding contrastive pairs...", flush=True)
    subprocess.run(cmd, check=True)

    # Persist BEFORE analysis so a later failure never costs the GPU work.
    src = f"data/prism/{EMB_NAME}"
    shutil.copy(src, f"{VOL_DIR}/{EMB_NAME}")
    volume.commit()
    print(f"  persisted {src} -> {VOL_DIR}/{EMB_NAME}", flush=True)

    print("\n[2/2] running the concept-user experiment...", flush=True)
    r = subprocess.run(["python", "PRISM/synthetic_concept_users.py"],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("stderr:\n" + r.stderr, flush=True)

    os.makedirs(f"{VOL_DIR}/results", exist_ok=True)
    with open(f"{VOL_DIR}/results/concept_users.txt", "w") as fh:
        fh.write(r.stdout + "\n" + (r.stderr or ""))
    csv_bytes = b""
    csv_path = "results/synthetic/concept_users.csv"
    if os.path.exists(csv_path):
        shutil.copy(csv_path, f"{VOL_DIR}/results/concept_users.csv")
        csv_bytes = open(csv_path, "rb").read()
    volume.commit()

    with open(f"{VOL_DIR}/{EMB_NAME}", "rb") as fh:
        emb_bytes = fh.read()
    return emb_bytes, csv_bytes, r.stdout


@app.local_entrypoint()
def main(smoke: bool = False):
    print(f"Submitting {'SMOKE ' if smoke else ''}concept-user run to Modal...")
    emb_bytes, csv_bytes, stdout = embed_and_run.remote(smoke)

    os.makedirs("data/prism", exist_ok=True)
    dest = f"data/prism/{'smoke_' if smoke else ''}{EMB_NAME}"
    with open(dest, "wb") as fh:
        fh.write(emb_bytes)
    print(f"\nSynced {dest} ({len(emb_bytes)/1e6:.1f} MB)")

    if csv_bytes:
        os.makedirs("results/synthetic", exist_ok=True)
        with open("results/synthetic/concept_users.csv", "wb") as fh:
            fh.write(csv_bytes)
        print("Synced results/synthetic/concept_users.csv")
    print("\n" + stdout[-2500:])
