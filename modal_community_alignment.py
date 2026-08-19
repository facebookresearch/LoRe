"""
Modal runner: LoRe on Community Alignment, end to end.

Everything runs in the cloud (the dataset download is ~1.9GB and is much faster from Modal than
locally): prepare pairs -> embed batched on an A10G -> train LoRe -> report diagnostics.

Why this dataset: synthetic_recovery.py located a phase transition at ~50 pairs/user; PRISM has ~15
and nulled out, while Community Alignment (English) has a median of 201 pairs/user. This is the first
run where the data volume is actually sufficient for LoRe to work.

Cost control: the full eligible set (941 users x ~200 pairs) would be ~250k forward passes through an
8B model. --max_users / --max_pairs_per_user bound that; the defaults give ~200 users x 120 pairs,
which is comfortably above the phase transition while keeping the embed to roughly half an hour.

Run:
  modal run modal_community_alignment.py --smoke     # 6 users x 12 pairs, minutes, proves the wiring
  modal run modal_community_alignment.py             # real run
"""
import os
import shutil
import subprocess

import modal

app = modal.App("lore-community-alignment")
volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1", "transformers==4.47.0", "accelerate==1.2.1",
                 "datasets==3.2.0", "tqdm", "numpy", "pandas", "pyarrow")
    .add_local_dir(".", remote_path="/workspace",
                   ignore=["**/*.pkl", "data/prism/*.pkl", ".git", "**/__pycache__",
                           "data/community_alignment/**", "new_embeddings/**"])
)

VOL_DIR = "/vol/community_alignment"


@app.function(image=image, gpu="A10G", timeout=86400,
              volumes={"/vol": volume}, secrets=[modal.Secret.from_name("huggingface")])
def run_ca(max_users: int, max_pairs_per_user: int, min_pairs: int, smoke: bool):
    os.chdir("/workspace")
    os.makedirs(VOL_DIR, exist_ok=True)
    os.makedirs("data/community_alignment", exist_ok=True)
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    pairs_path = "data/community_alignment/pairs.json"
    emb_path = "data/community_alignment/embeddings.pt"

    print("[1/3] building pairs (English, data-rich users)...", flush=True)
    subprocess.run(["python", "PRISM/community_alignment_prep.py",
                    "--min_pairs", str(min_pairs),
                    "--max_users", str(max_users),
                    "--max_pairs_per_user", str(max_pairs_per_user),
                    "--out", pairs_path], check=True)

    print("\n[2/3] embedding (batched, de-duplicated)...", flush=True)
    embed_cmd = ["python", "PRISM/embed_community_alignment.py",
                 "--pairs", pairs_path, "--out", emb_path,
                 "--batch_size", "8" if smoke else "16"]
    # Reuse the previous (smaller-cap) run's embeddings: keys are content hashes, so only the
    # newly-included texts hit the GPU.
    prev = f"{VOL_DIR}/embeddings.pt"
    if os.path.exists(prev):
        volume.reload()
        embed_cmd += ["--resume", prev]
    subprocess.run(embed_cmd, check=True)
    # Persist BEFORE analysis so a later failure never discards the GPU work.
    shutil.copy(pairs_path, f"{VOL_DIR}/pairs.json")
    shutil.copy(emb_path, f"{VOL_DIR}/embeddings.pt")
    volume.commit()
    print(f"  persisted pairs + embeddings to {VOL_DIR}", flush=True)

    print("\n[3/3] running LoRe + diagnostics...", flush=True)
    cmd = ["python", "PRISM/community_alignment_lore.py",
           "--pairs", pairs_path, "--emb", emb_path,
           "--min_pairs", str(min(min_pairs, max_pairs_per_user))]
    if smoke:
        cmd += ["--ranks", "1", "5", "--iters", "300"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("stderr:\n" + r.stderr, flush=True)

    os.makedirs(f"{VOL_DIR}/results", exist_ok=True)
    with open(f"{VOL_DIR}/results/lore_results.txt", "w") as fh:
        fh.write(r.stdout + "\n" + (r.stderr or ""))
    csv_bytes = b""
    csv_path = "results/community_alignment/lore_results.csv"
    if os.path.exists(csv_path):
        shutil.copy(csv_path, f"{VOL_DIR}/results/lore_results.csv")
        csv_bytes = open(csv_path, "rb").read()
    volume.commit()
    return csv_bytes, r.stdout


@app.local_entrypoint()
def main(max_users: int = 200, max_pairs_per_user: int = 400,
         min_pairs: int = 100, smoke: bool = False):
    if smoke:
        max_users, max_pairs_per_user, min_pairs = 6, 12, 12
    print(f"Community Alignment on Modal: max_users={max_users}, "
          f"max_pairs_per_user={max_pairs_per_user}, min_pairs={min_pairs}")
    csv_bytes, stdout = run_ca.remote(max_users, max_pairs_per_user, min_pairs, smoke)

    if csv_bytes:
        os.makedirs("results/community_alignment", exist_ok=True)
        with open("results/community_alignment/lore_results.csv", "wb") as fh:
            fh.write(csv_bytes)
        print("\nSynced results/community_alignment/lore_results.csv")
    print("\n" + stdout[-3000:])
