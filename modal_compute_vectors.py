import modal
import subprocess
import os

app = modal.App("lore-prism-concept-vectors")

# Define the persistent volume where data is stored
prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "tqdm"
    )
    .add_local_dir(".", remote_path="/workspace")
)

@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",          # Nvidia GPU required for fast embedding extraction
    timeout=86400,       # Max timeout (24 hours)
    secrets=[modal.Secret.from_name("huggingface")]
)
def run_compute_vectors():
    print("Computing Concept Vectors on Modal...")
    
    os.chdir("/workspace")
    
    # Symlink project data directory to persistent volume
    if not os.path.exists("data"):
        os.symlink("/vol/data", "data")
        
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")
    
    # Run the script
    subprocess.run([
        "python", "PRISM/compute_concept_vectors.py"
    ], check=True)
    
    print("\nConcept vectors computed in the cloud. Syncing back to local...")
    with open("data/prism/concept_vectors.pt", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    import os
    print("Submitting Concept Vector Computation to Modal Cloud...")
    file_bytes = run_compute_vectors.remote()
    
    os.makedirs("data/prism", exist_ok=True)
    with open("data/prism/concept_vectors.pt", "wb") as f:
        f.write(file_bytes)
    print("Successfully synced concept_vectors.pt back to your local machine!")
