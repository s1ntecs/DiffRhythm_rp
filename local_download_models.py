#!/usr/bin/env python3
"""
Скачивает веса DiffRhythm ЛОКАЛЬНО к себе на машину
Потом их можно копировать в Docker образ
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

# Папка куда сохранять (можешь изменить)
MODELS_DIR = "./pretrained"


def download_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("🔽 Downloading DiffRhythm weights locally...")
    print(f"📁 Destination: {os.path.abspath(MODELS_DIR)}\n")
    
    # 1. Base model (95 seconds)
    print("[1/4] DiffRhythm-1_2 (base - 95s) - 3.5GB")
    hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-1_2",
        filename="cfm_model.pt",
        local_dir=f"{MODELS_DIR}/DiffRhythm-1_2",
        local_dir_use_symlinks=False
    )
    print("✅ Done\n")
    
    # 2. Full model (285 seconds)
    print("[2/4] DiffRhythm-1_2-full (long audio - 285s) - 3.5GB")
    hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-1_2-full",
        filename="cfm_model.pt",
        local_dir=f"{MODELS_DIR}/DiffRhythm-1_2-full",
        local_dir_use_symlinks=False
    )
    print("✅ Done\n")
    
    # 3. VAE
    print("[3/4] DiffRhythm-vae - 400MB")
    hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        local_dir=f"{MODELS_DIR}/DiffRhythm-vae",
        local_dir_use_symlinks=False
    )
    print("✅ Done\n")
    
    # 4. MuQ-MuLan
    print("[4/4] MuQ-MuLan-large - 1.5GB")
    snapshot_download(
        repo_id="OpenMuQ/MuQ-MuLan-large",
        local_dir=f"{MODELS_DIR}/MuQ-MuLan-large",
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"]
    )
    print("✅ Done\n")
    
    print("=" * 60)
    print("✅ ALL WEIGHTS DOWNLOADED!")
    print("=" * 60)
    print(f"\nLocation: {os.path.abspath(MODELS_DIR)}")
    print("\nНext step:")
    print("  Update your Dockerfile to copy these models")
    
if __name__ == "__main__":
    try:
        download_all()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure you have installed: pip install huggingface_hub")