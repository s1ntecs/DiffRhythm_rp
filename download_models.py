#!/usr/bin/env python3
"""
Скачивает все необходимые модели DiffRhythm с HuggingFace
Использовать перед сборкой Docker образа для ускорения cold start
"""
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

CACHE_DIR = "./pretrained"


def download_models(include_full=True):
    """
    Загружает все модели DiffRhythm

    Args:
        include_full: Загружать ли full модель (285 сек) - требует +3.5GB
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("📥 Downloading DiffRhythm Models from HuggingFace")
    print("=" * 60)

    # 1. DiffRhythm-1_2 (base model для 95 секунд)
    print("\n[1/4] Downloading DiffRhythm-1_2 (base model)...")
    print("      Size: ~3.5GB")
    try:
        path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir=CACHE_DIR
        )
        print(f"      ✅ Downloaded to: {path}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

    # 2. DiffRhythm-1_2-full (опционально, для 285 секунд)
    if include_full:
        print("\n[2/4] Downloading DiffRhythm-1_2-full (for long audio)...")
        print("      Size: ~3.5GB")
        try:
            path = hf_hub_download(
                repo_id="ASLP-lab/DiffRhythm-1_2-full",
                filename="cfm_model.pt",
                cache_dir=CACHE_DIR
            )
            print(f"      ✅ Downloaded to: {path}")
        except Exception as e:
            print(f"      ⚠️  Skipping full model: {e}")
    else:
        print("\n[2/4] Skipping DiffRhythm-1_2-full (use --no-full to skip)")

    # 3. VAE model (обязательно)
    print("\n[3/4] Downloading DiffRhythm-vae (required)...")
    print("      Size: ~400MB")
    try:
        path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir=CACHE_DIR
        )
        print(f"      ✅ Downloaded to: {path}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

    # 4. MuQ-MuLan (для style prompts)
    print("\n[4/4] Downloading MuQ-MuLan-large (for style prompts)...")
    print("      Size: ~1.5GB")
    try:
        path = snapshot_download(
            repo_id="OpenMuQ/MuQ-MuLan-large",
            cache_dir=CACHE_DIR,
            ignore_patterns=["*.md", "*.txt"]  # Пропускаем документацию
        )
        print(f"      ✅ Downloaded to: {path}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
    print("=" * 60)

    # Показываем размер
    try:
        import subprocess
        size = subprocess.check_output(["du", "-sh", CACHE_DIR]).decode().split()[0]
        print(f"\nTotal size: {size}")
    except:
        pass

    print(f"Location: {os.path.abspath(CACHE_DIR)}")

    return True


if __name__ == "__main__":
    # Парсинг аргументов
    include_full = "--no-full" not in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python3 download_models.py [--no-full]")
        print("\nOptions:")
        print("  --no-full    Skip downloading full model (saves ~3.5GB)")
        print("\nDownloads models to ./pretrained directory")
        sys.exit(0)

    print("Configuration:")
    print(f"  - Base model (95s): YES")
    print(f"  - Full model (285s): {'YES' if include_full else 'NO'}")
    print(f"  - VAE model: YES")
    print(f"  - MuQ-MuLan: YES")
    print(f"\nEstimated download: {'~9GB' if include_full else '~5.5GB'}")
    print("\nPress Ctrl+C to cancel, or wait 3 seconds...")

    try:
        import time
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)

    success = download_models(include_full=include_full)

    if success:
        print("\n🎉 Ready to build Docker image!")
        sys.exit(0)
    else:
        print("\n❌ Download failed. Check your internet connection.")
        sys.exit(1)