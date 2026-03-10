"""
Download Stable Diffusion v1.5 weights from Hugging Face Hub.

Prerequisites:
    1. pip install -r requirements.txt
    2. Run `hf auth login` or set HF_TOKEN env var

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --token YOUR_HF_TOKEN
"""

import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    parser = argparse.ArgumentParser(
        description="Download SD1.5 weights to the project model directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (falls back to cached login).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_dir = project_root / cfg["paths"]["sd15_model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Repo       : {REPO_ID}")
    print(f"Destination: {model_dir}")
    print("Downloading (this may take a while) ...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(model_dir),
        token=token,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "feature_extractor/preprocessor_config.json",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    )

    print(f"\nDone. Model saved to {model_dir}")


if __name__ == "__main__":
    main()
