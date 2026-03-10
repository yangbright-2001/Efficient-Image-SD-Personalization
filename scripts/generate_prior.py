"""
Generate class-prior images using the base Stable Diffusion v1.5 model.

The generated images are saved to data/raw/prior/teddy_bear/ and can later
be preprocessed with scripts/preprocess.py.

Usage:
    python scripts/generate_prior.py                       # default 100 images
    python scripts/generate_prior.py --num_images 50       # generate 50
    python scripts/generate_prior.py --batch_size 2        # smaller batch for low VRAM
    python scripts/generate_prior.py --seed 42             # reproducible
"""

import argparse
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    parser = argparse.ArgumentParser(
        description="Generate class-prior images with SD1.5."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Total number of prior images to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Images per batch (lower if GPU OOM).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Denoising steps per image.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prior prompt from config.",
    )
    args = parser.parse_args()

    # ── load config ──────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_path = project_root / cfg["paths"]["sd15_model"]
    output_dir = project_root / cfg["paths"]["raw_prior_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt or cfg["prompts"]["prior"]
    size = cfg["training"]["image_size"]

    print(f"Model path : {model_path}")
    print(f"Output dir : {output_dir}")
    print(f"Prompt     : {prompt}")
    print(f"Num images : {args.num_images}")
    print(f"Batch size : {args.batch_size}")
    print(f"Resolution : {size}x{size}")

    # ── load pipeline ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # ── generate ─────────────────────────────────────────────────────
    generated = 0
    batch_idx = 0

    while generated < args.num_images:
        current_batch = min(args.batch_size, args.num_images - generated)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + batch_idx
        )

        images = pipe(
            prompt=[prompt] * current_batch,
            height=size,
            width=size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images

        for img in images:
            out_path = output_dir / f"prior_{generated:04d}.png"
            img.save(out_path)
            print(f"  Saved {out_path.name}")
            generated += 1

        batch_idx += 1

    print(f"\nDone. {generated} prior images saved to {output_dir}")


if __name__ == "__main__":
    main()
