"""
Inference & Evaluation — generate images with the personalized model and
compute CLIP-T, CLIP-I, and DINO similarity metrics.

Pipeline:
  1. Load base SD1.5.
  2. Register the learned <mybear> token embedding (Stage 1).
  3. Load LoRA weights (Stage 2).
  4. Generate images for each test prompt.
  5. Evaluate:
       - CLIP-T  (text alignment):  cosine(CLIP_img(gen), CLIP_txt(prompt))
       - CLIP-I  (image alignment): cosine(CLIP_img(gen), CLIP_img(instance))
       - DINO-I  (identity):        cosine(DINO(gen), DINO(instance))
  6. Print summary table and save results.

Usage:
    python scripts/inference.py
    python scripts/inference.py --num_images_per_prompt 4
    python scripts/inference.py --skip_eval
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


# ── Image generation ─────────────────────────────────────────────────

def build_pipeline(model_path, tokenizer_path, embed_path, lora_path,
                   placeholder, device, dtype):
    """Load SD1.5, register the learned token, and attach LoRA."""
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # register placeholder token + learned embedding
    pipe.tokenizer.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    placeholder_id = pipe.tokenizer.convert_tokens_to_ids(placeholder)

    learned = torch.load(str(embed_path), map_location="cpu", weights_only=True)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[placeholder_id] = learned[placeholder]

    # load tokenizer that was saved with the embedding (has the new token)
    if Path(tokenizer_path).exists():
        from transformers import CLIPTokenizer
        pipe.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path))

    # load LoRA weights
    if lora_path and Path(lora_path).exists():
        pipe.unet.load_attn_procs(str(lora_path))
        print(f"  Loaded LoRA weights from {lora_path}")
    else:
        print("  WARNING: No LoRA weights found — running with TI only.")

    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def generate_images(pipe, prompts, output_dir, num_per_prompt, seed, size):
    """Generate images for each prompt and return {prompt: [PIL.Image]}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for prompt in prompts:
        images = []
        safe_name = prompt.replace(" ", "_")[:80]
        for i in range(num_per_prompt):
            generator = torch.Generator(device=pipe.device).manual_seed(seed + i)
            img = pipe(
                prompt,
                height=size,
                width=size,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            fname = f"{safe_name}_{i}.png"
            img.save(output_dir / fname)
            images.append(img)
        results[prompt] = images
        print(f"  Generated {num_per_prompt} image(s) for: {prompt}")

    return results


# ── Evaluation helpers ───────────────────────────────────────────────

def load_instance_images(instance_dir: Path, size: int = 224):
    """Load and preprocess instance reference images."""
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    images = []
    for p in sorted(instance_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))
    return torch.stack(images)


@torch.no_grad()
def compute_clip_metrics(generated: dict, instance_dir: Path,
                         placeholder: str, device: str):
    """
    Compute CLIP-T and CLIP-I.
      CLIP-T: cosine(CLIP_image(gen), CLIP_text(prompt_without_placeholder))
      CLIP-I: cosine(CLIP_image(gen), mean(CLIP_image(instance_imgs)))
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # precompute instance image features
    inst_imgs = []
    for p in sorted(instance_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            inst_imgs.append(Image.open(p).convert("RGB"))

    inst_inputs = clip_processor(images=inst_imgs, return_tensors="pt").to(device)
    inst_features = clip_model.get_image_features(**inst_inputs)
    inst_features = inst_features / inst_features.norm(dim=-1, keepdim=True)
    inst_mean = inst_features.mean(dim=0, keepdim=True)
    inst_mean = inst_mean / inst_mean.norm(dim=-1, keepdim=True)

    clip_t_scores = []
    clip_i_scores = []

    for prompt, images in generated.items():
        # remove placeholder for text alignment eval
        clean_prompt = prompt.replace(placeholder, "").replace("  ", " ").strip()

        for img in images:
            # image features
            img_inputs = clip_processor(images=[img], return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # CLIP-T
            txt_inputs = clip_processor(text=[clean_prompt], return_tensors="pt",
                                        padding=True).to(device)
            txt_feat = clip_model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            clip_t = (img_feat * txt_feat).sum().item()
            clip_t_scores.append(clip_t)

            # CLIP-I
            clip_i = (img_feat * inst_mean).sum().item()
            clip_i_scores.append(clip_i)

    del clip_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "CLIP-T": sum(clip_t_scores) / len(clip_t_scores),
        "CLIP-I": sum(clip_i_scores) / len(clip_i_scores),
        "CLIP-T_per_prompt": clip_t_scores,
        "CLIP-I_per_prompt": clip_i_scores,
    }


@torch.no_grad()
def compute_dino_similarity(generated: dict, instance_dir: Path, device: str):
    """
    Compute DINO-I: cosine(DINO(gen), mean(DINO(instance_imgs))).
    Uses DINOv2-small for efficiency.
    """
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dino_model.eval()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # instance features
    inst_feats = []
    for p in sorted(instance_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            feat = dino_model(tensor)
            inst_feats.append(feat)
    inst_feats = torch.cat(inst_feats)
    inst_feats = inst_feats / inst_feats.norm(dim=-1, keepdim=True)
    inst_mean = inst_feats.mean(dim=0, keepdim=True)
    inst_mean = inst_mean / inst_mean.norm(dim=-1, keepdim=True)

    dino_scores = []
    for images in generated.values():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            feat = dino_model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (feat * inst_mean).sum().item()
            dino_scores.append(sim)

    del dino_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "DINO-I": sum(dino_scores) / len(dino_scores),
        "DINO-I_per_image": dino_scores,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference & Evaluation")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--num_images_per_prompt", type=int, default=3,
                        help="Number of images to generate per test prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, only generate images.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Override LoRA weights path.")
    parser.add_argument("--embed_path", type=str, default=None,
                        help="Override learned embedding path.")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Override test prompts file.")
    args = parser.parse_args()

    # ── config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_path = project_root / cfg["paths"]["sd15_model"]
    instance_dir = project_root / cfg["paths"]["processed_instance_dir"]
    output_dir = project_root / cfg["paths"]["inference_samples"]
    output_dir.mkdir(parents=True, exist_ok=True)

    placeholder = cfg["placeholder_token"]
    size = cfg["training"]["image_size"]

    embed_path = Path(args.embed_path) if args.embed_path else (
        project_root / cfg["paths"]["textual_inversion_output"] / "learned_embeds.pt"
    )
    tokenizer_path = (
        project_root / cfg["paths"]["textual_inversion_output"] / "tokenizer"
    )
    lora_path = Path(args.lora_path) if args.lora_path else (
        project_root / cfg["paths"]["lora_output"] / "final"
    )

    prompts_file = Path(args.prompts_file) if args.prompts_file else (
        project_root / "data" / "prompts" / "test_prompts.txt"
    )
    prompts = [
        line.strip() for line in prompts_file.read_text().splitlines()
        if line.strip()
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("Inference & Evaluation")
    print("=" * 60)
    print(f"  Model       : {model_path}")
    print(f"  Embedding   : {embed_path}")
    print(f"  LoRA        : {lora_path}")
    print(f"  Prompts     : {prompts_file} ({len(prompts)} prompts)")
    print(f"  Per prompt  : {args.num_images_per_prompt} images")
    print(f"  Output      : {output_dir}")
    print(f"  Device      : {device}")
    print()

    # ── build pipeline ───────────────────────────────────────────────
    print("[1/3] Building personalized pipeline ...")
    pipe = build_pipeline(
        model_path, tokenizer_path, embed_path, lora_path,
        placeholder, device, dtype,
    )

    # ── generate ─────────────────────────────────────────────────────
    print(f"[2/3] Generating images ...")
    generated = generate_images(
        pipe, prompts, output_dir,
        num_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        size=size,
    )

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    total_gen = sum(len(imgs) for imgs in generated.values())
    print(f"  Total: {total_gen} images saved to {output_dir}")

    # ── evaluate ─────────────────────────────────────────────────────
    if args.skip_eval:
        print("\nSkipping evaluation (--skip_eval).")
        return

    print(f"[3/3] Computing evaluation metrics ...")

    print("  Computing CLIP-T and CLIP-I ...")
    clip_results = compute_clip_metrics(generated, instance_dir, placeholder, device)

    print("  Computing DINO-I ...")
    dino_results = compute_dino_similarity(generated, instance_dir, device)

    # ── print summary ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  CLIP-T  (text alignment)     : {clip_results['CLIP-T']:.4f}")
    print(f"  CLIP-I  (image alignment)    : {clip_results['CLIP-I']:.4f}")
    print(f"  DINO-I  (identity fidelity)  : {dino_results['DINO-I']:.4f}")
    print("=" * 60)

    # ── save metrics to JSON ─────────────────────────────────────────
    metrics = {
        "CLIP-T": clip_results["CLIP-T"],
        "CLIP-I": clip_results["CLIP-I"],
        "DINO-I": dino_results["DINO-I"],
        "num_prompts": len(prompts),
        "num_images_per_prompt": args.num_images_per_prompt,
        "total_images": total_gen,
        "prompts": prompts,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
