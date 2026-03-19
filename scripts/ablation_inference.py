"""
Ablation Inference & Evaluation.

Run inference under different ablation settings and compute metrics.
Results are saved to separate output directories for comparison.

Modes:
  --mode base      : Base SD1.5, no personalization at all.
                     Uses prompts with placeholder replaced by class name.
  --mode ti_only   : Stage 1 only (learned embedding, no LoRA).

Usage:
    python scripts/ablation_inference.py --mode base
    python scripts/ablation_inference.py --mode ti_only
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor


# ── Pipeline builders ────────────────────────────────────────────────

def build_base_pipeline(model_path, device, dtype):
    """Plain SD1.5 — no embedding, no LoRA."""
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
    return pipe


def build_ti_only_pipeline(model_path, embed_path, tokenizer_path,
                           placeholder, device, dtype):
    """SD1.5 + learned token embedding, but NO LoRA."""
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.tokenizer.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    placeholder_id = pipe.tokenizer.convert_tokens_to_ids(placeholder)

    learned = torch.load(str(embed_path), map_location="cpu", weights_only=True)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[placeholder_id] = learned[placeholder]

    if Path(tokenizer_path).exists():
        from transformers import CLIPTokenizer
        pipe.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path))

    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


# ── Generation ───────────────────────────────────────────────────────

def generate_images(pipe, prompts, output_dir, num_per_prompt, seed, size):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for prompt in prompts:
        images = []
        safe_name = prompt.replace(" ", "_")[:80]
        for i in range(num_per_prompt):
            generator = torch.Generator(device=pipe.device).manual_seed(seed + i)
            img = pipe(
                prompt, height=size, width=size,
                num_inference_steps=30, guidance_scale=7.5,
                generator=generator,
            ).images[0]
            fname = f"{safe_name}_{i}.png"
            img.save(output_dir / fname)
            images.append(img)
        results[prompt] = images
        print(f"  Generated {num_per_prompt} image(s) for: {prompt}")
    return results


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_clip_metrics(generated, instance_dir, placeholder, device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    inst_imgs = []
    for p in sorted(instance_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            inst_imgs.append(Image.open(p).convert("RGB"))
    inst_inputs = clip_processor(images=inst_imgs, return_tensors="pt").to(device)
    inst_features = clip_model.get_image_features(**inst_inputs)
    inst_features = inst_features / inst_features.norm(dim=-1, keepdim=True)
    inst_mean = inst_features.mean(dim=0, keepdim=True)
    inst_mean = inst_mean / inst_mean.norm(dim=-1, keepdim=True)

    clip_t_scores, clip_i_scores = [], []
    for prompt, images in generated.items():
        clean_prompt = prompt.replace(placeholder, "").replace("  ", " ").strip()
        for img in images:
            img_inputs = clip_processor(images=[img], return_tensors="pt").to(device)
            img_feat = clip_model.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_inputs = clip_processor(text=[clean_prompt], return_tensors="pt",
                                        padding=True).to(device)
            txt_feat = clip_model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            clip_t_scores.append((img_feat * txt_feat).sum().item())
            clip_i_scores.append((img_feat * inst_mean).sum().item())

    del clip_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "CLIP-T": sum(clip_t_scores) / len(clip_t_scores),
        "CLIP-I": sum(clip_i_scores) / len(clip_i_scores),
    }


@torch.no_grad()
def compute_dino_similarity(generated, instance_dir, device):
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dino_model.eval()
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    inst_feats = []
    for p in sorted(instance_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            tensor = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            inst_feats.append(dino_model(tensor))
    inst_feats = torch.cat(inst_feats)
    inst_feats = inst_feats / inst_feats.norm(dim=-1, keepdim=True)
    inst_mean = inst_feats.mean(dim=0, keepdim=True)
    inst_mean = inst_mean / inst_mean.norm(dim=-1, keepdim=True)

    dino_scores = []
    for images in generated.values():
        for img in images:
            tensor = tf(img).unsqueeze(0).to(device)
            feat = dino_model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            dino_scores.append((feat * inst_mean).sum().item())

    del dino_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"DINO-I": sum(dino_scores) / len(dino_scores)}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation Inference & Eval")
    parser.add_argument("--mode", required=True, choices=["base", "ti_only"],
                        help="'base' = plain SD1.5; 'ti_only' = TI without LoRA.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--num_images_per_prompt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_path = project_root / cfg["paths"]["sd15_model"]
    instance_dir = project_root / cfg["paths"]["processed_instance_dir"]
    placeholder = cfg["placeholder_token"]
    size = cfg["training"]["image_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # output goes to a mode-specific directory
    output_dir = project_root / "outputs" / f"samples_{args.mode}" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load test prompts
    prompts_file = project_root / "data" / "prompts" / "test_prompts.txt"
    raw_prompts = [l.strip() for l in prompts_file.read_text().splitlines() if l.strip()]

    if args.mode == "base":
        # replace <mybear> with "a" so prompt becomes e.g. "a photo of a teddy bear ..."
        prompts = [p.replace(placeholder, "a") for p in raw_prompts]
        prompts = [p.replace("  ", " ") for p in prompts]
    else:
        prompts = raw_prompts

    print("=" * 60)
    print(f"Ablation: {args.mode}")
    print("=" * 60)
    print(f"  Output : {output_dir}")
    print(f"  Prompts: {len(prompts)}")
    for p in prompts:
        print(f"    {p}")
    print()

    # ── build pipeline ───────────────────────────────────────────────
    if args.mode == "base":
        print("[1/3] Loading base SD1.5 (no personalization) ...")
        pipe = build_base_pipeline(model_path, device, dtype)
    elif args.mode == "ti_only":
        print("[1/3] Loading SD1.5 + TI embedding (no LoRA) ...")
        embed_path = (project_root / cfg["paths"]["textual_inversion_output"]
                      / "learned_embeds.pt")
        tokenizer_path = (project_root / cfg["paths"]["textual_inversion_output"]
                          / "tokenizer")
        pipe = build_ti_only_pipeline(
            model_path, embed_path, tokenizer_path, placeholder, device, dtype,
        )

    # ── generate ─────────────────────────────────────────────────────
    print("[2/3] Generating images ...")
    generated = generate_images(
        pipe, prompts, output_dir,
        num_per_prompt=args.num_images_per_prompt, seed=args.seed, size=size,
    )
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    total = sum(len(v) for v in generated.values())
    print(f"  Total: {total} images saved to {output_dir}")

    if args.skip_eval:
        print("\nSkipping evaluation.")
        return

    # ── evaluate ─────────────────────────────────────────────────────
    print("[3/3] Computing metrics ...")
    # for CLIP-T, always strip placeholder so base and ti_only are comparable
    clip_res = compute_clip_metrics(generated, instance_dir, placeholder, device)
    dino_res = compute_dino_similarity(generated, instance_dir, device)

    print()
    print("=" * 60)
    print(f"Ablation Results  [{args.mode}]")
    print("=" * 60)
    print(f"  CLIP-T  : {clip_res['CLIP-T']:.4f}")
    print(f"  CLIP-I  : {clip_res['CLIP-I']:.4f}")
    print(f"  DINO-I  : {dino_res['DINO-I']:.4f}")
    print("=" * 60)

    metrics = {
        "mode": args.mode,
        "CLIP-T": clip_res["CLIP-T"],
        "CLIP-I": clip_res["CLIP-I"],
        "DINO-I": dino_res["DINO-I"],
        "num_prompts": len(prompts),
        "num_images_per_prompt": args.num_images_per_prompt,
        "total_images": total,
        "prompts": prompts,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
