"""
Stage 2 — Targeted LoRA Fine-Tuning.

Algorithm (from the project proposal):
  (a) Insert LoRA blocks (rank r) in selected UNet cross-attention projections.
  (b) Freeze vc and every base weight; optimize only LoRA parameters ψ.
  (c) For each training step:
        - Sample half a batch from D (instance) and half from Dclass (prior).
        - Augment instance images.
        - Total loss  Lt = Linst(ψ) + λ_prior · Lclass(ψ).
        - Update ψ via gradient descent.
  (d) Every N_val steps, generate a validation grid.

Usage:
    python scripts/train_lora.py
    python scripts/train_lora.py --steps 1500 --lr 1e-4
    python scripts/train_lora.py --rank 16 --batch_size 2
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig


# ── Datasets ─────────────────────────────────────────────────────────

class PersonalizationDataset(Dataset):
    """Dataset that loads preprocessed 512x512 images with a fixed prompt."""

    def __init__(self, image_dir: Path, prompt: str, tokenizer,
                 size: int = 512, augment: bool = False):
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.images = [Image.open(p).convert("RGB") for p in image_paths]
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.size = size

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        pixel_values = self.transform(img)

        input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ── Helpers ──────────────────────────────────────────────────────────

def load_learned_embedding(text_encoder, tokenizer, embed_path: Path,
                           placeholder: str):
    """Load the Stage-1 learned token embedding and register it."""
    num_added = tokenizer.add_tokens([placeholder])
    text_encoder.resize_token_embeddings(len(tokenizer))

    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    learned = torch.load(embed_path, map_location="cpu", weights_only=True)
    emb_vec = learned[placeholder]
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[placeholder_id] = emb_vec

    print(f"  Loaded embedding for '{placeholder}' from {embed_path}")
    return placeholder_id


def discover_target_modules(unet, block_prefixes):
    """
    Find cross-attention projection names in specified UNet blocks.
    Returns module names relative to the UNet, suitable for peft LoraConfig.
    """
    cross_attn_suffixes = (".to_q", ".to_k", ".to_v", ".to_out.0")
    targets = []
    for name, _ in unet.named_modules():
        in_block = any(name.startswith(bp) for bp in block_prefixes)
        is_proj = "attn2" in name and name.endswith(cross_attn_suffixes)
        if in_block and is_proj:
            targets.append(name)
    return targets


def compute_noise_loss(pixel_values, input_ids, vae, text_encoder, unet,
                       noise_scheduler, device, dtype):
    """Shared helper: encode -> diffuse -> predict noise -> MSE."""
    pixel_values = pixel_values.to(device, dtype=dtype)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=device,
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.cuda.amp.autocast():
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")


def generate_validation_grid(pipe, placeholder, save_path, seed=42):
    """Generate a small set of images to visually check training progress."""
    prompts = [
        f"a photo of {placeholder} teddy bear",
        f"a photo of {placeholder} teddy bear on the beach",
        f"a shiny {placeholder} teddy bear",
        f"a photo of {placeholder} teddy bear in the snow",
    ]
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    images = []
    for p in prompts:
        img = pipe(p, num_inference_steps=25, guidance_scale=7.5,
                   generator=generator).images[0]
        images.append(img)

    grid_w = images[0].width * len(images)
    grid_h = images[0].height
    grid = Image.new("RGB", (grid_w, grid_h))
    for i, img in enumerate(images):
        grid.paste(img, (i * img.width, 0))
    grid.save(save_path)
    print(f"  Validation grid -> {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Targeted LoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Total LoRA training steps.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for LoRA parameters.")
    parser.add_argument("--rank", type=int, default=None,
                        help="LoRA rank (overrides config).")
    parser.add_argument("--prior_weight", type=float, default=None,
                        help="λ_prior (overrides config).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-source batch size (total = 2x this).")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps.")
    parser.add_argument("--val_every", type=int, default=500,
                        help="Generate validation grid every N steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_path", type=str, default=None,
                        help="Path to learned embedding (default: auto from config).")
    args = parser.parse_args()

    # ── config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_path = project_root / cfg["paths"]["sd15_model"]
    instance_dir = project_root / cfg["paths"]["processed_instance_dir"]
    prior_dir = project_root / cfg["paths"]["processed_prior_dir"]
    output_dir = project_root / cfg["paths"]["lora_output"]
    val_dir = project_root / cfg["paths"]["validation_samples"]
    output_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    placeholder = cfg["placeholder_token"]
    instance_prompt = cfg["prompts"]["instance"]
    prior_prompt = cfg["prompts"]["prior"]
    size = cfg["training"]["image_size"]
    rank = args.rank or cfg["training"]["lora_rank"]
    lambda_prior = args.prior_weight if args.prior_weight is not None else cfg["training"]["prior_weight"]

    embed_path = Path(args.embed_path) if args.embed_path else (
        project_root / cfg["paths"]["textual_inversion_output"] / "learned_embeds.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Stage 2 — Targeted LoRA Fine-Tuning")
    print("=" * 60)
    print(f"  Model         : {model_path}")
    print(f"  Instance dir  : {instance_dir}")
    print(f"  Prior dir     : {prior_dir}")
    print(f"  Embedding     : {embed_path}")
    print(f"  Placeholder   : {placeholder}")
    print(f"  LoRA rank     : {rank}")
    print(f"  λ_prior       : {lambda_prior}")
    print(f"  Steps         : {args.steps}")
    print(f"  LR            : {args.lr}")
    print(f"  Batch size    : {args.batch_size} inst + {args.batch_size} prior")
    print(f"  Device        : {device}  dtype: {dtype}")
    print()

    # ── load model components ────────────────────────────────────────
    print("[1/6] Loading model components ...")
    tokenizer = CLIPTokenizer.from_pretrained(str(model_path), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        str(model_path), subfolder="text_encoder", variant="fp16", torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        str(model_path), subfolder="vae", variant="fp16", torch_dtype=dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(model_path), subfolder="unet", variant="fp16", torch_dtype=dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        str(model_path), subfolder="scheduler",
    )

    # ── load Stage 1 embedding ───────────────────────────────────────
    print("[2/6] Loading learned embedding from Stage 1 ...")
    placeholder_id = load_learned_embedding(
        text_encoder, tokenizer, embed_path, placeholder,
    )

    # ── freeze base weights ──────────────────────────────────────────
    print("[3/6] Freezing base weights and adding LoRA ...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Discover cross-attention projections in targeted blocks:
    #   mid_block + the two deepest down/up blocks with cross-attention
    target_block_prefixes = ["mid_block", "down_blocks.2", "up_blocks.1"]
    target_modules = discover_target_modules(unet, target_block_prefixes)
    print(f"  Targeted {len(target_modules)} cross-attention projections:")
    for m in target_modules:
        print(f"    {m}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    unet.add_adapter(lora_config)

    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    # ── datasets ─────────────────────────────────────────────────────
    print("[4/6] Building datasets ...")
    instance_dataset = PersonalizationDataset(
        instance_dir, instance_prompt, tokenizer, size, augment=True,
    )
    prior_dataset = PersonalizationDataset(
        prior_dir, prior_prompt, tokenizer, size, augment=False,
    )
    instance_loader = DataLoader(
        instance_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    prior_loader = DataLoader(
        prior_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    print(f"  {len(instance_dataset)} instance images, {len(prior_dataset)} prior images")

    # ── optimizer ────────────────────────────────────────────────────
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)

    # ── training loop ────────────────────────────────────────────────
    print(f"[5/6] Training for {args.steps} steps ...")
    print()

    unet.train()
    inst_iter = iter(instance_loader)
    prior_iter = iter(prior_loader)
    running_loss = 0.0
    running_inst = 0.0
    running_prior = 0.0

    for step in range(1, args.steps + 1):
        # cycle iterators
        try:
            inst_batch = next(inst_iter)
        except StopIteration:
            inst_iter = iter(instance_loader)
            inst_batch = next(inst_iter)

        try:
            prior_batch = next(prior_iter)
        except StopIteration:
            prior_iter = iter(prior_loader)
            prior_batch = next(prior_iter)

        # L_inst: instance loss (with <mybear> prompt)
        loss_inst = compute_noise_loss(
            inst_batch["pixel_values"], inst_batch["input_ids"],
            vae, text_encoder, unet, noise_scheduler, device, dtype,
        )

        # L_class: prior-preservation loss (generic prompt)
        loss_prior = compute_noise_loss(
            prior_batch["pixel_values"], prior_batch["input_ids"],
            vae, text_encoder, unet, noise_scheduler, device, dtype,
        )

        # total loss = L_inst + λ_prior * L_class
        loss = loss_inst + lambda_prior * loss_prior

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        running_inst += loss_inst.item()
        running_prior += loss_prior.item()

        if step % 50 == 0:
            n = 50
            print(f"  step {step:5d}/{args.steps}  "
                  f"loss={running_loss/n:.4f}  "
                  f"inst={running_inst/n:.4f}  "
                  f"prior={running_prior/n:.4f}")
            running_loss = 0.0
            running_inst = 0.0
            running_prior = 0.0

        # save checkpoint
        if step % args.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unet.save_attn_procs(str(ckpt_dir))
            print(f"  Saved LoRA checkpoint -> {ckpt_dir}")

        # validation grid
        if step % args.val_every == 0:
            unet.eval()
            pipe = StableDiffusionPipeline.from_pretrained(
                str(model_path), torch_dtype=dtype, variant="fp16",
                safety_checker=None, requires_safety_checker=False,
            )
            pipe.tokenizer = tokenizer
            pipe.text_encoder = text_encoder
            pipe.unet = unet
            pipe = pipe.to(device)
            if device == "cuda":
                pipe.enable_attention_slicing()

            grid_path = val_dir / f"val_step{step}.png"
            generate_validation_grid(pipe, placeholder, grid_path, seed=args.seed)

            del pipe
            if device == "cuda":
                torch.cuda.empty_cache()
            unet.train()

    # ── final save ───────────────────────────────────────────────────
    print("[6/6] Saving final LoRA weights ...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unet.save_attn_procs(str(final_dir))
    print(f"  Saved LoRA weights -> {final_dir}")

    print(f"\nDone. LoRA training complete.")
    print(f"  LoRA weights : {final_dir}")
    print(f"  Embedding    : {embed_path}")
    print(f"  Validation   : {val_dir}")


if __name__ == "__main__":
    main()
