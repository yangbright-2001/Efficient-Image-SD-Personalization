"""
Stage 1 — Textual Inversion: learn a placeholder token embedding for <mybear>.

Algorithm (from the project proposal):
  (a) Append a new embedding row vc to the text-encoder vocabulary.
  (b) Freeze every parameter except vc.
  (c) For each training step:
        - Sample a mini-batch from the instance set; apply mild augmentation.
        - Encode latents z0 = E(a(x)); pick timestep k ~ U[1,...,T].
        - Diffuse to zk; compute noise loss.
        - Update vc via gradient descent.

Usage:
    python scripts/train_textual_inversion.py
    python scripts/train_textual_inversion.py --steps 2000 --lr 5e-4
    python scripts/train_textual_inversion.py --save_every 250
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# ── Dataset ──────────────────────────────────────────────────────────

class InstanceDataset(Dataset):
    """Loads preprocessed 512x512 instance images with augmentation."""

    def __init__(self, image_dir: Path, prompt: str, tokenizer, size: int = 512):
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

        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        pixel_values = self.aug(img)

        input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ── Helpers ──────────────────────────────────────────────────────────

def add_placeholder_token(tokenizer, text_encoder, placeholder: str, initializer: str):
    """Add <placeholder> to the tokenizer and initialize its embedding."""
    num_added = tokenizer.add_tokens([placeholder])
    if num_added == 0:
        print(f"  Token '{placeholder}' already exists in tokenizer.")

    text_encoder.resize_token_embeddings(len(tokenizer))

    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    init_ids = tokenizer.encode(initializer, add_special_tokens=False)
    with torch.no_grad():
        init_emb = text_encoder.get_input_embeddings().weight[init_ids].mean(dim=0)
        text_encoder.get_input_embeddings().weight[placeholder_id] = init_emb

    return placeholder_id


def freeze_all_except_token(text_encoder, vae, unet, placeholder_id):
    """Freeze every parameter; only the placeholder row will receive gradients."""
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    token_embeds = text_encoder.get_input_embeddings()
    token_embeds.weight.requires_grad_(True)

    return token_embeds


def save_embedding(text_encoder, placeholder_id, placeholder: str, save_path: Path):
    """Save the learned token embedding to disk."""
    emb = text_encoder.get_input_embeddings().weight[placeholder_id].detach().cpu()
    torch.save({placeholder: emb}, save_path)
    print(f"  Saved embedding -> {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Textual Inversion")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps from config.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for the token embedding.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save a checkpoint every N steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initializer_word", type=str, default="teddy",
                        help="Word whose embedding initializes the placeholder.")
    args = parser.parse_args()

    # ── config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    model_path = project_root / cfg["paths"]["sd15_model"]
    instance_dir = project_root / cfg["paths"]["processed_instance_dir"]
    output_dir = project_root / cfg["paths"]["textual_inversion_output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    placeholder = cfg["placeholder_token"]
    prompt = cfg["prompts"]["instance"]
    total_steps = args.steps or cfg["training"]["textual_inversion_steps"]
    size = cfg["training"]["image_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Stage 1 — Textual Inversion")
    print("=" * 60)
    print(f"  Model        : {model_path}")
    print(f"  Instance dir : {instance_dir}")
    print(f"  Placeholder  : {placeholder}")
    print(f"  Prompt       : {prompt}")
    print(f"  Steps        : {total_steps}")
    print(f"  LR           : {args.lr}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Device       : {device}  dtype: {dtype}")
    print()

    # ── load model components ────────────────────────────────────────
    print("[1/5] Loading model components ...")
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

    # ── add placeholder token ────────────────────────────────────────
    print("[2/5] Adding placeholder token ...")
    placeholder_id = add_placeholder_token(
        tokenizer, text_encoder, placeholder, args.initializer_word,
    )
    print(f"  '{placeholder}' -> token id {placeholder_id}")

    # ── freeze ───────────────────────────────────────────────────────
    print("[3/5] Freezing all parameters except placeholder embedding ...")
    token_embeds = freeze_all_except_token(text_encoder, vae, unet, placeholder_id)
    token_embeds.to(torch.float32)

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    # ── dataset / dataloader ─────────────────────────────────────────
    print("[4/5] Building dataset ...")
    dataset = InstanceDataset(instance_dir, prompt, tokenizer, size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"  {len(dataset)} instance images, batch size {args.batch_size}")

    # ── optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [token_embeds.weight],
        lr=args.lr,
        weight_decay=1e-2,
    )

    # ── training loop ────────────────────────────────────────────────
    print(f"[5/5] Training for {total_steps} steps ...")
    print()

    orig_embeds = token_embeds.weight.data.clone()

    global_step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)

    while global_step < total_steps:
        # cycle through the small dataset
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        input_ids = batch["input_ids"].to(device)

        # encode image -> latent
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

        # sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device,
        ).long()

        # diffuse: z0 -> zt
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # forward pass under autocast (fp32 embedding feeds into fp16 layers)
        with torch.cuda.amp.autocast():
            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # MSE loss in fp32 for numerical stability
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        loss.backward()

        # zero out gradients for all tokens except the placeholder
        with torch.no_grad():
            grads = token_embeds.weight.grad
            if grads is not None:
                mask = torch.ones_like(grads, dtype=torch.bool)
                mask[placeholder_id] = False
                grads[mask] = 0.0

        optimizer.step()
        optimizer.zero_grad()

        # restore original embeddings for all tokens except placeholder
        with torch.no_grad():
            token_embeds.weight[: placeholder_id] = orig_embeds[: placeholder_id]
            if placeholder_id + 1 < len(orig_embeds):
                token_embeds.weight[placeholder_id + 1:] = orig_embeds[placeholder_id + 1:]

        running_loss += loss.item()
        global_step += 1

        if global_step % 50 == 0:
            avg = running_loss / 50
            print(f"  step {global_step:5d}/{total_steps}  loss={avg:.4f}")
            running_loss = 0.0

        if global_step % args.save_every == 0:
            ckpt_path = output_dir / f"learned_embeds_step{global_step}.pt"
            save_embedding(text_encoder, placeholder_id, placeholder, ckpt_path)

    # ── final save ───────────────────────────────────────────────────
    final_path = output_dir / "learned_embeds.pt"
    save_embedding(text_encoder, placeholder_id, placeholder, final_path)

    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    print(f"  Saved tokenizer -> {output_dir / 'tokenizer'}")

    print(f"\nDone. Final embedding saved to {final_path}")


if __name__ == "__main__":
    main()
