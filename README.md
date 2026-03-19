# Efficient SD Personalization

Compute-efficient few-shot personalization pipeline for Stable Diffusion v1.5, combining **Textual Inversion** and **Targeted LoRA** to teach the model a specific subject (bear plushie) from only a handful of images.

## Method Overview

The pipeline personalizes Stable Diffusion in two lightweight stages:

1. **Stage 1 — Textual Inversion**: A new token `<mybear>` is added to the text encoder vocabulary. Only this single embedding vector is trained while all other parameters remain frozen, teaching the model to associate `<mybear>` with the target subject's visual appearance.
2. **Stage 2 — Targeted LoRA Fine-Tuning**: Low-rank adaptation matrices (LoRA) are inserted into selected UNet cross-attention layers (`mid_block`, `down_blocks.2`, `up_blocks.1`). Only the LoRA parameters are optimized, with a prior-preservation loss to prevent the model from forgetting what a generic teddy bear looks like.

After training, the model can generate the specific subject in novel scenes by simply using prompts containing `<mybear>`.

### Evaluation Metrics

- **CLIP-T** — text alignment: cosine similarity between CLIP image features of generated images and CLIP text features of the prompt.
- **CLIP-I** — image alignment: cosine similarity between CLIP image features of generated images and the mean CLIP image features of instance references.
- **DINO-I** — identity fidelity: cosine similarity between DINOv2 features of generated images and the mean DINOv2 features of instance references.

## Directory Layout

Outputs and ablation studies for the model with `rank=8`, `prior weight=1.0` is already included in `outputs/`

```text
.
├── configs/
│   └── project.yaml                # centralized project configuration
├── data/
│   ├── prompts/                    # prompt text files (see below)
│   │   ├── instance_prompt.txt
│   │   ├── prior_prompt.txt
│   │   └── test_prompts.txt
│   ├── raw/
│   │   ├── instance/bear_plushie/  # original instance photos
│   │   └── prior/teddy_bear/       # generated class-prior images
│   └── processed/
│       ├── instance_512/           # center-cropped & resized instance images
│       └── prior_512/              # center-cropped & resized prior images
├── docs/                           # notes & experiment writeups
├── logs/                           # training logs
├── models/
│   └── base/sd15/                  # local Stable Diffusion v1.5 weights (fp16)
├── notebooks/                      # exploratory notebooks
├── outputs/
│   ├── textual_inversion/          # learned embeddings & tokenizer
│   ├── lora/                       # LoRA checkpoints
│   ├── samples/
│   │   ├── inference/              # final generated images + metrics.json
│   │   └── validation/             # validation grids from LoRA training
│   ├── samples_base/               # ablation: base SD1.5 outputs
│   ├── samples_ti_only/            # ablation: TI-only outputs
│   └── samples_r8_pw1.0/           # experiment variant outputs
├── scripts/                        # all runnable scripts (see Workflow)
├── requirements.txt
└── README.md
```

## Prompts

All prompts are stored as plain text files under `data/prompts/` and referenced by the config.

| File                    | Content                                                                                           | Used By                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `instance_prompt.txt` | `a photo of <mybear> teddy bear`                                                                | Textual Inversion & LoRA training (instance branch)   |
| `prior_prompt.txt`    | `a photo of a teddy bear`                                                                       | Prior image generation & LoRA training (prior branch) |
| `test_prompts.txt`    | 5 evaluation prompts placing `<mybear>` in varied scenes (jungle, snow, beach, mountain, shiny) | Inference & ablation scripts                          |

The placeholder token `<mybear>` is defined in `configs/project.yaml` and automatically substituted at runtime.

## Configuration

All paths, prompts, and training hyperparameters are centralized in `configs/project.yaml`:

```yaml
placeholder_token: "<mybear>"
prompts:
  instance: "a photo of <mybear> teddy bear"
  prior: "a photo of a teddy bear"
training:
  image_size: 512
  textual_inversion_steps: 1500
  lora_rank: 8
  prior_weight: 1.0
```

Every script reads this config via `--config configs/project.yaml` (default), so we rarely need to pass CLI flags unless overriding specific values.

## Workflow

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU (tested on a single NVIDIA GPU with >= 12 GB VRAM). CPU execution is supported but significantly slower.

### 1. Download Stable Diffusion v1.5

Downloads the fp16 variant of SD1.5 from Hugging Face Hub into `models/base/sd15/`. Requires a Hugging Face token (via `huggingface-cli login` or `--token`).

```bash
python scripts/download_model.py
python scripts/download_model.py --token YOUR_HF_TOKEN   # explicit token
```

### 2. Prepare instance images

Place photos of the target subject into `data/raw/instance/bear_plushie/`, 5 images of `bear_plushie` from `Google/Dreambooth` dataset are placed.

### 3. Generate class-prior images

Uses the base SD1.5 to generate generic teddy bear images for prior-preservation regularization during LoRA training.

```bash
python scripts/generate_prior.py                     # default: 100 images
python scripts/generate_prior.py --num_images 50     # fewer images
python scripts/generate_prior.py --batch_size 2      # lower batch for limited VRAM
```

Output: `data/raw/prior/teddy_bear/`

### 4. Preprocess images

Center-crops to square and resizes both instance and prior images to 512x512.

```bash
python scripts/preprocess.py                  # process both instance & prior
python scripts/preprocess.py --only instance  # instance only
python scripts/preprocess.py --only prior     # prior only
python scripts/preprocess.py --size 256       # custom resolution
```

Output: `data/processed/instance_512/` and `data/processed/prior_512/`

### 5. Stage 1 — Textual Inversion

Learns the `<mybear>` token embedding. All model weights are frozen; only the new embedding vector is trained.

```bash
python scripts/train_textual_inversion.py
python scripts/train_textual_inversion.py --steps 2000 --lr 5e-4
python scripts/train_textual_inversion.py --save_every 250
```

Output: `outputs/textual_inversion/learned_embeds.pt` and `outputs/textual_inversion/tokenizer/`

### 6. Stage 2 — Targeted LoRA Fine-Tuning

Inserts LoRA into UNet cross-attention projections and trains with instance + prior-preservation loss. Generates validation grids periodically.

```bash
python scripts/train_lora.py
python scripts/train_lora.py --steps 1500 --lr 1e-4
python scripts/train_lora.py --rank 16 --batch_size 2
python scripts/train_lora.py --prior_weight 0.5      # adjust λ_prior
```

Output: `outputs/lora/final/` (LoRA weights) and `outputs/samples/validation/` (validation grids)

### 7. Inference & Evaluation

Loads the full personalized pipeline (SD1.5 + learned embedding + LoRA) and generates images for each test prompt. Computes CLIP-T, CLIP-I, and DINO-I metrics.

```bash
python scripts/inference.py
python scripts/inference.py --num_images_per_prompt 4
python scripts/inference.py --skip_eval               # generate only, skip metrics
```

Output: `outputs/samples/inference/` (images + `metrics.json`)

### 8. Ablation Studies

Run inference under ablation settings to isolate the contribution of each stage:

```bash
python scripts/ablation_inference.py --mode base      # plain SD1.5, no personalization
python scripts/ablation_inference.py --mode ti_only   # TI embedding only, no LoRA
```

Output: `outputs/samples_base/` and `outputs/samples_ti_only/` (images + `metrics.json`)

Compare the three conditions (base / TI-only / TI+LoRA) to quantify the improvement from each stage.

## References

- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
