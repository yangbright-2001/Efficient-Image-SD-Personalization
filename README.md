# ECE285 DGM Final Project

Minimal project skeleton for a compute-efficient few-shot personalization pipeline based on:

- `Stable Diffusion v1.5`
- `Textual Inversion`
- `Targeted LoRA`
- subject: `bear_plushie`

## Directory layout

```text
.
├── configs/                      # YAML/JSON config files
├── data/
│   ├── prompts/                  # training and evaluation prompts
│   ├── raw/
│   │   ├── instance/bear_plushie # original instance images
│   │   └── prior/teddy_bear      # generated or collected class prior images
│   └── processed/
│       ├── instance_512          # resized/cropped instance images
│       └── prior_512             # resized/cropped prior images
├── docs/                         # notes, experiment writeups
├── logs/                         # training logs
├── models/
│   └── base/sd15                 # local Stable Diffusion v1.5 weights
├── notebooks/                    # exploratory notebooks
├── outputs/
│   ├── lora/                     # LoRA checkpoints
│   ├── samples/
│   │   ├── inference             # final generated images
│   │   └── validation            # validation grids during training
│   └── textual_inversion         # learned token embeddings
└── scripts/                      # download, preprocess, train, infer scripts
```

## Planned prompts

- instance prompt: `a photo of <mybear> teddy bear`
- prior prompt: `a photo of a teddy bear`

## Recommended workflow

1. Put the `bear_plushie` instance images into `data/raw/instance/bear_plushie/`.
2. Download SD1.5 into `models/base/sd15/`.
3. Generate or collect teddy-bear prior images into `data/raw/prior/teddy_bear/`.
4. Preprocess both datasets into the `data/processed/` folders.
5. Run Textual Inversion and save outputs to `outputs/textual_inversion/`.
6. Run LoRA fine-tuning and save outputs to `outputs/lora/`.
7. Save validation and inference samples under `outputs/samples/`.
