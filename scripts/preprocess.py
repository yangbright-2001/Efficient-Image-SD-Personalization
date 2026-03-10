"""
Preprocess raw images: center-crop to square then resize to 512x512.

Usage:
    python scripts/preprocess.py                          # process both instance & prior
    python scripts/preprocess.py --only instance          # process instance only
    python scripts/preprocess.py --only prior             # process prior only
    python scripts/preprocess.py --size 256               # custom resolution
"""

import argparse
from pathlib import Path

import yaml
from PIL import Image


def center_crop_and_resize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to the largest square, then resize to (size, size)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def process_directory(src_dir: Path, dst_dir: Path, size: int) -> int:
    """Process all images in *src_dir* and write results to *dst_dir*.
    Returns the number of images processed.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    count = 0
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue
        img = Image.open(img_path).convert("RGB")
        img = center_crop_and_resize(img, size)
        out_path = dst_dir / f"{img_path.stem}.png"
        img.save(out_path, format="PNG")
        count += 1
        print(f"  {img_path.name} -> {out_path.name}  ({size}x{size})")
    return count


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw images to 512x512.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Target resolution (overrides config).",
    )
    parser.add_argument(
        "--only",
        choices=["instance", "prior"],
        default=None,
        help="Process only instance or prior images.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    size = args.size or cfg["training"]["image_size"]
    project_root = Path(args.config).resolve().parent.parent

    tasks = []
    if args.only != "prior":
        tasks.append((
            project_root / cfg["paths"]["raw_instance_dir"],
            project_root / cfg["paths"]["processed_instance_dir"],
            "instance",
        ))
    if args.only != "instance":
        tasks.append((
            project_root / cfg["paths"]["raw_prior_dir"],
            project_root / cfg["paths"]["processed_prior_dir"],
            "prior",
        ))

    for src, dst, label in tasks:
        print(f"\n[preprocess] {label}: {src} -> {dst}")
        if not src.exists():
            print(f"  WARNING: source directory {src} does not exist, skipping.")
            continue
        n = process_directory(src, dst, size)
        print(f"  Processed {n} images.")

    print("\nDone.")


if __name__ == "__main__":
    main()
