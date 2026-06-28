"""
Dataset generator for LoRA training.

Creates synthetic dataset pairs for testing training pipeline.

Usage:
    # Generate random pairs
    python -m training.prepare_dataset generate --count 100 --output ./data/test
    
    # Validate existing dataset
    python -m training.prepare_dataset validate --path ./data/test
    
    # Split dataset into train/val
    python -m training.prepare_dataset split --path ./data/test --ratio 0.9
"""

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("training.dataset_generator")


# =============================================================================
# Image Generators
# =============================================================================

def generate_isometric_tile(
    size: int = 512,
    style: str = "isometric_nyc",
    complexity: int = 5,
) -> Image.Image:
    """
    Generate a synthetic isometric tile.
    
    This creates a basic isometric-style image for testing.
    """
    # Create base
    img = Image.new("RGB", (size, size), (60, 60, 70))
    draw = ImageDraw.Draw(img)
    
    # Draw random buildings/structures
    for _ in range(complexity):
        # Random position
        x = random.randint(0, size - 100)
        y = random.randint(0, size - 100)
        
        # Random size
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        
        # Random color (NYC palette)
        colors = [
            (180, 160, 140),  # Tan building
            (120, 110, 100),  # Brown building
            (80, 90, 100),    # Blue-gray building
            (100, 80, 70),    # Dark brown
            (140, 130, 120),  # Light gray
        ]
        color = random.choice(colors)
        
        # Draw building shape
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(50, 50, 60))
        
        # Add windows
        for wy in range(y + 10, y + h - 10, 15):
            for wx in range(x + 10, x + w - 10, 15):
                if random.random() > 0.3:
                    window_color = (200, 190, 150) if random.random() > 0.5 else (80, 80, 100)
                    draw.rectangle([wx, wy, wx + 8, wy + 8], fill=window_color)
    
    # Add some ground/road
    road_color = (80, 80, 90)
    draw.rectangle([0, size - 80, size, size], fill=road_color)
    
    # Add some greenery
    for _ in range(10):
        gx = random.randint(0, size - 20)
        gy = random.randint(0, size - 20)
        g_size = random.randint(10, 30)
        draw.ellipse([gx, gy, gx + g_size, gy + g_size], fill=(60, 100, 60))
    
    return img


def generate_control_image(
    target: Image.Image,
    mode: str = "blur",
) -> Image.Image:
    """
    Generate a control/input image from target.
    
    Modes:
        - blur: Apply gaussian blur
        - sketch: Extract edges
        - grayscale: Convert to grayscale
        - reduced: Reduce detail
    """
    if mode == "blur":
        return target.filter(ImageFilter.GaussianBlur(radius=5))
    
    elif mode == "sketch":
        # Convert to grayscale
        gray = target.convert("L")
        # Edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        # Invert
        edges = Image.eval(edges, lambda x: 255 - x)
        # Convert back to RGB
        return Image.merge("RGB", [edges, edges, edges])
    
    elif mode == "grayscale":
        gray = target.convert("L")
        return Image.merge("RGB", [gray, gray, gray])
    
    elif mode == "reduced":
        # Reduce color palette
        result = target.convert("P", palette=Image.ADAPTIVE, colors=16)
        return result.convert("RGB")
    
    else:
        return target


# =============================================================================
# Dataset Generator
# =============================================================================

class DatasetGenerator:
    """Generate synthetic datasets for LoRA training."""
    
    def __init__(
        self,
        output_dir: str,
        count: int = 100,
        size: int = 512,
        prompt_template: str = "Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, seamlessly blending edges with surrounding areas, maintaining consistent isometric perspective, shadow direction, lighting, pixel density, and color harmony while preserving structural integrity and removing all border artifacts",
    ):
        self.output_dir = Path(output_dir)
        self.count = count
        self.size = size
        self.prompt_template = prompt_template
        
        # Create directories
        self.images_dir = self.output_dir / "images"
        self.controls_dir = self.output_dir / "controls"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.controls_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, control_mode: str = "blur") -> str:
        """
        Generate dataset.
        
        Returns path to metadata file.
        """
        log.info(f"Generating {self.count} image pairs...")
        
        metadata = []
        
        for i in range(self.count):
            # Generate target image
            target = generate_isometric_tile(
                size=self.size,
                complexity=random.randint(3, 8),
            )
            
            # Generate control image
            control = generate_control_image(target, mode=control_mode)
            
            # Save images
            img_name = f"image_{i:04d}.png"
            ctrl_name = f"control_{i:04d}.png"
            
            target.save(self.images_dir / img_name)
            control.save(self.controls_dir / ctrl_name)
            
            # Add to metadata
            prompt = f"{self.prompt_template} with {random.choice(['buildings', 'streets', 'parks', 'shops'])}"
            metadata.append({
                "image": f"images/{img_name}",
                "control_image": f"controls/{ctrl_name}",
                "prompt": prompt,
            })
            
            if (i + 1) % 10 == 0:
                log.info(f"  Generated {i + 1}/{self.count}")
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.csv"
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["image", "control_image", "prompt"]
            )
            writer.writeheader()
            writer.writerows(metadata)
        
        log.info(f"Dataset saved to: {self.output_dir}")
        log.info(f"  Images: {self.images_dir}")
        log.info(f"  Controls: {self.controls_dir}")
        log.info(f"  Metadata: {metadata_path}")
        
        return str(metadata_path)


def validate_dataset(path: str) -> Dict:
    """Validate a dataset."""
    from training.dataset import validate_dataset
    
    base_path = Path(path)
    
    # Check for metadata
    metadata_path = base_path / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = base_path / "metadata.jsonl"
    
    if not metadata_path.exists():
        log.error(f"No metadata file found in {path}")
        return {"error": "No metadata file"}
    
    # Validate
    stats = validate_dataset(
        base_path=str(base_path),
        metadata_path=str(metadata_path),
    )
    
    # Print results
    log.info("=" * 50)
    log.info("DATASET VALIDATION RESULTS")
    log.info("=" * 50)
    log.info(f"  Total entries:    {stats['total']}")
    log.info(f"  Valid entries:    {stats['valid']}")
    log.info(f"  Invalid:          {stats['total'] - stats['valid']}")
    
    if stats["missing_images"]:
        log.warning(f"  Missing images:   {len(stats['missing_images'])}")
        for p in stats["missing_images"][:5]:
            log.warning(f"    - {p}")
    
    if stats["missing_controls"]:
        log.warning(f"  Missing controls: {len(stats['missing_controls'])}")
        for p in stats["missing_controls"][:5]:
            log.warning(f"    - {p}")
    
    if stats["missing_prompts"]:
        log.warning(f"  Missing prompts:  {len(stats['missing_prompts'])}")
    
    log.info("=" * 50)
    
    return stats


def split_dataset(
    path: str,
    train_ratio: float = 0.9,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Split dataset into train/validation sets.
    
    Returns (train_metadata_path, val_metadata_path)
    """
    from training.dataset import IsometricDataset
    
    base_path = Path(path)
    
    if output_dir is None:
        output_dir = base_path.parent / f"{base_path.name}_split"
    else:
        output_dir = Path(output_dir)
    
    # Find metadata
    metadata_path = base_path / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = base_path / "metadata.jsonl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata file in {path}")
    
    # Load entries
    if metadata_path.suffix == ".csv":
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            entries = list(reader)
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
    
    # Shuffle
    random.shuffle(entries)
    
    # Split
    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    def copy_images(entries_list, dest_dir):
        for entry in entries_list:
            # Copy image
            src_img = base_path / entry["image"]
            dst_img = dest_dir / entry["image"]
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Copy control
            src_ctrl = base_path / entry["control_image"]
            dst_ctrl = dest_dir / entry["control_image"]
            dst_ctrl.parent.mkdir(parents=True, exist_ok=True)
            if src_ctrl.exists():
                shutil.copy2(src_ctrl, dst_ctrl)
    
    copy_images(train_entries, train_dir)
    copy_images(val_entries, val_dir)
    
    # Save metadata
    train_meta = train_dir / "metadata.csv"
    val_meta = val_dir / "metadata.csv"
    
    with open(train_meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "control_image", "prompt"])
        writer.writeheader()
        writer.writerows(train_entries)
    
    with open(val_meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "control_image", "prompt"])
        writer.writeheader()
        writer.writerows(val_entries)
    
    log.info(f"Dataset split complete!")
    log.info(f"  Train: {len(train_entries)} samples -> {train_meta}")
    log.info(f"  Val:   {len(val_entries)} samples -> {val_meta}")
    
    return str(train_meta), str(val_meta)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset preparation tools for LoRA training"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--count", type=int, default=100, help="Number of pairs")
    gen_parser.add_argument("--size", type=int, default=512, help="Image size")
    gen_parser.add_argument("--output", type=str, required=True, help="Output directory")
    gen_parser.add_argument("--control-mode", type=str, default="blur", 
                          choices=["blur", "sketch", "grayscale", "reduced"],
                          help="Control image generation mode")
    
    # Validate
    val_parser = subparsers.add_parser("validate", help="Validate dataset")
    val_parser.add_argument("--path", type=str, required=True, help="Dataset path")
    
    # Split
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--path", type=str, required=True, help="Dataset path")
    split_parser.add_argument("--ratio", type=float, default=0.9, help="Train ratio")
    split_parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "generate":
        gen = DatasetGenerator(
            output_dir=args.output,
            count=args.count,
            size=args.size,
        )
        gen.generate(control_mode=args.control_mode)
    
    elif args.command == "validate":
        validate_dataset(args.path)
    
    elif args.command == "split":
        split_dataset(args.path, train_ratio=args.ratio, output_dir=args.output)


if __name__ == "__main__":
    main()
