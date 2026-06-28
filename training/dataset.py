"""
Dataset handling for LoRA training.

Supports:
    - CSV metadata files
    - JSONL metadata files
    - Direct folder of image pairs
    - Validation with sample generation
"""

import io
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers.utils import make_image_grid

log = logging.getLogger("training.dataset")


# =============================================================================
# Dataset Entry
# =============================================================================

@dataclass
class DatasetEntry:
    """Single training example."""
    
    image_path: str          # Path to target/output image
    control_image_path: str   # Path to input/control image
    prompt: str              # Text prompt
    image: Optional[Image.Image] = None  # Loaded PIL image
    control_image: Optional[Image.Image] = None  # Loaded PIL image


# =============================================================================
# Isometric Dataset
# =============================================================================

class IsometricDataset(Dataset):
    """
    Dataset for isometric pixel art LoRA training.
    
    Format:
        - metadata.csv/jsonl with columns: image, control_image, prompt
        - Images in base_path directory
    
    Usage:
        dataset = IsometricDataset(
            base_path="./data",
            metadata_path="./data/metadata.csv",
            max_pixels=1048576,
            transform=train_transform,
        )
    """
    
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        image_column: str = "image",
        control_image_column: str = "control_image",
        prompt_column: str = "prompt",
        max_pixels: int = 1048576,
        min_pixels: Optional[int] = None,
        transform: Optional[Callable] = None,
        repeat: int = 1,
    ):
        self.base_path = Path(base_path)
        self.metadata_path = Path(metadata_path)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.transform = transform
        self.repeat = repeat
        
        # Column names
        self.image_col = image_column
        self.control_col = control_image_column
        self.prompt_col = prompt_column
        
        # Load metadata
        self.entries = self._load_metadata()
        
        log.info(f"Loaded {len(self.entries)} entries from {metadata_path}")
    
    def _load_metadata(self) -> List[DatasetEntry]:
        """Load metadata from CSV or JSONL file."""
        ext = self.metadata_path.suffix.lower()
        
        if ext == ".csv":
            return self._load_csv()
        elif ext in (".json", ".jsonl"):
            return self._load_jsonl()
        else:
            raise ValueError(f"Unsupported metadata format: {ext}")
    
    def _load_csv(self) -> List[DatasetEntry]:
        """Load CSV metadata."""
        import csv
        
        entries = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = DatasetEntry(
                    image_path=str(self.base_path / row[self.image_col]),
                    control_image_path=str(self.base_path / row[self.control_col]),
                    prompt=row[self.prompt_col],
                )
                entries.append(entry)
        
        return entries
    
    def _load_jsonl(self) -> List[DatasetEntry]:
        """Load JSONL metadata."""
        entries = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                entry = DatasetEntry(
                    image_path=str(self.base_path / data[self.image_col]),
                    control_image_path=str(self.base_path / data[self.control_col]),
                    prompt=data[self.prompt_col],
                )
                entries.append(entry)
        
        return entries
    
    def _load_image(self, path: str) -> Image.Image:
        """Load and preprocess image."""
        img = Image.open(path).convert("RGB")
        
        # Resize to fit max_pixels constraint
        w, h = img.size
        max_side = int(np.sqrt(self.max_pixels))
        
        if max(w, h) > max_side:
            if w > h:
                new_w, new_h = max_side, int(h * max_side / w)
            else:
                new_w, new_h = int(w * max_side / h), max_side
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        return img
    
    def __len__(self) -> int:
        return len(self.entries) * self.repeat
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training example."""
        # Handle repeat
        idx = idx % len(self.entries)
        
        entry = self.entries[idx]
        
        # Load images
        image = self._load_image(entry.image_path)
        control_image = self._load_image(entry.control_image_path)
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
            control_image = self.transform(control_image)
        
        return {
            "image": image,  # Target output
            "control_image": control_image,  # Input
            "prompt": entry.prompt,
            "image_path": entry.image_path,
        }
    
    def get_sample(self, idx: int) -> Tuple[Image.Image, Image.Image, str]:
        """Get a sample as PIL images for validation."""
        entry = self.entries[idx % len(self.entries)]
        
        image = self._load_image(entry.image_path)
        control_image = self._load_image(entry.control_image_path)
        
        return image, control_image, entry.prompt


# =============================================================================
# Validation Dataset
# =============================================================================

class ValidationDataset:
    """
    Simple validation dataset for generating samples during training.
    
    Loads a few examples and generates images for visual inspection.
    """
    
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        num_samples: int = 4,
        image_column: str = "image",
        control_image_column: str = "control_image",
        prompt_column: str = "prompt",
    ):
        self.dataset = IsometricDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            image_column=image_column,
            control_image_column=control_image_column,
            prompt_column=prompt_column,
        )
        self.num_samples = min(num_samples, len(self.dataset))
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get validation samples."""
        indices = random.sample(range(len(self.dataset)), self.num_samples)
        samples = []
        
        for idx in indices:
            img, control, prompt = self.dataset.get_sample(idx)
            samples.append({
                "image": img,
                "control_image": control,
                "prompt": prompt,
            })
        
        return samples
    
    def create_comparison_grid(
        self, 
        inputs: List[Image.Image], 
        outputs: List[Image.Image],
        prompts: List[str],
    ) -> Image.Image:
        """Create a comparison grid of input -> output."""
        grids = []
        for inp, out, prompt in zip(inputs, outputs, prompts):
            # Stack input and output horizontally
            row = Image.new("RGB", (inp.width * 2 + 10, inp.height), (128, 128, 128))
            row.paste(inp, (0, 0))
            row.paste(out, (inp.width + 10, 0))
            grids.append(row)
        
        # Stack all rows vertically
        total_height = sum(g.height for g in grids) + (len(grids) - 1) * 10
        total_width = max(g.width for g in grids)
        
        grid = Image.new("RGB", (total_width, total_height), (128, 128, 128))
        y_offset = 0
        for g in grids:
            grid.paste(g, (0, y_offset))
            y_offset += g.height + 10
        
        return grid


# =============================================================================
# Dataset Creation Helpers
# =============================================================================

def create_dataset_from_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    base_path: Optional[str] = None,
) -> str:
    """
    Create a dataset metadata file from a CSV.
    
    If base_path is not specified, uses the directory containing the CSV.
    """
    import csv
    
    csv_path = Path(csv_path)
    
    if base_path is None:
        base_path = csv_path.parent
    
    if output_path is None:
        output_path = csv_path.parent / "metadata.csv"
    
    # Read existing CSV to validate
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Validate columns
    if not rows:
        raise ValueError("CSV file is empty")
    
    required_cols = {"image", "control_image", "prompt"}
    existing_cols = set(rows[0].keys())
    missing = required_cols - existing_cols
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Copy to output path if different
    if str(output_path) != str(csv_path):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        
        log.info(f"Created dataset metadata: {output_path}")
    
    return str(output_path)


def validate_dataset(
    base_path: str,
    metadata_path: str,
    image_column: str = "image",
    control_image_column: str = "control_image",
    prompt_column: str = "prompt",
) -> Dict[str, Any]:
    """
    Validate a dataset and return statistics.
    
    Returns:
        Dict with keys: total, valid, missing_images, missing_controls, missing_prompts
    """
    import csv
    
    base_path = Path(base_path)
    metadata_path = Path(metadata_path)
    
    stats = {
        "total": 0,
        "valid": 0,
        "missing_images": [],
        "missing_controls": [],
        "missing_prompts": [],
    }
    
    ext = metadata_path.suffix.lower()
    
    if ext == ".csv":
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["total"] += 1
                
                img_path = base_path / row[image_column]
                ctrl_path = base_path / row[control_image_column]
                prompt = row.get(prompt_column, "").strip()
                
                if not img_path.exists():
                    stats["missing_images"].append(str(img_path))
                if not ctrl_path.exists():
                    stats["missing_controls"].append(str(ctrl_path))
                if not prompt:
                    stats["missing_prompts"].append(str(img_path))
                
                if img_path.exists() and ctrl_path.exists() and prompt:
                    stats["valid"] += 1
    
    elif ext in (".json", ".jsonl"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                stats["total"] += 1
                
                img_path = base_path / data[image_column]
                ctrl_path = base_path / data[control_image_column]
                prompt = data.get(prompt_column, "").strip()
                
                if not img_path.exists():
                    stats["missing_images"].append(str(img_path))
                if not ctrl_path.exists():
                    stats["missing_controls"].append(str(ctrl_path))
                if not prompt:
                    stats["missing_prompts"].append(str(img_path))
                
                if img_path.exists() and ctrl_path.exists() and prompt:
                    stats["valid"] += 1
    
    return stats


def create_example_dataset(output_dir: str, num_examples: int = 10):
    """
    Create an example dataset for testing.
    
    This generates random images as placeholders.
    """
    from PIL import Image, ImageDraw
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images folder
    images_dir = output_dir / "images"
    controls_dir = output_dir / "controls"
    images_dir.mkdir(exist_ok=True)
    controls_dir.mkdir(exist_ok=True)
    
    rows = []
    
    for i in range(num_examples):
        # Create random control image (input)
        ctrl_img = Image.new("RGB", (512, 512), (
            random.randint(50, 150),
            random.randint(50, 150),
            random.randint(50, 150),
        ))
        ctrl_draw = ImageDraw.Draw(ctrl_img)
        
        # Add some random shapes
        for _ in range(random.randint(5, 15)):
            x1, y1 = random.randint(0, 480), random.randint(0, 480)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
            ctrl_draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Create target image (slightly modified)
        target_img = ctrl_img.copy()
        target_draw = ImageDraw.Draw(target_img)
        
        # Add more shapes to make it different
        for _ in range(random.randint(5, 10)):
            x1, y1 = random.randint(0, 480), random.randint(0, 480)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
            target_draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Save images
        ctrl_path = controls_dir / f"control_{i:04d}.png"
        target_path = images_dir / f"image_{i:04d}.png"
        
        ctrl_img.save(ctrl_path)
        target_img.save(target_path)
        
        # Create row
        rows.append({
            "image": f"images/image_{i:04d}.png",
            "control_image": f"controls/control_{i:04d}.png",
            "prompt": f"Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, seamlessly blending edges with surrounding areas, maintaining consistent isometric perspective, shadow direction, lighting, pixel density, and color harmony while preserving structural integrity and removing all border artifacts",
        })
    
    # Save metadata
    import csv
    
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "control_image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    
    log.info(f"Created example dataset at {output_dir}")
    log.info(f"  - {num_examples} image pairs")
    log.info(f"  - Metadata: {metadata_path}")
    
    return str(metadata_path)


# =============================================================================
# Default transforms
# =============================================================================

def get_train_transform(max_pixels: int = 1048576):
    """Get default training transform."""
    # Calculate target size
    target_size = int(np.sqrt(max_pixels))
    
    return transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])


def get_train_transform_augmented(max_pixels: int = 1048576, flip_p: float = 0.5):
    """Get training transform with augmentation."""
    target_size = int(np.sqrt(max_pixels))
    
    return transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.ToTensor(),
    ])
