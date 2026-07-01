"""
ControlNet Dataset for training with control images.

NEW STRUCTURE (v2.0):
    dataset/
        images/
            image_001.jpg   (target image - pixel art)
            image_001.txt   (caption for target)
            image_002.jpg
            image_002.txt
            ...
        control/
            image_001.jpg   (control image - template with red border)
            image_002.jpg
            ...
        prompts/
            prompt.txt         (COMMON prompt - used by all)
        dataset_mapping.csv
        dataset_metadata.json
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def throw_one(probability: float) -> int:
    """Return 1 with given probability, else 0."""
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    """Resize image maintaining aspect ratio."""
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))


def c_crop(image):
    """Center crop to square."""
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def crop_to_aspect_ratio(image, ratio="16:9"):
    """Crop image to target aspect ratio."""
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box)


def normalize_image(img):
    """Convert PIL image to normalized tensor."""
    img_array = np.array(img)
    img_tensor = torch.from_numpy((img_array / 127.5) - 1)
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor


def resize_to_multiple_of_32(img, max_size=512):
    """Resize image and ensure dimensions are multiples of 32."""
    img = image_resize(img, max_size)
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    img = img.resize((new_w, new_h))
    return img


class ControlDataset(Dataset):
    """
    Dataset for training with control images.

    Expected structure:
        img_dir/           - contains .jpg files (targets)
        control_dir/       - contains .jpg files (controls)
        prompts_dir/       - contains prompt.txt (common prompt)

    Also supports reading from dataset_mapping.csv for explicit mapping.
    """

    def __init__(
        self,
        img_dir: str,
        img_size: int = 512,
        caption_type: str = 'txt',
        random_ratio: bool = False,
        caption_dropout_rate: float = 0.1,
        cached_text_embeddings: dict = None,
        cached_image_embeddings: dict = None,
        control_dir: str = None,
        cached_image_embeddings_control: dict = None,
        prompts_dir: str = None,
        use_common_prompt: bool = True,
    ):
        self.img_dir = Path(img_dir)
        self.control_dir = Path(control_dir) if control_dir else None
        self.prompts_dir = Path(prompts_dir) if prompts_dir else self.img_dir.parent / 'prompts'
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        self.use_common_prompt = use_common_prompt

        # Load images (only .jpg files from img_dir)
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # Load common prompt if exists
        self.common_prompt = ""
        common_prompt_path = self.prompts_dir / 'prompt.txt'
        if common_prompt_path.exists():
            with open(common_prompt_path, 'r', encoding='utf-8') as f:
                self.common_prompt = f.read().strip()

        print(f"[ControlDataset] Found {len(self.images)} images")
        print(f"[ControlDataset] img_dir: {self.img_dir}")
        print(f"[ControlDataset] control_dir: {self.control_dir}")
        print(f"[ControlDataset] prompts_dir: {self.prompts_dir}")
        if self.common_prompt:
            print(f"[ControlDataset] Common prompt loaded ({len(self.common_prompt)} chars)")
        else:
            print("[ControlDataset] WARNING: No common prompt found!")

    def __len__(self):
        return len(self.images)

    def _load_and_process_image(self, img_path: Image.Image) -> torch.Tensor:
        """Load and process an image to tensor."""
        if self.random_ratio:
            ratio = random.choice(["16:9", "default", "1:1", "4:3"])
            if ratio != "default":
                img_path = crop_to_aspect_ratio(img_path, ratio)

        img_path = resize_to_multiple_of_32(img_path, self.img_size)
        return normalize_image(img_path)

    def _get_caption(self, img_name: str) -> str:
        """Get caption for the image."""
        if self.cached_text_embeddings is not None:
            # Return empty string when using cached embeddings
            return ""

        caption_name = img_name.rsplit('.', 1)[0] + '.' + self.caption_type
        caption_path = self.img_dir / caption_name

        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # Fallback to common prompt
            return self.common_prompt

    def _get_control_image_path(self, img_name: str) -> str:
        """Get the control image path for this target image."""
        if self.control_dir is None:
            return None

        # Control images have same name as target images
        return str(self.control_dir / img_name)

    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]

            # Load target image
            if self.cached_image_embeddings is None:
                img_path = self.img_dir / img_name
                img = Image.open(img_path).convert('RGB')
                target_tensor = self._load_and_process_image(img)
            else:
                target_tensor = self.cached_image_embeddings[img_name]

            # Load control image
            if self.cached_control_image_embeddings is None:
                control_path = self._get_control_image_path(img_name)
                if control_path and os.path.exists(control_path):
                    control_img = Image.open(control_path).convert('RGB')
                    control_tensor = self._load_and_process_image(control_img)
                else:
                    # Fallback: use target as control
                    control_tensor = target_tensor
            else:
                control_tensor = self.cached_control_image_embeddings[img_name]

            # Get caption/prompt
            caption = self._get_caption(img_name)

            # Apply caption dropout
            if self.caption_dropout_rate > 0 and throw_one(self.caption_dropout_rate):
                caption = " "  # Empty prompt for dropout

            return target_tensor, caption, control_tensor

        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            # Return random sample on error
            new_idx = random.randint(0, len(self.images) - 1)
            return self.__getitem__(new_idx)


def loader(train_batch_size, num_workers, **kwargs):
    """Create a DataLoader from ControlDataset."""
    dataset = ControlDataset(**kwargs)
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )


# Backward compatibility alias
CustomImageDataset = ControlDataset
