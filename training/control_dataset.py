"""
CSV-based dataset loader for LoRA training.
Uses dataset.csv as single source of truth.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# Backward compatibility alias
CustomImageDataset = None  # Will be set after CSVDataset is defined


class CSVDataset(Dataset):
    """Dataset that loads from CSV index file."""

    def __init__(
        self,
        csv_path,
        dataset_dir,
        img_size=1024,
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=None,
        cached_image_embeddings=None,
        cached_control_embeddings=None,
        txt_cache_dir=None,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_embeddings = cached_control_embeddings
        self.txt_cache_dir = txt_cache_dir

        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} samples from {self.csv_path}")

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img_path = self.dataset_dir / rel_path
        img = Image.open(img_path).convert('RGB')
        if self.random_ratio:
            w, h = img.size
            min_dim = min(w, h)
            img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return self._to_tensor(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            # Get paths
            target_rel = row['target_image']  # e.g., "targets/tile_..._target.png"
            template_rel = row['template_image']  # e.g., "templates/tile_..._template.png"
            sample_id = row['sample_id']

            # Load target image
            if self.cached_image_embeddings is not None:
                target_img = self.cached_image_embeddings.get(
                    target_rel,
                    torch.zeros(16, 128, 128)
                )
            else:
                target_img = self._load_image(target_rel)

            # Load control (template) image
            if self.cached_control_embeddings is not None:
                control_img = self.cached_control_embeddings.get(
                    template_rel,
                    torch.zeros(16, 128, 128)
                )
            else:
                control_img = self._load_image(template_rel)

            # Load prompt embeddings
            if self.cached_text_embeddings is not None:
                # Check for caption dropout
                empty_key = sample_id + '_empty'
                use_empty = random.random() < self.caption_dropout_rate

                if use_empty and empty_key in self.cached_text_embeddings:
                    emb = self.cached_text_embeddings[empty_key]
                    pem = emb.get('prompt_embeds')
                    pem_mask = emb.get('prompt_embeds_mask')
                    # Handle None mask
                    if pem_mask is None:
                        pem_mask = torch.ones(pem.shape[:2], dtype=torch.int32)
                    return target_img, pem, pem_mask, control_img

                if sample_id in self.cached_text_embeddings:
                    emb = self.cached_text_embeddings[sample_id]
                    pem = emb.get('prompt_embeds')
                    pem_mask = emb.get('prompt_embeds_mask')
                    if pem_mask is None:
                        pem_mask = torch.ones(pem.shape[:2], dtype=torch.int32)
                    return target_img, pem, pem_mask, control_img

                print(f"[WARNING] Cache key not found: {sample_id}")
                return target_img, torch.zeros(1, 4096), torch.ones(1, dtype=torch.int32), control_img

            elif self.txt_cache_dir:
                empty_cache_path = os.path.join(self.txt_cache_dir, sample_id + '_empty.pt')
                use_empty = random.random() < self.caption_dropout_rate

                if use_empty and os.path.exists(empty_cache_path):
                    emb = torch.load(empty_cache_path, map_location='cpu')
                    pem = emb.get('prompt_embeds')
                    pem_mask = emb.get('prompt_embeds_mask')
                    if pem_mask is None:
                        pem_mask = torch.ones(pem.shape[:2], dtype=torch.int32)
                    return target_img, pem, pem_mask, control_img

                cache_path = os.path.join(self.txt_cache_dir, sample_id + '.txt.pt')
                if os.path.exists(cache_path):
                    emb = torch.load(cache_path, map_location='cpu')
                    pem = emb.get('prompt_embeds')
                    pem_mask = emb.get('prompt_embeds_mask')
                    if pem_mask is None:
                        pem_mask = torch.ones(pem.shape[:2], dtype=torch.int32)
                    return target_img, pem, pem_mask, control_img

                print(f"[WARNING] Disk cache not found: {sample_id}")
                return target_img, torch.zeros(1, 4096), torch.ones(1, dtype=torch.int32), control_img
            else:
                # Load prompt from txt file (not cached)
                prompt_path_rel = row.get('prompt_path', '')
                prompt_full_path = self.dataset_dir / prompt_path_rel

                if prompt_full_path.exists():
                    prompt = prompt_full_path.read_text(encoding='utf-8')
                else:
                    prompt = "isometric pixel art tile"

                # Caption dropout
                if random.random() < self.caption_dropout_rate:
                    prompt = " "

                return target_img, prompt, control_img

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__(random.randint(0, len(self) - 1))


def collate_fn(batch):
    """Collate batch - handles both cached latents and raw images."""
    if len(batch[0]) == 4:
        # Cached mode: target_img, prompt_embeds, prompt_embeds_mask, control_img
        target_img, prompt_embeds, prompt_embeds_mask, control_img = zip(*batch)

        return (
            torch.stack(target_img),
            torch.stack(prompt_embeds),
            torch.stack(prompt_embeds_mask),
            torch.stack(control_img),
        )
    else:
        # Raw mode: target_img, prompt, control_img
        target_img, prompts, control_img = zip(*batch)
        return (
            torch.stack(target_img),
            list(prompts),
            torch.stack(control_img),
        )


def loader(
    csv_path=None,
    dataset_dir=None,
    cached_text_embeddings=None,
    cached_image_embeddings=None,
    cached_image_embeddings_control=None,
    train_batch_size=1,
    num_workers=4,
    img_size=1024,
    txt_cache_dir=None,
    caption_dropout_rate=0.1,
    random_ratio=False,
    **kwargs,
):
    """
    Create DataLoader from CSV index.

    Args:
        csv_path: Path to dataset.csv (e.g., "lora_dataset/dataset.csv")
        dataset_dir: Root directory containing targets/, templates/, prompts/ subdirs
        cached_text_embeddings: Precomputed text embeddings dict
        cached_image_embeddings: Precomputed target image embeddings dict
        cached_image_embeddings_control: Precomputed template image embeddings dict
        train_batch_size: Batch size
        num_workers: Number of workers for DataLoader
        img_size: Image size for loading
        txt_cache_dir: Directory for text embedding cache
        caption_dropout_rate: Rate for caption dropout
        random_ratio: Whether to randomly crop to different aspect ratios
    """
    # Default paths if not provided
    if csv_path is None and dataset_dir is not None:
        csv_path = os.path.join(dataset_dir, 'dataset.csv')
    if dataset_dir is None and csv_path is not None:
        dataset_dir = os.path.dirname(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dataset = CSVDataset(
        csv_path=csv_path,
        dataset_dir=dataset_dir,
        img_size=img_size,
        random_ratio=random_ratio,
        caption_dropout_rate=caption_dropout_rate,
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        cached_control_embeddings=cached_image_embeddings_control,
        txt_cache_dir=txt_cache_dir,
    )

    use_collate = cached_text_embeddings is not None or txt_cache_dir is not None

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn if use_collate else None,
    )


# Backward compatibility alias
CustomImageDataset = CSVDataset
