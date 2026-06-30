"""
ControlNet-style dataset loader for LoRA training.
Loads pairs of (control_image, target_image) with optional cached embeddings.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


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


class CustomImageDataset(Dataset):
    """Dataset for isometric map infilling - pairs control (template) + target images."""

    def __init__(
        self,
        img_dir,
        img_size=1024,
        caption_type='txt',
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=None,
        cached_image_embeddings=None,
        control_dir=None,
        cached_image_embeddings_control=None,
    ):
        self.img_dir = Path(img_dir) if img_dir else None
        self.control_dir = Path(control_dir) if control_dir else None
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_embeddings = cached_image_embeddings_control

        # Build sample list from control images
        self.samples = []
        if self.control_dir and self.control_dir.exists():
            for ctrl_file in sorted(self.control_dir.glob("*.png")):
                sample = self._build_sample(ctrl_file)
                if sample:
                    self.samples.append(sample)

        print(f'Loaded {len(self.samples)} samples from control_dir')

    def _build_sample(self, ctrl_file: Path):
        """Build a sample dict by matching control to target."""
        ctrl_name = ctrl_file.stem
        parts = ctrl_name.split('_')

        # Format: tile_{x}_{y}_{hash}_{mask_type}_{variant}_template
        # Need to extract: tile_{x}_{y}_{hash}_target
        if len(parts) >= 6 and parts[-2] == 'template':
            tile_prefix = '_'.join(parts[:4])  # tile_{x}_{y}_{hash}
            target_name = f"{tile_prefix}_target.png"
        elif len(parts) >= 4:
            tile_prefix = '_'.join(parts[:3])  # tile_{x}_{y}
            target_name = f"{tile_prefix}_target.png"
        else:
            target_name = ctrl_file.name.replace('_template', '_target')

        target_path = self.img_dir / target_name if self.img_dir else None

        return {
            'control': str(ctrl_file),
            'target': str(target_path) if target_path else ctrl_file.name,
            'id': ctrl_name,
        }

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        if self.random_ratio:
            # Simple square crop for isometric maps
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return self._to_tensor(img)

    def __len__(self):
        return len(self.samples) if self.samples else 999999

    def __getitem__(self, idx):
        try:
            # Infinite loop support for random sampling
            if not self.samples:
                idx = idx % max(1, len(self.img_dir.glob("*.png"))) if self.img_dir else 0
                img_name = sorted([i for i in os.listdir(self.img_dir) if '.png' in i or '.jpg' in i])[idx]
                img_path = self.img_dir / img_name
                ctrl_path = img_path  # Fallback: same as target
            else:
                if len(self) == 999999:
                    idx = idx % len(self.samples)
                sample = self.samples[idx]
                img_path = sample['target']
                ctrl_path = sample['control']

            # Load target image
            if self.cached_image_embeddings is not None:
                target_img = self.cached_image_embeddings.get(
                    img_path.split('/')[-1],
                    torch.zeros(1, 16, 64, 64)
                )
            else:
                target_img = self._load_image(img_path)

            # Load control image
            if self.cached_control_embeddings is not None:
                control_img = self.cached_control_embeddings.get(
                    ctrl_path.split('/')[-1],
                    torch.zeros(1, 16, 64, 64)
                )
            else:
                control_img = self._load_image(ctrl_path)

            # Load caption/prompt
            txt_path = img_path.rsplit('.', 1)[0] + '.' + self.caption_type

            if self.cached_text_embeddings is not None:
                txt_key = txt_path.split('/')[-1]
                empty_key = txt_key + 'empty_embedding'

                if empty_key in self.cached_text_embeddings and \
                   (np.random.random() < self.caption_dropout_rate):
                    emb = self.cached_text_embeddings[empty_key]
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                if txt_key in self.cached_text_embeddings:
                    emb = self.cached_text_embeddings[txt_key]
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                # Fallback: empty
                return target_img, torch.zeros(1, 768), torch.zeros(1, dtype=torch.int32), control_img
            else:
                try:
                    prompt = open(txt_path, encoding='utf-8').read()
                except:
                    prompt = "isometric pixel art tile"
                return target_img, prompt, control_img

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


def loader(
    cached_text_embeddings=None,
    cached_image_embeddings=None,
    cached_image_embeddings_control=None,
    control_dir="lora_dataset/templates",
    img_dir="lora_dataset/targets",
    caption_dir=None,
    train_batch_size=1,
    num_workers=4,
    img_size=1024,
    **kwargs,
):
    """Create DataLoader from control/target image pairs."""
    dataset = CustomImageDataset(
        img_dir=img_dir,
        img_size=img_size,
        caption_type='txt',
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        control_dir=control_dir,
        cached_image_embeddings_control=cached_image_embeddings_control,
    )

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True if cached_text_embeddings is None else True,
    )
