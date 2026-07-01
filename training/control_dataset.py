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


def _build_cache_key(ctrl_filename):
    """Build consistent cache key from control filename.
    
    Format: tile_{x}_{y}_{hash}_{mask}_{variant}_template.png
    -> tile_{x}_{y}_{hash}_{mask}_{variant} (strip _template)
    
    Format: tile_{x}_{y}_{hash}_target.png
    -> tile_{x}_{y}_{hash} (strip _target)
    """
    name = ctrl_filename.rsplit('.', 1)[0]  # Remove extension
    if name.endswith('_template'):
        return name[:-9]  # Remove _template (9 chars)
    elif name.endswith('_target'):
        return name[:-7]  # Remove _target (7 chars)
    return name


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
        txt_cache_dir=None,
        prompts_dir=None,
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
        self.txt_cache_dir = txt_cache_dir
        self.prompts_dir = Path(prompts_dir) if prompts_dir else (Path(img_dir).parent / 'prompts' if img_dir else None)

        # Build sample list from control images
        self.samples = []
        if self.control_dir and self.control_dir.exists():
            for ctrl_file in sorted(self.control_dir.glob("*.png")):
                sample = self._build_sample(ctrl_file)
                if sample:
                    self.samples.append(sample)

        print(f'Loaded {len(self.samples)} samples from control_dir')
        if self.prompts_dir and self.prompts_dir.exists():
            prompt_files = list(self.prompts_dir.glob("*.txt"))
            print(f'Found {len(prompt_files)} prompt files in {self.prompts_dir}')

    def _build_sample(self, ctrl_file: Path):
        """Build a sample dict by matching control to target."""
        ctrl_name = ctrl_file.stem
        
        # Format: tile_{x}_{y}_{hash}_{mask_type}_{variant}_template
        # Example: tile_+0_-34_4b43a01d_quadrant_br_00_template
        # Extract tile prefix: tile_{x}_{y}_{hash}
        if ctrl_name.endswith('_template'):
            base_name = ctrl_name[:-9]
            parts = base_name.split('_')
            if len(parts) >= 4:
                tile_prefix = '_'.join(parts[:4])
            else:
                tile_prefix = base_name
            target_name = f"{tile_prefix}_target.png"
        else:
            parts = ctrl_name.split('_')
            if len(parts) >= 4:
                tile_prefix = '_'.join(parts[:4])
            elif len(parts) >= 3:
                tile_prefix = '_'.join(parts[:3])
            else:
                tile_prefix = ctrl_name
            target_name = f"{tile_prefix}_target.png"

        target_path = self.img_dir / target_name if self.img_dir else None

        return {
            'control': str(ctrl_file),
            'target': str(target_path) if target_path else ctrl_file.name,
            'id': ctrl_name,
            'tile_prefix': tile_prefix,
        }

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        if self.random_ratio:
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        return self._to_tensor(img)

    def _get_prompt_path(self, tile_prefix: str) -> Path:
        """Get path to prompt file for this tile prefix."""
        if self.prompts_dir and self.prompts_dir.exists():
            prompt_path = self.prompts_dir / f"{tile_prefix}.txt"
            if prompt_path.exists():
                return prompt_path
        return None

    def __len__(self):
        return len(self.samples) if self.samples else 999999

    def __getitem__(self, idx):
        try:
            if not self.samples:
                idx = idx % max(1, len(self.img_dir.glob("*.png"))) if self.img_dir else 0
                img_name = sorted([i for i in os.listdir(self.img_dir) if '.png' in i or '.jpg' in i])[idx]
                img_path = self.img_dir / img_name
                ctrl_path = img_path
                tile_prefix = _build_cache_key(img_name)
            else:
                if len(self) == 999999:
                    idx = idx % len(self.samples)
                sample = self.samples[idx]
                img_path = sample['target']
                ctrl_path = sample['control']
                tile_prefix = sample.get('tile_prefix', _build_cache_key(ctrl_path.split('/')[-1]))

            ctrl_filename = ctrl_path.split('/')[-1]
            target_filename = img_path.split('/')[-1]
            
            # Cache key matches precompute: control filename without _template
            cache_key = _build_cache_key(ctrl_filename)
            empty_key = cache_key + '_empty.pt'

            # Load target image
            if self.cached_image_embeddings is not None:
                target_img = self.cached_image_embeddings.get(
                    target_filename,
                    torch.zeros(16, 128, 128)
                )
            else:
                target_img = self._load_image(img_path)

            # Load control image
            if self.cached_control_embeddings is not None:
                control_img = self.cached_control_embeddings.get(
                    ctrl_filename,
                    torch.zeros(16, 128, 128)
                )
            else:
                control_img = self._load_image(ctrl_path)

            # Load caption/prompt embeddings
            if self.cached_text_embeddings is not None:
                if empty_key in self.cached_text_embeddings and \
                   (np.random.random() < self.caption_dropout_rate):
                    emb = self.cached_text_embeddings[empty_key]
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                if cache_key in self.cached_text_embeddings:
                    emb = self.cached_text_embeddings[cache_key]
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                print(f"[WARNING] Cache key not found: {cache_key}")
                return target_img, torch.zeros(1, 4096), torch.zeros(1, dtype=torch.int32), control_img

            elif self.txt_cache_dir:
                empty_cache_path = os.path.join(self.txt_cache_dir, empty_key)
                if os.path.exists(empty_cache_path) and np.random.random() < self.caption_dropout_rate:
                    emb = torch.load(empty_cache_path, map_location='cpu')
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                cache_path = os.path.join(self.txt_cache_dir, cache_key + '.txt.pt')
                if os.path.exists(cache_path):
                    emb = torch.load(cache_path, map_location='cpu')
                    return target_img, emb['prompt_embeds'], emb['prompt_embeds_mask'], control_img

                print(f"[WARNING] Disk cache not found: {cache_key}")
                return target_img, torch.zeros(1, 4096), torch.zeros(1, dtype=torch.int32), control_img
            else:
                # Load prompt from txt file directly
                prompt_path = self._get_prompt_path(tile_prefix)
                if prompt_path:
                    prompt = prompt_path.read_text(encoding='utf-8')
                else:
                    prompt = "isometric pixel art tile"
                return target_img, prompt, control_img

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__(np.random.randint(0, len(self)))


def collate_fn(batch):
    """Collate batch - handles both cached latents and raw images."""
    if len(batch[0]) == 4:
        target_img, prompt_embeds, prompt_embeds_mask, control_img = zip(*batch)
        
        target_img_stacked = torch.stack(target_img)
        control_img_stacked = torch.stack(control_img)
        
        prompt_embeds_list = []
        prompt_masks_list = []
        for pe, pm in zip(prompt_embeds, prompt_embeds_mask):
            if isinstance(pe, torch.Tensor):
                prompt_embeds_list.append(pe)
            else:
                raise ValueError("String prompts not supported with cached embeddings")
            prompt_masks_list.append(pm)
        
        return (
            target_img_stacked,
            torch.stack(prompt_embeds_list),
            torch.stack(prompt_masks_list),
            control_img_stacked,
        )
    else:
        target_img, prompts, control_img = zip(*batch)
        return (
            torch.stack([t if isinstance(t, torch.Tensor) else torch.from_numpy(np.array(t)) for t in target_img]),
            list(prompts),
            torch.stack([c if isinstance(c, torch.Tensor) else torch.from_numpy(np.array(c)) for c in control_img]),
        )


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
    txt_cache_dir=None,
    prompts_dir=None,
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
        txt_cache_dir=txt_cache_dir,
        prompts_dir=prompts_dir,
    )

    use_collate = cached_text_embeddings is not None or txt_cache_dir is not None

    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn if use_collate else None,
    )
