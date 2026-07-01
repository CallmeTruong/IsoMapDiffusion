"""
Dataset Preparator - Main pipeline for training data preparation.
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict
from dataclasses import dataclass, field
from PIL import Image

from .omni import (
    OmniMasker, TEMPLATE_SIZE, QUADRANT_SIZE, DISTRIBUTION, ALL_TYPES, TYPE_FULL,
)
from ..constants import DEFAULT_PROMPT


@dataclass
class TrainingSample:
    sample_id: str
    target_path: str  # path to target image (pixel art)
    control_path: str  # path to control image (template with red border)
    caption_path: str  # path to caption text
    prompt_path: str  # path to common prompt
    mask_type: str
    metadata: dict = field(default_factory=dict)


class DatasetPreparator:
    """
    Creates dataset in NEW structure (v2.0):
        dataset/
            images/          - target images (jpg) + captions (txt)
            control/         - control images (jpg)
            prompts/        - common prompt.txt
            dataset_mapping.csv
            dataset_metadata.json
    """

    def __init__(self, renders_dir: Path, generations_dir: Path, output_dir: Path,
                 omni: OmniMasker = None):
        self.renders_dir = Path(renders_dir)
        self.generations_dir = Path(generations_dir)
        self.output_dir = Path(output_dir)
        self.omni = omni or OmniMasker()

        # NEW directory structure
        self.images_dir = self.output_dir / 'images'
        self.control_dir = self.output_dir / 'control'
        self.prompts_dir = self.output_dir / 'prompts'

        for d in [self.images_dir, self.control_dir, self.prompts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_tile_hash(self, filename: str) -> str:
        return filename.rsplit('_', 1)[-1].replace('.png', '')

    def _get_tile_coords(self, filename: str) -> Tuple[int, int]:
        parts = filename.replace('.png', '').split('_')
        return int(parts[1]), int(parts[2])

    def _find_matching_pair(self, render_file: Path) -> Optional[Path]:
        render_hash = self._get_tile_hash(render_file.name)
        for gen_file in self.generations_dir.glob('*_*.png'):
            if gen_file.name.endswith(f'_{render_hash}.png'):
                return gen_file
            if self._get_tile_hash(gen_file.name) == render_hash:
                return gen_file
        return None

    def _save_common_prompt(self, prompt: str) -> Path:
        """Save common prompt to prompts/prompt.txt"""
        prompt_path = self.prompts_dir / 'prompt.txt'
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return prompt_path

    def prepare_pairs(self, min_quality: int = 50) -> List[Tuple[Path, Path, dict]]:
        """Find valid render/pixel-art pairs."""
        pairs = []
        renders = list(self.renders_dir.glob('tile_*.png'))

        for render_path in renders:
            if render_path.suffix != '.png' or render_path.stat().st_size < min_quality * 1024:
                continue
            pixel_art_path = self._find_matching_pair(render_path)
            if pixel_art_path is None:
                continue
            coords = self._get_tile_coords(render_path.name)
            meta = {
                'x': coords[0], 'y': coords[1],
                'render_hash': self._get_tile_hash(render_path.name),
            }
            pairs.append((render_path, pixel_art_path, meta))

        return pairs

    def prepare(
        self,
        resize_to: Tuple[int, int] = (TEMPLATE_SIZE, TEMPLATE_SIZE),
        variants_per_pair: int = 5,
        max_pairs: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        prompt: str = DEFAULT_PROMPT,
    ) -> dict:
        """
        Create dataset in NEW structure (v2.0).

        Returns metadata dict with structure:
        {
            'total_samples': int,
            'total_pairs': int,
            'samples': [TrainingSample, ...],
            ...
        }
        """
        pairs = self.prepare_pairs()
        if max_pairs:
            pairs = pairs[:max_pairs]

        print(f"Processing {len(pairs)} pairs with {variants_per_pair} variants each")

        # Save common prompt
        self._save_common_prompt(prompt)

        all_samples = []
        mask_type_counts: Dict[str, int] = {}
        image_index = 1

        for idx, (render_path, pixel_art_path, meta) in enumerate(pairs):
            try:
                # Load and resize images
                render = Image.open(render_path).resize(resize_to, Image.Resampling.LANCZOS)
                pixel_art = Image.open(pixel_art_path).resize(resize_to, Image.Resampling.LANCZOS)
                coords = (meta['x'], meta['y'])

                # Select diverse template types
                template_types = random.sample(ALL_TYPES, min(variants_per_pair, len(ALL_TYPES)))

                for variant_index, template_type in enumerate(template_types):
                    region = self.omni.get_input_region(resize_to[0], resize_to[1], template_type)

                    # Create control image (template with red border)
                    control_image = self.omni.create_infill_template(pixel_art, render, region)

                    # Create target image (full pixel art)
                    target_image = pixel_art.copy()

                    # Save files with sequential numbering
                    img_name = f"image_{image_index:03d}"

                    # Save target image (jpg) to images/
                    target_jpg_path = self.images_dir / f"{img_name}.jpg"
                    target_image_rgb = target_image.convert('RGB')
                    target_image_rgb.save(target_jpg_path, quality=95)

                    # Save caption (txt) to images/
                    caption_path = self.images_dir / f"{img_name}.txt"
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(f"isometric pixel art tile at x={coords[0]}, y={coords[1]}")

                    # Save control image to control/
                    control_jpg_path = self.control_dir / f"{img_name}.jpg"
                    control_image_rgb = control_image.convert('RGB')
                    control_image_rgb.save(control_jpg_path, quality=95)

                    # Create sample
                    sample = TrainingSample(
                        sample_id=img_name,
                        target_path=f"images/{img_name}.jpg",
                        control_path=f"control/{img_name}.jpg",
                        caption_path=f"images/{img_name}.txt",
                        prompt_path="prompts/prompt.txt",
                        mask_type=template_type,
                        metadata={
                            'tile_coords': f"{coords[0]},{coords[1]}",
                            'variant_index': variant_index,
                        },
                    )
                    all_samples.append(sample)
                    mask_type_counts[template_type] = mask_type_counts.get(template_type, 0) + 1
                    image_index += 1

            except Exception as e:
                print(f"Error processing pair: {e}")
                continue

            if progress_callback:
                progress_callback(idx + 1, len(pairs))

        # Create metadata
        dataset_meta = {
            'total_samples': len(all_samples),
            'total_pairs': len(pairs),
            'variants_per_pair': variants_per_pair,
            'resize_to': list(resize_to),
            'prompt': prompt,
            'common_prompt_file': 'prompts/prompt.txt',
            'structure_version': '2.0',
            'mask_type_counts': mask_type_counts,
            'samples': [
                {
                    'sample_id': s.sample_id,
                    'target_path': s.target_path,
                    'control_path': s.control_path,
                    'caption_path': s.caption_path,
                    'prompt_path': s.prompt_path,
                    'mask_type': s.mask_type,
                    'tile_coords': s.metadata.get('tile_coords', ''),
                }
                for s in all_samples
            ],
        }

        # Save metadata JSON
        meta_path = self.output_dir / 'dataset_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

        # Create mapping CSV
        csv_path = self.output_dir / 'dataset_mapping.csv'
        import csv as csv_module
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_module.DictWriter(f, fieldnames=['caption_path', 'control_path', 'prompt_path'])
            writer.writeheader()
            for sample in all_samples:
                writer.writerow({
                    'caption_path': sample.caption_path,
                    'control_path': sample.control_path,
                    'prompt_path': sample.prompt_path,
                })

        return dataset_meta
