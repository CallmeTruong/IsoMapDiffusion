"""
Dataset Preparator — Creates training dataset from rendered tiles + pixel-art generations.

Two entry points:
    python -m src.dataset.prepare          # via main()
    python prepare_dataset.py             # thin wrapper (kept for backward compat)

Output structure (v2.0):
    dataset/
        images/          — target images (pixel art) + captions (.txt)
        control/         — control images (template with red border)
        prompts/         — common prompt.txt
        dataset_mapping.csv
        dataset_metadata.json
"""

import argparse
import csv as csv_module
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from .omni import (
    ALL_TYPES,
    OmniMasker,
    TEMPLATE_SIZE,
)
from ..constants import DEFAULT_PROMPT


@dataclass
class TrainingSample:
    sample_id: str
    target_path: str
    control_path: str
    caption_path: str
    prompt_path: str
    mask_type: str
    metadata: dict = field(default_factory=dict)


class DatasetPreparator:
    """
    Creates dataset in NEW structure (v2.0):
        dataset/
            images/          - target images (jpg) + captions (txt)
            control/         - control images (jpg)
            prompts/         - common prompt.txt
            dataset_mapping.csv
            dataset_metadata.json
    """

    def __init__(
        self,
        renders_dir: Path,
        generations_dir: Path,
        output_dir: Path,
        omni: Optional[OmniMasker] = None,
    ):
        self.renders_dir = Path(renders_dir)
        self.generations_dir = Path(generations_dir)
        self.output_dir = Path(output_dir)
        self.omni = omni or OmniMasker()

        self.images_dir   = self.output_dir / 'images'
        self.control_dir  = self.output_dir / 'control'
        self.prompts_dir  = self.output_dir / 'prompts'

        for d in [self.images_dir, self.control_dir, self.prompts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # helpers

    def _get_tile_hash(self, filename: str) -> str:
        return filename.rsplit('_', 1)[-1].replace('.png', '')

    def _get_tile_coords(self, filename: str) -> Tuple[int, int]:
        parts = filename.replace('.png', '').split('_')
        return int(parts[1]), int(parts[2])

    def _find_matching_pair(self, render_file: Path) -> Optional[Path]:
        render_hash = self._get_tile_hash(render_file.name)
        for gen_file in self.generations_dir.glob('*_*.png'):
            if gen_file.name.endswith('_' + render_hash + '.png'):
                return gen_file
            if self._get_tile_hash(gen_file.name) == render_hash:
                return gen_file
        return None

    def _save_common_prompt(self, prompt: str) -> Path:
        prompt_path = self.prompts_dir / 'prompt.txt'
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return prompt_path

    # public API

    def prepare_pairs(self, min_quality: int = 50) -> List[Tuple[Path, Path, dict]]:
        """Find valid render/pixel-art pairs."""
        pairs = []
        for render_path in self.renders_dir.glob('tile_*.png'):
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
        min_quality: int = 50,
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
        pairs = self.prepare_pairs(min_quality=min_quality)
        if max_pairs:
            pairs = pairs[:max_pairs]

        print(f"Processing {len(pairs)} pairs with {variants_per_pair} variants each")

        self._save_common_prompt(prompt)

        all_samples: List[TrainingSample] = []
        mask_type_counts: Dict[str, int] = {}
        image_index = 1

        for idx, (render_path, pixel_art_path, meta) in enumerate(pairs):
            try:
                render = Image.open(render_path).resize(resize_to, Image.Resampling.LANCZOS)
                pixel_art = Image.open(pixel_art_path).resize(resize_to, Image.Resampling.LANCZOS)
                coords = (meta['x'], meta['y'])

                template_types = random.sample(ALL_TYPES, min(variants_per_pair, len(ALL_TYPES)))

                for variant_index, template_type in enumerate(template_types):
                    region = self.omni.get_input_region(
                        resize_to[0], resize_to[1], template_type
                    )
                    control_image = self.omni.create_infill_template(pixel_art, render, region)
                    target_image = pixel_art.copy()

                    img_name = f"image_{image_index:03d}"

                    # Target image (jpg)
                    target_jpg_path = self.images_dir / (img_name + ".jpg")
                    target_image.convert('RGB').save(target_jpg_path, quality=95)

                    # Caption (txt)
                    caption_path = self.images_dir / (img_name + ".txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(f"isometric pixel art tile at x={coords[0]}, y={coords[1]}")

                    # Control image (jpg)
                    control_jpg_path = self.control_dir / (img_name + ".jpg")
                    control_image.convert('RGB').save(control_jpg_path, quality=95)

                    sample = TrainingSample(
                        sample_id=img_name,
                        target_path="images/" + img_name + ".jpg",
                        control_path="control/" + img_name + ".jpg",
                        caption_path="images/" + img_name + ".txt",
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

        # Metadata JSON
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

        meta_path = self.output_dir / 'dataset_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

        # Mapping CSV
        csv_path = self.output_dir / 'dataset_mapping.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_module.DictWriter(
                f, fieldnames=['caption_path', 'control_path', 'prompt_path']
            )
            writer.writeheader()
            for sample in all_samples:
                writer.writerow({
                    'caption_path': sample.caption_path,
                    'control_path': sample.control_path,
                    'prompt_path': sample.prompt_path,
                })

        return dataset_meta


# CLI entry point

DEFAULT_RENDERS_DIR   = Path(__file__).parent.parent.parent / 'output' / 'renders'
DEFAULT_GENERATIONS_DIR = Path(__file__).parent.parent.parent / 'generate'
DEFAULT_OUTPUT_DIR   = Path(__file__).parent.parent.parent / 'dataset'


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LoRA dataset from rendered tiles + pixel-art generations.'
    )
    parser.add_argument(
        '--renders', type=str,
        help='Path to renders directory (default: output/renders)',
    )
    parser.add_argument(
        '--generations', type=str,
        help='Path to generations directory (default: generate)',
    )
    parser.add_argument(
        '--output', type=str,
        help='Path to output dataset directory (default: dataset)',
    )
    parser.add_argument(
        '--size', type=int, default=1024,
        help='Image size in px (default: 1024)',
    )
    parser.add_argument(
        '--variants', type=int, default=5,
        help='Template variants per pair (default: 5)',
    )
    parser.add_argument(
        '--max-pairs', type=int, default=None,
        help='Maximum number of render/generation pairs to process',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Custom prompt text (default: use DEFAULT_PROMPT)',
    )
    parser.add_argument(
        '--min-quality-kb', type=int, default=50,
        help='Min render file size to be considered valid KB (default: 50)',
    )
    args = parser.parse_args()

    renders_dir    = Path(args.renders)    if args.renders    else DEFAULT_RENDERS_DIR
    generations_dir = Path(args.generations) if args.generations else DEFAULT_GENERATIONS_DIR
    output_dir     = Path(args.output)     if args.output     else DEFAULT_OUTPUT_DIR
    prompt         = args.prompt if args.prompt else DEFAULT_PROMPT

    print("=" * 60)
    print("LoRA Dataset Preparation (v2.0)")
    print("=" * 60)
    print(f"  renders:     {renders_dir}")
    print(f"  generations: {generations_dir}")
    print(f"  output:      {output_dir}")
    print(f"  size:        {args.size}x{args.size}")
    print(f"  variants:    {args.variants}")
    print(f"  max_pairs:   {args.max_pairs or 'all'}")
    print(f"  seed:        {args.seed}")
    print(f"  min_quality: {args.min_quality_kb} KB")
    print("=" * 60)

    if not renders_dir.exists():
        print(f"ERROR: Renders directory not found: {renders_dir}")
        print("Please run render pipeline first or specify --renders")
        return

    if not generations_dir.exists():
        print(f"WARNING: Generations directory not found: {generations_dir}")
        print("Will try to find matching pairs anyway...")

    random.seed(args.seed)
    omni = OmniMasker(seed=args.seed)
    preparator = DatasetPreparator(
        renders_dir=renders_dir,
        generations_dir=generations_dir,
        output_dir=output_dir,
        omni=omni,
    )

    result = preparator.prepare(
        resize_to=(args.size, args.size),
        variants_per_pair=args.variants,
        max_pairs=args.max_pairs,
        prompt=prompt,
        min_quality=args.min_quality_kb,
    )

    print(f"\n{'=' * 60}")
    print(f"Dataset created: {output_dir}")
    print(f"Total samples: {result['total_samples']}")
    print(f"  images/  : {len(list((output_dir / 'images').glob('*.jpg')))} target images")
    print(f"  control/ : {len(list((output_dir / 'control').glob('*.jpg')))} control images")
    print(f"  prompts/ : prompts/prompt.txt")
    print(f"  CSV rows : {len(result['samples'])}")
    print(f"  JSON     : dataset_metadata.json")
    print(f"{'=' * 60}")
    print("Next steps:")
    print("  1. Review images/ and control/ folders")
    print("  2. Edit prompts/prompt.txt if needed")
    print("  3. Run training: python -m training.lora_train --config training/config.yaml")


if __name__ == '__main__':
    main()
