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
    template_path: str
    target_path: str
    mask_type: str
    metadata: dict = field(default_factory=dict)


class DatasetPreparator:
    def __init__(self, renders_dir: Path, generations_dir: Path, output_dir: Path,
                 omni: OmniMasker = None, clean: bool = True):
        self.renders_dir = Path(renders_dir)
        self.generations_dir = Path(generations_dir)
        self.output_dir = Path(output_dir)
        self.omni = omni or OmniMasker()
        self.clean = clean
        self.templates_dir = self.output_dir / 'templates'
        self.targets_dir = self.output_dir / 'targets'
        for d in [self.templates_dir, self.targets_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _cleanup_tile(self, output_prefix: str):
        """Remove old templates and targets for this tile prefix."""
        for f in self.templates_dir.glob(f"{output_prefix}_*_template.png"):
            f.unlink()
        for f in self.targets_dir.glob(f"{output_prefix}_target.png"):
            f.unlink()

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

    def _select_diverse_types(self, num_variants: int = 5) -> List[str]:
        """Select num_variants unique types randomly from ALL_TYPES, respecting DISTRIBUTION."""
        selected = []
        remaining = list(ALL_TYPES)
        
        if TYPE_FULL in remaining and num_variants > 1:
            selected.append(TYPE_FULL)
            remaining.remove(TYPE_FULL)
        
        while len(selected) < num_variants and remaining:
            weights = []
            for t in remaining:
                cat = t.split('_')[0] if '_' in t else t
                if cat in DISTRIBUTION:
                    weights.append(DISTRIBUTION[cat])
                else:
                    weights.append(0.1)
            
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(remaining)] * len(remaining)
            
            chosen = random.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        
        random.shuffle(selected)
        return selected

    def prepare_pairs(self, target_size: Tuple[int, int] = (512, 512), min_quality: int = 50) -> List[Tuple[Path, Path, dict]]:
        pairs = []
        renders = list(self.renders_dir.glob('tile_*.png'))
        print(f"Found {len(renders)} renders")

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
                'render_size': render_path.stat().st_size,
                'pixel_art_hash': self._get_tile_hash(pixel_art_path.name),
                'pixel_art_size': pixel_art_path.stat().st_size,
            }
            pairs.append((render_path, pixel_art_path, meta))

        print(f"Found {len(pairs)} valid pairs")
        return pairs

    def process_pair(self, render_path: Path, pixel_art_path: Path, output_prefix: str,
                     resize_to: Tuple[int, int] = (TEMPLATE_SIZE, TEMPLATE_SIZE),
                     num_variants: int = 5, template_types: List[str] = None,
                     apply_corruption: bool = True, corruption_params: dict = None,
                     prompt: str = DEFAULT_PROMPT) -> List[TrainingSample]:
        samples = []
        try:
            if self.clean:
                self._cleanup_tile(output_prefix)

            render = Image.open(render_path).resize(resize_to, Image.Resampling.LANCZOS)
            pixel_art = Image.open(pixel_art_path).resize(resize_to, Image.Resampling.LANCZOS)
            target_path = self.targets_dir / f"{output_prefix}_target.png"
            pixel_art.save(target_path)

            if template_types is None:
                template_types = self._select_diverse_types(num_variants)

            for variant_index, template_type in enumerate(template_types):
                region = self.omni.get_input_region(resize_to[0], resize_to[1], template_type)
                template = self.omni.create_infill_template(pixel_art, render, region)
                template_path = self.templates_dir / f"{output_prefix}_{template_type}_{variant_index:02d}_template.png"
                template.save(template_path)
                
                # Store relative paths for GPU compatibility
                rel_template_path = f"templates/{template_path.name}"
                rel_target_path = f"targets/{target_path.name}"
                
                samples.append(TrainingSample(
                    sample_id=f"{output_prefix}_{template_type}_{variant_index}",
                    template_path=rel_template_path,
                    target_path=rel_target_path,
                    mask_type=template_type,
                    metadata={
                        'variant_index': variant_index,
                        'template_type': template_type,
                        'apply_corruption': False,
                    },
                ))
        except Exception as e:
            print(f"Error processing {output_prefix}: {e}")
        return samples

    def prepare(self, resize_to: Tuple[int, int] = (TEMPLATE_SIZE, TEMPLATE_SIZE),
                variants_per_pair: int = 5, max_pairs: Optional[int] = None,
                progress_callback: Optional[Callable[[int, int], None]] = None,
                apply_corruption: bool = True, corruption_params: dict = None,
                prompt: str = DEFAULT_PROMPT) -> dict:
        pairs = self.prepare_pairs(target_size=resize_to[0])
        if max_pairs:
            pairs = pairs[:max_pairs]

        print(f"Processing {len(pairs)} pairs with {variants_per_pair} variants each")
        print(f"Expected samples: {len(pairs) * variants_per_pair}")

        all_samples = []
        mask_type_counts: Dict[str, int] = {}

        for idx, (render_path, pixel_art_path, meta) in enumerate(pairs):
            coords = meta['x'], meta['y']
            output_prefix = f"tile_{coords[0]:+d}_{coords[1]:+d}_{meta['render_hash']}"
            samples = self.process_pair(
                render_path, pixel_art_path, output_prefix,
                resize_to=resize_to, num_variants=variants_per_pair,
                apply_corruption=apply_corruption, corruption_params=corruption_params,
                prompt=prompt,
            )
            for sample in samples:
                all_samples.append(sample)
                mask_type_counts[sample.mask_type] = mask_type_counts.get(sample.mask_type, 0) + 1
            if progress_callback:
                progress_callback(idx + 1, len(pairs))

        dataset_meta = {
            'total_samples': len(all_samples),
            'total_pairs': len(pairs),
            'variants_per_pair': variants_per_pair,
            'resize_to': resize_to,
            'mask_type_counts': mask_type_counts,
            'samples': [
                {'sample_id': s.sample_id, 'template_path': s.template_path,
                 'target_path': s.target_path, 'mask_type': s.mask_type}
                for s in all_samples
            ],
        }
        meta_path = self.output_dir / 'dataset_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(dataset_meta, f, indent=2)
        return dataset_meta
