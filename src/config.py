"""
src/config.py — Central configuration for isometric pipeline (Python).
"""

import os
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Handle both package import and direct script execution
try:
    from .helpers import get_project_root
except ImportError:
    # Fallback for direct execution / when run from project root
    _src_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(_src_dir))
    from helpers import get_project_root

PROJECT_ROOT = get_project_root()


@dataclass
class DatasetConfig:
    size: int = 512
    variants: int = 5
    max_pairs: Optional[int] = None
    format: str = "sd"
    seed: int = 42
    use_variance: bool = True
    desaturation_factor: float = 0.5
    noise_intensity: int = 100
    gamma_value: float = 1.2
    blur_radius: float = 0.5
    brightness_factor: float = 0.9
    contrast_factor: float = 0.8
    mask_probs: dict = field(default_factory=lambda: {
        'full': 0.15, 'quadrant': 0.20, 'half': 0.20, 'strip': 0.20, 'rect': 0.25,
    })


@dataclass
class DZIConfig:
    vips_bin_path: str = ""
    strip_height: int = 4096
    tile_size: int = 512
    compression: str = "lzw"
    jpeg_quality: int = 95
    input_json: str = "output/map_plan.json"
    output_name: str = "output/gigapixel_map"
    vips_progress: bool = False
    vips_concurrency: int = 2
    cache_max_mem_mb: int = 2000
    cache_max_files: int = 500


@dataclass
class RenderConfig:
    size_px: int = 1024
    azimuth: int = 180
    elevation: int = -45
    altitude: int = 200
    sse: int = 10
    blank_variance_thr: int = 800
    blank_edge_thr: float = 0.15
    blank_mean_thr: tuple = (60, 110)
    blank_size_kb: int = 30
    tile_wait_ms: int = 12000
    settle_poll_ms: int = 300
    settle_max_ms: int = 4000


@dataclass
class PathsConfig:
    districts: str = "./districts"
    water: str = "./geo/water.geojson"
    infra: str = "./geo/infra.geojson"
    output: str = "./output"
    renders: str = "./output/renders"
    generate: str = "./generate_sample"
    lora_dataset: str = "./lora_dataset"
    meta: str = "meta"


@dataclass
class InfillConfig:
    """Infill/template parameters used by inference pipeline.

    Source of truth for the AI-edit request template (red border, max area,
    default prompt). Keeps inference module in sync with src/.
    """
    border_width: int = 2
    border_color: tuple = (255, 0, 0, 255)
    seam_color: tuple = (255, 0, 0, 255)
    seam_thickness_px: int = 1
    max_infill_area_ratio: float = 0.5  # 50% of template_size^2
    default_prompt: str = (
        "Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, "
        "seamlessly blending edges with surrounding areas, maintaining consistent isometric perspective, "
        "shadow direction, lighting, pixel density, and color harmony while preserving structural integrity "
        "and removing all border artifacts"
    )


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dzi: DZIConfig = field(default_factory=DZIConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    infill: InfillConfig = field(default_factory=InfillConfig)


def _apply_env_overrides(cfg: Config) -> Config:
    """Apply environment variable overrides to config."""
    for section_name in ['dataset', 'dzi', 'render']:
        section = getattr(cfg, section_name)
        prefix = f"{section_name.upper()}_"

        for field_name in dir(section):
            if field_name.startswith('_'):
                continue

            env_key = prefix + field_name.upper()
            env_val = os.environ.get(env_key)
            if env_val is None:
                continue

            try:
                current = getattr(section, field_name)
                if isinstance(current, bool):
                    setattr(section, field_name, env_val.lower() in ('true', '1', 'yes'))
                elif isinstance(current, int):
                    setattr(section, field_name, int(env_val))
                elif isinstance(current, float):
                    setattr(section, field_name, float(env_val))
                elif isinstance(current, str):
                    setattr(section, field_name, env_val)
                elif isinstance(current, tuple):
                    setattr(section, field_name, tuple(int(x) for x in env_val.split(',')))
                elif isinstance(current, dict):
                    setattr(section, field_name, json.loads(env_val))
            except (ValueError, json.JSONDecodeError):
                pass
    return cfg


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance (lazy singleton)."""
    global _config
    if _config is None:
        _config = _apply_env_overrides(Config())
    return _config


def get_dzi_config() -> DZIConfig:
    """Get DZI config section."""
    return get_config().dzi


def reset_config():
    """Reset global config (for testing)."""
    global _config
    _config = None
