"""
inference/config.py - Central configuration cho inference pipeline.

Load tat ca tu src/config.py (Python) + src/config.mjs (JS config source
of truth), KHONG hardcode bat ky gia tri nao.

Source of truth: src/config.mjs (vi day la noi dinh nghia chinh)
  TILE.sizePx              -> tile_size_px       (1024)
  TILE.cameraMoveStep      -> camera_move_step   (0.5)
  TILE_STEP_PX             -> stride_px          (512)
  CELL_SIZE_M              -> cell_size_m        (200)
  MAX_INFILL_AREA          -> max_infill_area    (524288 = 50% of 1024^2)
  border_width             -> border_width       (2, tu isometric-nyc)
  border_color             -> border_color       ((255,0,0,255), tu isometric-nyc)

Moi file trong inference (template.py, traversal.py, scripts/*) nen dung
InferenceConfig thay vi hardcode constants.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Project root = parent of inference/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow imports when run as script (e.g. python -m inference.config)
_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _parse_config_mjs() -> dict:
    """
    Parse src/config.mjs de lay cac defaults.

    config.mjs dang:
        TILE: { sizePx: 1024, cameraMoveStep: 0.5, ... }
        CELL_SIZE_M = 200;

    Parse kieu don gian: regex extract pattern `const NAME = VALUE` hoac
    pattern `SECTION: { KEY: VALUE }` trong object literal.
    """
    mjs_path = _SRC_DIR / "config.mjs"
    if not mjs_path.exists():
        return {}

    try:
        text = mjs_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    cfg: dict = {}

    # 1) Parse const NAME = VALUE (number, string, array, object)
    #    Only top-level `const NAME = ...` (no leading spaces inside the match)
    const_re = re.compile(
        r"^const\s+([A-Z_][A-Z0-9_]*)\s*=\s*([^;]+);",
        re.MULTILINE,
    )
    for m in const_re.finditer(text):
        name = m.group(1)
        raw = m.group(2).strip()
        # Try eval as JSON-like literal (replace single quotes not used here)
        try:
            # Use ast.literal_eval for safety
            import ast
            value = ast.literal_eval(raw)
            cfg[name] = value
        except Exception:
            # Best effort: extract a number
            num_m = re.search(r"^-?\d+(\.\d+)?", raw)
            if num_m:
                try:
                    cfg[name] = float(num_m.group(0))
                except Exception:
                    pass

    # 2) Parse DEFAULTS = { TILE: { sizePx: 1024, cameraMoveStep: 0.5, ... } }
    defaults_re = re.compile(
        r"const\s+DEFAULTS\s*=\s*\{(?P<body>.*?)\n\};",
        re.DOTALL,
    )
    defaults_m = defaults_re.search(text)
    if defaults_m:
        body = defaults_m.group("body")
        # Match each section:  SECTION: { ... }
        section_re = re.compile(
            r"^\s{2}([A-Z_]+):\s*\{(?P<inner>.*?)\n\s{2}\},?",
            re.MULTILINE | re.DOTALL,
        )
        for sm in section_re.finditer(body):
            section = sm.group(1)
            inner = sm.group("inner")
            section_dict: dict = {}
            # Match key: value pairs
            kv_re = re.compile(
                r"([a-zA-Z_]\w*)\s*:\s*([^\n,]+?)(?=,\s*[a-zA-Z_]|\n\s+}|\Z)",
                re.MULTILINE,
            )
            for kv in kv_re.finditer(inner):
                key = kv.group(1).strip()
                val_raw = kv.group(2).strip().rstrip(",")
                # Strip comments
                if "//" in val_raw:
                    val_raw = val_raw.split("//")[0].strip()
                try:
                    import ast
                    section_dict[key] = ast.literal_eval(val_raw)
                except Exception:
                    num_m = re.search(r"-?\d+(\.\d+)?", val_raw)
                    if num_m:
                        try:
                            section_dict[key] = float(num_m.group(0))
                        except Exception:
                            section_dict[key] = val_raw
            cfg[section] = section_dict

    return cfg


# Cache parsed config
_MJS_CACHE: Optional[dict] = None


def _load_mjs_config() -> dict:
    global _MJS_CACHE
    if _MJS_CACHE is None:
        _MJS_CACHE = _parse_config_mjs()
    return _MJS_CACHE


def _get_mjs(key_path: str, default: Any = None) -> Any:
    """
    Get value from parsed config.mjs by dot path.
    e.g. _get_mjs("TILE.sizePx") or _get_mjs("CELL_SIZE_M")
    """
    cfg = _load_mjs_config()
    parts = key_path.split(".")
    cur: Any = cfg
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


@dataclass
class InferenceConfig:
    """
    Inference-specific config, derived from src/config.mjs (source of truth).

    Tat ca gia tri duoc load tu src/config.mjs qua _get_mjs().
    Neu khong parse duoc (vd khong co config.mjs), fallback ve defaults
    mirror tu src/config.mjs.
    """

    # ---- Tile / quadrant sizes ----
    @property
    def tile_size_px(self) -> int:
        # TILE.sizePx trong src/config.mjs (default 1024)
        val = _get_mjs("TILE.sizePx", 1024)
        return int(val)

    @property
    def quadrant_size_px(self) -> int:
        # Moi tile = 2x2 quadrants (theo isometric-nyc/infill_template.py)
        return self.tile_size_px // 2

    @property
    def camera_move_step(self) -> float:
        # TILE.cameraMoveStep trong src/config.mjs (default 0.5)
        val = _get_mjs("TILE.cameraMoveStep", 0.5)
        return float(val)

    @property
    def stride_px(self) -> int:
        # TILE_STEP_PX = TILE.sizePx * TILE.cameraMoveStep (src/config.mjs:288)
        # Neu khong co derived constant, tinh truc tiep
        val = _get_mjs("TILE_STEP_PX", None)
        if val is None:
            val = int(round(self.tile_size_px * self.camera_move_step))
        return int(val)

    @property
    def cell_size_m(self) -> int:
        # CELL_SIZE_M trong src/config.mjs:278 (default 200m)
        val = _get_mjs("CELL_SIZE_M", 200)
        return int(val)

    @property
    def max_infill_area(self) -> int:
        # MAX_INFILL_AREA = TEMPLATE_SIZE^2 // 2 (isometric-nyc:44)
        return self.tile_size_px * self.tile_size_px // 2

    @property
    def border_width(self) -> int:
        # src/config.py.InfillConfig (default 2)
        try:
            from src.config import get_config
            return int(get_config().infill.border_width)
        except Exception:
            return 2

    @property
    def border_color(self) -> tuple:
        # src/config.py.InfillConfig (default (255, 0, 0, 255))
        try:
            from src.config import get_config
            col = get_config().infill.border_color
            return tuple(col) if not isinstance(col, tuple) else col
        except Exception:
            return (255, 0, 0, 255)

    @property
    def seam_color(self) -> tuple:
        # src/config.py.InfillConfig (default (255, 0, 0, 255))
        try:
            from src.config import get_config
            col = get_config().infill.seam_color
            return tuple(col) if not isinstance(col, tuple) else col
        except Exception:
            return (255, 0, 0, 255)

    @property
    def seam_thickness_px(self) -> int:
        # src/config.py.InfillConfig (default 1)
        try:
            from src.config import get_config
            return int(get_config().infill.seam_thickness_px)
        except Exception:
            return 1

    @property
    def max_infill_area_ratio(self) -> float:
        # src/config.py.InfillConfig (default 0.5)
        try:
            from src.config import get_config
            return float(get_config().infill.max_infill_area_ratio)
        except Exception:
            return 0.5

    @property
    def default_prompt(self) -> str:
        # src/config.py.InfillConfig.default_prompt
        try:
            from src.config import get_config
            return str(get_config().infill.default_prompt)
        except Exception:
            return (
                "Fill in the outlined section with coherent pixels matching the "
                "<isometric pixel art> style, seamlessly blending edges with "
                "surrounding areas, maintaining consistent isometric perspective, "
                "shadow direction, lighting, pixel density, and color harmony "
                "while preserving structural integrity and removing all border artifacts"
            )

    # ---- Paths (tu src/config.mjs.PATHS) ----
    @property
    def renders_dir(self) -> str:
        rel = _get_mjs("PATHS.renders", "./output/renders")
        if rel.startswith("./"):
            rel = rel[2:]
        return str(PROJECT_ROOT / rel)

    @property
    def output_dir(self) -> str:
        rel = _get_mjs("PATHS.output", "./output")
        if rel.startswith("./"):
            rel = rel[2:]
        return str(PROJECT_ROOT / rel)

    @property
    def paths(self) -> dict:
        return {
            "renders": self.renders_dir,
            "output": self.output_dir,
        }

    # ---- Server / HTTP (tu inference/.env hoac shell env) ----
    @property
    def endpoint(self) -> str:
        return os.environ.get("INFERENCE_ENDPOINT", "http://127.0.0.1:10100")

    @property
    def base_model(self) -> str:
        return os.environ.get("BASE_MODEL", "Qwen/Qwen-Image-Edit")

    @property
    def lora_path(self) -> str:
        return os.environ.get("LORA_PATH", "")

    @property
    def lora_weight(self) -> float:
        try:
            return float(os.environ.get("LORA_WEIGHT", "1.0"))
        except ValueError:
            return 1.0

    # ---- Model generation defaults ----
    @property
    def default_steps(self) -> int:
        return int(os.environ.get("INFERENCE_STEPS", "14"))

    @property
    def default_guidance(self) -> float:
        try:
            return float(os.environ.get("INFERENCE_GUIDANCE", "3.0"))
        except ValueError:
            return 3.0

    @property
    def default_true_cfg_scale(self) -> float:
        try:
            return float(os.environ.get("INFERENCE_TRUE_CFG", "2.0"))
        except ValueError:
            return 2.0


# Singleton instance
_config: Optional[InferenceConfig] = None


def get_inference_config() -> InferenceConfig:
    """Get the global inference config instance (lazy singleton)."""
    global _config
    if _config is None:
        _config = InferenceConfig()
    return _config


def reset_config() -> None:
    """Reset global config (for testing)."""
    global _config
    _config = None
    global _MJS_CACHE
    _MJS_CACHE = None


if __name__ == "__main__":
    cfg = get_inference_config()
    print("InferenceConfig:")
    print(f"  tile_size_px        = {cfg.tile_size_px}")
    print(f"  quadrant_size_px    = {cfg.quadrant_size_px}")
    print(f"  camera_move_step    = {cfg.camera_move_step}")
    print(f"  stride_px           = {cfg.stride_px}")
    print(f"  cell_size_m         = {cfg.cell_size_m}")
    print(f"  max_infill_area     = {cfg.max_infill_area}")
    print(f"  border_width        = {cfg.border_width}")
    print(f"  border_color        = {cfg.border_color}")
    print(f"  endpoint            = {cfg.endpoint}")
    print(f"  base_model          = {cfg.base_model}")
    print(f"  lora_path           = {cfg.lora_path}")
    print(f"  default_steps       = {cfg.default_steps}")
    print(f"  default_guidance    = {cfg.default_guidance}")
    print(f"  renders_dir         = {cfg.renders_dir}")
    print(f"  output_dir          = {cfg.output_dir}")
