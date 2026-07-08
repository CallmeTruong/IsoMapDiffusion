"""Build map_plan.json tu model_generate/*.png de feed cho src.dzi.builder.

Tuong duong src/dzi/export_plan.mjs nhung pure Python.

Quy uoc tile:
    tile_[+-]<qx>_[+-]<qy>_<8hex>.png

Layout:
    - stride = 512 px (50% overlap, tile 1024 px, cameraMoveStep=0.5)
    - canvas = ((max_qx - min_qx + 1) * 512 + padding*2, ...)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")
DEFAULT_PADDING = 100
DEFAULT_STRIDE = 512


def export_dzi_plan(gen_dir, out_plan, padding=DEFAULT_PADDING, stride=DEFAULT_STRIDE):
    """Quet gen_dir, ghi JSON plan cho src.dzi.builder."""
    tiles = []
    min_qx = min_qy = float("inf")
    max_qx = max_qy = float("-inf")

    for p in Path(gen_dir).glob("tile_+*_+*_*.png"):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        if p.stat().st_size < 30 * 1024:
            continue
        qx, qy = int(m.group(1)), int(m.group(2))
        if qx < min_qx: min_qx = qx
        if qx > max_qx: max_qx = qx
        if qy < min_qy: min_qy = qy
        if qy > max_qy: max_qy = qy
        tiles.append({"path": str(p.resolve()), "qx": qx, "qy": qy})

    if not tiles:
        raise FileNotFoundError(f"No valid tiles in {gen_dir}")

    map_w = (max_qx - min_qx + 1) * stride + 2 * padding
    map_h = (max_qy - min_qy + 1) * stride + 2 * padding

    plan_tiles = []
    for t in tiles:
        dx = (t["qx"] - min_qx) * stride + padding
        dy = (max_qy - t["qy"]) * stride + padding
        plan_tiles.append({"path": t["path"], "x": dx, "y": dy})

    plan = {
        "canvasWidth": int(map_w),
        "canvasHeight": int(map_h),
        "tiles": plan_tiles,
    }

    out_plan = Path(out_plan)
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {
        "tiles": len(plan_tiles),
        "canvas": [int(map_w), int(map_h)],
        "plan_path": str(out_plan.resolve()),
    }
