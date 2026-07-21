"""Build map_plan.json tu model_generate/*.png de feed cho src.dzi.builder.

Tuong duong src/dzi/export_plan.mjs nhung pure Python.

Quy uoc tile:
    tile_[+-]<qx>_[+-]<qy>_<8hex>.png

Layout (must match stitch_generated.py):
    - tile_size  = 1024 px
    - stride     = 512 px (50% overlap)
    - canvas_w   = (max_qx - min_qx + 1) * stride + (tile_size - stride)
    - canvas_h   = (max_qy - min_qy + 1) * stride + (tile_size - stride)

Coordinate convention (MUST MATCH stitch_generated.py):
    - qx increases → x increases → moves RIGHT on canvas
    - qy increases → y increases → moves DOWN on canvas
    - x = (qx - min_qx) * stride
    - y = (qy - min_qy) * stride
    This is the SAME direct top-down system used by stitch_generated.py.

Output JSON (compatible voi src/dzi/builder.py):
    {
        "canvasWidth":  int,
        "canvasHeight": int,
        "tiles": [
            {"path": str, "x": int, "y": int}
        ]
    }
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")

# Defaults matching stitch_generated.py and src/config.mjs
DEFAULT_TILE_SIZE = 1024
DEFAULT_STRIDE = 512


def export_dzi_plan(gen_dir, out_plan, stride=None, tile_size=None):
    """
    Quet gen_dir, ghi JSON plan cho src.dzi.builder.

    Canvas size and tile placement use the SAME formula as
    stitch_generated.py to ensure pixel-perfect alignment:
        canvas_w = (max_qx - min_qx + 1) * stride + (tile_size - stride)
        canvas_h = (max_qy - min_qy + 1) * stride + (tile_size - stride)
        tile_x   = (qx - min_qx) * stride
        tile_y   = (qy - min_qy) * stride

    Args:
        gen_dir:   Path toi folder chua tile_<qx>_<qy>_<hash>.png.
        out_plan:  Path output JSON file.
        stride:    Buoc nhay pixel giua 2 tile ke nhau (default: 512).
        tile_size: Kich thuoc tile theo pixel (default: 1024).

    Returns:
        dict voi keys: tiles, canvas, bounds, stride, tile_size, plan_path

    Raises:
        FileNotFoundError: neu khong co tile hop le trong gen_dir.
    """
    if stride is None:
        stride = DEFAULT_STRIDE
    if tile_size is None:
        tile_size = DEFAULT_TILE_SIZE

    tiles = []
    min_qx = min_qy = float("inf")
    max_qx = max_qy = float("-inf")

    for p in Path(gen_dir).glob("tile_*_*_*.png"):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        if p.stat().st_size < 1024:
            continue
        qx, qy = int(m.group(1)), int(m.group(2))
        if qx < min_qx: min_qx = qx
        if qx > max_qx: max_qx = qx
        if qy < min_qy: min_qy = qy
        if qy > max_qy: max_qy = qy
        tiles.append({"path": str(p.resolve()), "qx": qx, "qy": qy})

    if not tiles:
        raise FileNotFoundError(f"No valid tiles in {gen_dir}")

    # Canvas size: SAME formula as stitch_generated.py lines 77-78
    #   full_w = (max_qx - min_qx + 1) * stride + (TILE_SIZE - stride)
    #   full_h = (max_qy - min_qy + 1) * stride + (TILE_SIZE - stride)
    map_w = (max_qx - min_qx + 1) * stride + (tile_size - stride)
    map_h = (max_qy - min_qy + 1) * stride + (tile_size - stride)

    # Sort tiles by row (qy) then column (qx) matching stitch_generated.py
    tiles.sort(key=lambda t: (t["qy"], t["qx"]))

    plan_tiles = []
    for t in tiles:
        dx = (t["qx"] - min_qx) * stride
        dy = (t["qy"] - min_qy) * stride
        plan_tiles.append({"path": t["path"], "x": dx, "y": dy})

    # Format khop src/dzi/builder.py: canvasWidth / canvasHeight / tiles[{path,x,y}]
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
        "bounds": {
            "min_qx": int(min_qx),
            "max_qx": int(max_qx),
            "min_qy": int(min_qy),
            "max_qy": int(max_qy),
        },
        "stride": int(stride),
        "tile_size": int(tile_size),
        "plan_path": str(out_plan.resolve()),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Export DZI plan JSON tu tile PNGs (src/dzi/builder compatible)."
    )
    p.add_argument("--input", "-i", required=True,
                   help="Path toi folder chua tile_*.png (gen_dir).")
    p.add_argument("--output", "-o", required=True,
                   help="Path output JSON plan (vd output/model_map_plan.json).")
    p.add_argument("--stride", type=int, default=None,
                   help="Override stride -- default: 512.")
    p.add_argument("--tile-size", type=int, default=None,
                   help="Override tile_size -- default: 1024.")
    args = p.parse_args()

    result = export_dzi_plan(
        gen_dir=args.input,
        out_plan=args.output,
        stride=args.stride,
        tile_size=args.tile_size,
    )
    print(f"Plan written: {result['plan_path']}")
    print(f"  tiles   = {result['tiles']}")
    print(f"  canvas  = {result['canvas'][0]} x {result['canvas'][1]}")
    print(f"  bounds  = {result['bounds']}")
    print(f"  stride  = {result['stride']}")