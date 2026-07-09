"""Build map_plan.json tu model_generate/*.png de feed cho src.dzi.builder.

Tuong duong src/dzi/export_plan.mjs nhung pure Python.

Quy uoc tile:
    tile_[+-]<qx>_[+-]<qy>_<8hex>.png

Layout (mirror src/config.mjs):
    - tile_size  = 1024 px (TILE.sizePx, source of truth)
    - stride     = TILE.sizePx * TILE.cameraMoveStep (default 0.5 -> 512px,
                    i.e. 50% overlap giua cac tile 1024 lien ke)
    - canvas     = ((max_qx - min_qx + 1) * stride + padding*2, ...)

Output JSON (compatible voi src/dzi/builder.py):
    {
        "canvasWidth":  int,
        "canvasHeight": int,
        "tiles": [
            {"path": str, "x": int, "y": int}
        ]
    }

Y-axis convention (IMPORTANT - bottom-up):
    - tile coords (qx, qy) tang theo chieu X phai, chieu Y XUONG (top-down)
      trong file naming va generation plan.
    - Canvas (PIL / DZI) dung chieu Y DI XUONG (top-left origin), nhung
      isometric map lay qy lon nhat lam "top of world" (vi the gioi isometric
      nhin tu tren xuong, qy cang lon cang o phia Bac / xa camera).
    - De dat tile qy lon o phia TREN canvas: y_canvas = (max_qy - qy) * stride + padding.
      Day la quy uoc bottom-up: tile o qy=max nam o y_canvas = padding (top).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")
DEFAULT_PADDING = 100


def _get_inference_config():
    """
    Lazy import de tranh hard import inference.config (giup goi duoc tu moi context).
    """
    try:
        from inference.config import get_inference_config as _gic
        return _gic()
    except Exception:
        # Fallback: hardcoded mirror cua src/config.mjs defaults
        class _Fallback:
            tile_size_px = 1024
            camera_move_step = 0.5
            @property
            def stride_px(self) -> int:
                return int(round(self.tile_size_px * self.camera_move_step))
        return _Fallback()


def export_dzi_plan(gen_dir, out_plan, padding=DEFAULT_PADDING, stride=None, tile_size=None):
    """
    Quet gen_dir, ghi JSON plan cho src.dzi.builder.

    Args:
        gen_dir:  Path toi folder chua tile_<qx>_<qy>_<hash>.png.
        out_plan: Path output JSON file.
        padding:  So pixel padding 2 canh canvas (default 100).
        stride:   Buoc nhay pixel giua 2 tile ke nhau (default: tile_size * 0.5 = 512).
        tile_size: Kich thuoc tile theo pixel (default: 1024, tu src/config.mjs TILE.sizePx).

    Returns:
        dict voi keys:
            tiles:   int   - so tile trong plan
            canvas:  [w, h] - kich thuoc canvas
            bounds:  dict  - {min_qx, max_qx, min_qy, max_qy} trong quadrant coords
            plan_path: str - absolute path toi JSON vua ghi

    Raises:
        FileNotFoundError: neu khong co tile hop le trong gen_dir.
    """
    cfg = _get_inference_config()

    # Lay stride/tile_size tu config neu caller khong truyen.
    # stride_px = tile_size_px * camera_move_step (src/config.mjs: TILE_STEP_PX).
    if stride is None:
        stride = cfg.stride_px
    if tile_size is None:
        tile_size = cfg.tile_size_px

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

    # Canvas size: (so tile theo truc X * stride) + 2 canh padding
    map_w = (max_qx - min_qx + 1) * stride + 2 * padding
    map_h = (max_qy - min_qy + 1) * stride + 2 * padding

    # Y-axis bottom-up: qy=max (xa camera nhat) -> y_canvas = padding (top of canvas).
    plan_tiles = []
    for t in tiles:
        dx = (t["qx"] - min_qx) * stride + padding
        dy = (max_qy - t["qy"]) * stride + padding
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
        "padding": int(padding),
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
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING,
                   help="Canvas padding 2 canh (default: 100).")
    p.add_argument("--stride", type=int, default=None,
                   help="Override stride (default: tile_size * camera_move_step tu config).")
    p.add_argument("--tile-size", type=int, default=None,
                   help="Override tile_size (default: tu src/config.mjs TILE.sizePx).")
    args = p.parse_args()

    result = export_dzi_plan(
        gen_dir=args.input,
        out_plan=args.output,
        padding=args.padding,
        stride=args.stride,
        tile_size=args.tile_size,
    )
    print(f"Plan written: {result['plan_path']}")
    print(f"  tiles   = {result['tiles']}")
    print(f"  canvas  = {result['canvas'][0]} x {result['canvas'][1]}")
    print(f"  bounds  = {result['bounds']}")
    print(f"  stride  = {result['stride']}")
    print(f"  padding = {result['padding']}")