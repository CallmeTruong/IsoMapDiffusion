#!/usr/bin/env python3
"""Stitch all generated tile_*.png trong 1 directory thanh 1 map lon.

Tuong duong logic src/tile/stitch/compose.mjs::stitchTiles nhung thuan Python.
stride = 512px (50% overlap voi tile 1024px), weighted-blend o v?ng bien.

Usage:
    python -m inference.scripts.stitch_generated ^
        --input ./generate ^
        --output ./isometric_world.png
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


log = logging.getLogger("stitch_gen")

TILE_SIZE = 1024
DEFAULT_STRIDE = 512
DEFAULT_BACKGROUND = (255, 255, 255)


def sign_int(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


TILE_FILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")


def discover_tiles(gen_dir: Path, must_be_nonempty: bool = True):
    """Qu?t directory, tr? v? {(qx, qy): path}."""
    if not gen_dir.exists():
        raise FileNotFoundError(gen_dir)
    out: dict = {}
    for f in gen_dir.glob("tile_*_*_*.png"):
        m = TILE_FILE_RE.match(f.name)
        if not m:
            continue
        if must_be_nonempty and f.stat().st_size < 30 * 1024:
            continue
        qx, qy = int(m.group(1)), int(m.group(2))
        out[(qx, qy)] = f
    return out


def compute_bounds(tiles: dict):
    if not tiles:
        return 0, -1, 0, -1
    qxs = [k[0] for k in tiles]
    qys = [k[1] for k in tiles]
    return min(qxs), max(qxs), min(qys), max(qys)


def stitch_all(gen_dir: Path, output_path: Path,
               stride: int = DEFAULT_STRIDE,
               scale: float = 0.5,
               background: tuple = DEFAULT_BACKGROUND) -> dict:
    """Stitch tat ca tile trong `gen_dir` -> `output_path`. Tra ve stats dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = discover_tiles(gen_dir)
    if not tiles:
        log.error(f"No tiles found in {gen_dir}")
        return {"count": 0}

    min_qx, max_qx, min_qy, max_qy = compute_bounds(tiles)
    log.info(f"Found {len(tiles)} tiles, qx in [{min_qx},{max_qx}] qy in [{min_qy},{max_qy}]")

    full_w = (max_qx - min_qx + 1) * stride + (TILE_SIZE - stride)
    full_h = (max_qy - min_qy + 1) * stride + (TILE_SIZE - stride)

    # Auto-adjust scale if canvas would exceed 1.5 GB memory
    est_mem_gb = (full_w * scale) * (full_h * scale) * 4 * 4 / (1024**3)
    if est_mem_gb > 1.5 and scale > 0.1:
        auto_scale = round(scale * (1.5 / est_mem_gb) ** 0.5, 2)
        auto_scale = max(0.1, auto_scale)
        log.warning(f"Full resolution ({full_w}x{full_h}) requires {est_mem_gb:.1f} GB RAM. Auto-adjusting --scale from {scale} to {auto_scale}.")
        scale = auto_scale

    tile_size_s = max(1, int(TILE_SIZE * scale))
    stride_s = max(1, int(stride * scale))
    width_px = (max_qx - min_qx + 1) * stride_s + (tile_size_s - stride_s)
    height_px = (max_qy - min_qy + 1) * stride_s + (tile_size_s - stride_s)

    log.info(f"Output preview size: {width_px}x{height_px} (scale={scale:.2f})")

    canvas = np.full((height_px, width_px, 3), background, dtype=np.float32)
    weight = np.zeros((height_px, width_px), dtype=np.float32)

    tile_weight = np.ones((tile_size_s, tile_size_s), dtype=np.float32)
    edge = max(1, int(scale))
    tile_weight[:edge] *= np.linspace(0.5, 1.0, edge).reshape(-1, 1)
    tile_weight[-edge:] *= np.linspace(1.0, 0.5, edge).reshape(-1, 1)
    tile_weight[:, :edge] *= np.linspace(0.5, 1.0, edge).reshape(1, -1)
    tile_weight[:, -edge:] *= np.linspace(1.0, 0.5, edge).reshape(1, -1)

    for (qx, qy), path in sorted(tiles.items()):
        # Handle concurrent tile updates where pipeline renames old tile hash
        img_pil = None
        target_path = path
        if not target_path.exists():
            matches = list(gen_dir.glob(f"tile_{sign_int(qx)}_{sign_int(qy)}_*.png"))
            if matches:
                target_path = matches[0]

        if target_path.exists():
            try:
                img_pil = Image.open(target_path).convert("RGB")
            except Exception:
                img_pil = None

        if img_pil is None:
            continue

        if scale != 1.0 or img_pil.size != (TILE_SIZE, TILE_SIZE):
            img_pil = img_pil.resize((tile_size_s, tile_size_s), Image.Resampling.BILINEAR)
        img = np.asarray(img_pil, dtype=np.float32)

        x0 = (qx - min_qx) * stride_s
        y0 = (qy - min_qy) * stride_s
        x1 = x0 + tile_size_s
        y1 = y0 + tile_size_s

        for c in range(3):
            canvas[y0:y1, x0:x1, c] += img[:, :, c] * tile_weight
        weight[y0:y1, x0:x1] += tile_weight

    weight = np.where(weight > 0, weight, 1.0)
    canvas = canvas / weight[:, :, None]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(canvas)
    out_img.save(output_path, format="PNG", optimize=True)

    size_kb = output_path.stat().st_size / 1024
    log.info(f"Saved: {output_path} ({size_kb:.1f} KB, {out_img.size[0]}x{out_img.size[1]})")

    return {
        "count": len(tiles),
        "width": out_img.size[0],
        "height": out_img.size[1],
        "size_kb": size_kb,
        "bounds": {"min_qx": min_qx, "max_qx": max_qx,
                   "min_qy": min_qy, "max_qy": max_qy},
        "stride": stride,
    }


def main():
    parser = argparse.ArgumentParser(description="Stitch generated tiles into a map")
    parser.add_argument("--input", type=str, required=True,
                        help="Gen dir chua tile_*_*_*.png")
    parser.add_argument("--output", type=str, required=True,
                        help="Output PNG path")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"Pixel stride giua 2 tile -- default: {DEFAULT_STRIDE}")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scale factor for preview output (default: 0.5 for lower RAM footprint)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    t0 = time.monotonic()
    stats = stitch_all(
        gen_dir=Path(args.input),
        output_path=Path(args.output),
        stride=args.stride,
        scale=args.scale,
    )
    log.info(f"Done in {time.monotonic()-t0:.1f}s ? {stats}")


if __name__ == "__main__":
    main()
