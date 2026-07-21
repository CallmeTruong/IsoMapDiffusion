#!/usr/bin/env python3
"""Postprocess all map tiles: Unified Color Palette Quantization & Color Balancing.

Adapts NYC pixel art color normalization for our isometric map tiles:
1. Samples representative pixels from all non-black regions across generated tiles.
2. Builds a unified 256-color palette.
3. Quantizes every tile to the unified palette so grass, pavement, roofs, and shadows
   have 100% identical RGB values across all tiles, eliminating color drift and seam lines.

Usage:
    python -m inference.scripts.postprocess_map ^
        --input output/model_generate ^
        --output output/model_generate_processed ^
        --num-colors 256
"""
import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("postprocess_map")


def sample_colors_from_tiles(
    tile_paths: list[Path],
    sample_tiles_count: int = 200,
    pixels_per_tile: int = 1000,
) -> list[tuple[int, int, int]]:
    """Sample non-black colors across tiles to build a representative color set."""
    if len(tile_paths) > sample_tiles_count:
        sampled_paths = random.sample(tile_paths, sample_tiles_count)
    else:
        sampled_paths = tile_paths

    sampled_colors: list[tuple[int, int, int]] = []

    for path in sampled_paths:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            # Mask out black / unrendered pixels (RGB <= 5)
            non_black_mask = (arr > 5).any(axis=2)
            valid_pixels = arr[non_black_mask]
            if len(valid_pixels) == 0:
                continue

            if len(valid_pixels) > pixels_per_tile:
                idx = np.random.choice(len(valid_pixels), size=pixels_per_tile, replace=False)
                sub_pixels = valid_pixels[idx]
            else:
                sub_pixels = valid_pixels

            for p in sub_pixels:
                sampled_colors.append((int(p[0]), int(p[1]), int(p[2])))
        except Exception as e:
            log.warning("Could not read %s for palette sampling: %s", path, e)

    return sampled_colors


def build_unified_palette(colors: list[tuple[int, int, int]], num_colors: int = 256) -> Image.Image:
    """Build a unified P-mode palette image from sampled colors."""
    if not colors:
        # Fallback to standard 256 color adaptive palette
        dummy = Image.new("RGB", (256, 256), (128, 128, 128))
        return dummy.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)

    side = int(len(colors) ** 0.5) + 1
    composite = Image.new("RGB", (side, side), (0, 0, 0))
    pixels = composite.load()

    for i, color in enumerate(colors):
        x = i % side
        y = i // side
        if y < side:
            pixels[x, y] = color

    palette_img = composite.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
    return palette_img


def process_single_tile(
    tile_path_str: str,
    output_dir_str: str,
    palette_path_str: str,
    dither: bool = False,
) -> str:
    """Worker task: quantize one tile to the shared palette, keeping black pixels black."""
    tile_path = Path(tile_path_str)
    output_dir = Path(output_dir_str)
    out_path = output_dir / tile_path.name

    try:
        img = Image.open(tile_path).convert("RGB")
        arr = np.asarray(img)
        non_black_mask = (arr > 5).any(axis=2)

        # Load unified palette image with its complete RGB color palette table
        palette_img = Image.open(palette_path_str)

        # Quantize image to palette
        quantized = img.quantize(palette=palette_img, dither=1 if dither else 0).convert("RGB")
        q_arr = np.asarray(quantized, dtype=np.uint8)

        # Preserve black unrendered pixels
        final_arr = np.zeros_like(arr)
        final_arr[non_black_mask] = q_arr[non_black_mask]

        final_img = Image.fromarray(final_arr, mode="RGB")
        final_img.save(out_path, format="PNG", optimize=True)
        return str(out_path)
    except Exception as e:
        return f"ERROR: {tile_path.name} - {e}"


def postprocess_all(
    input_dir: Path,
    output_dir: Path,
    num_colors: int = 256,
    dither: bool = False,
    workers: int = 8,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = list(input_dir.glob("tile_*_*_*.png"))
    if not tile_paths:
        log.error("No tile PNGs found in %s", input_dir)
        return {"count": 0}

    log.info("Found %d tiles in %s. Building unified %d-color palette...", len(tile_paths), input_dir, num_colors)
    colors = sample_colors_from_tiles(tile_paths, sample_tiles_count=200, pixels_per_tile=1000)
    palette_img = build_unified_palette(colors, num_colors=num_colors)

    palette_path = output_dir / "unified_palette.png"
    palette_img.save(palette_path)
    log.info("Saved unified palette to %s", palette_path)

    log.info("Processing %d tiles in parallel with %d workers...", len(tile_paths), workers)

    processed_count = 0
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_tile,
                str(p),
                str(output_dir),
                str(palette_path),
                dither,
            ): p
            for p in tile_paths
        }

        for future in as_completed(futures):
            res = future.result()
            if not res.startswith("ERROR"):
                processed_count += 1
            else:
                log.warning(res)

    elapsed = time.monotonic() - t0
    log.info("Done postprocessing %d/%d tiles in %.1fs -> %s", processed_count, len(tile_paths), elapsed, output_dir)

    return {
        "count": processed_count,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "palette": str(palette_path),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Postprocess tiles with unified palette quantization")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input gen dir (output/model_generate)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output dir for postprocessed tiles (output/model_generate_processed)")
    parser.add_argument("--num-colors", type=int, default=256,
                        help="Palette colors count (default: 256)")
    parser.add_argument("--dither", action="store_true",
                        help="Enable dithering during quantization")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes (default: 8)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    postprocess_all(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        num_colors=args.num_colors,
        dither=args.dither,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
