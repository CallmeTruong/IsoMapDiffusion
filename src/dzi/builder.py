"""
DZI Builder — Creates Deep Zoom Image tiles from pixel art tiles.

Usage:
    python -m src.dzi.builder
    python -m src.dzi.builder --input output/map_plan.json --output output/gigapixel_map

    # Or via config defaults:
    python -m src.dzi.builder
"""

import os
import gc
import sys
import json
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

try:
    from ..config import get_dzi_config, PROJECT_ROOT
    from ..helpers import resolve_path
except ImportError:
    # Fallback for direct execution / when run from project root
    import sys
    from pathlib import Path
    # Add parent of 'src' to path so 'src.xxx' resolves
    _src_dir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(_src_dir.parent))
    from src.config import get_dzi_config, PROJECT_ROOT
    from src.helpers import resolve_path

# Apply VIPS settings from config
cfg = get_dzi_config()

# VIPS path setup
vips_bin = cfg.vips_bin_path
if not vips_bin:
    # Auto-detect common locations
    possible_paths = [
        PROJECT_ROOT / "vips" / "vips-dev-8.18" / "bin",
        Path("D:/isometric-map/vips/vips-dev-8.18/bin"),
    ]
    for p in possible_paths:
        if p.exists():
            vips_bin = str(p)
            break

if vips_bin:
    os.environ['PATH'] = vips_bin + ';' + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(vips_bin)

if cfg.vips_progress:
    os.environ['VIPS_PROGRESS'] = '1'

os.environ['VIPS_CONCURRENCY'] = str(cfg.vips_concurrency)

import pyvips

# VIPS cache settings
pyvips.cache_set_max_mem(cfg.cache_max_mem_mb * 1024 * 1024)
pyvips.cache_set_max(cfg.cache_max_files)


# ─── Builder ──────────────────────────────────────────────────────────────────

def composite_strip(tiles_info, canvas_w, strip_y0, strip_y1, strip_height):
    """Composite tiles into a horizontal strip."""
    strip_h = strip_y1 - strip_y0

    strip_tiles = [
        t for t in tiles_info
        if os.path.exists(t['path'])
        and t['y'] < strip_y1
        and t['y'] + t.get('height', 1024) > strip_y0
    ]

    background = pyvips.Image.black(canvas_w, strip_h, bands=4).copy(interpretation='srgb')

    if not strip_tiles:
        return background

    images, xs, ys = [], [], []
    for t in strip_tiles:
        img = pyvips.Image.new_from_file(t['path'], access='sequential')
        if img.bands < 4:
            img = img.addalpha()
        if img.interpretation != 'srgb':
            img = img.colourspace('srgb')

        # Make unrendered black pixels (RGB <= 2) transparent (alpha = 0)
        # so they never overwrite valid content underneath in DZI composite.
        non_black = (img[0] > 2) | (img[1] > 2) | (img[2] > 2)
        alpha = non_black.ifthenelse(img[3], 0)
        img = img[0:3].bandjoin(alpha)

        images.append(img)
        xs.append(t['x'])
        ys.append(t['y'] - strip_y0)

    return background.composite(images, [2] * len(images), x=xs, y=ys)


def build_dzi_from_plan(json_path, output_dzi_name, strip_height=None):
    """
    Build DZI from a map plan JSON.

    Args:
        json_path: Path to map_plan.json
        output_dzi_name: Output DZI basename (without extension)
        strip_height: Override strip height from config
    """
    cfg = get_dzi_config()
    strip_height = strip_height or cfg.strip_height

    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Error: Missing file:\n   {json_path}")
        sys.exit(1)

    print(f"Reading: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    canvas_w = plan['canvasWidth']
    canvas_h = plan['canvasHeight']
    tiles_info = plan['tiles']

    print(f"Canvas size: {canvas_w} x {canvas_h}")
    print(f"Tiles: {len(tiles_info)}")

    temp_tiff = str(output_dzi_name) + "_temp.tif"

    if os.path.exists(temp_tiff):
        print(f"\nFound existing TIFF, skipping Step 1...")
    else:
        import math
        num_strips = math.ceil(canvas_h / strip_height)
        strip_tiffs = []

        print(f"\nStep 1: Writing {num_strips} strips to TIFF...")

        for i in range(num_strips):
            strip_y0 = i * strip_height
            strip_y1 = min(strip_y0 + strip_height, canvas_h)
            print(f"  Strip {i+1}/{num_strips}: y={strip_y0}~{strip_y1}")

            strip_img = composite_strip(tiles_info, canvas_w, strip_y0, strip_y1, strip_height)

            strip_path = str(output_dzi_name) + f"_strip_{i}.tif"
            strip_img.tiffsave(
                strip_path,
                tile=True,
                tile_width=cfg.tile_size,
                tile_height=cfg.tile_size,
                compression=cfg.compression,
                bigtiff=True,
            )
            strip_tiffs.append(strip_path)
            del strip_img

        # Merge strips
        print(f"\nMerging {len(strip_tiffs)} strips into single TIFF...")
        strip_images = []
        for p in strip_tiffs:
            img = pyvips.Image.new_from_file(p, access='sequential')
            strip_images.append(img)

        full_image = pyvips.Image.arrayjoin(strip_images, across=1)

        full_image.tiffsave(
            temp_tiff,
            tile=True,
            tile_width=cfg.tile_size,
            tile_height=cfg.tile_size,
            compression=cfg.compression,
            bigtiff=True,
        )
        del full_image, strip_images
        gc.collect()

        # Cleanup strip files
        for p in strip_tiffs:
            try:
                os.remove(p)
            except PermissionError:
                print(f"   Warning: cannot delete {p}")

        print(f"  TIFF done.")

    # Export DZI from TIFF
    print(f"\nStep 2: Exporting DZI from TIFF...")
    tiff_img = pyvips.Image.new_from_file(temp_tiff, access='sequential')
    tiff_img.dzsave(
        str(output_dzi_name),
        overlap=0,
        tile_size=cfg.tile_size,
        suffix=f'.jpg[Q={cfg.jpeg_quality}]',
    )

    del tiff_img
    gc.collect()

    print(f"\nCleaning up temp file...")
    try:
        os.remove(temp_tiff)
    except PermissionError:
        print(f"   Warning: cannot delete temp file at:\n   {temp_tiff}")

    print(f"\nComplete! DZI saved at:\n   {output_dzi_name}.dzi")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Build DZI from map plan')

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to map_plan.json (default: from config)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output DZI basename (default: from config)'
    )

    parser.add_argument(
        '--strip-height',
        type=int,
        help='Strip height in pixels (default: from config)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_dzi_config()

    # Resolve paths
    input_path = args.input or cfg.input_json
    output_name = args.output or cfg.output_name

    # Make paths absolute if relative
    input_path = resolve_path(input_path)
    output_name = resolve_path(output_name)

    print("=" * 60)
    print("DZI Builder")
    print("=" * 60)
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_name}")
    print(f"  Strip:   {args.strip_height or cfg.strip_height}px")
    print("=" * 60)

    build_dzi_from_plan(input_path, output_name, args.strip_height)


if __name__ == "__main__":
    main()
