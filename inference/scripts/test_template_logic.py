"""
inference/scripts/test_template_logic.py

Test TemplateBuilder local (khong can GPU server).
Verify logic ghep template moi (port tu isometric-nyc) chay dung.

Su dung:
    python -m inference.scripts.test_template_logic \\
        --renders output/renders \\
        --output /tmp/test_template_output

Output:
    - In ra placement info cho moi tile
    - Luu template_*.png vao /tmp/test_template_output/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from inference import (
    TemplateBuilder,
    InfillRegion,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
    get_inference_config,
)


def test_seed_tile(renders_dir: Path, gen_dir: Path, output_dir: Path):
    """
    Test 1: seed tile (0, 0) - khong co neighbor nao, full template.
    """
    print("\n=== Test 1: seed tile (0, 0) full template ===")
    cfg = get_inference_config()
    render_cache: dict = {}
    generation_cache: dict = {}

    region = InfillRegion.full_tile(0, 0)
    print(f"  region = {region}")
    print(f"  is_valid_size = {region.is_valid_size()}")
    print(f"  is_full_tile = {region.is_full_tile()}")
    print(f"  overlapping_quadrants = {region.overlapping_quadrants()}")

    generated_set = scan_generated_set(gen_dir)
    has_gen = make_has_generation(generated_set)
    get_render = make_render_provider(renders_dir, render_cache)
    get_generation = make_generation_provider(gen_dir, generation_cache)

    builder = TemplateBuilder(
        infill_region=region,
        has_generation=has_gen,
        get_render=get_render,
        get_generation=get_generation,
    )
    print(f"  has_left_gen  = {builder._has_generated_context('left')}")
    print(f"  has_right_gen = {builder._has_generated_context('right')}")
    print(f"  has_top_gen   = {builder._has_generated_context('top')}")
    print(f"  has_bottom_gen= {builder._has_generated_context('bottom')}")

    result = builder.build()
    if result is None:
        print(f"  [ERROR] no valid placement: {builder._last_validation_error}")
        return None
    template, placement = result
    print(f"  placement = {placement}")
    print(f"  template.size = {template.size}")
    print(f"  template.mode = {template.mode}")

    out_path = output_dir / "template_seed_0_0.png"
    template.save(out_path)
    print(f"  Saved: {out_path}")
    return template, placement


def test_neighbor_tile(renders_dir: Path, gen_dir: Path, output_dir: Path,
                        seed_qx: int, seed_qy: int,
                        neighbor_qx: int, neighbor_qy: int):
    """
    Test 2: tile (neighbor_qx, neighbor_qy) co seed la (seed_qx, seed_qy) da gen.

    Seed phai co san trong gen_dir (manual test: copy from renders/ hoac generate
    1 fake). Day la test logic placement.
    """
    print(f"\n=== Test 2: tile ({neighbor_qx}, {neighbor_qy}) "
          f"with seed ({seed_qx}, {seed_qy}) generated ===")
    cfg = get_inference_config()
    render_cache: dict = {}
    generation_cache: dict = {}

    # Fake seed generation: copy render thanh "generation" de test logic
    seed_render = renders_dir / f"tile_{seed_qx >= 0 and '+' or ''}{seed_qx}_" \
                                   f"{seed_qy >= 0 and '+' or ''}{seed_qy}_*.png"
    seed_render_files = list(renders_dir.glob(
        f"tile_{'+' if seed_qx >= 0 else ''}{seed_qx}_"
        f"{'+' if seed_qy >= 0 else ''}{seed_qy}_*.png"
    ))
    if not seed_render_files:
        # Try alternate sign format
        seed_render_files = list(renders_dir.glob(
            f"tile_{seed_qx}_{seed_qy}_*.png"
        ))
    if not seed_render_files:
        print(f"  [SKIP] no render file for seed ({seed_qx}, {seed_qy})")
        return None

    # Copy seed render thanh fake generation (de test context lookup)
    fake_gen_path = gen_dir / f"tile_{'+' if seed_qx >= 0 else ''}{seed_qx}_" \
                                 f"{'+' if seed_qy >= 0 else ''}{seed_qy}_seed.png"
    fake_gen_path.parent.mkdir(parents=True, exist_ok=True)
    seed_img = Image.open(seed_render_files[0])
    seed_img.save(fake_gen_path)
    print(f"  Created fake generation: {fake_gen_path}")

    # Re-scan generated set
    generated_set = scan_generated_set(gen_dir)
    print(f"  generated_set = {generated_set}")
    has_gen = make_has_generation(generated_set)

    # Test placement for (neighbor_qx, neighbor_qy)
    region = InfillRegion.full_tile(neighbor_qx, neighbor_qy)
    get_render = make_render_provider(renders_dir, render_cache)
    get_generation = make_generation_provider(gen_dir, generation_cache)

    builder = TemplateBuilder(
        infill_region=region,
        has_generation=has_gen,
        get_render=get_render,
        get_generation=get_generation,
    )
    print(f"  has_left_gen  = {builder._has_generated_context('left')}")
    print(f"  has_right_gen = {builder._has_generated_context('right')}")
    print(f"  has_top_gen   = {builder._has_generated_context('top')}")
    print(f"  has_bottom_gen= {builder._has_generated_context('bottom')}")

    result = builder.build()
    if result is None:
        print(f"  [ERROR] no valid placement: {builder._last_validation_error}")
        # Day co the la expected behavior neu seed o sat mep
        return None
    template, placement = result
    print(f"  placement = {placement}")

    out_path = output_dir / f"template_{neighbor_qx}_{neighbor_qy}_with_seed_{seed_qx}_{seed_qy}.png"
    template.save(out_path)
    print(f"  Saved: {out_path}")
    return template, placement


def main():
    parser = argparse.ArgumentParser(
        description="Test TemplateBuilder local (no GPU).")
    parser.add_argument("--renders", type=str, default="output/renders",
                        help="Path to renders dir (default: output/renders)")
    parser.add_argument("--output", type=str, default="/tmp/test_template_output",
                        help="Path to output dir for template PNGs")
    args = parser.parse_args()

    renders_dir = PROJECT_ROOT / args.renders
    output_dir = Path(args.output)
    gen_dir = PROJECT_ROOT / "model_generate_test"  # Use existing test dir

    if not renders_dir.exists():
        print(f"ERROR: renders dir not found: {renders_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_inference_config()
    print("InferenceConfig:")
    print(f"  tile_size_px        = {cfg.tile_size_px}")
    print(f"  quadrant_size_px    = {cfg.quadrant_size_px}")
    print(f"  camera_move_step    = {cfg.camera_move_step}")
    print(f"  stride_px           = {cfg.stride_px}")
    print(f"  max_infill_area     = {cfg.max_infill_area}")
    print(f"  border_width        = {cfg.border_width}")
    print(f"  renders_dir         = {renders_dir}")

    # Discover tiles
    from inference.scripts.run_inference_pipeline import discover_tiles
    tiles = discover_tiles(renders_dir)
    print(f"\nDiscovered {len(tiles)} tiles: {tiles[:5]}{'...' if len(tiles) > 5 else ''}")

    if not tiles:
        print("No tiles to test. Exiting.")
        sys.exit(1)

    # Test 1: seed tile (no neighbors)
    test_seed_tile(renders_dir, gen_dir, output_dir)

    # Test 2: tile with 1 neighbor generated (use first tile as fake seed)
    seed = min(tiles, key=lambda t: abs(t[0]) + abs(t[1]))
    # Pick 1 neighbor that exists in tiles
    for neighbor in [(seed[0] + 1, seed[1]), (seed[0], seed[1] + 1),
                     (seed[0] - 1, seed[1]), (seed[0], seed[1] - 1)]:
        if neighbor in set(tiles):
            test_neighbor_tile(renders_dir, gen_dir, output_dir,
                               seed[0], seed[1], neighbor[0], neighbor[1])
            break

    print(f"\n=== Done. Output: {output_dir} ===")


if __name__ == "__main__":
    main()
