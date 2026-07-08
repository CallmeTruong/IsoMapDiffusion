"""Local test: verify per-quadrant mode (TemplateBuilder) se khong bi seam."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from inference import (
    TemplateBuilder, InfillRegion, scan_generated_set,
    make_has_generation, make_render_provider, make_generation_provider,
    get_inference_config,
)

renders_dir = PROJECT_ROOT / "output" / "renders"
gen_dir = PROJECT_ROOT / "model_generate_test"
out_dir = Path("D:/tmp/test_template_output")
out_dir.mkdir(parents=True, exist_ok=True)
gen_dir.mkdir(exist_ok=True)


def sign(n):
    return "+%d" % n if n >= 0 else str(n)


# Cleanup old fake
for old in gen_dir.glob("tile_*_fake.png"):
    old.unlink()

# Fake TILE (0, 0) world - chua quadrants (0,0), (1,0), (0,1), (1,1)
# Tile (1, 0) world - chua quadrants (2,0), (3,0), (2,1), (3,1)
# Tile (0, 1) world - chua quadrants (0,2), (1,2), (0,3), (1,3)
for tile_qx, tile_qy in [(0, 0), (1, 0), (0, 1)]:
    src_files = list(renders_dir.glob("tile_%s_%s_*.png" % (sign(tile_qx), sign(tile_qy))))
    if not src_files:
        print("[SKIP] tile %s,%s not found in renders" % (tile_qx, tile_qy))
        continue
    fake = gen_dir / ("tile_%s_%s_fake.png" % (sign(tile_qx), sign(tile_qy)))
    Image.open(src_files[0]).save(fake)
    print("Created fake tile %s: %s" % ((tile_qx, tile_qy), fake))

print()

# Show generated_set
generated_set = scan_generated_set(gen_dir)
print("generated_set (quadrants):", sorted(generated_set))

print()
print("=== Test 1: quadrant (1, 0) - top-right of tile (0, 0) ===")
render_cache = {}
generation_cache = {}
region = InfillRegion.from_quadrant(1, 0)
print("  region =", region)
print("  is_full_tile =", region.is_full_tile())
print("  overlapping_quadrants =", region.overlapping_quadrants())

has_gen = make_has_generation(generated_set)
get_render = make_render_provider(renders_dir, render_cache)
get_generation = make_generation_provider(gen_dir, generation_cache)

builder = TemplateBuilder(region, has_gen, get_render, get_generation)
left = builder._has_generated_context("left")
right = builder._has_generated_context("right")
top = builder._has_generated_context("top")
bottom = builder._has_generated_context("bottom")
print("  has_left_gen   =", left)
print("  has_right_gen  =", right)
print("  has_top_gen    =", top)
print("  has_bottom_gen =", bottom)

result = builder.build()
if result is None:
    print("  [ERR]", builder._last_validation_error)
else:
    template, placement = result
    print("  placement =", placement)
    out_path = out_dir / "template_quadrant_1_0_with_context.png"
    template.save(out_path)
    print("  Saved:", out_path)

print()
print("=== Test 2: quadrant (0, 1) - bottom-left of tile (0, 0) ===")
render_cache = {}
generation_cache = {}
region2 = InfillRegion.from_quadrant(0, 1)
print("  overlapping_quadrants =", region2.overlapping_quadrants())

builder2 = TemplateBuilder(region2, has_gen, get_render, get_generation)
left2 = builder2._has_generated_context("left")
right2 = builder2._has_generated_context("right")
top2 = builder2._has_generated_context("top")
bottom2 = builder2._has_generated_context("bottom")
print("  has_left_gen   =", left2)
print("  has_right_gen  =", right2)
print("  has_top_gen    =", top2)
print("  has_bottom_gen =", bottom2)

result2 = builder2.build()
if result2 is None:
    print("  [ERR]", builder2._last_validation_error)
else:
    template2, placement2 = result2
    print("  placement =", placement2)
    out_path2 = out_dir / "template_quadrant_0_1_with_context.png"
    template2.save(out_path2)
    print("  Saved:", out_path2)

print()
print("=== Cleanup fake files ===")
for old in gen_dir.glob("tile_*_fake.png"):
    old.unlink()
    print("  Removed", old)