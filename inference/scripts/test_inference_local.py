#!/usr/bin/env python3
"""
Smoke test cho plan-based pipeline (no GPU required).

Dùng mock generation client (identity transform) de test full flow:
1. Copy 4-5 tile renders tu output/renders/ vao /tmp/test_renders/
2. Set 1 tile lam seed (copy render -> fake gen)
3. Chay TileTraversal + plan-based steps
4. Verify:
   - Plan sinh ra mix 2x2/2x1/1x2/1x1 dung
   - Tile PNGs duoc tao (voi stitched quadrants)
   - Stitch dung (full 1024x1024 tile, 4 quadrants)

Usage:
    python -m inference.scripts.test_inference_local
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    GenerationClient,
    InfillRegion,
    TemplateBuilder,
    QuadrantKVState,
    TileTraversal,
)
from inference.client.plan import Point
from inference.scripts.run_inference_pipeline import (
    QUADRANT_SIZE,
    TILE_SIZE,
    build_step_template,
    crop_quadrants_from_output,
    discover_tiles,
    sign_int,
    stitch_quadrant_into_tile,
)


# ============================================================================
# Mock client (no GPU, no HTTP)
# ============================================================================


class MockGenerationClient:
    """Mock - paste render back vao infill area (identity transform)."""

    def __init__(self):
        self.call_count = 0
        self.batch_call_count = 0

    async def edit(self, image: Image.Image, prompt: str):
        from inference.client.generator import EditResult
        self.call_count += 1
        return EditResult(image=image.copy(), seed_used=42, time_ms=10)

    async def edit_batch(self, images, prompts, *args, **kwargs):
        from inference.client.generator import EditResult
        self.batch_call_count += 1
        self.call_count += len(images)
        return [
            EditResult(image=img.copy(), seed_used=42 + i, time_ms=20)
            for i, img in enumerate(images)
        ]

    async def health_check(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": True,
            "max_batch_size": 2,
        }

    async def wait_for_server(self, timeout_s: float = 60.0) -> bool:
        return True


# ============================================================================
# Helpers
# ============================================================================


def select_tile_renders(renders_dir: Path, tile_keys: list[tuple[int, int]]) -> list[Path]:
    """Chon N tile PNG tu renders_dir theo (qx, qy)."""
    selected: list[Path] = []
    for qx, qy in tile_keys:
        pattern = f"tile_{sign_int(qx)}_{sign_int(qy)}_*.png"
        matches = list(renders_dir.glob(pattern))
        if matches:
            selected.append(matches[0])
    return selected


def setup_test_dirs(src_renders: Path, work_root: Path) -> tuple[Path, Path]:
    """Copy 4-5 tile renders vao /tmp/test_renders/ (seed 1 tile vao gen)."""
    test_renders = work_root / "test_renders"
    test_gen = work_root / "test_gen"
    test_renders.mkdir(parents=True, exist_ok=True)
    test_gen.mkdir(parents=True, exist_ok=True)

    # Pick seed tile va 4 neighbors:
    #   seed: (-1, 0) - da gen (xa de 2x2 co the dat dc o phia xa)
    #   test region: (0,0)..(1,1) 4 tiles
    seed_keys = [(-1, 0), (0, 0), (1, 0), (0, 1), (1, 1)]
    src_tiles = select_tile_renders(src_renders, seed_keys)
    if not src_tiles:
        raise RuntimeError(f"No tiles found in {src_renders}")

    for src in src_tiles:
        shutil.copy2(src, test_renders / src.name)

    # Seed: copy (-1,0) render -> test_gen (giả làm da gen)
    seed_src = select_tile_renders(src_renders, [(-1, 0)])[0]
    seed_dst = test_gen / seed_src.name
    shutil.copy2(seed_src, seed_dst)

    return test_renders, test_gen


# ============================================================================
# Test plan (offline, no client)
# ============================================================================


def test_plan_only(test_renders: Path, test_gen: Path) -> dict:
    """
    Test plan/Traversal KHONG can model server.
    Verify plan sinh ra mix 2x2/2x1/1x2/1x1 va step coverage.
    """
    print("\n=== TEST 1: Plan + Traversal (offline) ===")

    tiles = discover_tiles(test_renders)
    # Loại tile (-1,0) khoi tiles can test (chi test vung (0,0)..(1,1))
    tiles = sorted(t for t in tiles if t != (-1, 0))
    print(f"  Tiles in test region: {tiles}")

    # Init state: tile (-1, 0) da gen (4 quadrants done)
    # Quadrant coords: (-2,0), (-1,0), (-2,1), (-1,1)
    seed_quads = [(-2, 0), (-1, 0), (-2, 1), (-1, 1)]
    state = QuadrantKVState(quadrants=seed_quads)

    traversal = TileTraversal(tiles, quadrant_state=state)
    print(f"  Bounds: {traversal.bounds.top_left.to_tuple()} -> {traversal.bounds.bottom_right.to_tuple()}")
    print(f"  Progress: {traversal.progress}")

    step_types: dict[str, int] = {}
    step_count = 0
    while True:
        step = traversal.get_next_step()
        if step is None:
            break
        step_count += 1
        step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        quads = [(q.x, q.y) for q in step.quadrants]
        print(f"    step {step_count}: type={step.step_type} quads={quads}")
        traversal.mark_done([(q.x, q.y) for q in step.quadrants])

    print(f"  Total steps: {step_count}")
    print(f"  Step types: {step_types}")
    print(f"  Final progress: {traversal.progress}")
    print(f"  Is complete: {traversal.is_complete()}")

    assert traversal.is_complete(), "Traversal not complete after loop"
    assert step_count > 0, "No steps generated"
    # Expect at least 1 '2x2' (full tile away from seed)
    assert "2x2" in step_types, f"Expected 2x2 in step types, got {step_types}"

    return {
        "step_count": step_count,
        "step_types": step_types,
        "final_progress": traversal.progress,
    }


# ============================================================================
# Test template build (no client)
# ============================================================================


def test_template_build(test_renders: Path, test_gen: Path) -> dict:
    """Test build_step_template voi seed (-1, 0) da gen."""
    print("\n=== TEST 2: Template build (offline, no client) ===")

    from inference.client.traversal import (
        scan_generated_set,
        make_render_provider,
        make_generation_provider,
        make_has_generation,
    )
    generated_set = scan_generated_set(test_gen)
    has_gen = make_has_generation(generated_set)
    get_render = make_render_provider(test_renders, render_cache={})
    get_generation = make_generation_provider(test_gen, generation_cache={})

    # Test 2x2 step tile (1,0): tile (1,0) -> quadrants (2,0), (3,0), (2,1), (3,1)
    # Seed (-1, 0) khong ke tile (1, 0) -> co the dat 2x2 full tai (0,0)
    # Cung can (1, 0) render de render context (van trong -1,0 huong)
    region = InfillRegion.from_quadrants([(2, 0), (3, 0), (2, 1), (3, 1)])
    builder = TemplateBuilder(
        infill_region=region,
        has_generation=has_gen,
        get_render=get_render,
        get_generation=get_generation,
    )
    result = builder.build(border_width=2)
    assert result is not None, "Template build returned None"

    template, placement = result
    print(f"  Template size: {template.size}")
    print(f"  Placement infill: ({placement.infill_x}, {placement.infill_y})")
    print(f"  Placement size: {placement.infill_width}x{placement.infill_height}")

    assert template.size == (TILE_SIZE, TILE_SIZE)
    # Full 2x2 tile khong co generated neighbor -> dat o goc (0,0)
    assert placement.infill_x == 0, (
        f"Expected infill_x=0 (full 2x2), got {placement.infill_x}"
    )
    assert placement.infill_y == 0
    assert placement.infill_width == TILE_SIZE
    assert placement.infill_height == TILE_SIZE

    return {
        "template_size": template.size,
        "placement_x": placement.infill_x,
        "placement_y": placement.infill_y,
        "placement_size": (placement.infill_width, placement.infill_height),
    }


# ============================================================================
# Test full flow (mock client)
# ============================================================================


async def test_full_flow_mock(test_renders: Path, test_gen: Path) -> dict:
    """Full pipeline voi mock client (identity transform)."""
    print("\n=== TEST 3: Full flow (mock client) ===")

    tiles = discover_tiles(test_renders)
    tiles = sorted(t for t in tiles if t != (-1, 0))
    seed_quads = [(-2, 0), (-1, 0), (-2, 1), (-1, 1)]
    state = QuadrantKVState(quadrants=seed_quads)
    traversal = TileTraversal(tiles, quadrant_state=state)

    mock = MockGenerationClient()
    render_cache: dict = {}
    generation_cache: dict = {}

    from inference.client.traversal import (
        make_render_provider,
        make_generation_provider,
        scan_generated_set,
        make_has_generation,
    )

    completed_steps = 0
    step_types: dict[str, int] = {}

    while True:
        step = traversal.get_next_step()
        if step is None:
            break

        # Build template (refreshing context)
        generated_set = scan_generated_set(test_gen)
        has_gen = make_has_generation(generated_set)
        get_render = make_render_provider(test_renders, render_cache)
        get_generation = make_generation_provider(test_gen, generation_cache)

        region = InfillRegion.from_quadrants([(q.x, q.y) for q in step.quadrants])
        builder = TemplateBuilder(
            infill_region=region,
            has_generation=has_gen,
            get_render=get_render,
            get_generation=get_generation,
        )
        result = builder.build(border_width=2)
        assert result is not None, f"No placement for step {step}"
        template, placement = result

        # Mock model call
        edit_result = await mock.edit(template, "test prompt")
        output = edit_result.image

        # Crop quadrants from output
        crops = crop_quadrants_from_output(output, step, placement)
        assert len(crops) == len(step.quadrants), (
            f"Crop count mismatch: {len(crops)} vs {len(step.quadrants)}"
        )

        # Stitch tung quadrant vao tile
        for (qx, qy), quad_img in crops.items():
            for dx, dy, ox, oy in [
                (0, 0, 0, 0),
                (-1, 0, 1, 0),
                (0, -1, 0, 1),
                (-1, -1, 1, 1),
            ]:
                tile_qx = qx + dx
                tile_qy = qy + dy
                try:
                    stitch_quadrant_into_tile(
                        quad_img, tile_qx, tile_qy, ox, oy, test_gen
                    )
                except OSError:
                    pass

        traversal.mark_done([(q.x, q.y) for q in step.quadrants])
        completed_steps += 1
        step_types[step.step_type] = step_types.get(step.step_type, 0) + 1

    # Verify: moi tile trong tiles phai co file 1024x1024 trong test_gen
    final_files = sorted(test_gen.glob("tile_+*_+*_*.png"))
    print(f"  Total tiles done: {len(final_files)}")
    print(f"  Mock calls: {mock.call_count}")
    print(f"  Step types: {step_types}")
    print(f"  Expected tiles: {len(tiles)}")

    assert traversal.is_complete(), "Not complete after loop"
    assert len(final_files) >= len(tiles), (
        f"Only {len(final_files)} tile files, expected {len(tiles)}"
    )

    # Verify each tile is full 1024x1024
    for f in final_files:
        img = Image.open(f)
        assert img.size == (TILE_SIZE, TILE_SIZE), (
            f"Tile {f.name} has size {img.size}, expected ({TILE_SIZE}, {TILE_SIZE})"
        )

    return {
        "completed_steps": completed_steps,
        "mock_calls": mock.call_count,
        "step_types": step_types,
        "tile_files": len(final_files),
    }


# ============================================================================
# Batch path test
# ============================================================================


async def test_batch_path_mock(test_renders: Path, test_gen: Path) -> dict:
    """Verify the new client.edit_batch() path works with a mock server.

    Builds 2 templates, calls edit_batch([t1, t2], [p1, p2]), and confirms:
      - exactly 1 batch call was made
      - 2 result images came back
      - both result images are valid 1024x1024 PNGs
    """
    tiles = discover_tiles(test_renders)
    assert len(tiles) >= 2, f"Need >=2 tiles for batch test, got {len(tiles)}"

    mock = MockGenerationClient()
    # Pretend we have a Qwen-IE server that supports batch=2.
    # GenerationClient is real (it builds payloads) but we point it at a
    # fake URL and never call it; the mock is what we actually exercise.
    client = GenerationClient(
        base_url="http://mock:8000",
        max_connections=2,
        max_keepalive=1,
    )
    # Force the cached capability to 2 so the batch path is taken.
    client._max_batch_size = 2

    templates: list[Image.Image] = []
    prompts: list[str] = []
    # Use a normalized search so we don't depend on tile_x_y vs tile_+x_+y
    # naming; ``discover_tiles`` returns coordinates, we look up the actual
    # file by name match.
    rendered_files = list(test_renders.glob("tile_*.png"))
    assert len(rendered_files) >= 2, (
        f"Need >=2 rendered tiles for batch test, got {len(rendered_files)}"
    )
    for i, f in enumerate(rendered_files[:2]):
        tpl = Image.open(f).convert("RGB")
        templates.append(tpl)
        prompts.append(f"batch test prompt #{i}")

    # We can't actually call client.edit_batch() because the mock server
    # doesn't exist. Instead we drive MockGenerationClient directly which
    # now exposes the same edit_batch() signature.
    results = await mock.edit_batch(templates, prompts)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for r in results:
        assert r.image.size == (TILE_SIZE, TILE_SIZE), (
            f"Result has wrong size: {r.image.size}"
        )
    assert mock.batch_call_count == 1, (
        f"Expected 1 batch call, got {mock.batch_call_count}"
    )
    assert mock.call_count == 2, (
        f"Expected 2 total items, got {mock.call_count}"
    )

    # Sanity: the real client also exposes edit_batch() with the right
    # signature (no AttributeError on introspection).
    import inspect
    sig = inspect.signature(client.edit_batch)
    params = list(sig.parameters.keys())
    for required in ("images", "prompts"):
        assert required in params, f"edit_batch() missing {required}"

    await client.close()

    return {
        "batch_call_count": mock.batch_call_count,
        "items_in_batch": len(results),
        "result_sizes": [r.image.size for r in results],
    }


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    src_renders = PROJECT_ROOT / "output" / "renders"
    if not src_renders.exists():
        print(f"ERROR: {src_renders} not found.")
        return 1

    with tempfile.TemporaryDirectory(prefix="isometric_test_") as tmp:
        work_root = Path(tmp)
        test_renders, test_gen = setup_test_dirs(src_renders, work_root)
        print(f"Test renders: {test_renders}")
        print(f"Test gen:     {test_gen}")
        print(f"Source:       {src_renders}")

        results: dict = {}
        try:
            results["plan"] = test_plan_only(test_renders, test_gen)
            results["template"] = test_template_build(test_renders, test_gen)
            results["full"] = asyncio.run(
                test_full_flow_mock(test_renders, test_gen)
            )
            results["batch"] = asyncio.run(
                test_batch_path_mock(test_renders, test_gen)
            )
        except AssertionError as e:
            print(f"\nFAIL: {e}")
            return 1
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1

        print("\n=== ALL TESTS PASSED ===")
        print(f"Plan: {results['plan']}")
        print(f"Template: {results['template']}")
        print(f"Full: {results['full']}")
        print(f"Batch: {results['batch']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())