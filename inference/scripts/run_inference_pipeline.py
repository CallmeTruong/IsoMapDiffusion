#!/usr/bin/env python3
"""
Inference pipeline (plan-based) - 3 PLACEMENT RULES tu isometric-nyc.

Loop through renders/, generate via Qwen-Image-Edit + LoRA server theo plan
sinh boi `inference.client.plan.create_rectangle_plan`. Moi step la 1 request:
- 2x2 (full tile, 4 quadrants): KHONG co generated neighbor nao o 8 exterior
  quadrants xung quanh tile.
- 2x1 (2 quadrants ngang): long side co CA 2 generated, short side KHONG generated.
- 1x2 (2 quadrants doc): tuong tu nhung theo truc doc.
- 1x1 (1 quadrant): 3/4 quadrants cua it nhat 1 2x2 block chua no da generated.

Moi step: build 1024x1024 template (qua TemplateBuilder), POST den server, crop
quadrant 512x512 tu output theo placement, stitch vao tile PNG tren disk.

Pipeline:
  Step 1: uv run python -m inference.scripts.run_server --port 8000
  Step 2: uv run python -m inference.scripts.run_inference_pipeline --renders output/renders --output model_generate
  Step 3: uv run python -m inference.scripts.export_plan --input model_generate --output output/model_map_plan.json
          uv run python -m src.dzi.builder --input output/model_map_plan.json --output output/model_map
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    GenerationClient,
    InfillRegion,
    TemplateBuilder,
    QuadrantKVState,
    TileTraversal,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
)
from inference.client.plan import (
    Point,
    GenerationStep,
    get_2x2_quadrants,
)
from inference.config import get_inference_config

log = logging.getLogger("inference.pipeline")

TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")
_cfg = get_inference_config()
TILE_SIZE: int = _cfg.tile_size_px          # 1024
QUADRANT_SIZE: int = _cfg.quadrant_size_px  # 512

STATE_VERSION = 1  # bump khi format thay doi


# ============================================================================
# Helpers
# ============================================================================


def sign_int(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


def discover_tiles(renders_dir: Path) -> list[tuple[int, int]]:
    """Parse tile coords tu `tile_<qx>_<qy>_<hash>.png` filenames."""
    tiles: set[tuple[int, int]] = set()
    for f in renders_dir.glob("tile_+*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            tiles.add((int(m.group(1)), int(m.group(2))))
        except ValueError:
            continue
    for f in renders_dir.glob("tile_-*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            tiles.add((int(m.group(1)), int(m.group(2))))
        except ValueError:
            continue
    return sorted(tiles)


# ============================================================================
# State
# ============================================================================


def load_state(state_path: Path) -> dict:
    """Load .state.json. Returns empty default on first run / old format."""
    if not state_path.exists():
        return {"version": STATE_VERSION, "done_quadrants": {}}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("State file unreadable (%s); starting fresh.", e)
        return {"version": STATE_VERSION, "done_quadrants": {}}

    # Migrate old format ({"done": {"qx,qy": "filename"}, "failed": ...})
    if "version" not in data and "done" in data:
        log.info("Migrating old state format (per-tile) -> per-quadrant.")
        data = {
            "version": STATE_VERSION,
            "done_quadrants": {},  # se duoc dien tu on-disk scan
        }
    if "done_quadrants" not in data:
        data["done_quadrants"] = {}
    return data


def save_state(state_path: Path, state: dict) -> None:
    """Atomically write .state.json."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(state_path)


def scan_done_from_disk(gen_dir: Path) -> set[tuple[int, int]]:
    """
    Re-scan gen_dir cho already-saved tiles (full 1024x1024 -> 4 quadrants).

    Heuristic: 1 file `tile_<tqx>_<tqy>_*.png` duoc coi la "full tile done"
    (4 quadrants generated). Day la fallback cho state cu bi mat.
    """
    done: set[tuple[int, int]] = set()
    if not gen_dir.exists():
        return done
    for f in gen_dir.glob("tile_+*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            tqx = int(m.group(1))
            tqy = int(m.group(2))
            for ox in (0, 1):
                for oy in (0, 1):
                    done.add((tqx * 2 + ox, tqy * 2 + oy))
        except ValueError:
            continue
    for f in gen_dir.glob("tile_-*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            tqx = int(m.group(1))
            tqy = int(m.group(2))
            for ox in (0, 1):
                for oy in (0, 1):
                    done.add((tqx * 2 + ox, tqy * 2 + oy))
        except ValueError:
            continue
    return done


# ============================================================================
# Quadrant-level save / stitch
# ============================================================================


def _find_tile_path(gen_dir: Path, tile_qx: int, tile_qy: int) -> Optional[Path]:
    pattern = f"tile_{sign_int(tile_qx)}_{sign_int(tile_qy)}_*.png"
    matches = list(gen_dir.glob(pattern))
    return matches[0] if matches else None


def _load_or_init_tile(gen_dir: Path, tile_qx: int, tile_qy: int) -> Image.Image:
    """Load existing tile 1024x1024 tu gen_dir; black neu missing."""
    existing = _find_tile_path(gen_dir, tile_qx, tile_qy)
    if existing is not None:
        return Image.open(existing).convert("RGB")
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))


def stitch_quadrant_into_tile(
    quad_img: Image.Image,
    tile_qx: int,
    tile_qy: int,
    quad_ox: int,
    quad_oy: int,
    gen_dir: Path,
) -> Path:
    """
    Paste 1 newly-generated quadrant (512x512) vao parent tile 1024x1024 tren disk.

    Returns new tile PNG path (hash trong filename).
    """
    tile_img = _load_or_init_tile(gen_dir, tile_qx, tile_qy)
    tile_img.paste(
        quad_img,
        (quad_ox * QUADRANT_SIZE, quad_oy * QUADRANT_SIZE),
    )

    old_path = _find_tile_path(gen_dir, tile_qx, tile_qy)
    bio = BytesIO()
    tile_img.save(bio, format="PNG", optimize=False)
    png_bytes = bio.getvalue()
    digest = hashlib.sha256(png_bytes).hexdigest()[:8]
    new_path = gen_dir / f"tile_{sign_int(tile_qx)}_{sign_int(tile_qy)}_{digest}.png"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(png_bytes)
    if old_path is not None and old_path != new_path:
        try:
            old_path.unlink()
        except OSError:
            pass
    return new_path


# ============================================================================
# Template building
# ============================================================================


def _refresh_context(
    gen_dir: Path,
    renders_dir: Path,
    render_cache: dict,
    generation_cache: dict,
):
    """Build fresh context providers cho 1 generation request."""
    generated_set = QuadrantKVState()  # placeholder; will use real set below
    # Caller passes generated set; we read from disk each call.
    from inference.client.traversal import scan_generated_set
    generated_set_quads = scan_generated_set(gen_dir)
    has_gen = make_has_generation(generated_set_quads)
    get_render = make_render_provider(renders_dir, render_cache)
    get_generation = make_generation_provider(gen_dir, generation_cache)
    return has_gen, get_render, get_generation


def build_step_template(
    step: GenerationStep,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
    allow_expansion: bool = False,
) -> tuple[Optional[Image.Image], Optional[object], str]:
    """
    Build 1024x1024 template cho 1 step (1/2/4 quadrants).

    Tra ve (template, placement, message).
    """
    # Quadrants -> world coords: (qx*512, qy*512, width, height)
    min_qx = min(q.x for q in step.quadrants)
    max_qx = max(q.x for q in step.quadrants)
    min_qy = min(q.y for q in step.quadrants)
    max_qy = max(q.y for q in step.quadrants)

    has_gen, get_render, get_generation = _refresh_context(
        gen_dir, renders_dir, render_cache, generation_cache
    )

    region = InfillRegion.from_quadrants(
        [(q.x, q.y) for q in step.quadrants]
    )

    builder = TemplateBuilder(
        infill_region=region,
        has_generation=has_gen,
        get_render=get_render,
        get_generation=get_generation,
    )
    result = builder.build(border_width=2, allow_expansion=allow_expansion)
    if result is None:
        return None, None, builder._last_validation_error or "no valid placement"

    template, placement = result
    summary = (
        f"step={step.step_type} quads={len(step.quadrants)} "
        f"infill=({placement.infill_x},{placement.infill_y}) "
        f"size=({placement.infill_width}x{placement.infill_height})"
    )
    return template, placement, summary


def crop_quadrants_from_output(
    output_img: Image.Image,
    step: GenerationStep,
    placement,
) -> dict[tuple[int, int], Image.Image]:
    """
    Crop tung quadrant 512x512 tu model output dua vao placement.

    Placement.infill_x/y/width/height cho biet vi tri infill trong template 1024x1024.
    Step.quadrants cho biet 1/2/4 quadrants nam o vi tri nao trong infill rect
    (theo quadrant grid alignment).
    """
    if output_img.size != (TILE_SIZE, TILE_SIZE):
        output_img = output_img.resize(
            (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS
        )

    # Vi tri infill trong output (sau khi model tra ve)
    infill_x = placement.infill_x
    infill_y = placement.infill_y
    infill_w = placement.infill_width
    infill_h = placement.infill_height

    # Quadrant grid: moi quadrant 512, can trong infill_w x infill_h
    # Step.quadrants duoc sort theo (y, x) de mapping on dinh
    sorted_quads = sorted(step.quadrants, key=lambda p: (p.y, p.x))

    # Bounds cua step quadrants trong quadrant coords
    min_qx = min(q.x for q in step.quadrants)
    max_qx = max(q.x for q in step.quadrants)
    min_qy = min(q.y for q in step.quadrants)
    max_qy = max(q.y for q in step.quadrants)
    w_quads = max_qx - min_qx + 1  # 1 hoac 2
    h_quads = max_qy - min_qy + 1  # 1 hoac 2

    quad_w = infill_w // w_quads
    quad_h = infill_h // h_quads

    crops: dict[tuple[int, int], Image.Image] = {}
    for q in sorted_quads:
        # Vi tri quadrant trong infill rect (0..w_quads-1, 0..h_quads-1)
        local_x = q.x - min_qx
        local_y = q.y - min_qy
        x0 = infill_x + local_x * quad_w
        y0 = infill_y + local_y * quad_h
        crop = output_img.crop((x0, y0, x0 + QUADRANT_SIZE, y0 + QUADRANT_SIZE))
        if crop.size != (QUADRANT_SIZE, QUADRANT_SIZE):
            crop = crop.resize(
                (QUADRANT_SIZE, QUADRANT_SIZE), Image.Resampling.LANCZOS
            )
        crops[(q.x, q.y)] = crop

    return crops


# ============================================================================
# Per-step generation
# ============================================================================


async def wait_for_server(client: GenerationClient, timeout_s: float = 300.0) -> None:
    """Block cho den khi server reports model_loaded=True."""
    log.info("Waiting for server at %s ...", client.base_url)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            health = await client.health_check()
            if health.get("model_loaded"):
                log.info("Server ready: %s", health)
                return
        except (httpx.HTTPError, httpx.RequestError):
            pass
        await asyncio.sleep(2.0)
    raise RuntimeError(
        f"Server at {client.base_url} did not become ready within {timeout_s:.0f}s"
    )


async def generate_step(
    client: GenerationClient,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
    prompt: str,
    step: GenerationStep,
    sem: asyncio.Semaphore,
) -> tuple[bool, str, Optional[dict[tuple[int, int], Image.Image]]]:
    """
    Generate 1 step (1/2/4 quadrants).

    Returns:
        (True, summary, {quad -> Image}) on success.
        (False, error_msg, None) on failure.
    """
    async with sem:
        try:
            template, placement, summary = build_step_template(
                step, renders_dir, gen_dir, render_cache, generation_cache
            )
            if template is None:
                log.warning("[step %s %s] no valid placement: %s",
                            step.step_type,
                            [(q.x, q.y) for q in step.quadrants],
                            summary)
                return False, summary, None

            result_img = await client.edit(template, prompt)

            crops = crop_quadrants_from_output(result_img, step, placement)
            return True, summary, crops

        except (httpx.HTTPError, httpx.RequestError, OSError) as e:
            return False, str(e), None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", None


# ============================================================================
# Main pipeline
# ============================================================================


async def run(args: argparse.Namespace) -> None:
    renders_dir = Path(args.renders)
    gen_dir = Path(args.output)
    state_path = gen_dir / args.state_file
    renders_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    log.info("renders  : %s", renders_dir)
    log.info("output   : %s", gen_dir)
    log.info("state    : %s", state_path)
    log.info("endpoint : %s", args.endpoint)

    tiles = discover_tiles(renders_dir)
    if not tiles:
        log.error("No tiles found in %s", renders_dir)
        return
    total_quads = len(tiles) * 4
    log.info("Discovered %d tiles (= %d quadrants).", len(tiles), total_quads)

    # Load state
    state = load_state(state_path) if args.resume else {"version": STATE_VERSION, "done_quadrants": {}}

    # Scan on-disk -> done quadrants
    on_disk_quads = scan_done_from_disk(gen_dir)
    done_quadrants: set[tuple[int, int]] = set()
    if args.resume:
        done_quadrants |= on_disk_quads
        # Add from state file
        for k in state.get("done_quadrants", {}).keys():
            try:
                qx, qy = k.split(",")
                done_quadrants.add((int(qx), int(qy)))
            except ValueError:
                continue
        log.info(
            "Resume: %d quadrants already done (disk + state).",
            len(done_quadrants),
        )

    # Build initial state dict for save
    state["version"] = STATE_VERSION
    state["done_quadrants"] = {f"{q[0]},{q[1]}": True for q in done_quadrants}

    # Init traversal
    quadrant_state = QuadrantKVState(quadrants=done_quadrants)
    traversal = TileTraversal(tiles, quadrant_state=quadrant_state)

    if traversal.is_complete():
        log.info("All %d tiles already generated. Nothing to do.", len(tiles))
        save_state(state_path, state)
        return

    log.info(
        "%d/%d quadrants remaining.",
        traversal.progress[1] - traversal.progress[0],
        traversal.progress[1],
    )

    # Server + caches
    client = GenerationClient(
        args.endpoint,
        timeout=args.timeout,
        max_connections=args.concurrency + 10,
        max_keepalive=args.concurrency,
    )
    await wait_for_server(client, timeout_s=args.server_wait)
    sem = asyncio.Semaphore(args.concurrency)

    render_cache: dict = {}
    generation_cache: dict = {}

    # Stats
    started = time.monotonic()
    last_save = started
    completed_quads = len(done_quadrants)
    completed_steps = 0
    failed_steps = 0
    step_types_count: dict[str, int] = {}

    # Main loop - SEQUENTIAL (deterministic cho tile_placement rules)
    # Note: concurrency param is for future parallelization via queued set in plan
    try:
        while not traversal.is_complete():
            step = traversal.get_next_step()
            if step is None:
                log.warning(
                    "Traversal stuck: no next step but %d quadrants remaining.",
                    traversal.progress[1] - traversal.progress[0],
                )
                break

            # Optional: skip step neu TOAN BO quadrants cua no da co trong state
            # (truong hop on-disk partial)
            already_done = all(
                (q.x, q.y) in done_quadrants for q in step.quadrants
            )
            if already_done:
                # Skip nhanh, khong goi server
                traversal.mark_done([(q.x, q.y) for q in step.quadrants])
                continue

            ok, summary, crops = await generate_step(
                client,
                renders_dir, gen_dir,
                render_cache, generation_cache,
                args.prompt,
                step,
                sem,
            )

            if ok and crops is not None:
                # Stitch tung quadrant vao tile PNG tuong ung
                stitch_failed = []
                for (qx, qy), quad_img in crops.items():
                    tile_qx = qx // 2
                    tile_qy = qy // 2
                    quad_ox = qx % 2
                    quad_oy = qy % 2
                    try:
                        stitch_quadrant_into_tile(
                            quad_img, tile_qx, tile_qy, quad_ox, quad_oy, gen_dir
                        )
                        done_quadrants.add((qx, qy))
                        state["done_quadrants"][f"{qx},{qy}"] = True
                    except OSError as e:
                        stitch_failed.append((qx, qy, str(e)))

                if stitch_failed:
                    # Stitch failed -> revert mark_done
                    for qx, qy, _ in stitch_failed:
                        state["done_quadrants"].pop(f"{qx},{qy}", None)
                    done_quadrants.difference_update((q[0], q[1]) for q in stitch_failed)
                    traversal.quadrant_state.mark_generated(
                        (q[0], q[1]) for q in stitch_failed  # remove them so plan re-places
                    )
                    traversal.quadrant_state._generated.difference_update(
                        (q[0], q[1]) for q in stitch_failed
                    )
                    traversal._current_plan = traversal._build_plan()
                    traversal._step_index = 0
                    failed_steps += 1
                    log.warning("[step %s] stitch failures: %s",
                                step.step_type, stitch_failed)
                    continue

                # Mark done (4/2/1 quadrants)
                traversal.mark_done([(q.x, q.y) for q in step.quadrants])
                completed_quads += len(step.quadrants)
                completed_steps += 1
                step_types_count[step.step_type] = (
                    step_types_count.get(step.step_type, 0) + 1
                )
                # Clear cache vi gen_dir vua thay doi
                generation_cache.clear()

                log.info(
                    "[step %d: %s quads=%d] %s | done %d/%d quads | tiles-remaining=%d",
                    completed_steps,
                    step.step_type,
                    len(step.quadrants),
                    summary,
                    completed_quads,
                    traversal.progress[1],
                    sum(
                        1 for tqx, tqy in tiles
                        if not all(
                            (tqx * 2 + ox, tqy * 2 + oy) in done_quadrants
                            for ox in (0, 1) for oy in (0, 1)
                        )
                    ),
                )
            else:
                failed_steps += 1
                log.warning(
                    "[step %s quads=%s] FAIL err=%s",
                    step.step_type,
                    [(q.x, q.y) for q in step.quadrants],
                    summary,
                )

            # Periodic state save
            now = time.monotonic()
            if now - last_save >= args.state_save_interval:
                save_state(state_path, state)
                last_save = now

    finally:
        # Cleanup: close HTTP client
        await client.close()

    save_state(state_path, state)

    elapsed = time.monotonic() - started
    log.info("=== Inference summary ===")
    log.info("  total tiles    : %d", len(tiles))
    log.info("  total quadrants: %d", total_quads)
    log.info("  done quadrants : %d", completed_quads)
    log.info("  steps run      : %d (failed=%d)", completed_steps, failed_steps)
    log.info("  step types     : %s", step_types_count)
    log.info("  elapsed        : %.1fs", elapsed)
    log.info("  gen dir        : %s", gen_dir)
    log.info("  state          : %s", state_path)

    log.info("Next step (run yourself):")
    log.info("  uv run python -m inference.scripts.export_plan \\")
    log.info("      --input  %s \\", gen_dir)
    log.info("      --output output/model_map_plan.json")
    log.info("  uv run python -m src.dzi.builder \\")
    log.info("      --input  output/model_map_plan.json \\")
    log.info("      --output output/model_map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inference-only pipeline (plan-based): generates via Qwen-Image-Edit + "
            "LoRA using 3 placement rules (2x2/2x1/1x2/1x1) from isometric-nyc."
        )
    )
    p.add_argument("--renders", type=str, required=True,
                   help="Path to renders/ directory (input).")
    p.add_argument("--output", type=str, required=True,
                   help="Path to gen output dir (tile_<qx>_<qy>_<hash>.png).")
    p.add_argument("--endpoint", type=str,
                   default=os.environ.get("INFERENCE_ENDPOINT", "http://localhost:8000"),
                   help="Inference server URL (default: from INFERENCE_ENDPOINT env var).")
    p.add_argument("--prompt", type=str, default=None,
                   help="Edit prompt (default: from src.constants.DEFAULT_PROMPT).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max in-flight requests (default: 4). Note: plan-based generation is sequential; this is for connection pooling.")
    p.add_argument("--tile-size", type=int, default=1024,
                   help="Tile size in pixels (default: 1024).")
    p.add_argument("--timeout", type=int, default=300,
                   help="HTTP timeout per request in seconds (default: 300).")
    p.add_argument("--server-wait", type=float, default=300.0,
                   help="Max seconds to wait for server model_loaded=true.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from .state.json + on-disk tiles.")
    p.add_argument("--state-file", type=str, default=".state.json",
                   help="State filename inside --output (default: .state.json).")
    p.add_argument("--state-save-interval", type=float, default=15.0,
                   help="Seconds between state flushes (default: 15).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.prompt is None:
        try:
            from src.constants import DEFAULT_PROMPT
            args.prompt = DEFAULT_PROMPT
        except ImportError:
            args.prompt = (
                "Fill in the outlined section with coherent pixels matching the "
                "<isometric pixel art> style, seamlessly blending edges with "
                "surrounding areas, maintaining consistent isometric perspective, "
                "shadow direction, lighting, pixel density, and color harmony."
            )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()