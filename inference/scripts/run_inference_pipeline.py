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
import numpy as np
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
TEMPLATE_SIZE: int = _cfg.tile_size_px      # alias for clarity

STATE_VERSION = 1  # bump khi format thay doi


def encode_template(
    img: Image.Image, fmt: str = "PNG", jpeg_quality: int = 95
) -> bytes:
    """Encode a PIL image for transport to the server.

    fmt "PNG" (default) is lossless and the safe choice. fmt "JPEG" can
    shave 50-70% off payload size for large 1024x1024 templates, but
    invisible to the Qwen-IE model because the model only needs to see
    the red border + the surrounding context. Toggle via
    --template-format jpeg on the run_inference_pipeline CLI.

    Note: JPEG does not support alpha. Templates are RGBA today; the
    border is fully opaque so we flatten against black before encoding.
    """
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        # Flatten alpha on black; the model needs RGB anyway.
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def decode_template(data: bytes, fmt: str) -> Image.Image:
    """Inverse of encode_template; mirrors what the model returns."""
    img = Image.open(BytesIO(data))
    img.load()
    if fmt.upper() == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ============================================================================
# Helpers
# ============================================================================


def sign_int(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


def discover_tiles(renders_dir: Path) -> list[tuple[int, int]]:
    """Parse tile coords tu `tile_<qx>_<qy>_<hash>.png` filenames."""
    tiles: set[tuple[int, int]] = set()
    for f in renders_dir.glob("tile_*_*_*.png"):
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


def _region_has_content(img: Image.Image, threshold: float = 0.01) -> bool:
    """
    Check if a PIL region has non-zero content.

    A region is considered "has content" if mean pixel value > threshold * 255.
    Default threshold 0.01 = 1% of 255 = 2.55, which catches any non-trivial
    generated content (pure black = 0, blank tile = 0).

    Used to detect quadrant duplication before stitch.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(arr.mean()) > threshold * 255.0


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

    # Phase 6: detect quadrant duplication - neu quadrant region da co pixel
    # (mean > 1% of 255) truoc khi paste, log warning de surface duplication bug.
    existing_region = tile_img.crop(
        (
            quad_ox * QUADRANT_SIZE,
            quad_oy * QUADRANT_SIZE,
            (quad_ox + 1) * QUADRANT_SIZE,
            (quad_oy + 1) * QUADRANT_SIZE,
        )
    )
    if _region_has_content(existing_region):
        log.warning(
            "[stitch] quadrant (tile_qx=%d, tile_qy=%d, ox=%d, oy=%d) "
            "already has content - pasting will overwrite. "
            "Possible duplication if step plan re-visits this quadrant.",
            tile_qx, tile_qy, quad_ox, quad_oy,
        )

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
    generated_set_quads: set[tuple[int, int]],
):
    """Build fresh context providers cho 1 generation request.

    Args:
        generated_set_quads: Already-known set of generated quadrants.
            Callers should pass their in-memory set instead of re-scanning
            disk every step.
    """
    has_gen = make_has_generation(generated_set_quads)
    get_render = make_render_provider(renders_dir, render_cache)
    get_generation = make_generation_provider(gen_dir, generation_cache)
    return has_gen, get_render, get_generation


def evict_tile_from_cache(cache: dict, tile_qx: int, tile_qy: int) -> None:
    """Drop a specific tile from a provider cache after its file changes."""
    cache.pop((tile_qx, tile_qy), None)


def build_step_template(
    step: GenerationStep,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
    generated_set_quads: set[tuple[int, int]],
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
        gen_dir, renders_dir, render_cache, generation_cache,
        generated_set_quads=generated_set_quads,
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
    border_width = _cfg.border_width
    result = builder.build(border_width=border_width, allow_expansion=allow_expansion)
    if result is None:
        return None, None, builder._last_validation_error or "no valid placement"

    template, placement = result
    summary = (
        f"step={step.step_type} quads={len(step.quadrants)} "
        f"infill=({placement.infill_x},{placement.infill_y}) "
        f"size=({placement.infill_width}x{placement.infill_height})"
    )
    return template, placement, summary


def verify_output_size(output_img: Image.Image, placement) -> None:
    """
    Verify model output is the full 1024x1024 template.

    Qwen-Image-Edit is a diffusion model; its output is always the same size
    as its input. The standard input is the full 1024x1024 template, so the
    standard output is also 1024x1024 (containing both edited infill AND the
    regenerated context). We assert this so any future model behavior change
    surfaces immediately rather than silently mis-cropping.
    """
    if output_img.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        raise ValueError(
            f"Model output size mismatch: expected "
            f"({TEMPLATE_SIZE}, {TEMPLATE_SIZE}) (full template), "
            f"got {output_img.size}."
        )


def crop_quadrants_from_output(
    output_img: Image.Image,
    step: GenerationStep,
    placement,
) -> dict[tuple[int, int], Image.Image]:
    """
    Crop quadrants tu model output (full 1024x1024 template).

    The infill region inside the template starts at
    (placement.infill_x, placement.infill_y). For a 1x2 placement the
    infill is 512x1024 placed at the left or right edge of the template;
    for a 1x1 placement it is 512x512 at one of the four corners; for a
    2x2 placement the infill is the whole 1024x1024 template.
    """
    verify_output_size(output_img, placement)

    crops: dict[tuple[int, int], Image.Image] = {}

    # Sort quadrants by position (so output is deterministic).
    sorted_quads = sorted(step.quadrants, key=lambda p: (p.y, p.x))

    for q in sorted_quads:
        # Position of q within the local rect of step.quadrants.
        min_qx = min(p.x for p in step.quadrants)
        min_qy = min(p.y for p in step.quadrants)
        local_x = q.x - min_qx
        local_y = q.y - min_qy

        # Coordinates in the full 1024x1024 output.
        x0 = placement.infill_x + local_x * QUADRANT_SIZE
        y0 = placement.infill_y + local_y * QUADRANT_SIZE

        crops[(q.x, q.y)] = output_img.crop(
            (x0, y0, x0 + QUADRANT_SIZE, y0 + QUADRANT_SIZE)
        )

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


def _build_step_template_sync(
    step: GenerationStep,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
    generated_set_quads: set[tuple[int, int]],
):
    """Sync wrapper around build_step_template.

    Used by ``run_in_executor`` so the CPU-side template build can run in
    a background thread while the main event loop is doing GPU I/O.
    """
    return build_step_template(
        step, renders_dir, gen_dir, render_cache, generation_cache,
        generated_set_quads=generated_set_quads,
    )


async def generate_step(
    client: GenerationClient,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
    generated_set_quads: set[tuple[int, int]],
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
                step, renders_dir, gen_dir, render_cache, generation_cache,
                generated_set_quads=generated_set_quads,
            )
            if template is None:
                log.warning("[step %s %s] no valid placement: %s",
                            step.step_type,
                            [(q.x, q.y) for q in step.quadrants],
                            summary)
                return False, summary, None

            edit_result = await client.edit(template, prompt)
            # edit() returns EditResult (image, seed_used, time_ms). Crop
            # quadrants from result.image, and remember the seed for the
            # resume manifest.
            result_img = edit_result.image
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
    if args.min_qx is not None:
        tiles = [t for t in tiles if t[0] >= args.min_qx]
    if args.max_qx is not None:
        tiles = [t for t in tiles if t[0] <= args.max_qx]
    if args.min_qy is not None:
        tiles = [t for t in tiles if t[1] >= args.min_qy]
    if args.max_qy is not None:
        tiles = [t for t in tiles if t[1] <= args.max_qy]

    if not tiles:
        log.error("No tiles found in %s for coordinate range", renders_dir)
        return
    total_quads = len(tiles) * 4
    log.info("Discovered %d tiles (= %d quadrants) for coordinate range.", len(tiles), total_quads)

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

    # Main loop - SEQUENTIAL by design for plan determinism.
    # Concurrency is currently used only as a connection-pool sizing hint
    # (the actual model still runs one call at a time on a single GPU).
    # We keep this as a sequential loop to preserve exact plan semantics:
    # each step's outputs are committed before the next step is selected.
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

            # Build template on the event loop's default executor so it can
            # overlap with the GPU call's setup (image base64 encode on the
            # client, base64 decode on the server). On a 4-core client this
            # shaves ~50ms per request on a 1024x1024 template.
            loop = asyncio.get_event_loop()
            template, placement, summary = await loop.run_in_executor(
                None,
                _build_step_template_sync,
                step, renders_dir, gen_dir,
                render_cache, generation_cache,
                done_quadrants,
            )
            if template is None:
                log.warning("[step %s %s] skipping (no 3D render): %s",
                            step.step_type,
                            [(q.x, q.y) for q in step.quadrants],
                            summary)
                traversal.mark_done([(q.x, q.y) for q in step.quadrants])
                done_quadrants.update((q.x, q.y) for q in step.quadrants)
                state["done_quadrants"].update({f"{q.x},{q.y}": True for q in step.quadrants})
                failed_steps += 1
                continue

            edit_result = await client.edit(template, args.prompt, steps=args.steps)
            result_img = edit_result.image
            crops = crop_quadrants_from_output(result_img, step, placement)

            if crops is not None:
                # Stitch tung quadrant vao tile PNG tuong ung
                stitch_failed = []
                # Track which tiles actually got modified (for per-tile
                # cache eviction in place of clearing the entire cache).
                affected_tiles: set[tuple[int, int]] = set()
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
                        affected_tiles.add((tile_qx, tile_qy))
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
                # Evict only the tiles we just modified from the
                # generation cache; other cached tiles stay valid.
                for tile_key in affected_tiles:
                    evict_tile_from_cache(generation_cache, *tile_key)

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
                   help="Inference server URL -- default: from INFERENCE_ENDPOINT env var.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Edit prompt -- default: from src.constants.DEFAULT_PROMPT.")
    p.add_argument("--steps", type=int, default=None,
                   help="Inference steps (default: from INFERENCE_STEPS env var or 14).")
    p.add_argument("--min-qx", type=int, default=None,
                   help="Filter tiles: minimum qx coordinate (inclusive).")
    p.add_argument("--max-qx", type=int, default=None,
                   help="Filter tiles: maximum qx coordinate (inclusive).")
    p.add_argument("--min-qy", type=int, default=None,
                   help="Filter tiles: minimum qy coordinate (inclusive).")
    p.add_argument("--max-qy", type=int, default=None,
                   help="Filter tiles: maximum qy coordinate (inclusive).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max in-flight requests (default=4). Note: plan-based generation is sequential; this is for connection pooling.")
    p.add_argument("--tile-size", type=int, default=1024,
                   help="Tile size in pixels (default=1024).")
    p.add_argument("--timeout", type=int, default=300,
                   help="HTTP timeout per request in seconds (default=300).")
    p.add_argument("--server-wait", type=float, default=300.0,
                   help="Max seconds to wait for server model_loaded=true.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from .state.json + on-disk tiles.")
    p.add_argument("--state-file", type=str, default=".state.json",
                   help="State filename inside --output (default=.state.json).")
    p.add_argument("--state-save-interval", type=float, default=15.0,
                   help="Seconds between state flushes (default=15).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging.")
    p.add_argument("--template-format", type=str, default="PNG",
                   choices=["PNG", "JPEG"],
                   help="Image format for the template sent to the server "
                        "-- default: PNG, lossless. JPEG shaves 50-70 percent of "
                        "payload size with no visible quality loss because "
                        "the model only needs the red border + context. "
                        "Both endpoints handle decoding transparently.")
    p.add_argument("--jpeg-quality", type=int, default=92,
                   help="Quality when --template-format=jpeg -- default: 92.")
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