#!/usr/bin/env python3
"""
Inference pipeline (inference-only) - REFACTORED to use TemplateBuilder
ported from isometric-nyc/infill_template.py.

Loop through renders/, generate quadrants via Qwen-Image-Edit + LoRA server
using quadrant-based template (TemplateBuilder), and save each TILE to
<output>/tile_<qx>_<qy>_<hash>.png (same regex as renders/).

Pipeline split (server -> inference -> user-built DZI):
  Step 1: uv run python -m inference.scripts.run_server --port 8000
  Step 2: uv run python -m inference.scripts.run_inference_pipeline --renders output/renders --output model_generate
  Step 3: uv run python -m inference.scripts.export_plan --input model_generate --output output/model_map_plan.json
          uv run python -m src.dzi.builder --input output/model_map_plan.json --output output/model_map

This script does NOT call stitch_all() or export_dzi_plan() -- those are
done in step 3 (you run yourself).

=== REFACTOR NOTES (vs old version) ===
- find_neighbor_gen (1 neighbor, wrong direction) -> REPLACED by
  scan_generated_set + make_has_generation (multi-neighbor, 4-connected)
- build_template (left/right split) -> REPLACED by TemplateBuilder.build()
  with has_generation/get_render/get_generation callables.
- Each tile (qx_tile, qy_tile) is generated as a FULL 1024x1024
  via 1 template build that infills the WHOLE 1024x1024 (mode=full).
  Reason: 1 tile = 1 generation = 1 save, simpler.
  Generated quadrants are saved in the same 1024x1024 file.
- The user can switch to per-quadrant mode by setting --per-quadrant flag.
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

from inference.client import (
    GenerationClient,
    TemplateBuilder,
    InfillRegion,
    TileTraversal,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
)
from inference.config import get_inference_config

_cfg = get_inference_config()
log = logging.getLogger("inference.pipeline")

# Tile filename regex
TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")


def sign_int(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


def discover_tiles(renders_dir: Path) -> list:
    """Parse tile coords from `tile_<qx>_<qy>_<hash>.png` filenames."""
    tiles = set()
    for f in renders_dir.glob("tile_+*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            tiles.add((int(m.group(1)), int(m.group(2))))
        except ValueError:
            continue
    return sorted(tiles)


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"done": {}, "failed": {}}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "done" not in data:
            data["done"] = {}
        if "failed" not in data:
            data["failed"] = {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("State file unreadable (%s); starting fresh.", e)
        return {"done": {}, "failed": {}}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(state_path)


def scan_done_from_disk(gen_dir: Path) -> dict:
    """Re-scan gen_dir for already-saved tiles."""
    done: dict = {}
    if not gen_dir.exists():
        return done
    for f in gen_dir.glob("tile_+*_+*_*.png"):
        m = TILE_RE.match(f.name)
        if not m:
            continue
        try:
            qx = int(m.group(1))
            qy = int(m.group(2))
            done[f"{qx},{qy}"] = f.name
        except ValueError:
            continue
    return done


def save_generated(img: Image.Image, qx: int, qy: int, out_dir: Path) -> Path:
    """Save generated tile as tile_<qx>_<qy>_<8hex>.png (same format as renders/)."""
    bio = BytesIO()
    img.convert("RGB").save(bio, format="PNG", optimize=False)
    png_bytes = bio.getvalue()
    digest = hashlib.sha256(png_bytes).hexdigest()[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tile_{sign_int(qx)}_{sign_int(qy)}_{digest}.png"
    out_path.write_bytes(png_bytes)
    return out_path


def build_tile_template(
    tile_qx: int,
    tile_qy: int,
    renders_dir: Path,
    gen_dir: Path,
    render_cache: dict,
    generation_cache: dict,
) -> Optional[tuple]:
    """
    Build template cho 1 TILE (qx, qy) su dung TemplateBuilder.

    Moi tile (qx, qy) = 2x2 quadrants bat dau tu (qx*2, qy*2) trong quadrant
    coords. Day la approach "full tile" (region = 1024x1024 = toan bo tile).

    Returns:
        (template_image, placement) hoac None neu khong co placement hop le.
    """
    # Full-tile infill region (1024x1024) cho tile (tile_qx, tile_qy)
    region = InfillRegion.full_tile(tile_qx, tile_qy)

    # has_generation: kiem tra 4 quadrants cua tile da co generation chua
    generated_set = scan_generated_set(gen_dir)
    has_gen = make_has_generation(generated_set)

    # get_render: load render 1024x1024 cua tile, crop 512x512 quadrant
    get_render = make_render_provider(renders_dir, render_cache)

    # get_generation: load generation 1024x1024 cua tile, crop 512x512 quadrant
    get_generation = make_generation_provider(gen_dir, generation_cache)

    builder = TemplateBuilder(
        infill_region=region,
        has_generation=has_gen,
        get_render=get_render,
        get_generation=get_generation,
    )
    result = builder.build(border_width=_cfg.border_width)
    return result


def build_quadrant_template(
    qx: int, qy: int, renders_dir: Path, gen_dir: Path,
    render_cache: dict, generation_cache: dict,
) -> Optional[tuple]:
    """
    Build template cho 1 QUADRANT (qx, qy) (512x512).

    Per-quadrant mode: gen 1 quadrant = 1 cuoc goi model. Useful cho fine-grained
    seam control nhung cham hon 4x so voi full-tile mode.

    Returns:
        (template_image, placement) hoac None neu khong co placement hop le.
    """
    region = InfillRegion.from_quadrant(qx, qy)
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
    return builder.build(border_width=_cfg.border_width)


async def wait_for_server(client: GenerationClient, timeout_s: float = 300.0) -> None:
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


async def generate_tile(
    client: GenerationClient,
    render_path: Path,
    gen_dir: Path,
    prompt: str,
    qx: int,
    qy: int,
    concurrency_sem: asyncio.Semaphore,
    per_quadrant: bool = False,
) -> tuple:
    """
    Generate 1 tile (or 1 quadrant) su dung TemplateBuilder moi.

    Returns:
        (success, mode, saved_path_or_error)
    """
    async with concurrency_sem:
        try:
            render_cache: dict = {}
            generation_cache: dict = {}

            if per_quadrant:
                # Per-quadrant mode: 4 cuoc goi model / tile
                saved_paths = []
                # Order: tl, tr, bl, br
                for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    qqx, qqy = qx * 2 + ox, qy * 2 + oy
                    result = build_quadrant_template(
                        qqx, qqy,
                        render_path.parent, gen_dir,
                        render_cache, generation_cache,
                    )
                    if result is None:
                        return False, f"quadrant-{ox}-{oy}", "no valid placement"
                    template, placement = result
                    # Gen 1 quadrant
                    gen_img = await client.edit(template, prompt)
                    # Lay quadrant tu result
                    result_quadrant = crop_quadrant(
                        gen_img.resize(
                            (_cfg.tile_size_px, _cfg.tile_size_px),
                            Image.Resampling.LANCZOS,
                        ),
                        qqx, qqy,
                    )
                    saved_paths.append(((qqx, qqy), result_quadrant, placement))

                # Stitch 4 quadrants thanh 1 tile 1024x1024
                tile_img = Image.new(
                    "RGB", (_cfg.tile_size_px, _cfg.tile_size_px), (0, 0, 0)
                )
                for (qqx, qqy), quad_img, _pl in saved_paths:
                    left = (qqx % 2) * _cfg.quadrant_size_px
                    top = (qqy % 2) * _cfg.quadrant_size_px
                    tile_img.paste(quad_img, (left, top))
                out_path = save_generated(tile_img, qx, qy, gen_dir)
                return True, "per-quadrant", out_path
            else:
                # Full-tile mode (default): 1 cuoc goi model / tile
                # NOTE: pass renders_dir (parent of render_path) for provider
                result = build_tile_template(
                    qx, qy,
                    render_path.parent, gen_dir,
                    render_cache, generation_cache,
                )
                if result is None:
                    # Full-tile validation failed (co generated neighbor ben ngoai).
                    # Fallback: gen theo quadrant.
                    log.info(
                        "[%d,%d] full-tile failed, fallback to per-quadrant",
                        qx, qy,
                    )
                    return await generate_tile(
                        client, render_path, gen_dir, prompt,
                        qx, qy, concurrency_sem, per_quadrant=True,
                    )
                template, placement = result
                gen_img = await client.edit(template, prompt)
                out_path = save_generated(gen_img, qx, qy, gen_dir)
                return True, f"full-tile@{placement.infill_x},{placement.infill_y}", out_path
        except (httpx.HTTPError, httpx.RequestError, OSError) as e:
            return False, None, e
        except Exception as e:
            return False, None, e


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
    log.info("mode     : %s", "per-quadrant" if args.per_quadrant else "full-tile")
    log.info("config   : tile=%dpx, quadrant=%dpx, border_w=%d",
             _cfg.tile_size_px, _cfg.quadrant_size_px, _cfg.border_width)

    # Discover tiles
    tiles = discover_tiles(renders_dir)
    if not tiles:
        log.error("No tiles found in %s", renders_dir)
        return
    log.info("Discovered %d tiles.", len(tiles))

    # Load state
    state = load_state(state_path) if args.resume else {"done": {}, "failed": {}}
    on_disk_done = scan_done_from_disk(gen_dir)
    for key, fname in on_disk_done.items():
        state["done"].setdefault(key, fname)
    if args.resume:
        log.info(
            "Resume: %d already done on disk, %d in state file.",
            len(on_disk_done),
            len(state["done"]),
        )

    pending = [t for t in tiles if f"{t[0]},{t[1]}" not in state["done"]]
    if not pending:
        log.info("All %d tiles already generated. Nothing to do.", len(tiles))
        return
    log.info("%d tiles remaining to generate.", len(pending))

    # Server + builder
    client = GenerationClient(
        args.endpoint,
        timeout=args.timeout,
        default_steps=_cfg.default_steps,
        default_guidance=_cfg.default_guidance,
    )
    await wait_for_server(client, timeout_s=args.server_wait)
    sem = asyncio.Semaphore(args.concurrency)

    # BFS traversal (seed = closest to origin)
    seed = min(tiles, key=lambda t: abs(t[0]) + abs(t[1]))
    traversal = TileTraversal(set(tiles), seed=seed)
    for qx, qy in state["done"]:
        try:
            qxi, qyi = (int(p) for p in qx.split(","))
            traversal.mark_done(qxi, qyi)
        except ValueError:
            continue

    # Stats
    started = time.monotonic()
    last_save = started
    completed = len(state["done"])
    failed = 0
    last_modes: dict = {}

    # Main loop
    while not traversal.is_complete:
        batch = traversal.get_next_batch(batch_size=args.batch_size)
        if not batch:
            break
        for qx, qy in batch:
            key = f"{qx},{qy}"
            if key in state["done"]:
                traversal.mark_done(qx, qy)
                continue
            render_files = list(
                renders_dir.glob(f"tile_{sign_int(qx)}_{sign_int(qy)}_*.png")
            )
            if not render_files:
                log.warning("[%d,%d] render not found; marking failed.", qx, qy)
                state["failed"][key] = "render-not-found"
                failed += 1
                traversal.mark_failed(qx, qy)
                continue
            render_path = render_files[0]

            ok, mode, result = await generate_tile(
                client,
                render_path,
                gen_dir,
                args.prompt,
                qx,
                qy,
                sem,
                per_quadrant=args.per_quadrant,
            )
            if ok:
                assert isinstance(result, Path)
                state["done"][key] = result.name
                completed += 1
                if isinstance(mode, str):
                    last_modes[mode] = last_modes.get(mode, 0) + 1
                traversal.mark_done(qx, qy)
                log.info(
                    "[%d,%d] ok  -> %s  (mode=%s, total=%d/%d)",
                    qx, qy, result.name, mode, completed, len(tiles),
                )
            else:
                state["failed"][key] = repr(result)
                failed += 1
                traversal.mark_failed(qx, qy)
                log.warning("[%d,%d] FAIL  err=%s", qx, qy, result)

            now = time.monotonic()
            if now - last_save >= args.state_save_interval:
                save_state(state_path, state)
                last_save = now

    # Final save
    save_state(state_path, state)

    elapsed = time.monotonic() - started
    log.info("=== Inference summary ===")
    log.info("  total     : %d", len(tiles))
    log.info("  completed : %d", completed)
    log.info("  failed    : %d", failed)
    log.info("  elapsed   : %.1fs", elapsed)
    log.info("  gen dir   : %s", gen_dir)
    log.info("  state     : %s", state_path)
    if last_modes:
        log.info("  modes     : %s", last_modes)

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
            "Inference-only pipeline: BFS gen tiles via Qwen-Image-Edit + LoRA "
            "and save to <output>/. No stitch / no DZI here."
        )
    )
    p.add_argument("--renders", type=str, required=True,
                   help="Path to renders/ directory (input).")
    p.add_argument("--output", type=str, required=True,
                   help="Path to gen output dir (each tile saved as tile_<qx>_<qy>_<hash>.png).")
    p.add_argument("--endpoint", type=str, default=_cfg.endpoint,
                   help=f"Inference server URL (default: {_cfg.endpoint}).")
    p.add_argument("--prompt", type=str, default=None,
                   help=f"Edit prompt (default: from src.constants.DEFAULT_PROMPT).")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max in-flight requests (default: 1; bump to 2 if VRAM allows).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Tiles per traversal batch (default: 1, strict BFS).")
    p.add_argument("--tile-size", type=int, default=_cfg.tile_size_px,
                   help=f"Tile size in pixels (default: {_cfg.tile_size_px} from config).")
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
    p.add_argument("--per-quadrant", action="store_true",
                   help="Use per-quadrant mode (4 calls/tile, finer seam control).")
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
        args.prompt = _cfg.default_prompt
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
