#!/usr/bin/env python3
"""
Inference pipeline (inference-only).

Loop through renders/, generate tiles via Qwen-Image-Edit + LoRA server,
and save each tile to <output>/tile_<qx>_<qy>_<hash>.png (same regex as renders/).

Pipeline is split into 3 steps (server -> inference -> user-built DZI):
  Step 1: uv run python -m inference.scripts.run_server --port 8000
  Step 2: uv run python -m inference.scripts.run_inference_pipeline --renders output/renders --output model_generate
  Step 3: uv run python -m inference.scripts.export_plan --input model_generate --output output/model_map_plan.json
          uv run python -m src.dzi.builder --input output/model_map_plan.json --output output/model_map

This script does NOT call stitch_all() or export_dzi_plan() — those are done
in step 3, which you run yourself.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

# Allow running as a module from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.client import GenerationClient, OmniTemplateBuilder, TileTraversal

log = logging.getLogger("inference.pipeline")

TILE_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")


def sign_int(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


def discover_tiles(renders_dir: Path) -> list[tuple[int, int]]:
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
    """Load .state.json. Returns empty dict on first run / corrupted file."""
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
    """Atomically write .state.json."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(state_path)


def scan_done_from_disk(gen_dir: Path) -> dict[str, str]:
    """Re-scan gen_dir for already-saved tiles. Used at startup to recover
    after a crash between save_generated() and save_state()."""
    done: dict[str, str] = {}
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
    """Save a generated tile as tile_<qx>_<qy>_<8hex>.png (same format as renders/)."""
    bio = BytesIO()
    img.convert("RGB").save(bio, format="PNG", optimize=False)
    png_bytes = bio.getvalue()
    digest = hashlib.sha256(png_bytes).hexdigest()[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tile_{sign_int(qx)}_{sign_int(qy)}_{digest}.png"
    out_path.write_bytes(png_bytes)
    return out_path


def find_neighbor_gen(
    qx: int, qy: int, gen_dir: Path
) -> tuple[Optional[Image.Image], Optional[str]]:
    """Find a generated neighbor (4-connected) to use as context.

    Returns (neighbor_image, side) where side ∈ {right, left, top, bottom}.
    """
    for dx, dy, side in [
        (1, 0, "right"),
        (-1, 0, "left"),
        (0, 1, "bottom"),
        (0, -1, "top"),
    ]:
        nqx, nqy = qx + dx, qy + dy
        pattern = f"tile_{sign_int(nqx)}_{sign_int(nqy)}_*.png"
        matches = list(gen_dir.glob(pattern))
        if matches:
            return Image.open(matches[0]).convert("RGB"), side
    return None, None


def build_template(
    builder: OmniTemplateBuilder,
    render: Image.Image,
    neighbor_gen: Optional[Image.Image],
    neighbor_side: Optional[str],
) -> tuple[Image.Image, str]:
    """Build the template image for the model. Returns (template, mode)."""
    if neighbor_gen is not None and neighbor_side is not None:
        template = builder.create_next_tile_template(
            generated_tile=neighbor_gen,
            next_render=render,
            side=neighbor_side,
        )
        return template, f"next:{neighbor_side}"
    template = builder.create_full_template(render)
    return template, "full"


async def wait_for_server(client: GenerationClient, timeout_s: float = 300.0) -> None:
    """Block until the server reports model_loaded=True, or raise."""
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
    builder: OmniTemplateBuilder,
    render_path: Path,
    gen_dir: Path,
    prompt: str,
    qx: int,
    qy: int,
    concurrency_sem: asyncio.Semaphore,
) -> tuple[bool, Optional[str], Optional[Path]]:
    """Generate a single tile. Returns (success, mode, saved_path)."""
    async with concurrency_sem:
        try:
            render = Image.open(render_path).convert("RGB")
            neighbor_gen, neighbor_side = find_neighbor_gen(qx, qy, gen_dir)
            template, mode = build_template(
                builder, render, neighbor_gen, neighbor_side
            )
            log.debug("[%d,%d] template mode=%s", qx, qy, mode)
            result_img = await client.edit(template, prompt)
            out_path = save_generated(result_img, qx, qy, gen_dir)
            return True, mode, out_path
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

    # Discover tiles
    tiles = discover_tiles(renders_dir)
    if not tiles:
        log.error("No tiles found in %s", renders_dir)
        return
    log.info("Discovered %d tiles.", len(tiles))

    # Load state (resume support)
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
    client = GenerationClient(args.endpoint, timeout=args.timeout)
    builder = OmniTemplateBuilder(tile_size=args.tile_size)
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
    last_modes: dict[str, int] = {}

    # Main loop
    while not traversal.is_complete:
        batch = traversal.get_next_batch(batch_size=args.batch_size)
        if not batch:
            break
        for qx, qy in batch:
            key = f"{qx},{qy}"
            if key in state["done"]:
                # Already done on a previous run; skip silently
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
                builder,
                render_path,
                gen_dir,
                args.prompt,
                qx,
                qy,
                sem,
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

            # Save state periodically
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

    # Final hand-off hint (no stitch / no DZI here)
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
    p.add_argument("--endpoint", type=str, default="http://localhost:8000",
                   help="Inference server URL.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Edit prompt (default: from src.constants.DEFAULT_PROMPT).")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max in-flight requests (default: 1; bump to 2 if VRAM allows).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Tiles per traversal batch (default: 1, strict BFS).")
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
        # Lazy import to avoid hard dependency
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