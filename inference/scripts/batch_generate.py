#!/usr/bin/env python3
"""
Batch generate tiles using the inference server.

Usage:
    python -m inference.scripts.batch_generate \
        --renders ./output/renders \
        --output ./generate \
        --prompt "Fill in the outlined section..." \
        --endpoint http://localhost:8000 \
        --seed 42
"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.client import GenerationClient, OmniTemplateBuilder, TileTraversal


@dataclass
class GenerationResult:
    """Result of a generation attempt."""

    qx: int
    qy: int
    success: bool
    error: Optional[str] = None


def create_db(db_path: Path) -> None:
    """Create the generations database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tiles (
            qx INTEGER NOT NULL,
            qy INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            has_render INTEGER DEFAULT 0,
            has_generation INTEGER DEFAULT 0,
            generation_b64 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (qx, qy)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles(status)")
    conn.commit()
    conn.close()


def get_tiles_from_renders(renders_dir: Path) -> list[tuple[int, int]]:
    """Parse tile coordinates from render filenames.

    Filename format: tile_[+-]<qx>_[+-]<qy>_<hex>.png
    """
    import re as _re
    pattern = _re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")
    tiles = set()
    for f in renders_dir.glob("tile_+*_+*_*.png"):
        m = pattern.match(f.name)
        if not m:
            continue
        try:
            qx = int(m.group(1))
            qy = int(m.group(2))
            tiles.add((qx, qy))
        except ValueError:
            continue
    return sorted(tiles)


async def generate_tile(
    client: GenerationClient,
    builder: OmniTemplateBuilder,
    render_path: Path,
    existing_gen_path: Optional[Path],
    renders_dir: Path,
    generations_dir: Optional[Path],
    prompt: str,
    qx: int,
    qy: int,
) -> GenerationResult:
    """
    Generate a single tile using correct overlap-based logic.

    Logic (theo src/tile/worker.mjs + cameraMoveStep=0.5):
    - 2 tile k? nhau ch?ng 50% (512px).
    - N?u kh?ng c? neighbor ?? gen ? full template.
    - N?u c? 1+ neighbor ?? gen ? d?ng create_next_tile_template v?i neighbor.
    """
    try:
        # Load render for current tile
        render = Image.open(render_path).convert("RGB")

        # 1. T?m neighbor ?? gen (4-connectivity)
        neighbor_gen = None
        neighbor_side = None  # "right", "left", "top", "bottom"

        for dx, dy, side in [(1, 0, "right"), (-1, 0, "left"),
                              (0, 1, "bottom"), (0, -1, "top")]:
            nqx, nqy = qx + dx, qy + dy
            sign_qx = "+" + str(nqx) if nqx >= 0 else str(nqx)
            sign_qy = "+" + str(nqy) if nqy >= 0 else str(nqy)
            # T?m trong generations dir tr??c, sau ?? renders
            search_dirs = []
            if (gen_path := existing_gen_path) and gen_path.exists() and gen_path.parent.exists():
                # Reuse the same parent dir as the current existing_gen_path
                search_dirs.append(gen_path.parent)
            # C?ng th? output dir m?c ??nh
            default_gen = Path(render_path.parent.parent) / "generate"
            if default_gen.exists():
                search_dirs.append(default_gen)

            for d in search_dirs:
                # Look for tile_nqx_nqy_*.png
                pattern = f"tile_{sign_qx}_{sign_qy}_*.png"
                matches = list(d.glob(pattern))
                if matches:
                    neighbor_gen = Image.open(matches[0]).convert("RGB")
                    neighbor_side = side
                    break
            if neighbor_gen:
                break

        # 2. Build template
        if neighbor_gen is not None:
            # D?ng neighbor ?? gen l?m context
            template = builder.create_next_tile_template(
                generated_tile=neighbor_gen,
                next_render=render,
                side=neighbor_side,
            )
            print(f"  [{qx},{qy}] Using neighbor ({neighbor_side}) as context")
        elif existing_gen_path and existing_gen_path.exists():
            # Fallback: n?u c? s?n existing_gen cho c?ng tile, d?ng quadrant template
            pixel_art = Image.open(existing_gen_path).convert("RGB")
            template = builder.create_quadrant_template(
                generated_tile=pixel_art,
                render=render,
                region_type="tr",
            )
            print(f"  [{qx},{qy}] Using existing same-tile gen (quadrant TR)")
        else:
            # First generation - no context
            template = builder.create_full_template(render)
            print(f"  [{qx},{qy}] Full template (no context)")

        # 3. Call API
        result_img = await client.edit(template, prompt)

        # 4. Save to generations_dir (or renders_dir/../generate) for downstream stitch
        save_dir = (Path(generations_dir) if generations_dir is not None
                    else Path(renders_dir).parent / "generate")
        save_dir.mkdir(parents=True, exist_ok=True)
        bio = BytesIO()
        result_img.convert("RGB").save(bio, format="PNG", optimize=False)
        png_bytes = bio.getvalue()
        digest = hashlib.sha256(png_bytes).hexdigest()[:8]
        sign = lambda n: f"+{n}" if n >= 0 else str(n)
        out_path = save_dir / f"tile_{sign(qx)}_{sign(qy)}_{digest}.png"
        out_path.write_bytes(png_bytes)
        print(f"  [{qx},{qy}] saved -> {out_path.name}")

        return GenerationResult(qx=qx, qy=qy, success=True)

    except Exception as e:
        print(f"  [{qx},{qy}] Error: {e}")
        return GenerationResult(qx=qx, qy=qy, success=False, error=str(e))


async def main():
    parser = argparse.ArgumentParser(description="Batch generate tiles")
    parser.add_argument(
        "--renders",
        type=str,
        required=True,
        help="Path to renders directory",
    )
    parser.add_argument(
        "--generations",
        type=str,
        default=None,
        help="Path to existing generations directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, seamlessly blending edges with surrounding areas.",
        help="Edit prompt",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Inference server endpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximum tiles to generate",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database (optional)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tiles per traversal batch (1 = strict BFS, fastest context propagation)",
    )

    args = parser.parse_args()

    renders_dir = Path(args.renders)
    generations_dir = Path(args.generations) if args.generations else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tiles from renders
    tiles = get_tiles_from_renders(renders_dir)
    if not tiles:
        print(f"No tiles found in {renders_dir}")
        return

    if args.max_tiles:
        tiles = tiles[: args.max_tiles]

    print(f"Found {len(tiles)} tiles to generate")
    print(f"Output: {output_dir}")

    # Initialize client
    client = GenerationClient(args.endpoint)
    builder = OmniTemplateBuilder(tile_size=args.tile_size)

    # Check server health
    print("Checking server health...")
    try:
        health = await client.health_check()
        print(f"  Server status: {health}")
    except Exception as e:
        print(f"Error: Cannot connect to server: {e}")
        print("Make sure the server is running: python -m inference.scripts.run_server")
        return

    # Generate tiles via priority traversal:
    # seed = tile closest to (0,0); expand BFS outward, prioritizing tiles
    # that already have a generated neighbor (so next_tile_template has context).
    seed = min(tiles, key=lambda t: abs(t[0]) + abs(t[1]))
    traversal = TileTraversal(tiles, seed=seed)
    print(f"Seed: {seed}  |  traversal queue: {len(tiles)} tiles")

    results = []
    completed = 0
    failed = 0

    print(f"\nGenerating {len(tiles)} tiles...")
    i = 0
    while not traversal.is_complete:
        batch = traversal.get_next_batch(batch_size=args.batch_size)
        if not batch:
            break
        for qx, qy in batch:
            i += 1
            print(f"[{i}/{len(tiles)}] Tile ({qx}, {qy})")

            # Find render file (sign-aware)
            sign = lambda n: f"+{n}" if n >= 0 else str(n)
            render_files = list(renders_dir.glob(f"tile_{sign(qx)}_{sign(qy)}_*.png"))
            if not render_files:
                print(f"  Render file not found for ({qx}, {qy})")
                failed += 1
                traversal.mark_failed(qx, qy)
                continue

            render_path = render_files[0]

            # Find existing generation if available
            existing_gen = None
            if generations_dir:
                gen_files = list(generations_dir.glob(f"tile_{qx}_{qy}_*.png"))
                if gen_files:
                    existing_gen = gen_files[0]

            # Generate
            result = await generate_tile(
                client,
                builder,
                render_path,
                existing_gen,
                renders_dir,
                generations_dir,
                args.prompt,
                qx,
                qy,
            )
            results.append(result)

            if result.success:
                completed += 1
                traversal.mark_done(qx, qy)
            else:
                failed += 1
                traversal.mark_failed(qx, qy)

            print(f"  Progress: {completed} done, {failed} failed")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Total: {len(tiles)}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")

    # Save results to DB if specified
    if args.db:
        db_path = Path(args.db)
        create_db(db_path)
        conn = sqlite3.connect(db_path)

        for result in results:
            status = "done" if result.success else "failed"
            conn.execute(
                """
                INSERT OR REPLACE INTO tiles
                (qx, qy, status, has_generation, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (result.qx, result.qy, status, 1 if result.success else 0),
            )

        conn.commit()
        conn.close()
        print(f"  Database: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
