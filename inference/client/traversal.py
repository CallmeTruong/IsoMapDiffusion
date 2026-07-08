"""Tile traversal logic + helper callables cho TemplateBuilder.

Traversal: BFS theo (qx, qy) quadrant index, moi tile = 2x2 quadrants
(voi tile_size=1024, quadrant_size=512).

Helpers (su dung voi TemplateBuilder moi):
- has_generation_quadrant: callable cho `has_generation(qx, qy)`
- render_provider: callable cho `get_render(qx, qy)` - tra ve 512x512 PIL Image
- generation_provider: callable cho `get_generation(qx, qy)` - tra ve 512x512 PIL Image
"""
from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from PIL import Image

from inference.config import get_inference_config

_cfg = get_inference_config()
QUADRANT_SIZE: int = _cfg.quadrant_size_px
TEMPLATE_SIZE: int = _cfg.tile_size_px

# Tile filename regex (mirror src/dzi/export_plan.mjs + inference/scripts/export_plan.py)
TILE_FILENAME_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")


class TileStatus(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TileInfo:
    qx: int
    qy: int
    status: TileStatus = TileStatus.PENDING
    priority: float = 0.0


class TileTraversal:
    """
    BFS traversal theo (qx, qy) quadrant index.

    Moi tile = 2x2 quadrants. Traversal priority dua vao:
    - So generated neighbors (nhieu hon = uu tien hon)
    - Distance tu seed (gan hon = uu tien hon)
    """

    def __init__(
        self,
        tiles: List[Tuple[int, int]],
        seed: Optional[Tuple[int, int]] = None,
    ):
        self.tiles = set(tiles)
        self.tile_infos: dict = {}
        self.completed: Set[Tuple[int, int]] = set()
        self.seed = seed or self._find_seed(tiles)

        for (qx, qy) in tiles:
            priority = self._calculate_priority(qx, qy)
            self.tile_infos[(qx, qy)] = TileInfo(qx, qy, TileStatus.PENDING, priority)

    def _find_seed(self, tiles: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not tiles:
            raise ValueError("No tiles provided")
        return min(tiles, key=lambda t: abs(t[0]) + abs(t[1]))

    def _calculate_priority(self, qx: int, qy: int) -> float:
        if (qx, qy) in self.completed:
            return -float("inf")

        # Count 4-connected generated neighbors
        neighbor_count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (qx + dx, qy + dy) in self.completed:
                neighbor_count += 1

        if self.seed:
            dist = abs(qx - self.seed[0]) + abs(qy - self.seed[1])
        else:
            dist = abs(qx) + abs(qy)

        return neighbor_count * 10 - dist * 0.1

    def can_generate(self, qx: int, qy: int) -> bool:
        if (qx, qy) in self.completed:
            return False
        if (qx, qy) == self.seed:
            return True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (qx + dx, qy + dy) in self.completed:
                return True
        return False

    def get_next_batch(self, batch_size: int = 1) -> List[Tuple[int, int]]:
        available = [
            (self.tile_infos[(qx, qy)].priority, qx, qy)
            for (qx, qy) in self.tiles
            if self.can_generate(qx, qy) and (qx, qy) not in self.completed
        ]
        if not available:
            return []
        available.sort(reverse=True)
        return [(qx, qy) for (_, qx, qy) in available[:batch_size]]

    def mark_done(self, qx: int, qy: int) -> None:
        self.completed.add((qx, qy))
        if (qx, qy) in self.tile_infos:
            self.tile_infos[(qx, qy)].status = TileStatus.DONE
        for tile in self.tiles:
            if tile not in self.completed:
                self.tile_infos[tile].priority = self._calculate_priority(*tile)

    def mark_failed(self, qx: int, qy: int) -> None:
        if (qx, qy) in self.tile_infos:
            self.tile_infos[(qx, qy)].status = TileStatus.FAILED

    @property
    def is_complete(self) -> bool:
        return self.completed == self.tiles

    @property
    def progress(self) -> Tuple[int, int, int]:
        completed = len(self.completed)
        failed = sum(
            1 for t in self.tiles
            if self.tile_infos[t].status == TileStatus.FAILED
        )
        remaining = len(self.tiles) - completed - failed
        return completed, failed, remaining


# ─────────────────────────────────────────────────────────────────────
# Helpers cho TemplateBuilder (port tu isometric-nyc/shared.py)
# ─────────────────────────────────────────────────────────────────────


def scan_generated_set(gen_dir: Path) -> Set[Tuple[int, int]]:
    """
    Scan gen_dir, tra ve set (qx, qy) QUADRANT coords cua tiles da gen thanh cong.

    Moi file `tile_<qx>_<qy>_*.png` trong repo nay la 1 TILE 1024x1024 chua
    4 quadrants:
        tile_qx = qx_tile, tile_qy = qy_tile
        quadrants: (qx_tile*2, qy_tile*2), (qx_tile*2+1, qy_tile*2),
                   (qx_tile*2, qy_tile*2+1), (qx_tile*2+1, qy_tile*2+1)

    Su dung cho `has_generation` callable.
    """
    out: Set[Tuple[int, int]] = set()
    if not gen_dir.exists():
        return out
    for f in gen_dir.glob("tile_+*_+*_*.png"):
        m = TILE_FILENAME_RE.match(f.name)
        if not m:
            continue
        try:
            tile_qx = int(m.group(1))
            tile_qy = int(m.group(2))
            # Expand tile -> 4 quadrants
            for ox in (0, 1):
                for oy in (0, 1):
                    out.add((tile_qx * 2 + ox, tile_qy * 2 + oy))
        except ValueError:
            continue
    return out


def _sign_int(n: int) -> str:
    """Format signed integer: positive -> '+n', negative/zero -> 'n'."""
    return f"+{n}" if n >= 0 else str(n)


def crop_quadrant(img: Image.Image, qx: int, qy: int) -> Image.Image:
    """
    Cat 1 quadrant (512x512) tu full tile (1024x1024) tai (qx, qy).

    Quadrant (qx, qy) nam o goc tren-trai tai (qx*512, qy*512) trong tile
    1024x1024. Day la cach isometric-nyc to chuc (xem infill_template.py).
    """
    if img.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        img = img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS)
    left = (qx % 2) * QUADRANT_SIZE
    top = (qy % 2) * QUADRANT_SIZE
    return img.crop((left, top, left + QUADRANT_SIZE, top + QUADRANT_SIZE))


def make_has_generation(
    generated_set: Set[Tuple[int, int]],
) -> Callable[[int, int], bool]:
    """Closure cho has_generation(qx, qy)."""
    def has_generation(qx: int, qy: int) -> bool:
        return (qx, qy) in generated_set
    return has_generation


def make_render_provider(
    renders_dir: Path,
    render_cache: Optional[dict] = None,
) -> Callable[[int, int], Optional[Image.Image]]:
    """
    Closure cho get_render(qx, qy).

    Moi tile (qx_tile, qy_tile) = 2x2 quadrants bat dau tu
    (qx_tile*2, qy_tile*2) trong quadrant coords.

    Returns:
        Callable tra ve 512x512 PIL Image (crop tu render 1024x1024) hoac None.
    """
    cache = render_cache if render_cache is not None else {}

    def get_render(qx: int, qy: int) -> Optional[Image.Image]:
        # Tile index = (qx // 2, qy // 2) (4 quadrants / tile)
        tile_qx = qx // 2
        tile_qy = qy // 2
        tile_key = (tile_qx, tile_qy)
        if tile_key in cache:
            full_tile = cache[tile_key]
        else:
            pattern = f"tile_{_sign_int(tile_qx)}_{_sign_int(tile_qy)}_*.png"
            matches = list(renders_dir.glob(pattern))
            if not matches:
                cache[tile_key] = None
                return None
            try:
                full_tile = Image.open(matches[0]).convert("RGB")
                if full_tile.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                    full_tile = full_tile.resize(
                        (TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS
                    )
                cache[tile_key] = full_tile
            except Exception:
                cache[tile_key] = None
                return None
        if full_tile is None:
            return None
        return crop_quadrant(full_tile, qx, qy)

    return get_render


def make_generation_provider(
    gen_dir: Path,
    generation_cache: Optional[dict] = None,
) -> Callable[[int, int], Optional[Image.Image]]:
    """
    Closure cho get_generation(qx, qy).

    Tra ve 512x512 PIL Image (crop tu generation 1024x1024 da save) hoac None.
    """
    cache = generation_cache if generation_cache is not None else {}

    def get_generation(qx: int, qy: int) -> Optional[Image.Image]:
        tile_qx = qx // 2
        tile_qy = qy // 2
        tile_key = (tile_qx, tile_qy)
        if tile_key in cache:
            full_tile = cache[tile_key]
        else:
            pattern = f"tile_{_sign_int(tile_qx)}_{_sign_int(tile_qy)}_*.png"
            matches = list(gen_dir.glob(pattern))
            if not matches:
                cache[tile_key] = None
                return None
            try:
                full_tile = Image.open(matches[0]).convert("RGB")
                if full_tile.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                    full_tile = full_tile.resize(
                        (TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS
                    )
                cache[tile_key] = full_tile
            except Exception:
                cache[tile_key] = None
                return None
        if full_tile is None:
            return None
        return crop_quadrant(full_tile, qx, qy)

    return get_generation


def quadrant_iteration_order(
    start_qx: int, start_qy: int
) -> List[Tuple[int, int, str]]:
    """
    Order generate quadrants within 1 tile (2x2).
    Tra ve (qx, qy, region_type) tuples:
    1. tl (top-left) - 1st, khong co context
    2. tr (top-right) - context tu tl
    3. bl (bottom-left) - context tu tl, tr
    4. br (bottom-right) - context tu ca 3
    """
    return [
        (start_qx * 2,     start_qy * 2,     "tl"),
        (start_qx * 2 + 1, start_qy * 2,     "tr"),
        (start_qx * 2,     start_qy * 2 + 1, "bl"),
        (start_qx * 2 + 1, start_qy * 2 + 1, "br"),
    ]


def build_neighbor_map(
    tiles: List[Tuple[int, int]]
) -> dict:
    """Build map tile -> 4-connected neighbors."""
    neighbor_map = {tile: [] for tile in tiles}
    for tile in tiles:
        qx, qy = tile
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (qx + dx, qy + dy)
            if neighbor in neighbor_map:
                neighbor_map[tile].append(neighbor)
    return neighbor_map
