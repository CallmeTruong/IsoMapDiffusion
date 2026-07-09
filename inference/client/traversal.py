"""Tile/quadrant traversal logic cho isometric pipeline.

Thay the SQLite bang in-memory `QuadrantKVState` (set cua generated quadrants).
Moi tile = 1024x102 = 2x2 quadrants (qx, qy) trong quadrant grid:
  tile (tile_qx, tile_qy) chua quadrants:
    (tile_qx*2,   tile_qy*2),
    (tile_qx*2+1, tile_qy*2),
    (tile_qx*2,   tile_qy*2+1),
    (tile_qx*2+1, tile_qy*2+1)

TileTraversal: quan ly tiles (list co toa do tile), seed, va goi `plan.py`
de sinh GenerationStep tiep theo (1/2/4 quadrants / request).

Helpers (dung cho TemplateBuilder moi):
- has_generation_quadrant(qx, qy) -> bool
- render_provider(qx, qy) -> 512x512 PIL Image (crop tu render 1024x1024)
- generation_provider(qx, qy) -> 512x512 PIL Image (crop tu gen 1024x1024)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

from inference.config import get_inference_config
from inference.client.plan import (
    Point,
    RectBounds,
    GenerationStep,
    create_rectangle_plan,
    get_2x2_quadrants,
)

_cfg = get_inference_config()
QUADRANT_SIZE: int = _cfg.quadrant_size_px
TEMPLATE_SIZE: int = _cfg.tile_size_px

# Tile filename regex (mirror src/dzi/export_plan.mjs + export_plan.py)
TILE_FILENAME_RE = re.compile(r"^tile_([+-]\d+)_([+-]\d+)_[a-f0-9]+\.png$")


# ============================================================================
# QuadrantKVState
# ============================================================================


class QuadrantKVState:
    """In-memory quadrant generation state.

    Quan ly set quadrant (qx, qy) da generated.
    Thay the SQLite cho project hien tai (data set <10k quadrants).
    """

    def __init__(
        self,
        quadrants: Optional[Iterable[tuple[int, int]]] = None,
    ):
        self._generated: set[tuple[int, int]] = set()
        if quadrants:
            self._generated.update(quadrants)

    def __contains__(self, q: tuple[int, int]) -> bool:
        return q in self._generated

    def __len__(self) -> int:
        return len(self._generated)

    def is_generated(self, qx: int, qy: int) -> bool:
        return (qx, qy) in self._generated

    def mark_generated(self, quadrants: Iterable[tuple[int, int]]) -> None:
        for q in quadrants:
            self._generated.add((int(q[0]), int(q[1])))

    def all_generated(self) -> set[tuple[int, int]]:
        return set(self._generated)

    def to_dict(self) -> dict:
        return {
            "quadrants": [list(q) for q in sorted(self._generated)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuadrantKVState":
        return cls(quadrants=tuple(tuple(q) for q in data.get("quadrants", [])))


# ============================================================================
# TileTraversal
# ============================================================================


class TileTraversal:
    """
    Tile-level state + plan-driven next-step generation.

    - tiles: list tile (qx, qy) trong tile coords (khong phai quadrant).
    - quadrant_state: QuadrantKVState theo doi per-quadrant.
    - get_next_step(): goi plan.py sinh GenerationStep tiep theo.
    - mark_done(quadrants): danh dau N quadrant vua gen xong.

    Moi step tuong ung 1 request den inference server.
    """

    def __init__(
        self,
        tiles: list[tuple[int, int]],
        quadrant_state: Optional[QuadrantKVState] = None,
    ):
        if not tiles:
            raise ValueError("At least one tile required")
        self.tiles: set[tuple[int, int]] = set(tiles)
        self.quadrant_state = quadrant_state or QuadrantKVState()

        # Bounds cua vung tile (tinh theo quadrant coords)
        all_quads: list[tuple[int, int]] = []
        for tqx, tqy in self.tiles:
            for q in get_2x2_quadrants(Point(tqx, tqy)):
                all_quads.append(q.to_tuple())
        if not all_quads:
            raise ValueError("No quadrants from tiles")
        self._bounds = RectBounds(
            top_left=Point(min(q[0] for q in all_quads), min(q[1] for q in all_quads)),
            bottom_right=Point(
                max(q[0] for q in all_quads), max(q[1] for q in all_quads)
            ),
        )

        # Track current step index in plan
        self._step_index = 0
        self._current_plan = self._build_plan()

    def _build_plan(self):
        """Build full plan cho toan vung tile bang plan.create_rectangle_plan."""
        return create_rectangle_plan(
            bounds=self._bounds,
            generated={
                Point(q[0], q[1]) for q in self.quadrant_state.all_generated()
            },
        )

    def get_next_step(self) -> Optional[GenerationStep]:
        """
        Tra ve GenerationStep tiep theo (1, 2, hoac 4 quadrants).

        Returns None neu da gen xong het quadrant trong bounds.
        """
        # Rebuild plan neu state da thay doi (e.g. sau khi mark_done)
        # De don gian va deterministic, ta rebuild khi step index het.
        # Tuy nhien, de toi uu, ta cap nhat in-memory generated set ngay
        # trong mark_done() va rebuild o day neu can.
        if self._current_plan is None or self._step_index >= len(self._current_plan.steps):
            # Thu rebuild (neu state da thay doi)
            new_plan = self._build_plan()
            if not new_plan.steps:
                return None
            # Neu plan moi khac plan cu (do state), reset index
            self._current_plan = new_plan
            self._step_index = 0

        if self._step_index >= len(self._current_plan.steps):
            return None

        step = self._current_plan.steps[self._step_index]
        self._step_index += 1

        # Double-check: cac quadrants trong step phai chua generated
        generated_set = self.quadrant_state.all_generated()
        for q in step.quadrants:
            if q in generated_set:
                # Quadrant da gen -> skip step nay, lay step tiep theo
                return self.get_next_step()

        return step

    def mark_done(self, quadrants: Iterable[tuple[int, int]]) -> None:
        """Danh dau N quadrants vua gen xong (step 2x2/2x1/1x2/1x1)."""
        self.quadrant_state.mark_generated(quadrants)
        # Rebuild plan de buoc tiep theo tinh chinh xac
        self._current_plan = self._build_plan()
        self._step_index = 0

    def is_complete(self) -> bool:
        """True neu tat ca quadrant trong bounds da generated."""
        for q in self._bounds.all_points():
            if not self.quadrant_state.is_generated(q.x, q.y):
                return False
        return True

    @property
    def bounds(self) -> RectBounds:
        return self._bounds

    @property
    def progress(self) -> tuple[int, int]:
        """(generated_quadrants, total_quadrants)."""
        total = self._bounds.area
        done = sum(
            1
            for q in self._bounds.all_points()
            if self.quadrant_state.is_generated(q.x, q.y)
        )
        return done, total


# ============================================================================
# Helpers cho TemplateBuilder
# ============================================================================


def scan_generated_set(gen_dir: Path) -> set[tuple[int, int]]:
    """
    Scan gen_dir, return set of quadrant (qx, qy) that are fully generated.

    CRITICAL: Mỗi file tile_<tile_qx>_<tile_qy>_*.png chứa 4 quadrants:
      (tile_qx*2,   tile_qy*2)   → top-left
      (tile_qx*2+1, tile_qy*2)   → top-right
      (tile_qx*2,   tile_qy*2+1) → bottom-left
      (tile_qx*2+1, tile_qy*2+1) → bottom-right
    """
    out: set[tuple[int, int]] = set()
    if not gen_dir.exists():
        return out

    for f in gen_dir.glob("tile_*_*_*.png"):
        m = TILE_FILENAME_RE.match(f.name)
        if not m:
            continue
        tile_qx = _sign_int_to_int(m.group(1))
        tile_qy = _sign_int_to_int(m.group(2))
        # Verify file is non-empty (avoid counting blank/failed tiles)
        if f.stat().st_size < 30 * 1024:  # 30KB threshold
            continue
        # Tile chứa 4 quadrants
        out.add((tile_qx * 2,     tile_qy * 2))
        out.add((tile_qx * 2 + 1, tile_qy * 2))
        out.add((tile_qx * 2,     tile_qy * 2 + 1))
        out.add((tile_qx * 2 + 1, tile_qy * 2 + 1))

    return out


def _sign_int(n: int) -> str:
    """Format signed integer: positive -> '+n', negative/zero -> 'n'."""
    return f"+{n}" if n >= 0 else str(n)


def _sign_int_to_int(s: str) -> int:
    """Parse signed-integer string (from filename) -> int.

    Inverse of `_sign_int`. Accepts strings like '+5', '5', '-3', '+0', '0'.
    """
    if s.startswith("+"):
        return int(s[1:])
    return int(s)


def crop_quadrant(img: Image.Image, qx: int, qy: int) -> Image.Image:
    """
    Crop 512x512 quadrant từ tile 1024x1024.

    Quadrant (qx, qy) offset trong tile:
      left = (qx % 2) * 512  → 0 hoặc 512
      top  = (qy % 2) * 512  → 0 hoặc 512
    """
    if img.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        img = img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS)
    left = (qx % 2) * QUADRANT_SIZE
    top  = (qy % 2) * QUADRANT_SIZE
    return img.crop((left, top, left + QUADRANT_SIZE, top + QUADRANT_SIZE))


def make_has_generation(
    generated_set: set[tuple[int, int]],
) -> Callable[[int, int], bool]:
    """Closure cho has_generation(qx, qy)."""

    def has_generation(qx: int, qy: int) -> bool:
        return (qx, qy) in generated_set

    return has_generation


def _make_provider(
    dir_path: Path,
    cache: Optional[dict],
    label: str,
) -> Callable[[int, int], Optional[Image.Image]]:
    """Generic provider: load tile 1024x1024 + crop quadrant 512x512."""
    cache = cache if cache is not None else {}

    def get_image(qx: int, qy: int) -> Optional[Image.Image]:
        tile_qx = qx // 2
        tile_qy = qy // 2
        tile_key = (tile_qx, tile_qy)
        if tile_key in cache:
            full_tile = cache[tile_key]
        else:
            pattern = f"tile_{_sign_int(tile_qx)}_{_sign_int(tile_qy)}_*.png"
            matches = list(dir_path.glob(pattern))
            if not matches:
                cache[tile_key] = None
                return None
            try:
                full_tile = Image.open(matches[0]).convert("RGB")
                if full_tile.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                    full_tile = full_tile.resize(
                        (TEMPLATE_SIZE, TEMPLATE_SIZE),
                        Image.Resampling.LANCZOS,
                    )
                cache[tile_key] = full_tile
            except Exception:
                cache[tile_key] = None
                return None
        if full_tile is None:
            return None
        return crop_quadrant(full_tile, qx, qy)

    return get_image


def make_render_provider(
    renders_dir: Path,
    render_cache: Optional[dict] = None,
) -> Callable[[int, int], Optional[Image.Image]]:
    """
    Closure cho get_render(qx, qy).

    Tra ve 512x512 PIL Image (crop tu render 1024x1024) hoac None.
    """
    return _make_provider(renders_dir, render_cache, "render")


def make_generation_provider(
    gen_dir: Path,
    generation_cache: Optional[dict] = None,
) -> Callable[[int, int], Optional[Image.Image]]:
    """
    Closure cho get_generation(qx, qy).

    Tra ve 512x512 PIL Image (crop tu generation 1024x1024) hoac None.
    """
    return _make_provider(gen_dir, generation_cache, "generation")


def quadrant_iteration_order(
    start_qx: int, start_qy: int
) -> list[tuple[int, int, str]]:
    """Order generate quadrants within 1 tile (2x2):
    tl, tr, bl, br - moi buoc sau co context tu buoc truoc.
    """
    return [
        (start_qx * 2,     start_qy * 2,     "tl"),
        (start_qx * 2 + 1, start_qy * 2,     "tr"),
        (start_qx * 2,     start_qy * 2 + 1, "bl"),
        (start_qx * 2 + 1, start_qy * 2 + 1, "br"),
    ]