"""
Generation plan cho isometric infill.

Port tu `isometric-nyc/src/isometric_nyc/generation/make_rectangle_plan.py`.
Logic goc: chon tile size (2x2, 2x1, 1x2, 1x1) theo 3 placement rules, dam bao
khong tao seam voi generated neighbors.

Khi nao gen 2x2 (full tile, 4 quadrants / request):
- 4 quadrants nam trong bounds
- KHONG quadrant nao cua tile da generated hoac scheduled
- KHONG bat ky exterior neighbor nao (8 quadrants xung quanh tile) da generated

Khi nao gen 2x1 ngang (2 quadrants / request):
- 2 quadrants lien ke ngang
- "Long side" (top hoac bottom) co CA 2 neighbor generated
- "Short side" (left va right) KHONG co neighbor generated

Khi nao gen 1x2 doc (2 quadrants / request):
- 2 quadrants lien ke doc
- "Long side" (left hoac right) co CA 2 neighbor generated
- "Short side" (top va bottom) KHONG co neighbor generated

Khi nao gen 1x1 (1 quadrant / request):
- It nhat 1/4 to 2x2 block chua quadrant nay co 3 quadrants khac da generated
- Dam bao co context day du khi gen
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class Point:
    """Toa do quadrant (qx, qy)."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: tuple[int, int]) -> "Point":
        return cls(t[0], t[1])

    @classmethod
    def from_string(cls, s: str) -> "Point":
        s = s.strip().replace("(", "").replace(")", "").replace(" ", "")
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid coordinate format: {s}")
        return cls(int(parts[0]), int(parts[1]))


@dataclass
class RectBounds:
    """Bounds cua 1 rectangle vung quadrant can gen."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """Mot buoc generation: 1, 2, hoac 4 quadrants cung 1 request."""

    quadrants: list[Point]
    step_type: str = ""  # "2x2" | "2x1" | "1x2" | "1x1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadrants": [q.to_tuple() for q in self.quadrants],
            "type": self.step_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationStep":
        return cls(
            quadrants=[Point.from_tuple(t) for t in data["quadrants"]],
            step_type=data.get("type", ""),
        )


@dataclass
class RectanglePlan:
    """Plan day du cho 1 rectangle."""

    bounds: RectBounds
    steps: list[GenerationStep] = field(default_factory=list)
    pre_generated: set[Point] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": {
                "top_left": self.bounds.top_left.to_tuple(),
                "bottom_right": self.bounds.bottom_right.to_tuple(),
            },
            "steps": [step.to_dict() for step in self.steps],
            "pre_generated": [p.to_tuple() for p in self.pre_generated],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RectanglePlan":
        bounds_dict = data["bounds"]
        return cls(
            bounds=RectBounds(
                Point.from_tuple(tuple(bounds_dict["top_left"])),
                Point.from_tuple(tuple(bounds_dict["bottom_right"])),
            ),
            steps=[GenerationStep.from_dict(s) for s in data.get("steps", [])],
            pre_generated={
                Point.from_tuple(t) for t in data.get("pre_generated", [])
            },
        )

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for step in self.steps:
            t = step.step_type or f"{len(step.quadrants)}-quad"
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "bounds": {
                "tl": self.bounds.top_left.to_tuple(),
                "br": self.bounds.bottom_right.to_tuple(),
                "width": self.bounds.width,
                "height": self.bounds.height,
            },
            "pre_generated_count": len(self.pre_generated),
            "total_steps": len(self.steps),
            "total_quadrants": sum(len(s.quadrants) for s in self.steps),
            "steps_by_type": by_type,
        }


# ============================================================================
# 2x2 Tile Placement
# ============================================================================


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """4 quadrants cua 2x2 tile cho top-left corner."""
    x, y = top_left.x, top_left.y
    return [
        Point(x, y),
        Point(x + 1, y),
        Point(x, y + 1),
        Point(x + 1, y + 1),
    ]


def get_2x2_neighbors(top_left: Point) -> list[Point]:
    """8 exterior neighbors cua 1 2x2 tile (Top/Bottom/Left/Right pairs)."""
    x, y = top_left.x, top_left.y
    return [
        # Top
        Point(x, y - 1),
        Point(x + 1, y - 1),
        # Bottom
        Point(x, y + 2),
        Point(x + 1, y + 2),
        # Left
        Point(x - 1, y),
        Point(x - 1, y + 1),
        # Right
        Point(x + 2, y),
        Point(x + 2, y + 1),
    ]


def can_place_2x2(
    top_left: Point,
    bounds: RectBounds,
    generated: set[Point],
    scheduled: set[Point],
    allow_adjacent_scheduled: bool = False,
) -> bool:
    """
    Check 2x2 tile co the dat tai top_left khong.

    Rules:
    - All 4 quadrants in bounds, khong generated, khong scheduled.
    - NO exterior neighbor duoc generated (seam prevention).
    - Neu allow_adjacent_scheduled=False, cung khong duoc co neighbor scheduled.
    """
    quadrants = get_2x2_quadrants(top_left)

    for q in quadrants:
        if not bounds.contains(q):
            return False
        if q in generated or q in scheduled:
            return False

    neighbors = get_2x2_neighbors(top_left)
    for n in neighbors:
        if n in generated:
            return False
        if not allow_adjacent_scheduled and n in scheduled:
            return False

    return True


def _find_valid_2x2_positions(
    bounds: RectBounds,
    generated: set[Point],
    scheduled: set[Point],
    allow_adjacent_scheduled: bool,
) -> list[Point]:
    valid: list[Point] = []
    # Can phai con cho cho 2x2 (vi vay range cuoi khong + 1)
    for y in range(bounds.top_left.y, bounds.bottom_right.y):
        for x in range(bounds.top_left.x, bounds.bottom_right.x):
            tl = Point(x, y)
            if can_place_2x2(tl, bounds, generated, scheduled, allow_adjacent_scheduled):
                valid.append(tl)
    return valid


def place_2x2_tiles(
    bounds: RectBounds,
    effective_generated: set[Point],
) -> tuple[list[GenerationStep], set[Point]]:
    """
    Dat toi da 2x2 tiles.

    Strategy:
    1. First pass: 2x2 tiles KHONG cham generated HOAC scheduled neighbors
       (tao pattern co gap de bridge sau).
    2. Second pass: fill 2x2 gaps con lai bang cach allow adjacent scheduled
       (dense packing cho vung trong rong).
    """
    steps: list[GenerationStep] = []
    scheduled: set[Point] = set()

    # First pass
    while True:
        valid = _find_valid_2x2_positions(
            bounds, effective_generated, scheduled, allow_adjacent_scheduled=False
        )
        if not valid:
            break
        tl = valid[0]
        quadrants = get_2x2_quadrants(tl)
        steps.append(GenerationStep(quadrants=quadrants, step_type="2x2"))
        scheduled.update(quadrants)

    # Second pass
    while True:
        valid = _find_valid_2x2_positions(
            bounds, effective_generated, scheduled, allow_adjacent_scheduled=True
        )
        if not valid:
            break
        tl = valid[0]
        quadrants = get_2x2_quadrants(tl)
        steps.append(GenerationStep(quadrants=quadrants, step_type="2x2"))
        scheduled.update(quadrants)

    return steps, scheduled


# ============================================================================
# 2x1 / 1x2 Tile Placement
# ============================================================================


def can_place_2x1_horizontal(
    left: Point,
    bounds: RectBounds,
    generated: set[Point],
    scheduled: set[Point],
) -> bool:
    """
    Check 2x1 horizontal (left + left+1) tai y=left.y.

    Rules:
    - 2 quadrants in bounds, unscheduled, ungenerated.
    - Top HOAC bottom co CA 2 neighbors generated/scheduled (long side context).
    - Left VA right KHONG generated/scheduled (transverse sides clear).
    """
    right = Point(left.x + 1, left.y)

    for q in (left, right):
        if not bounds.contains(q):
            return False
        if q in generated or q in scheduled:
            return False

    left_neighbor = Point(left.x - 1, left.y)
    right_neighbor = Point(right.x + 1, right.y)
    if left_neighbor in generated or left_neighbor in scheduled:
        return False
    if right_neighbor in generated or right_neighbor in scheduled:
        return False

    combined = generated | scheduled
    top_both = (
        Point(left.x, left.y - 1) in combined
        and Point(right.x, right.y - 1) in combined
    )
    bottom_both = (
        Point(left.x, left.y + 1) in combined
        and Point(right.x, right.y + 1) in combined
    )
    return top_both or bottom_both


def can_place_1x2_vertical(
    top: Point,
    bounds: RectBounds,
    generated: set[Point],
    scheduled: set[Point],
) -> bool:
    """
    Check 1x2 vertical (top + top+1) tai x=top.x.

    Rules:
    - 2 quadrants in bounds, unscheduled, ungenerated.
    - Left HOAC right co CA 2 neighbors generated/scheduled.
    - Top VA bottom KHONG generated/scheduled.
    """
    bottom = Point(top.x, top.y + 1)

    for q in (top, bottom):
        if not bounds.contains(q):
            return False
        if q in generated or q in scheduled:
            return False

    top_neighbor = Point(top.x, top.y - 1)
    bottom_neighbor = Point(bottom.x, bottom.y + 1)
    if top_neighbor in generated or top_neighbor in scheduled:
        return False
    if bottom_neighbor in generated or bottom_neighbor in scheduled:
        return False

    combined = generated | scheduled
    left_both = (
        Point(top.x - 1, top.y) in combined
        and Point(top.x - 1, bottom.y) in combined
    )
    right_both = (
        Point(top.x + 1, top.y) in combined
        and Point(top.x + 1, bottom.y) in combined
    )
    return left_both or right_both


def _find_valid_2x1_positions(
    bounds: RectBounds,
    generated: set[Point],
    scheduled: set[Point],
) -> list[tuple[Point, str]]:
    """Find valid 2x1 (horizontal) va 1x2 (vertical) positions.

    Returns list of (top-left, type). Uu tien 1x2 vertical truoc (bridge
    2x2 tiles), 2x1 horizontal sau (connect to generation edge).
    """
    valid: list[tuple[Point, str]] = []

    # Vertical 1x2 first (bridges between 2x2 tiles)
    for y in range(bounds.top_left.y, bounds.bottom_right.y):
        for x in range(bounds.top_left.x, bounds.bottom_right.x + 1):
            top = Point(x, y)
            if can_place_1x2_vertical(top, bounds, generated, scheduled):
                valid.append((top, "1x2"))

    # Horizontal 2x1 second
    for y in range(bounds.top_left.y, bounds.bottom_right.y + 1):
        for x in range(bounds.top_left.x, bounds.bottom_right.x):
            left = Point(x, y)
            if can_place_2x1_horizontal(left, bounds, generated, scheduled):
                valid.append((left, "2x1"))

    return valid


def place_2x1_tiles(
    bounds: RectBounds,
    effective_generated: set[Point],
    scheduled: set[Point],
) -> tuple[list[GenerationStep], set[Point]]:
    """Place 2x1 and 1x2 tiles o cac gap giua 2x2 tiles va generation edge."""
    steps: list[GenerationStep] = []
    new_scheduled = set(scheduled)

    while True:
        valid = _find_valid_2x1_positions(bounds, effective_generated, new_scheduled)
        if not valid:
            break

        pos, tile_type = valid[0]
        if tile_type == "2x1":
            quadrants = [pos, Point(pos.x + 1, pos.y)]
        else:  # "1x2"
            quadrants = [pos, Point(pos.x, pos.y + 1)]

        steps.append(GenerationStep(quadrants=quadrants, step_type=tile_type))
        new_scheduled.update(quadrants)

    return steps, new_scheduled


# ============================================================================
# 1x1 Tile Placement
# ============================================================================


def get_2x2_block_positions(p: Point) -> list[list[Point]]:
    """Lay 4 vi tri 2x2 block chua p (p o moi corner)."""
    return [
        # p as top-left
        [p, Point(p.x + 1, p.y), Point(p.x, p.y + 1), Point(p.x + 1, p.y + 1)],
        # p as top-right
        [Point(p.x - 1, p.y), p, Point(p.x - 1, p.y + 1), Point(p.x, p.y + 1)],
        # p as bottom-left
        [Point(p.x, p.y - 1), Point(p.x + 1, p.y - 1), p, Point(p.x + 1, p.y)],
        # p as bottom-right
        [Point(p.x - 1, p.y - 1), Point(p.x, p.y - 1), Point(p.x - 1, p.y), p],
    ]


def _count_in_block(block: list[Point], combined: set[Point]) -> int:
    return sum(1 for p in block if p in combined)


def has_valid_2x2_context(
    quadrants: list[Point],
    combined: set[Point],
) -> bool:
    """Check quadrants co the gen voi context 2x2 day du hay khong.

    Phai co it nhat 1/4 2x2 block (chua 1 trong cac quadrants) co CA 4 quadrants
    hoac la duoc generate (o quadrants list), hoac da generated/scheduled (trong
    combined set).
    """
    quadrant_set = set(quadrants)
    for q in quadrants:
        for block in get_2x2_block_positions(q):
            if all(p in quadrant_set or p in combined for p in block):
                return True
    return False


def can_place_1x1(p: Point, combined: set[Point]) -> bool:
    """Check 1x1 quadrant co the gen (co context tu 3 quadrant khac).

    Co the thuoc nhieu 2x2 blocks; it nhat 1 block phai co 3 quadrants khac
    (khong tinh p) da generated.
    """
    for block in get_2x2_block_positions(p):
        other_generated = sum(1 for q in block if q != p and q in combined)
        if other_generated >= 3:
            return True
    return False


def place_1x1_tiles(
    bounds: RectBounds,
    effective_generated: set[Point],
    scheduled: set[Point],
) -> list[GenerationStep]:
    """Fill gap con lai bang 1x1 (chi khi co context 2x2 day du)."""
    steps: list[GenerationStep] = []
    combined = effective_generated | scheduled
    new_scheduled = set(scheduled)

    remaining = [p for p in bounds.all_points() if p not in combined]

    # Sort theo priority: nhieu generated neighbors hon = uu tien hon
    def priority(p: Point) -> int:
        blocks = get_2x2_block_positions(p)
        max_generated = max(
            _count_in_block(block, combined | new_scheduled) for block in blocks
        )
        return -max_generated  # negative for descending sort

    remaining.sort(key=priority)

    changed = True
    while changed:
        changed = False
        for p in list(remaining):
            if p in new_scheduled:
                remaining.remove(p)
                continue
            if can_place_1x1(p, combined | new_scheduled):
                steps.append(GenerationStep(quadrants=[p], step_type="1x1"))
                new_scheduled.add(p)
                remaining.remove(p)
                changed = True

    return steps


# ============================================================================
# Main Algorithm
# ============================================================================


def create_rectangle_plan(
    bounds: RectBounds,
    generated: Optional[set[Point]] = None,
    queued: Optional[set[Point]] = None,
) -> RectanglePlan:
    """
    Tao plan day du cho 1 rectangle.

    Args:
        bounds: Vung rectangle (top-left -> bottom-right inclusive).
        generated: Set da generated (giong set trong DB).
        queued: Set dang gen (in-progress) - duoc treat nhu "se generated" cho
                seam detection.
    """
    generated = generated or set()
    queued = queued or set()
    effective_generated = generated | queued

    points_to_generate = set(bounds.all_points()) - effective_generated
    if not points_to_generate:
        return RectanglePlan(bounds=bounds, steps=[], pre_generated=generated)

    # Phase 1: 2x2 tiles (full tile, 4 quads / request)
    steps_2x2, scheduled = place_2x2_tiles(bounds, effective_generated)
    # Phase 2: 2x1/1x2 bridges (2 quads / request)
    steps_2x1, scheduled = place_2x1_tiles(bounds, effective_generated, scheduled)
    # Phase 3: 1x1 fills (1 quad / request)
    steps_1x1 = place_1x1_tiles(bounds, effective_generated, scheduled)

    return RectanglePlan(
        bounds=bounds,
        steps=steps_2x2 + steps_2x1 + steps_1x1,
        pre_generated=generated,
    )


def create_rectangle_plan_from_tuples(
    tl: tuple[int, int],
    br: tuple[int, int],
    generated: Optional[set[tuple[int, int]]] = None,
    queued: Optional[set[tuple[int, int]]] = None,
) -> RectanglePlan:
    """Tao plan tu toa do tuple."""
    bounds = RectBounds(Point.from_tuple(tl), Point.from_tuple(br))
    gen = {Point.from_tuple(t) for t in (generated or set())}
    que = {Point.from_tuple(t) for t in (queued or set())}
    return create_rectangle_plan(bounds, gen, que)


# ============================================================================
# Plan persistence
# ============================================================================


def save_plan(plan: RectanglePlan, path: str | Path) -> None:
    """Save plan thanh JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)


def load_plan(path: str | Path) -> RectanglePlan:
    """Load plan tu JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return RectanglePlan.from_dict(json.load(f))
