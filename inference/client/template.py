"""
Infill template generation cho isometric pipeline.

Port tu `isometric-nyc/src/isometric_nyc/generation/infill_template.py`,
giu nguyen logic seam-avoidance va placement optimization. Logic cot loi:

- Infill region (den 50% cua 1024x1024 tile) duoc dat trong template sao cho
  KHONG canh nao cua infill (left/right/top/bottom) cham generated neighbor
  (tranh seam noi dung).
- Neu co generated neighbor, placement se "day" infill sang phia doi dien de
  context quadrants (xung quanh) la generated quadrants.
- Neu khong tim duoc placement hoan hao, co gang expand infill (cho phep
  them padding quadrants) hoac fall back seam-tolerant placement.

Template luon 1024x1024 px. Quadrant 512x512 px (the isometric-nyc convention).

================================================================================
SEAM RULES (from PHASE 3.1 of the refactor plan, ported tu isometric-nyc)
================================================================================
Khi dat infill trong template 1024x1024, CANH cua infill KHONG duoc cham generated
quadrant ben ngoai (seam = noi dung bi "dung dot" giua render va generated).

Rule 1 - 2x2 Tile (full tile, 4 quads/request):
    + Ca 4 quadrants nam trong template bounds.
    + KHONG quadrant nao trong 4 da duoc generated.
    + KHONG exterior neighbor nao (8 xung quanh) da duoc generated.
    -> Dat full tile tai goc (0, 0) cua template.

Rule 2 - 2x1 Horizontal (2 quads/request, canh nhau theo chieu ngang):
    + 2 quadrants ngang lien ke (left, left+1) trong cung template.
    + Long side (top HOAC bottom) co DU 2 generated neighbors (context day du).
    + Short side (left, right) KHONG co neighbor generated (tranh seam).
    -> Dat infill o phia doi dien voi long-side generated.

Rule 3 - 1x2 Vertical (2 quads/request, canh nhau theo chieu doc):
    + 2 quadrants doc lien ke (top, top+1) trong cung template.
    + Long side (left HOAC right) co DU 2 generated neighbors.
    + Short side (top, bottom) KHONG co neighbor generated.
    -> Dat infill o phia doi dien voi long-side generated.

Rule 4 - 1x1 (1 quad/request):
    + It nhat 1 trong 4 quadrants cua 2x2 block chua quadrant co 3 quadrants
      khac da generated (dam bao context 2x2 day du).
    -> Dat infill o goc cua 2x2 block, de 3 generated quadrants o context.

Algorithm placement (template.py::find_optimal_placement):
    1. Try placement theo context preferences (push infill sang phia doi dien
       voi generated neighbors).
    2. Validate seams (neu edge cua infill cham generated neighbor -> reject).
    3. Neu thieu context quadrants, thu alternative placements.
    4. Last resort: seam-tolerant placement hoac expand infill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image, ImageDraw

from inference.config import get_inference_config

# Load config (singleton, lazy).
# Source of truth: src/config.mjs (TILE.sizePx, INFILL.borderWidth, ...).
# Do NOT hardcode nhung gia tri nay - chung co the thay doi khi config.mjs doi.
_cfg = get_inference_config()


def _tile_size() -> int:
    """Lazy lookup cho TILE.sizePx (default 1024)."""
    return _cfg.tile_size_px


def _quadrant_size() -> int:
    """Lazy lookup cho quadrant size = tile_size // 2 (default 512)."""
    return _cfg.quadrant_size_px


def _max_infill_area() -> int:
    """Lazy lookup cho max_infill_area (default 50% cua template = 524288)."""
    return _cfg.max_infill_area


# Backward-compatible constants (referenced by other modules).
# These are evaluated at import time; tests that need to override config
# should call reset_config() and reload this module.
TEMPLATE_SIZE: int = _tile_size()
QUADRANT_SIZE: int = _quadrant_size()
MAX_INFILL_AREA: int = _max_infill_area()

# Loai cua has_generation/get_render/get_generation callbacks
HasGeneration = Callable[[int, int], bool]
GetQuadrantImage = Callable[[int, int], Optional[Image.Image]]


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class InfillRegion:
    """Vung hinh chu nhat can duoc gen.

    Toa do la "world" pixel space, trong do:
    - (0, 0) la top-left cua quadrant (0, 0)
    - x tang sang phai, y tang xuong duoi
    - Moi quadrant 512x512 pixels
    """

    x: int  # World x (top-left)
    y: int  # World y (top-left)
    width: int  # Width (px)
    height: int  # Height (px)

    @classmethod
    def from_quadrant(cls, qx: int, qy: int) -> "InfillRegion":
        """Tao region cho 1 quadrant don le."""
        return cls(
            x=qx * QUADRANT_SIZE,
            y=qy * QUADRANT_SIZE,
            width=QUADRANT_SIZE,
            height=QUADRANT_SIZE,
        )

    @classmethod
    def from_quadrants(cls, quadrants: list[tuple[int, int]]) -> "InfillRegion":
        """Tao region chua nhieu quadrant (phai lien tuc thanh rectangle)."""
        if not quadrants:
            raise ValueError("At least one quadrant required")
        min_qx = min(q[0] for q in quadrants)
        max_qx = max(q[0] for q in quadrants)
        min_qy = min(q[1] for q in quadrants)
        max_qy = max(q[1] for q in quadrants)
        return cls(
            x=min_qx * QUADRANT_SIZE,
            y=min_qy * QUADRANT_SIZE,
            width=(max_qx - min_qx + 1) * QUADRANT_SIZE,
            height=(max_qy - min_qy + 1) * QUADRANT_SIZE,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def is_valid_size(self) -> bool:
        """<= 50% template, hoac chinh xac full tile (2x2 quadrants)."""
        return self.area <= MAX_INFILL_AREA or self.is_full_tile()

    def is_full_tile(self) -> bool:
        return self.width == TEMPLATE_SIZE and self.height == TEMPLATE_SIZE

    def overlapping_quadrants(self) -> list[tuple[int, int]]:
        """List (qx, qy) ma region nay chiem (khi aligned voi quadrant grid)."""
        start_qx = self.x // QUADRANT_SIZE
        end_qx = (self.right - 1) // QUADRANT_SIZE
        start_qy = self.y // QUADRANT_SIZE
        end_qy = (self.bottom - 1) // QUADRANT_SIZE
        return [
            (qx, qy)
            for qx in range(start_qx, end_qx + 1)
            for qy in range(start_qy, end_qy + 1)
        ]

    def __str__(self) -> str:
        return f"InfillRegion(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


@dataclass
class TemplatePlacement:
    """Vi tri dat infill trong template 1024x1024."""

    infill_x: int  # Vi tri (x, y) cua infill trong template (0..1024)
    infill_y: int
    world_offset_x: int  # Toa do world cua top-left template
    world_offset_y: int

    # Set sau khi construction
    _infill_width: int = 0
    _infill_height: int = 0
    _primary_quadrants: list[tuple[int, int]] = field(default_factory=list)
    _padding_quadrants: list[tuple[int, int]] = field(default_factory=list)
    _expanded_region: Optional[InfillRegion] = None

    @property
    def infill_width(self) -> int:
        return self._infill_width

    @property
    def infill_height(self) -> int:
        return self._infill_height

    @property
    def infill_right(self) -> int:
        return self.infill_x + self._infill_width

    @property
    def infill_bottom(self) -> int:
        return self.infill_y + self._infill_height

    @property
    def primary_quadrants(self) -> list[tuple[int, int]]:
        """Quadrants user chon ban dau."""
        return self._primary_quadrants

    @property
    def padding_quadrants(self) -> list[tuple[int, int]]:
        """Quadrants auto-them de cover missing context."""
        return self._padding_quadrants

    @property
    def all_infill_quadrants(self) -> list[tuple[int, int]]:
        """Tat ca quadrants se duoc fill voi render pixels."""
        return self._primary_quadrants + self._padding_quadrants

    @property
    def is_expanded(self) -> bool:
        return len(self._padding_quadrants) > 0


# ============================================================================
# TemplateBuilder
# ============================================================================


class TemplateBuilder:
    """Build template image cho infill generation.

    Dam bao:
    - Dat infill region toi uu de maximize context (generated neighbors).
    - Validate seam (neu edge cua infill cham generated neighbor -> reject).
    - Assembly template tu quadrant data (context = generation, infill = render).
    """

    def __init__(
        self,
        infill_region: InfillRegion,
        has_generation: HasGeneration,
        get_render: Optional[GetQuadrantImage] = None,
        get_generation: Optional[GetQuadrantImage] = None,
    ):
        """
        Args:
            infill_region: Vung can gen.
            has_generation: Callable(qx, qy) -> bool. True neu quadrant da generated.
            get_render: Callable(qx, qy) -> Image. Render 3D (dung cho infill).
            get_generation: Callable(qx, qy) -> Image. Generated pixels (context).
        """
        self.region = infill_region
        self.has_generation = has_generation
        self.get_render = get_render
        self.get_generation = get_generation
        self._last_validation_error = ""

        if not infill_region.is_valid_size():
            raise ValueError(
                f"Infill region too large: {infill_region.area} pixels "
                f"(max: {MAX_INFILL_AREA})"
            )

    def find_optimal_placement(
        self, allow_expansion: bool = False
    ) -> Optional[TemplatePlacement]:
        """
        Tim optimal placement cho infill region trong template.

        Returns None neu khong co placement nao tranh duoc seam.

        Algorithm (seam rules tu PHASE 3.1 cua plan):
            1. Thuc hien "try placement" theo context preferences (push infill
               sang phia doi dien voi generated neighbors). Day la phuong an
               "best effort" cho 4 seam rules (2x2, 2x1, 1x2, 1x1).
            2. Validate seams: neu edge cua infill cham generated neighbor
               ben ngoai template -> reject placement.
            3. Neu thieu context quadrants (quadrant trong template, ngoai
               infill, chua generated) -> thu alternative placements.
            4. Last resort: expand infill (neu allow_expansion) hoac seam-
               tolerant placement chap nhan 1 vai seam.
        """
        # Step 1: Try the "natural" placement (push infill away from generated
        # neighbors on every side). This is the baseline that satisfies the
        # seam rules for the common case.
        placement = self._try_placement_with_context_preferences(
            include_left=True,
            include_right=True,
            include_top=True,
            include_bottom=True,
        )

        # Step 2: No valid placement at all -> bail out.
        if placement is None:
            return None

        # Step 3: Check whether the template has any context quadrants that
        # are NOT yet generated. If everything is filled, we are done.
        missing = self._find_missing_context_quadrants(placement)
        if not missing:
            return placement

        # Step 4: Some context quadrants are missing. Try alternative
        # placements that exclude specific problem sides to see if we can
        # find a configuration where every context quadrant is generated.
        alternative = self._try_alternative_placements(missing, allow_expansion)
        if alternative is not None:
            return alternative

        # Step 5: Fallback - if expansion is allowed, grow the infill region
        # so that the previously-missing quadrants become part of the infill
        # (they will be generated by the model together with the primary
        # quadrants).
        if allow_expansion:
            expanded = self._expand_to_cover_missing(placement, missing)
            if expanded is not None:
                return expanded

        # Step 6: Nothing worked - record the reason and return None.
        missing_str = ", ".join(f"({qx}, {qy})" for qx, qy in missing)
        self._last_validation_error = (
            f"Context quadrants missing generations: {missing_str}"
        )
        return None

    def _try_placement_with_context_preferences(
        self,
        include_left: bool,
        include_right: bool,
        include_top: bool,
        include_bottom: bool,
    ) -> Optional[TemplatePlacement]:
        """
        Try placement theo context preferences (which sides to include).
        """
        margin_x = TEMPLATE_SIZE - self.region.width
        margin_y = TEMPLATE_SIZE - self.region.height

        has_left_gen = self._has_generated_context("left") if include_left else False
        has_right_gen = self._has_generated_context("right") if include_right else False
        has_top_gen = self._has_generated_context("top") if include_top else False
        has_bottom_gen = (
            self._has_generated_context("bottom") if include_bottom else False
        )

        # Horizontal positioning
        if has_left_gen and has_right_gen:
            infill_x = margin_x // 2
        elif has_left_gen:
            infill_x = margin_x
        elif has_right_gen:
            infill_x = 0
        else:
            # Khong co context -> dat phia xa generated
            actual_left = self._has_generated_context("left")
            actual_right = self._has_generated_context("right")
            if actual_right and not actual_left:
                infill_x = 0
            elif actual_left and not actual_right:
                infill_x = margin_x
            else:
                infill_x = 0

        # Vertical positioning
        if has_top_gen and has_bottom_gen:
            infill_y = margin_y // 2
        elif has_top_gen:
            infill_y = margin_y
        elif has_bottom_gen:
            infill_y = 0
        else:
            actual_top = self._has_generated_context("top")
            actual_bottom = self._has_generated_context("bottom")
            if actual_bottom and not actual_top:
                infill_y = 0
            elif actual_top and not actual_bottom:
                infill_y = margin_y
            else:
                infill_y = 0

        world_offset_x = self.region.x - infill_x
        world_offset_y = self.region.y - infill_y

        placement = TemplatePlacement(
            infill_x=infill_x,
            infill_y=infill_y,
            world_offset_x=world_offset_x,
            world_offset_y=world_offset_y,
        )
        placement._infill_width = self.region.width
        placement._infill_height = self.region.height

        is_valid, error = self._validate_placement_seams(placement)
        if not is_valid:
            self._last_validation_error = error
            return None

        return placement

    def _try_alternative_placements(
        self,
        missing: list[tuple[int, int]],
        allow_expansion: bool,
    ) -> Optional[TemplatePlacement]:
        """Thu cac placements khac de tranh missing context quadrants."""
        infill_quadrants = set(self.region.overlapping_quadrants())
        infill_min_qx = min(q[0] for q in infill_quadrants)
        infill_max_qx = max(q[0] for q in infill_quadrants)
        infill_min_qy = min(q[1] for q in infill_quadrants)
        infill_max_qy = max(q[1] for q in infill_quadrants)

        problem_sides = set()
        for qx, qy in missing:
            if qx < infill_min_qx:
                problem_sides.add("left")
            if qx > infill_max_qx:
                problem_sides.add("right")
            if qy < infill_min_qy:
                problem_sides.add("top")
            if qy > infill_max_qy:
                problem_sides.add("bottom")

        side_combinations: list[set[str]] = []
        if problem_sides:
            side_combinations.append(problem_sides)
        for side in problem_sides:
            side_combinations.append({side})

        for exclude_sides in side_combinations:
            placement = self._try_placement_with_context_preferences(
                include_left="left" not in exclude_sides,
                include_right="right" not in exclude_sides,
                include_top="top" not in exclude_sides,
                include_bottom="bottom" not in exclude_sides,
            )
            if placement is None:
                continue

            new_missing = self._find_missing_context_quadrants(placement)
            if not new_missing:
                return placement

            if allow_expansion and len(new_missing) < len(missing):
                expanded = self._expand_to_cover_missing(placement, new_missing)
                if expanded is not None:
                    return expanded

        # Last resort: seam-tolerant placement
        best = self._try_seam_tolerant_placement(problem_sides)
        if best is not None:
            return best

        return None

    def _try_seam_tolerant_placement(
        self, problem_sides: set[str]
    ) -> Optional[TemplatePlacement]:
        """Tim placement chap nhan seam (last resort)."""
        margin_x = TEMPLATE_SIZE - self.region.width
        margin_y = TEMPLATE_SIZE - self.region.height

        has_left_gen = self._has_generated_context("left")
        has_right_gen = self._has_generated_context("right")
        has_top_gen = self._has_generated_context("top")
        has_bottom_gen = self._has_generated_context("bottom")

        positions: list[tuple[int, int]] = []

        # Loai bo problem sides, day infill sang phia doi dien
        if "left" in problem_sides:
            infill_x = 0
            if has_top_gen and "top" not in problem_sides:
                infill_y = margin_y
            elif has_bottom_gen and "bottom" not in problem_sides:
                infill_y = 0
            else:
                infill_y = 0
            positions.append((infill_x, infill_y))

        if "right" in problem_sides:
            infill_x = margin_x
            if has_top_gen and "top" not in problem_sides:
                infill_y = margin_y
            elif has_bottom_gen and "bottom" not in problem_sides:
                infill_y = 0
            else:
                infill_y = 0
            positions.append((infill_x, infill_y))

        if "top" in problem_sides:
            infill_y = 0
            if has_right_gen and "right" not in problem_sides:
                infill_x = 0
            elif has_left_gen and "left" not in problem_sides:
                infill_x = margin_x
            else:
                infill_x = 0
            positions.append((infill_x, infill_y))

        if "bottom" in problem_sides:
            infill_y = margin_y
            if has_right_gen and "right" not in problem_sides:
                infill_x = 0
            elif has_left_gen and "left" not in problem_sides:
                infill_x = margin_x
            else:
                infill_x = 0
            positions.append((infill_x, infill_y))

        # Try 4 corners
        corners = [
            (0, 0),
            (margin_x, 0),
            (0, margin_y),
            (margin_x, margin_y),
        ]
        positions.extend(corners)

        for infill_x, infill_y in positions:
            world_offset_x = self.region.x - infill_x
            world_offset_y = self.region.y - infill_y

            placement = TemplatePlacement(
                infill_x=infill_x,
                infill_y=infill_y,
                world_offset_x=world_offset_x,
                world_offset_y=world_offset_y,
            )
            placement._infill_width = self.region.width
            placement._infill_height = self.region.height

            missing = self._find_missing_context_quadrants(placement)
            if not missing:
                return placement

        return None

    def _has_generated_context(self, side: str) -> bool:
        """Check ben canh (left/right/top/bottom) cua region co generated quadrant khong."""
        if side == "left":
            check_x = self.region.x - 1
            qx = check_x // QUADRANT_SIZE
            start_qy = self.region.y // QUADRANT_SIZE
            end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
            return any(self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1))

        if side == "right":
            check_x = self.region.right
            qx = check_x // QUADRANT_SIZE
            start_qy = self.region.y // QUADRANT_SIZE
            end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
            return any(self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1))

        if side == "top":
            check_y = self.region.y - 1
            qy = check_y // QUADRANT_SIZE
            start_qx = self.region.x // QUADRANT_SIZE
            end_qx = (self.region.right - 1) // QUADRANT_SIZE
            return any(self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1))

        if side == "bottom":
            check_y = self.region.bottom
            qy = check_y // QUADRANT_SIZE
            start_qx = self.region.x // QUADRANT_SIZE
            end_qx = (self.region.right - 1) // QUADRANT_SIZE
            return any(self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1))

        return False

    def _validate_placement_seams(
        self, placement: TemplatePlacement
    ) -> tuple[bool, str]:
        """
        Check placement khong tao seam.

        Definition (tu PHASE 3 cua plan):
            Seam = edge cua infill (template boundary) co generated neighbor
            ben ngoai template. Neu seam xay ra, model se nhin thay 2 vung
            noi dung khac nhau (generated + render) noi tiep nhau -> noi dung
            "bi dut doan" (anh khong lien mach).

        Validation rules (4 canh cua infill):
            - LEFT seam:   infill_x == 0 AND generated quadrant o ben trai
                           (ngoai template).
            - RIGHT seam:  infill_x + width == TEMPLATE_SIZE AND generated
                           quadrant o ben phai (ngoai template).
            - TOP seam:    infill_y == 0 AND generated quadrant o ben tren
                           (ngoai template).
            - BOTTOM seam: infill_y + height == TEMPLATE_SIZE AND generated
                           quadrant o ben duoi (ngoai template).

        Returns:
            (True, "") neu khong co seam.
            (False, "<reason>") neu co it nhat 1 seam.
        """
        # ---- Left seam: infill touches the left edge of the template AND
        # there is a generated quadrant to the left of the infill region.
        if placement.infill_x == 0 and self._has_generated_context("left"):
            return False, "Would create seam with generated pixels on left"

        # ---- Right seam: infill touches the right edge of the template AND
        # there is a generated quadrant to the right of the infill region.
        if placement.infill_x + self.region.width == TEMPLATE_SIZE and self._has_generated_context(
            "right"
        ):
            return False, "Would create seam with generated pixels on right"

        # ---- Top seam: infill touches the top edge of the template AND
        # there is a generated quadrant above the infill region.
        if placement.infill_y == 0 and self._has_generated_context("top"):
            return False, "Would create seam with generated pixels on top"

        # ---- Bottom seam: infill touches the bottom edge of the template AND
        # there is a generated quadrant below the infill region.
        if placement.infill_y + self.region.height == TEMPLATE_SIZE and self._has_generated_context(
            "bottom"
        ):
            return False, "Would create seam with generated pixels on bottom"

        # No seam detected on any of the 4 sides -> placement is valid.
        return True, ""

    def _find_missing_context_quadrants(
        self, placement: TemplatePlacement
    ) -> list[tuple[int, int]]:
        """Context quadrants (in template, not in infill) ma khong co generation."""
        template_world_left = placement.world_offset_x
        template_world_right = placement.world_offset_x + TEMPLATE_SIZE
        template_world_top = placement.world_offset_y
        template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

        start_qx = template_world_left // QUADRANT_SIZE
        end_qx = (template_world_right - 1) // QUADRANT_SIZE
        start_qy = template_world_top // QUADRANT_SIZE
        end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

        infill_quadrants = set(self.region.overlapping_quadrants())

        missing: list[tuple[int, int]] = []
        for qx in range(start_qx, end_qx + 1):
            for qy in range(start_qy, end_qy + 1):
                if (qx, qy) not in infill_quadrants:
                    if not self.has_generation(qx, qy):
                        missing.append((qx, qy))
        return missing

    def _expand_to_cover_missing(
        self,
        placement: TemplatePlacement,
        missing: list[tuple[int, int]],
    ) -> Optional[TemplatePlacement]:
        """Expand infill region de cover missing context quadrants."""
        primary_quadrants = self.region.overlapping_quadrants()
        all_quadrants = set(primary_quadrants + missing)

        min_qx = min(q[0] for q in all_quadrants)
        max_qx = max(q[0] for q in all_quadrants)
        min_qy = min(q[1] for q in all_quadrants)
        max_qy = max(q[1] for q in all_quadrants)

        expanded_region = InfillRegion(
            x=min_qx * QUADRANT_SIZE,
            y=min_qy * QUADRANT_SIZE,
            width=(max_qx - min_qx + 1) * QUADRANT_SIZE,
            height=(max_qy - min_qy + 1) * QUADRANT_SIZE,
        )

        if not expanded_region.is_valid_size():
            self._last_validation_error = (
                f"Cannot expand infill to cover missing quadrants: "
                f"expanded region would be {expanded_region.area} pixels "
                f"(max: {MAX_INFILL_AREA})"
            )
            return None

        expanded_builder = TemplateBuilder(
            expanded_region, self.has_generation
        )
        expanded_placement = expanded_builder.find_optimal_placement(
            allow_expansion=False
        )

        if expanded_placement is None:
            self._last_validation_error = expanded_builder._last_validation_error
            return None

        expanded_placement._primary_quadrants = list(primary_quadrants)
        expanded_placement._padding_quadrants = list(missing)
        expanded_placement._expanded_region = expanded_region

        return expanded_placement

    def build(
        self,
        border_width: Optional[int] = None,
        allow_expansion: bool = False,
    ) -> Optional[tuple[Image.Image, TemplatePlacement]]:
        """Build template image (1024x1024).

        Border width va color lay tu config (INFILL.borderWidth/borderColor)
        neu khong truyen tham so. Day la gia tri source-of-truth tu
        src/config.mjs -> src/config.py.InfillConfig.

        Returns:
            (template_image, placement) hoac None neu khong co valid placement.
        """
        if self.get_render is None or self.get_generation is None:
            raise ValueError("get_render and get_generation must be provided to build")

        # Step 1: Find the best placement of the infill region inside the
        # 1024x1024 template (see find_optimal_placement for seam rules).
        placement = self.find_optimal_placement(allow_expansion=allow_expansion)
        if placement is None:
            return None

        # Step 2: If the placement was created by expanding the infill to
        # cover missing context quadrants, use the expanded region here so
        # that the additional padding quadrants also get rendered.
        effective_region = (
            placement._expanded_region if placement._expanded_region is not None
            else self.region
        )

        # Step 3: Create a transparent 1024x1024 canvas. This is the "template"
        # that the model will see as input.
        template = Image.new("RGBA", (TEMPLATE_SIZE, TEMPLATE_SIZE), (0, 0, 0, 0))

        # Step 4: Compute which world quadrants intersect with the template
        # (template_window = [world_offset, world_offset + TEMPLATE_SIZE)).
        template_world_left = placement.world_offset_x
        template_world_right = placement.world_offset_x + TEMPLATE_SIZE
        template_world_top = placement.world_offset_y
        template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

        start_qx = template_world_left // QUADRANT_SIZE
        end_qx = (template_world_right - 1) // QUADRANT_SIZE
        start_qy = template_world_top // QUADRANT_SIZE
        end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

        infill_quadrants = set(effective_region.overlapping_quadrants())
        if not any(self.get_render(qx, qy) is not None for (qx, qy) in infill_quadrants):
            self._last_validation_error = "No 3D render available for infill region"
            return None

        # Step 5: For every quadrant inside the template window, decide
        # whether it is part of the infill (use render / 3D output) or
        # part of the context (use existing AI generation).
        for qx in range(start_qx, end_qx + 1):
            for qy in range(start_qy, end_qy + 1):
                quad_world_x = qx * QUADRANT_SIZE
                quad_world_y = qy * QUADRANT_SIZE
                template_x = quad_world_x - template_world_left
                template_y = quad_world_y - template_world_top

                if (qx, qy) in infill_quadrants:
                    # Infill quadrant -> source = 3D render.
                    quad_img = self.get_render(qx, qy)
                    if quad_img is None:
                        continue
                else:
                    # Context quadrant -> source = existing AI generation.
                    quad_img = self.get_generation(qx, qy)
                    if quad_img is None:
                        continue

                # Normalize to 512x512 RGBA. DZI deep-zoom tiles may come
                # in a different size, so resize with LANCZOS for quality.
                if quad_img.size != (QUADRANT_SIZE, QUADRANT_SIZE):
                    quad_img = quad_img.resize(
                        (QUADRANT_SIZE, QUADRANT_SIZE), Image.Resampling.LANCZOS
                    )
                if quad_img.mode != "RGBA":
                    quad_img = quad_img.convert("RGBA")

                # The quadrant may partially overlap the template window
                # (when the template straddles the tile boundary). In that
                # case we crop to the overlap region before pasting.
                crop_left = max(0, -template_x)
                crop_top = max(0, -template_y)
                crop_right = min(QUADRANT_SIZE, TEMPLATE_SIZE - template_x)
                crop_bottom = min(QUADRANT_SIZE, TEMPLATE_SIZE - template_y)

                if crop_left < crop_right and crop_top < crop_bottom:
                    cropped = quad_img.crop(
                        (crop_left, crop_top, crop_right, crop_bottom)
                    )
                    paste_x = max(0, template_x)
                    paste_y = max(0, template_y)
                    template.paste(cropped, (paste_x, paste_y))

        # Step 6: Draw the red border around the infill region so the model
        # knows which pixels to edit. Width/color come from config.
        effective_border_width = (
            border_width if border_width is not None else _cfg.border_width
        )
        template = self._draw_border(template, placement, effective_border_width)

        return template, placement

    def _draw_border(
        self,
        template: Image.Image,
        placement: TemplatePlacement,
        border_width: int,
    ) -> Image.Image:
        """Ve khung quanh infill region (color tu cfg.border_color)."""
        result = template.copy()
        draw = ImageDraw.Draw(result)

        # Color from config (src/config.mjs.INFILL.borderColor). Defaults to
        # opaque red, but tests may override it.
        border_color = tuple(_cfg.border_color)

        # Bounding box of the infill region inside the template, in
        # template-local coordinates (0..TEMPLATE_SIZE).
        left = placement.infill_x
        top = placement.infill_y
        right = placement.infill_x + self.region.width
        bottom = placement.infill_y + self.region.height

        # Draw N concentric outlines (border_width determines thickness).
        for i in range(border_width):
            draw.rectangle(
                [left + i, top + i, right - 1 - i, bottom - 1 - i],
                outline=border_color,
                fill=None,
            )
        return result

    def get_validation_info(self) -> dict:
        """Debug info cho placement validation."""
        return {
            "region": str(self.region),
            "area": self.region.area,
            "max_area": MAX_INFILL_AREA,
            "valid_size": self.region.is_valid_size(),
            "has_left_gen": self._has_generated_context("left"),
            "has_right_gen": self._has_generated_context("right"),
            "has_top_gen": self._has_generated_context("top"),
            "has_bottom_gen": self._has_generated_context("bottom"),
            "overlapping_quadrants": self.region.overlapping_quadrants(),
            "last_validation_error": self._last_validation_error,
        }


# ============================================================================
# Convenience: validate quadrant selection
# ============================================================================


def validate_quadrant_selection(
    quadrants: list[tuple[int, int]],
    has_generation: HasGeneration,
    allow_expansion: bool = False,
) -> tuple[bool, str, Optional[TemplatePlacement]]:
    """Validate 1 quadrant selection va tra ve optimal placement.

    Dac biet xu ly full tile (2x2):
    - Neu 1 so quadrant da generated, reduce selection chi con quadrants missing.
    - Neu KHONG co generated neighbor ngoai, dat full tile tai (0, 0) cua template.

    Returns:
        (is_valid, message, placement)
    """
    if not quadrants:
        return False, "No quadrants selected", None

    # Check quadrants form a contiguous rectangle
    min_qx = min(q[0] for q in quadrants)
    max_qx = max(q[0] for q in quadrants)
    min_qy = min(q[1] for q in quadrants)
    max_qy = max(q[1] for q in quadrants)

    expected_count = (max_qx - min_qx + 1) * (max_qy - min_qy + 1)
    if len(quadrants) != expected_count:
        return False, "Quadrants must form a contiguous rectangle", None

    expected = {
        (qx, qy)
        for qx in range(min_qx, max_qx + 1)
        for qy in range(min_qy, max_qy + 1)
    }
    if set(quadrants) != expected:
        return False, "Quadrants must form a contiguous rectangle", None

    region = InfillRegion.from_quadrants(quadrants)

    if not region.is_valid_size():
        return (
            False,
            f"Selection too large: {region.area} pixels (max: {MAX_INFILL_AREA} or full tile)",
            None,
        )

    # Full tile (2x2) special case
    if region.is_full_tile():
        generated_quadrants = [
            q for q in quadrants if has_generation(q[0], q[1])
        ]
        non_generated = [
            q for q in quadrants if not has_generation(q[0], q[1])
        ]

        if len(generated_quadrants) == 4:
            return False, "All quadrants already have generations", None

        if len(generated_quadrants) > 0:
            # Mot so quadrant da generated - recurse voi phan con lai
            return validate_quadrant_selection(
                non_generated, has_generation, allow_expansion
            )

        # Full tile, chua co quadrant nao generated - check khong co generated neighbor
        has_any_gen_neighbor = False
        for qx, qy in quadrants:
            if qx == min_qx:  # Left edge
                if has_generation(qx - 1, qy):
                    has_any_gen_neighbor = True
                    break
            if qx == max_qx:  # Right edge
                if has_generation(qx + 1, qy):
                    has_any_gen_neighbor = True
                    break
            if qy == min_qy:  # Top edge
                if has_generation(qx, qy - 1):
                    has_any_gen_neighbor = True
                    break
            if qy == max_qy:  # Bottom edge
                if has_generation(qx, qy + 1):
                    has_any_gen_neighbor = True
                    break

        if has_any_gen_neighbor:
            return (
                False,
                "Full tile (2x2) selection cannot have generated neighbors (would create seams)",
                None,
            )

        # Valid full tile - place at origin
        placement = TemplatePlacement(
            infill_x=0,
            infill_y=0,
            world_offset_x=region.x,
            world_offset_y=region.y,
        )
        placement._infill_width = region.width
        placement._infill_height = region.height
        placement._primary_quadrants = list(quadrants)
        return True, "Valid selection (full tile)", placement

    # Partial region - su TemplateBuilder thong thuong
    builder = TemplateBuilder(region, has_generation)
    placement = builder.find_optimal_placement(allow_expansion=allow_expansion)

    if placement is None:
        info = builder.get_validation_info()
        if info["last_validation_error"]:
            return False, info["last_validation_error"], None
        if info["has_left_gen"]:
            return False, "Would create seam with generated pixels on left", None
        if info["has_right_gen"]:
            return False, "Would create seam with generated pixels on right", None
        if info["has_top_gen"]:
            return False, "Would create seam with generated pixels on top", None
        if info["has_bottom_gen"]:
            return False, "Would create seam with generated pixels on bottom", None
        return False, "No valid placement found", None

    if not placement._primary_quadrants:
        placement._primary_quadrants = list(quadrants)

    if placement.is_expanded:
        padding_str = ", ".join(
            f"({qx}, {qy})" for qx, qy in placement._padding_quadrants
        )
        return True, f"Valid selection (expanded to cover: {padding_str})", placement

    return True, "Valid selection", placement