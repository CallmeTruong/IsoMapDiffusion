"""
Template builder cho omni infill generation.

PORT TU: isometric-nyc/src/isometric_nyc/generation/infill_template.py

Logic ghep tile khi gen:
- Template 1024x1024 pixel
- Moi tile chia thanh 2x2 quadrants (512x512 moi quadrant)
- Infill region la mot rectangle (1 quadrant hoac nhieu quadrant ke nhau)
- Vi tri infill trong template duoc toi uu (push ve phia doi dien generated
  neighbors) de toi da context, tranh seam.
- Red border (255,0,0,255) ve quanh infill rect, border_width=2.
- Context quadrants paste generation (pixel art da gen)
- Infill quadrants paste render (3D render)

Coords:
- (qx, qy) = world quadrant index
- quadrant (qx, qy) = o (qx*512, qy*512) den (qx*512+512, qy*512+512) trong world
- tile (qx_tile, qy_tile) = 2x2 quadrants bat dau tu (qx_tile*2, qy_tile*2)
  (1 tile = 4 quadrants, moi tile 1024x1024, moi quadrant 512x512)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image, ImageDraw

from inference.config import get_inference_config

# Lay config tu src/config.mjs (khong hardcode)
_cfg = get_inference_config()

# Constants tu InferenceConfig (mirror src/config.mjs + isometric-nyc)
TEMPLATE_SIZE: int = _cfg.tile_size_px            # 1024
QUADRANT_SIZE: int = _cfg.quadrant_size_px        # 512
MAX_INFILL_AREA: int = _cfg.max_infill_area       # 524288 = 50% cua 1024^2
BORDER_COLOR: tuple = _cfg.border_color           # (255, 0, 0, 255)
BORDER_WIDTH: int = _cfg.border_width             # 2

RegionType = str  # "full" | "tl" | "tr" | "bl" | "br" | "left" | "right" | "top" | "bottom"


@dataclass
class InfillRegion:
    """
    Vung rect can dien (infill) trong world pixel coords.

    Coords (x, y) la world pixel (khong phai quadrant index).
    Moi quadrant = 512x512 = QUADRANT_SIZE.
    """
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_quadrant(cls, qx: int, qy: int) -> "InfillRegion":
        """Mot quadrant (512x512) tai (qx, qy)."""
        return cls(
            x=qx * QUADRANT_SIZE,
            y=qy * QUADRANT_SIZE,
            width=QUADRANT_SIZE,
            height=QUADRANT_SIZE,
        )

    @classmethod
    def from_quadrants(cls, quadrants: list[tuple[int, int]]) -> "InfillRegion":
        """Bounding rect cua nhieu quadrant ke nhau (phai contiguous rectangle)."""
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

    @classmethod
    def full_tile(cls, tile_qx: int, tile_qy: int) -> "InfillRegion":
        """Full tile 1024x102x tai (tile_qx, tile_qy) (4 quadrants)."""
        return cls(
            x=tile_qx * TEMPLATE_SIZE,
            y=tile_qy * TEMPLATE_SIZE,
            width=TEMPLATE_SIZE,
            height=TEMPLATE_SIZE,
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

    def is_full_tile(self) -> bool:
        """True neu region = 1024x1024 (full tile)."""
        return self.width == TEMPLATE_SIZE and self.height == TEMPLATE_SIZE

    def is_valid_size(self) -> bool:
        """True neu area <= 50% template HOAC full tile (100%)."""
        return self.area <= MAX_INFILL_AREA or self.is_full_tile()

    def overlapping_quadrants(self) -> list[tuple[int, int]]:
        """List (qx, qy) overlap voi region (theo quadrant grid)."""
        quadrants = []
        start_qx = self.x // QUADRANT_SIZE
        end_qx = (self.right - 1) // QUADRANT_SIZE
        start_qy = self.y // QUADRANT_SIZE
        end_qy = (self.bottom - 1) // QUADRANT_SIZE
        for qx in range(start_qx, end_qx + 1):
            for qy in range(start_qy, end_qy + 1):
                quadrants.append((qx, qy))
        return quadrants

    def __str__(self) -> str:
        return f"InfillRegion(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


@dataclass
class TemplatePlacement:
    """Vi tri dat infill region trong template 1024x1024."""
    infill_x: int           # toa do x cua infill trong template (0..1024-w)
    infill_y: int           # toa do y cua infill trong template
    world_offset_x: int     # world x cua goc tren-trai template
    world_offset_y: int     # world y cua goc tren-trai template

    _infill_width: int = field(default=0, init=False)
    _infill_height: int = field(default=0, init=False)
    _primary_quadrants: list = field(default_factory=list, init=False)
    _padding_quadrants: list = field(default_factory=list, init=False)
    _expanded_region: Optional["InfillRegion"] = field(default=None, init=False)

    @property
    def infill_right(self) -> int:
        return self.infill_x + self._infill_width

    @property
    def infill_bottom(self) -> int:
        return self.infill_y + self._infill_height

    @property
    def primary_quadrants(self) -> list:
        return self._primary_quadrants

    @property
    def padding_quadrants(self) -> list:
        return self._padding_quadrants

    @property
    def is_expanded(self) -> bool:
        return len(self._padding_quadrants) > 0

    def __str__(self) -> str:
        return (
            f"TemplatePlacement(infill=({self.infill_x},{self.infill_y}) "
            f"{self._infill_width}x{self._infill_height}, "
            f"world_offset=({self.world_offset_x},{self.world_offset_y}))"
        )


class TemplateBuilder:
    """
    Build 1024x1024 template image cho infill generation.

    Logic ghep:
    1. Tinh placement (vi tri infill trong template) dua vao generated neighbors:
       - Neu co generated ben trai: push infill sang phai
       - Neu co generated ben phai: push infill sang trai
       - Neu ca 2: can giua
       - Tuong tu cho top/bottom
    2. Validate: neu infill cham mep template VA co generated ben ngoai mep do -> INVALID (seam)
    3. Build template:
       - Tao RGBA 1024x1024
       - Voi moi quadrant overlap voi template:
         + Neu quadrant trong infill: paste render (co the corrupt)
         + Else: paste generation (context)
       - Ve red border quanh infill rect
    """

    def __init__(
        self,
        infill_region: InfillRegion,
        has_generation: Callable[[int, int], bool],
        get_render: Optional[Callable[[int, int], Optional[Image.Image]]] = None,
        get_generation: Optional[Callable[[int, int], Optional[Image.Image]]] = None,
    ):
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

    def _has_generated_context(self, side: str) -> bool:
        """
        Kiem tra co quadrant generated ke canh voi infill region o phia 'side'.
        side in {left, right, top, bottom}.
        """
        if side == "left":
            check_x = self.region.x - 1
            qx = check_x // QUADRANT_SIZE
            start_qy = self.region.y // QUADRANT_SIZE
            end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
            return any(
                self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1)
            )
        elif side == "right":
            check_x = self.region.right
            qx = check_x // QUADRANT_SIZE
            start_qy = self.region.y // QUADRANT_SIZE
            end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
            return any(
                self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1)
            )
        elif side == "top":
            check_y = self.region.y - 1
            qy = check_y // QUADRANT_SIZE
            start_qx = self.region.x // QUADRANT_SIZE
            end_qx = (self.region.right - 1) // QUADRANT_SIZE
            return any(
                self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1)
            )
        elif side == "bottom":
            check_y = self.region.bottom
            qy = check_y // QUADRANT_SIZE
            start_qx = self.region.x // QUADRANT_SIZE
            end_qx = (self.region.right - 1) // QUADRANT_SIZE
            return any(
                self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1)
            )
        return False

    def _try_placement_with_context_preferences(
        self,
        include_left: bool = True,
        include_right: bool = True,
        include_top: bool = True,
        include_bottom: bool = True,
    ) -> Optional[TemplatePlacement]:
        """
        Tim placement hop le dua vao context preferences.
        Push infill ve phia doi dien generated de max context, tranh seam.
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
            infill_x = margin_x  # push sang phai
        elif has_right_gen:
            infill_x = 0  # push sang trai
        else:
            # No horizontal context - kiem tra that su
            actual_left_gen = self._has_generated_context("left")
            actual_right_gen = self._has_generated_context("right")
            if actual_right_gen and not actual_left_gen:
                infill_x = 0
            elif actual_left_gen and not actual_right_gen:
                infill_x = margin_x
            else:
                infill_x = 0

        # Vertical positioning
        if has_top_gen and has_bottom_gen:
            infill_y = margin_y // 2
        elif has_top_gen:
            infill_y = margin_y  # push xuong
        elif has_bottom_gen:
            infill_y = 0  # push len
        else:
            actual_top_gen = self._has_generated_context("top")
            actual_bottom_gen = self._has_generated_context("bottom")
            if actual_bottom_gen and not actual_top_gen:
                infill_y = 0
            elif actual_top_gen and not actual_bottom_gen:
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

        # Validate seams
        is_valid, error = self._validate_placement_seams(placement)
        if not is_valid:
            self._last_validation_error = error
            return None

        return placement

    def _validate_placement_seams(
        self, placement: TemplatePlacement
    ) -> tuple[bool, str]:
        """
        Validate: neu infill cham mep template VA co generated ben ngoai -> INVALID.
        """
        if placement.infill_x == 0:
            if self._has_generated_context("left"):
                return False, "Would create seam with generated pixels on left"
        if placement.infill_x + self.region.width == TEMPLATE_SIZE:
            if self._has_generated_context("right"):
                return False, "Would create seam with generated pixels on right"
        if placement.infill_y == 0:
            if self._has_generated_context("top"):
                return False, "Would create seam with generated pixels on top"
        if placement.infill_y + self.region.height == TEMPLATE_SIZE:
            if self._has_generated_context("bottom"):
                return False, "Would create seam with generated pixels on bottom"
        return True, ""

    def _find_missing_context_quadrants(
        self, placement: TemplatePlacement
    ) -> list[tuple[int, int]]:
        """Context quadrants (in template, NOT in infill) ma khong co generation."""
        missing = []
        template_world_left = placement.world_offset_x
        template_world_right = placement.world_offset_x + TEMPLATE_SIZE
        template_world_top = placement.world_offset_y
        template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

        start_qx = template_world_left // QUADRANT_SIZE
        end_qx = (template_world_right - 1) // QUADRANT_SIZE
        start_qy = template_world_top // QUADRANT_SIZE
        end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

        infill_quadrants = set(self.region.overlapping_quadrants())

        for qx in range(start_qx, end_qx + 1):
            for qy in range(start_qy, end_qy + 1):
                if (qx, qy) not in infill_quadrants:
                    if not self.has_generation(qx, qy):
                        missing.append((qx, qy))
        return missing

    def find_optimal_placement(
        self, allow_expansion: bool = False
    ) -> Optional[TemplatePlacement]:
        """Tim vi tri tot nhat cho infill region trong template."""
        placement = self._try_placement_with_context_preferences(
            include_left=True,
            include_right=True,
            include_top=True,
            include_bottom=True,
        )
        if placement is not None:
            missing = self._find_missing_context_quadrants(placement)
            if not missing:
                return placement
            # TODO: alternative placements / expansion
            missing_str = ", ".join(f"({qx}, {qy})" for qx, qy in missing)
            self._last_validation_error = (
                f"Context quadrants missing generations: {missing_str}"
            )
            return None
        return None

    def build(
        self,
        border_width: int = BORDER_WIDTH,
        allow_expansion: bool = False,
    ) -> Optional[tuple[Image.Image, TemplatePlacement]]:
        """
        Build 1024x1024 template image.

        Returns:
            (template_image, placement) hoac None neu khong co placement hop le.
        """
        if self.get_render is None or self.get_generation is None:
            raise ValueError("get_render and get_generation must be provided to build")

        placement = self.find_optimal_placement(allow_expansion=allow_expansion)
        if placement is None:
            return None

        # Effective region (expanded if applicable)
        effective_region = (
            placement._expanded_region
            if placement._expanded_region is not None
            else self.region
        )

        # Tao RGBA 1024x1024 transparent
        template = Image.new("RGBA", (TEMPLATE_SIZE, TEMPLATE_SIZE), (0, 0, 0, 0))

        # Determine quadrants trong template
        template_world_left = placement.world_offset_x
        template_world_right = placement.world_offset_x + TEMPLATE_SIZE
        template_world_top = placement.world_offset_y
        template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

        start_qx = template_world_left // QUADRANT_SIZE
        end_qx = (template_world_right - 1) // QUADRANT_SIZE
        start_qy = template_world_top // QUADRANT_SIZE
        end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

        infill_quadrants = set(effective_region.overlapping_quadrants())

        # Fill quadrants
        for qx in range(start_qx, end_qx + 1):
            for qy in range(start_qy, end_qy + 1):
                quad_world_x = qx * QUADRANT_SIZE
                quad_world_y = qy * QUADRANT_SIZE
                template_x = quad_world_x - template_world_left
                template_y = quad_world_y - template_world_top

                if (qx, qy) in infill_quadrants:
                    quad_img = self.get_render(qx, qy)
                else:
                    quad_img = self.get_generation(qx, qy)

                if quad_img is None:
                    continue

                # Resize if needed
                if quad_img.size != (QUADRANT_SIZE, QUADRANT_SIZE):
                    quad_img = quad_img.resize(
                        (QUADRANT_SIZE, QUADRANT_SIZE), Image.Resampling.LANCZOS
                    )
                if quad_img.mode != "RGBA":
                    quad_img = quad_img.convert("RGBA")

                # Crop if quadrant extends outside template
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

        # Ve red border quanh infill rect
        template = self._draw_border(template, placement, border_width)

        return template.convert("RGB"), placement

    def _draw_border(
        self,
        template: Image.Image,
        placement: TemplatePlacement,
        border_width: int,
    ) -> Image.Image:
        """Ve red border quanh infill region."""
        result = template.copy()
        draw = ImageDraw.Draw(result)
        left = placement.infill_x
        top = placement.infill_y
        right = placement.infill_x + self.region.width
        bottom = placement.infill_y + self.region.height
        for i in range(border_width):
            draw.rectangle(
                [left + i, top + i, right - 1 - i, bottom - 1 - i],
                outline=BORDER_COLOR,
                fill=None,
            )
        return result

    def get_validation_info(self) -> dict:
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


def validate_quadrant_selection(
    quadrants: list[tuple[int, int]],
    has_generation: Callable[[int, int], bool],
    allow_expansion: bool = False,
) -> tuple[bool, str, Optional[TemplatePlacement]]:
    """
    Validate quadrant selection va tim optimal placement.

    Special handling cho full tiles (2x2):
    - Neu 1 so quadrants da co generation, giam xuong con lai
    - Generated quadrants tro thanh context cho missing ones
    """
    if not quadrants:
        return False, "No quadrants selected", None

    # Check quadrants form rectangle
    min_qx = min(q[0] for q in quadrants)
    max_qx = max(q[0] for q in quadrants)
    min_qy = min(q[1] for q in quadrants)
    max_qy = max(q[1] for q in quadrants)

    expected_count = (max_qx - min_qx + 1) * (max_qy - min_qy + 1)
    if len(quadrants) != expected_count:
        return False, "Quadrants must form a contiguous rectangle", None

    expected = set()
    for qx in range(min_qx, max_qx + 1):
        for qy in range(min_qy, max_qy + 1):
            expected.add((qx, qy))

    if set(quadrants) != expected:
        return False, "Quadrants must form a contiguous rectangle", None

    region = InfillRegion.from_quadrants(quadrants)

    if not region.is_valid_size():
        return (
            False,
            f"Selection too large: {region.area} pixels (max: {MAX_INFILL_AREA} or full tile)",
            None,
        )

    # Full tile special handling
    if region.is_full_tile():
        generated_quadrants = [q for q in quadrants if has_generation(q[0], q[1])]
        non_generated_quadrants = [
            q for q in quadrants if not has_generation(q[0], q[1])
        ]

        if len(generated_quadrants) == 4:
            return False, "All quadrants already have generations", None

        if len(generated_quadrants) > 0:
            return validate_quadrant_selection(
                non_generated_quadrants, has_generation, allow_expansion
            )

        # Full tile with no internal gens - check neighbors
        has_any_gen_neighbor = False
        for qx, qy in quadrants:
            if qx == min_qx and has_generation(qx - 1, qy):
                has_any_gen_neighbor = True
                break
            if qx == max_qx and has_generation(qx + 1, qy):
                has_any_gen_neighbor = True
                break
            if qy == min_qy and has_generation(qx, qy - 1):
                has_any_gen_neighbor = True
                break
            if qy == max_qy and has_generation(qx, qy + 1):
                has_any_gen_neighbor = True
                break

        if has_any_gen_neighbor:
            return (
                False,
                "Full tile (2x2) selection cannot have generated neighbors (would create seams)",
                None,
            )

        # Valid - place flush
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

    # Partial region
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


# Backward-compat class alias (ten cu: OmniTemplateBuilder)
class OmniTemplateBuilder(TemplateBuilder):
    """
    Alias cho ten cu. Moi code cu dung `OmniTemplateBuilder(...)` se van chay.
    Logic ghep moi (TemplateBuilder) tu isometric-nyc thay the logic cu.
    """

    def __init__(
        self,
        tile_size: int = TEMPLATE_SIZE,
        border_width: int = BORDER_WIDTH,
    ):
        # NOTE: Constructor signature cu nhan (tile_size, border_width).
        # Su dung backward-compat wrapper de khong break code cu.
        # Tuy nhien, TemplateBuilder that su can `region` + `has_generation`
        # nen caller can su dung `TemplateBuilder` truc tiep.
        # Constructor nay chi de instantiate config.
        self.tile_size = tile_size
        self.border_width = border_width

    # Backward-compat shim: build full template tu render (khi khong co neighbor)
    def create_full_template(self, render: Image.Image) -> Image.Image:
        """Backward-compat: tra ve render voi red border toan tile."""
        template = render.copy().convert("RGBA")
        # Ve red border full
        draw = ImageDraw.Draw(template)
        for i in range(self.border_width):
            draw.rectangle(
                [i, i, self.tile_size - 1 - i, self.tile_size - 1 - i],
                outline=BORDER_COLOR,
                fill=None,
            )
        return template.convert("RGB")

    # Backward-compat shim: build next tile template (cu, KHONG dung cho logic moi)
    def create_next_tile_template(
        self,
        generated_tile: Image.Image,
        next_render: Image.Image,
        side: str = "right",
    ) -> Image.Image:
        """
        BACKWARD-COMPAT ONLY.
        Logic cu sai huong - KHONG nen dung.
        Moi code moi phai dung `TemplateBuilder` voi `has_generation`/`get_generation`/
        `get_render` callables de co logic ghep dung.
        """
        import warnings
        warnings.warn(
            "create_next_tile_template is DEPRECATED. Use TemplateBuilder with "
            "has_generation/get_render/get_generation callables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        gen = generated_tile.convert("RGBA")
        rnd = next_render.convert("RGBA")
        # Old (incorrect) logic - kept for backward compat
        W, H = self.tile_size, self.tile_size
        half = W // 2
        template = Image.new("RGBA", (W, H))
        if side == "right":
            template.paste(gen.crop((half, 0, W, H)), (0, 0))
            template.paste(rnd.crop((half, 0, W, H)), (half, 0))
        elif side == "left":
            template.paste(rnd.crop((0, 0, half, H)), (0, 0))
            template.paste(gen.crop((0, 0, half, H)), (half, 0))
        elif side == "top":
            template.paste(rnd.crop((0, 0, W, half)), (0, half))
            template.paste(gen.crop((0, 0, W, half)), (0, 0))
        elif side == "bottom":
            template.paste(gen.crop((0, half, W, H)), (0, 0))
            template.paste(rnd.crop((0, half, W, H)), (0, half))
        # Red border toan tile
        draw = ImageDraw.Draw(template)
        for i in range(self.border_width):
            draw.rectangle(
                [i, i, W - 1 - i, H - 1 - i],
                outline=BORDER_COLOR,
                fill=None,
            )
        return template.convert("RGB")
