"""
Template builder for omni infill generation.

Theo logic của `src/tile/stitch/compose.mjs`, `src/tile/worker.mjs` và
`isometric-nyc/src/isometric_nyc/generation/infill_template.py`:

- 1 tile = 1024x1024 pixels = 200m x 200m world.
- cameraMoveStep = 0.5 → stride = 512px (100m) → 2 tile kề nhau chồng 50%.
- Mỗi tile render chứa 1 vùng world rộng 200m, tâm tại ((qx+0.5)*100m, (qy+0.5)*100m).
- Template = 1024x1024, gồm context (đã gen) + infill (render mới) + red border.

Pattern sử dụng:
- create_next_tile_template: tạo input cho tile world kế tiếp (gen 1 tile mới,
  dùng 512px overlap từ tile đã gen làm context).
- create_quadrant_template: tạo input cho 1 quadrant (512x512) trong 1 tile 1024x1024.
- create_full_template: tạo input cho tile chưa có neighbor (toàn bộ 1024x1024).
"""

import random
from typing import Literal, Tuple

from PIL import Image, ImageDraw

# Red border color for masked regions
BORDER_COLOR = (255, 0, 0)
BORDER_WIDTH = 2

# Region type constants (cho quadrant-based gen)
REGION_FULL = "full"
REGION_TL = "tl"
REGION_TR = "tr"
REGION_BL = "bl"
REGION_BR = "br"
REGION_LEFT = "left"
REGION_RIGHT = "right"
REGION_TOP = "top"
REGION_BOTTOM = "bottom"

RegionType = Literal["full", "tl", "tr", "bl", "br", "left", "right", "top", "bottom"]

# Template size constants
TEMPLATE_SIZE = 1024
QUADRANT_SIZE = 512
TILE_OVERLAP_PX = 512  # 50% overlap giữa 2 tile kề nhau


class OmniTemplateBuilder:
    """
    Tạo template cho omni infill generation với Qwen-Image-Edit.

    2 use case chính:
    1. Gen tile world kế tiếp (chồng 50% với tile đã gen)
       → create_next_tile_template(generated_tile, next_render)
    2. Gen quadrant trong 1 tile (chồng 50% với quadrant đã gen)
       → create_quadrant_template(generated_quadrant, render_quadrant, position)
    """

    def __init__(self, tile_size: int = TEMPLATE_SIZE, border_width: int = BORDER_WIDTH):
        self.tile_size = tile_size
        self.border_width = border_width

    # ─────────────────────────────────────────────────────────────────────
    # Public API: gen tile kế tiếp
    # ─────────────────────────────────────────────────────────────────────

    def create_next_tile_template(
        self,
        generated_tile: Image.Image,
        next_render: Image.Image,
        side: Literal["right", "left", "top", "bottom"] = "right",
    ) -> Image.Image:
        """
        Tạo template input để gen 1 tile world kế tiếp.

        Do cameraMoveStep=0.5, 2 tile kề nhau CHỒNG 50% (512px).
        Template 1024x1024 gồm:
        - Phần context: 512px bên "side" của generated_tile (đã gen, dùng làm anchor)
        - Phần infill: 512px bên "side" của next_render (phần MỚI, cần gen)

        Args:
            generated_tile: 1024x1024 pixel art đã gen (output trước đó của model)
            next_render: 1024x1024 ảnh render tại world position kế tiếp
            side: hướng gen tile kế tiếp
                - "right": gen tile bên phải (mặc định)
                - "left": gen tile bên trái
                - "top": gen tile phía trên
                - "bottom": gen tile phía dưới

        Returns:
            Template 1024x1024 RGB với red border bao quanh vùng infill.
        """
        gen = self._ensure_size(generated_tile)
        rnd = self._ensure_size(next_render)
        W, H = self.tile_size, self.tile_size
        half = W // 2  # 512

        template = Image.new("RGB", (W, H))

        if side == "right":
            # Context: nửa phải của generated (đã gen)
            # Infill: nửa phải của next_render (phần mới, không overlap)
            template.paste(gen.crop((half, 0, W, H)), (0, 0))
            template.paste(rnd.crop((half, 0, W, H)), (half, 0))
            infill_box = (half, 0, W, H)
        elif side == "left":
            # Context: nửa trái của generated
            # Infill: nửa trái của next_render
            template.paste(rnd.crop((0, 0, half, H)), (0, 0))
            template.paste(gen.crop((0, 0, half, H)), (half, 0))
            infill_box = (0, 0, half, H)
        elif side == "top":
            # Context: nửa trên của generated
            # Infill: nửa trên của next_render
            template.paste(rnd.crop((0, 0, W, half)), (0, half))
            template.paste(gen.crop((0, 0, W, half)), (0, 0))
            infill_box = (0, 0, W, half)
        elif side == "bottom":
            # Context: nửa dưới của generated
            # Infill: nửa dưới của next_render
            template.paste(gen.crop((0, half, W, H)), (0, 0))
            template.paste(rnd.crop((0, half, W, H)), (0, half))
            infill_box = (0, half, W, H)
        else:
            raise ValueError(f"Unknown side: {side}")

        return self._draw_red_border(template, infill_box)

    def create_full_template(self, render: Image.Image) -> Image.Image:
        """
        Tạo template full (toàn bộ 1024x1024) cho tile không có context.

        Khi tile chưa có neighbor đã gen, toàn bộ tile cần gen → full template.
        """
        rnd = self._ensure_size(render)
        template = rnd.copy().convert("RGBA")
        return self._draw_red_border(
            template.convert("RGB"), (0, 0, self.tile_size, self.tile_size)
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API: gen quadrant trong tile (dùng cho training augmentation)
    # ─────────────────────────────────────────────────────────────────────

    def get_region_box(self, region_type: RegionType) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2) cho 1 region type."""
        half = self.tile_size // 2
        region_map = {
            REGION_FULL: (0, 0, self.tile_size, self.tile_size),
            REGION_TL: (0, 0, half, half),
            REGION_TR: (half, 0, self.tile_size, half),
            REGION_BL: (0, half, half, self.tile_size),
            REGION_BR: (half, half, self.tile_size, self.tile_size),
            REGION_LEFT: (0, 0, half, self.tile_size),
            REGION_RIGHT: (half, 0, self.tile_size, self.tile_size),
            REGION_TOP: (0, 0, self.tile_size, half),
            REGION_BOTTOM: (0, half, self.tile_size, self.tile_size),
        }
        if region_type not in region_map:
            raise ValueError(f"Unknown region type: {region_type}")
        return region_map[region_type]

    def create_quadrant_template(
        self,
        generated_tile: Image.Image,
        render: Image.Image,
        region_type: RegionType,
    ) -> Image.Image:
        """
        Tạo template cho training augmentation (giống src/dataset/omni.py).

        Template = generated_tile (full) với vùng region_type thay bằng render + red border.

        Args:
            generated_tile: 1024x1024 pixel art đã gen (full)
            render: 1024x1024 ảnh render
            region_type: vùng nào của tile sẽ thay bằng render (full/tl/tr/bl/br/...)
        """
        gen = self._ensure_size(generated_tile)
        rnd = self._ensure_size(render)

        template = gen.copy().convert("RGBA")
        x1, y1, x2, y2 = self.get_region_box(region_type)
        render_crop = rnd.convert("RGBA").crop((x1, y1, x2, y2))
        template.paste(render_crop, (x1, y1))
        return self._draw_red_border(template.convert("RGB"), (x1, y1, x2, y2))

    def create_random_template(
        self,
        generated_tile: Image.Image,
        render: Image.Image,
        rng: random.Random | None = None,
    ) -> Tuple[Image.Image, str]:
        """
        Random chọn region_type (full / quadrant) và build template.
        Trả về (template, region_type_used).
        """
        if rng is None:
            rng = random.Random()

        # Theo distribution của src/dataset/omni.py
        region_type = rng.choice(
            [REGION_FULL, REGION_TL, REGION_TR, REGION_BL, REGION_BR,
             REGION_LEFT, REGION_RIGHT, REGION_TOP, REGION_BOTTOM]
        )

        if region_type == REGION_FULL:
            return self.create_full_template(render), region_type
        else:
            return self.create_quadrant_template(generated_tile, render, region_type), region_type

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_size(self, img: Image.Image) -> Image.Image:
        """Resize ảnh về (tile_size, tile_size) nếu cần."""
        if img.size != (self.tile_size, self.tile_size):
            return img.resize(
                (self.tile_size, self.tile_size), Image.Resampling.LANCZOS
            )
        return img

    def _draw_red_border(
        self, img: Image.Image, box: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Vẽ red border quanh vùng box=(x1, y1, x2, y2)."""
        result = img.copy()
        draw = ImageDraw.Draw(result)
        x1, y1, x2, y2 = box
        for i in range(self.border_width):
            draw.rectangle(
                [x1 + i, y1 + i, x2 - 1 - i, y2 - 1 - i],
                outline=BORDER_COLOR,
            )
        return result
