"""
Omni masking for training data augmentation.
"""

import numpy as np
from PIL import Image, ImageDraw
import random
from typing import Tuple


TEMPLATE_SIZE = 1024
QUADRANT_SIZE = 512
RED_BORDER_COLOR = (255, 0, 0, 255)
BORDER_WIDTH = 2

TYPE_FULL = "full"
TYPE_QUADRANT_TL = "quadrant_tl"
TYPE_QUADRANT_TR = "quadrant_tr"
TYPE_QUADRANT_BL = "quadrant_bl"
TYPE_QUADRANT_BR = "quadrant_br"
TYPE_HALF_LEFT = "half_left"
TYPE_HALF_RIGHT = "half_right"
TYPE_HALF_TOP = "half_top"
TYPE_HALF_BOTTOM = "half_bottom"
TYPE_MIDDLE_VERTICAL = "middle_vertical"
TYPE_MIDDLE_HORIZONTAL = "middle_horizontal"
TYPE_STRIP_VERTICAL = "strip_vertical"
TYPE_STRIP_HORIZONTAL = "strip_horizontal"
TYPE_RECT_INFILL = "rect_infill"

FULL_TYPES = [TYPE_FULL]
QUADRANT_TYPES = [TYPE_QUADRANT_TL, TYPE_QUADRANT_TR, TYPE_QUADRANT_BL, TYPE_QUADRANT_BR]
HALF_TYPES = [TYPE_HALF_LEFT, TYPE_HALF_RIGHT, TYPE_HALF_TOP, TYPE_HALF_BOTTOM]
MIDDLE_TYPES = [TYPE_MIDDLE_VERTICAL, TYPE_MIDDLE_HORIZONTAL]
STRIP_TYPES = [TYPE_STRIP_VERTICAL, TYPE_STRIP_HORIZONTAL]
INFILL_TYPES = [TYPE_RECT_INFILL]
ALL_TYPES = FULL_TYPES + QUADRANT_TYPES + HALF_TYPES + MIDDLE_TYPES + STRIP_TYPES + INFILL_TYPES

DISTRIBUTION = {
    "full": 0.20, "quadrant": 0.18, "half": 0.17,
    "middle": 0.15, "strip": 0.10, "infill": 0.20,
}


def apply_noise(image: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    noise_multiplier = np.random.normal(loc=1.0, scale=intensity * 0.15, size=img_array.shape)
    noisy_array = np.clip(img_array * noise_multiplier, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def apply_desaturation(image: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    gray_rgb = np.stack([gray, gray, gray], axis=2)
    result = (1 - intensity) * img_array + intensity * gray_rgb
    return Image.fromarray(result.astype(np.uint8))


def apply_gamma_shift(image: Image.Image, intensity: float) -> Image.Image:
    if intensity <= 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0
    gamma = 1.0 + (intensity * 0.8)
    crushed = np.power(img_array, gamma)
    if intensity > 0.5:
        crushed = crushed * (1.0 - (intensity - 0.5) * 0.6)
    return Image.fromarray(np.clip(crushed * 255, 0, 255).astype(np.uint8))


def apply_preprocessing(img: Image.Image, desaturation: float = 0.0, noise: float = 0.0, gamma_shift: float = 0.0) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if desaturation > 0:
        img = apply_desaturation(img, desaturation)
    if noise > 0:
        img = apply_noise(img, noise)
    if gamma_shift > 0:
        img = apply_gamma_shift(img, gamma_shift)
    return img


class OmniMasker:
    MASK_TYPES = ['full', 'quadrant', 'half', 'middle', 'strip', 'rect']

    def __init__(self, seed: int = 42, border_width: int = BORDER_WIDTH):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.border_width = border_width

    def create_mask(self, size: Tuple[int, int], mask_type: str = 'full') -> Image.Image:
        w, h = size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if mask_type == 'full':
            draw.rectangle([0, 0, w-1, h-1], fill=255)
        elif mask_type == 'quadrant':
            qx, qy = self.rng.integers(0, 2, 2)
            x1 = 0 if qx == 0 else w // 2
            y1 = 0 if qy == 0 else h // 2
            x2 = w // 2 if qx == 0 else w
            y2 = h // 2 if qy == 0 else h
            draw.rectangle([x1, y1, x2-1, y2-1], fill=255)
        elif mask_type == 'half':
            if self.rng.random() < 0.5:
                draw.rectangle([0, 0, w-1, h//2-1], fill=255)
            else:
                draw.rectangle([0, 0, w//2-1, h-1], fill=255)
        elif mask_type == 'middle':
            if self.rng.random() < 0.5:
                sw = w // 2
                draw.rectangle([(w-sw)//2, 0, (w+sw)//2-1, h-1], fill=255)
            else:
                sh = h // 2
                draw.rectangle([0, (h-sh)//2, w-1, (h+sh)//2-1], fill=255)
        elif mask_type == 'strip':
            if self.rng.random() < 0.5:
                sw = int(w * self.rng.uniform(0.25, 0.60))
                x = self.rng.integers(0, max(1, w - sw))
                draw.rectangle([x, 0, x+sw-1, h-1], fill=255)
            else:
                sh = int(h * self.rng.uniform(0.25, 0.60))
                y = self.rng.integers(0, max(1, h - sh))
                draw.rectangle([0, y, w-1, y+sh-1], fill=255)
        elif mask_type == 'rect':
            total_area = w * h
            target_area = self.rng.integers(int(total_area*0.25), int(total_area*0.60)+1)
            aspect = self.rng.uniform(0.5, 2.0)
            rh = int((target_area / aspect) ** 0.5)
            rw = int(rh * aspect)
            rw = max(w//4, min(rw, w-32))
            rh = max(h//4, min(rh, h-32))
            x = self.rng.integers(16, max(17, w-rw-16))
            y = self.rng.integers(16, max(17, h-rh-16))
            draw.rectangle([x, y, x+rw-1, y+rh-1], fill=255)

        return mask

    def select_template_type(self) -> str:
        categories = list(DISTRIBUTION.keys())
        probs = list(DISTRIBUTION.values())
        category = random.choices(categories, weights=probs, k=1)[0]

        type_map = {
            'full': TYPE_FULL, 'quadrant': random.choice(QUADRANT_TYPES),
            'half': random.choice(HALF_TYPES), 'middle': random.choice(MIDDLE_TYPES),
            'strip': random.choice(STRIP_TYPES), 'infill': TYPE_RECT_INFILL,
        }
        return type_map.get(category, TYPE_FULL)

    def get_input_region(self, width: int, height: int, template_type: str) -> Tuple[int, int, int, int]:
        hw, hh = width // 2, height // 2

        if template_type == TYPE_FULL:
            return (0, 0, width, height)
        elif template_type == TYPE_QUADRANT_TL:
            return (0, 0, hw, hh)
        elif template_type == TYPE_QUADRANT_TR:
            return (hw, 0, hw, hh)
        elif template_type == TYPE_QUADRANT_BL:
            return (0, hh, hw, hh)
        elif template_type == TYPE_QUADRANT_BR:
            return (hw, hh, hw, hh)
        elif template_type == TYPE_HALF_LEFT:
            return (0, 0, hw, height)
        elif template_type == TYPE_HALF_RIGHT:
            return (hw, 0, hw, height)
        elif template_type == TYPE_HALF_TOP:
            return (0, 0, width, hh)
        elif template_type == TYPE_HALF_BOTTOM:
            return (0, hh, width, hh)
        elif template_type == TYPE_MIDDLE_VERTICAL:
            sw = width // 2
            return ((width-sw)//2, 0, sw, height)
        elif template_type == TYPE_MIDDLE_HORIZONTAL:
            sh = height // 2
            return (0, (height-sh)//2, width, sh)
        elif template_type == TYPE_STRIP_VERTICAL:
            sw = int(width * self.rng.uniform(0.25, 0.60))
            x = self.rng.integers(0, max(1, width-sw))
            return (x, 0, sw, height)
        elif template_type == TYPE_STRIP_HORIZONTAL:
            sh = int(height * self.rng.uniform(0.25, 0.60))
            y = self.rng.integers(0, max(1, height-sh))
            return (0, y, width, sh)
        elif template_type == TYPE_RECT_INFILL:
            total_area = width * height
            target_area = self.rng.integers(int(total_area*0.25), int(total_area*0.60)+1)
            aspect = self.rng.uniform(0.5, 2.0)
            rh = int((target_area / aspect) ** 0.5)
            rw = int(rh * aspect)
            rw = max(width//4, min(rw, width-32))
            rh = max(height//4, min(rh, height-32))
            x = self.rng.integers(16, max(17, width-rw-16))
            y = self.rng.integers(16, max(17, height-rh-16))
            return (x, y, rw, rh)
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def apply_perfect_corruption(self, image: Image.Image, desaturation: float = 0.5, noise: float = 1.0, gamma_shift: float = 1.0) -> Image.Image:
        return apply_preprocessing(image, desaturation, noise, gamma_shift)

    def create_infill_template(self, pixel_art: Image.Image, render: Image.Image, mask_region: Tuple[int, int, int, int]) -> Image.Image:
        if pixel_art.size != (TEMPLATE_SIZE, TEMPLATE_SIZE):
            pixel_art = pixel_art.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS)
            render = render.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.LANCZOS)

        x, y, mw, mh = mask_region
        template = pixel_art.convert('RGBA')
        render_crop = render.convert('RGBA').crop((x, y, x+mw, y+mh))
        template.paste(render_crop, (x, y))
        return self._draw_red_border(template, x, y, mw, mh)

    def _draw_red_border(self, template: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
        result = template.copy()
        draw = ImageDraw.Draw(result)
        right, bottom = x + w, y + h
        for i in range(self.border_width):
            draw.rectangle([x+i, y+i, right-1-i, bottom-1-i], outline=RED_BORDER_COLOR)
        return result

    def create_omni_template(self, pixel_art: Image.Image, render: Image.Image, template_type: str, corruption_params: dict = None) -> Image.Image:
        region = self.get_input_region(TEMPLATE_SIZE, TEMPLATE_SIZE, template_type)
        if corruption_params is None:
            corruption_params = {'desaturation': 0.5, 'noise': 1.0, 'gamma_shift': 1.0}
        corrupted_render = self.apply_perfect_corruption(
            render,
            desaturation=corruption_params.get('desaturation', 0.5),
            noise=corruption_params.get('noise', 1.0),
            gamma_shift=corruption_params.get('gamma_shift', 1.0),
        )
        return self.create_infill_template(pixel_art, corrupted_render, region)
