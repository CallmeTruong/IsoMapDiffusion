"""
Perfect Corruption - Synthetic data generation from render + pixel art pairs.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random


class PerfectCorruption:
    DEFAULT_PARAMS = {
        'desaturation_factor': 0.5,
        'noise_intensity': 100,
        'gamma_value': 1.2,
        'blur_radius': 0.5,
        'brightness_factor': 0.9,
        'contrast_factor': 0.8,
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def corrupt(self, image: Image.Image, params: dict = None) -> Image.Image:
        params = params or self.DEFAULT_PARAMS
        img = image.convert('RGB')

        if params.get('desaturation_factor', 1.0) < 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(params['desaturation_factor'])

        if params.get('brightness_factor', 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(params['brightness_factor'])

        if params.get('contrast_factor', 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(params['contrast_factor'])

        if params.get('gamma_value', 1.0) != 1.0:
            img = self._apply_gamma(img, params['gamma_value'])

        if params.get('blur_radius', 0) > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_radius']))

        if params.get('noise_intensity', 0) > 0:
            img = self._add_noise(img, params['noise_intensity'])

        if params.get('color_jitter', True):
            img = self._color_jitter(img)

        return img

    def corrupt_with_variance(self, image: Image.Image, variance: float = 0.2) -> Image.Image:
        params = self.DEFAULT_PARAMS.copy()
        params['desaturation_factor'] = max(0.3, params['desaturation_factor'] + self.rng.uniform(-variance, variance))
        params['noise_intensity'] = max(0, params['noise_intensity'] + self.rng.integers(-50, 50))
        params['gamma_value'] = max(0.8, min(1.5, params['gamma_value'] + self.rng.uniform(-variance*0.3, variance*0.3)))
        params['brightness_factor'] = max(0.7, min(1.1, params['brightness_factor'] + self.rng.uniform(-variance*0.2, variance*0.2)))
        params['contrast_factor'] = max(0.6, min(1.0, params['contrast_factor'] + self.rng.uniform(-variance*0.2, variance*0.2)))
        return self.corrupt(image, params)

    def _apply_gamma(self, image: Image.Image, gamma: float) -> Image.Image:
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _add_noise(self, image: Image.Image, intensity: int) -> Image.Image:
        arr = np.array(image).astype(np.float32)
        noise = self.rng.normal(0, intensity, arr.shape)
        arr = (arr + noise).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        arr = np.array(image).astype(np.float32)
        bright_shift = self.rng.uniform(-10, 10)
        arr[:, :, 0] = (arr[:, :, 0] + bright_shift).clip(0, 255)
        arr[:, :, 1] = (arr[:, :, 1] + bright_shift * 0.8).clip(0, 255)
        arr[:, :, 2] = (arr[:, :, 2] + bright_shift * 0.6).clip(0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def quick_corrupt(image: Image.Image) -> Image.Image:
        return PerfectCorruption().corrupt(image)
