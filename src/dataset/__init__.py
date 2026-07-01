"""
Dataset module for training data preparation.

NEW STRUCTURE (v2.0):
    dataset/
        images/
            image_001.jpg   (target image - pixel art)
            image_001.txt   (caption for target)
            image_002.jpg
            image_002.txt
            ...
        control/
            image_001.jpg   (control image - template with red border)
            image_002.jpg
            ...
        prompts/
            prompt.txt         (COMMON prompt - used by all)
        dataset_mapping.csv    (mapping: caption_path, control_path, prompt_path)
        dataset_metadata.json  (metadata)
"""

from .omni import (
    OmniMasker,
    apply_noise,
    apply_desaturation,
    apply_gamma_shift,
    apply_preprocessing,
    TEMPLATE_SIZE,
    QUADRANT_SIZE,
    DISTRIBUTION,
    RED_BORDER_COLOR,
    TYPE_FULL,
    TYPE_QUADRANT_TL,
    TYPE_QUADRANT_TR,
    TYPE_QUADRANT_BL,
    TYPE_QUADRANT_BR,
    QUADRANT_TYPES,
    HALF_TYPES,
    MIDDLE_TYPES,
    STRIP_TYPES,
    TYPE_RECT_INFILL,
)
from .prepare import DatasetPreparator, TrainingSample

__all__ = [
    'OmniMasker',
    'apply_noise',
    'apply_desaturation',
    'apply_gamma_shift',
    'apply_preprocessing',
    'TEMPLATE_SIZE',
    'QUADRANT_SIZE',
    'DISTRIBUTION',
    'RED_BORDER_COLOR',
    'TYPE_FULL',
    'TYPE_QUADRANT_TL',
    'TYPE_QUADRANT_TR',
    'TYPE_QUADRANT_BL',
    'TYPE_QUADRANT_BR',
    'QUADRANT_TYPES',
    'HALF_TYPES',
    'MIDDLE_TYPES',
    'STRIP_TYPES',
    'TYPE_RECT_INFILL',
    'DatasetPreparator',
    'TrainingSample',
]
