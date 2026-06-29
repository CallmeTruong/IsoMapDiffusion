"""
Training module for LoRA fine-tuning Qwen-Image-Edit.

Usage:
    # Local training
    python -m training.train --dataset-path ./data/metadata.csv --epochs 5 --lora-rank 16

    # Or with config file
    python -m training.train --config training/configs/default.yaml
"""

from .config import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    LoRAConfig,
    get_preset,
    PRESETS,
)
from .dataset import (
    IsometricDataset,
    ValidationDataset,
    create_dataset_from_csv,
    validate_dataset,
    get_train_transform,
    get_train_transform_augmented,
)
from .train import train_lora, LoRATrainer

__all__ = [
    # Config
    "TrainingConfig",
    "DatasetConfig",
    "ModelConfig",
    "LoRAConfig",
    "get_preset",
    "PRESETS",
    # Dataset
    "IsometricDataset",
    "ValidationDataset",
    "create_dataset_from_csv",
    "validate_dataset",
    "get_train_transform",
    "get_train_transform_augmented",
    # Training
    "train_lora",
    "LoRATrainer",
]
