"""
Training module for LoRA fine-tuning Qwen-Image-Edit on RunPod.

Usage:
    # Local training
    python -m training.train --config training/configs/default.yaml

    # Deploy and train on RunPod
    python -m training.runpod_deploy deploy --dataset ./my_dataset

    # Check training status
    python -m training.runpod_deploy logs <pod_id>

    # Generate synthetic dataset for testing
    python -m training.prepare_dataset generate --count 100 --output ./data/test
"""

from .config import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    LoRAConfig,
    RunPodConfig,
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
    "RunPodConfig",
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
