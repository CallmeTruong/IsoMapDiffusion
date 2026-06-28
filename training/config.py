"""
Configuration management for LoRA training.

Handles:
    - Training hyperparameters
    - Model configuration
    - Dataset configuration
    - LoRA settings
"""

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from omegaconf import OmegaConf


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Base model configuration."""
    
    # HuggingFace model path
    model_id: str = "Qwen/Qwen-Image-Edit"
    
    # Model variant (for Qwen-Image-Edit-2511)
    model_variant: Optional[str] = None
    
    # Precision
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    
    # Use local model files if specified
    local_model_path: Optional[str] = None
    
    def get_model_id(self) -> str:
        """Get the actual model ID to use."""
        return self.local_model_path or self.model_id


# =============================================================================
# LoRA Configuration
# =============================================================================

@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    
    # LoRA rank (higher = more parameters, better quality, more VRAM)
    rank: int = 32
    
    # LoRA alpha (usually 2x rank)
    alpha: int = 64
    
    # Target modules for LoRA injection
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q",           # Query projection
        "to_k",           # Key projection
        "to_v",           # Value projection
        "add_q_proj",     # Cross-attention query
        "add_k_proj",     # Cross-attention key
        "add_v_proj",     # Cross-attention value
        "to_out.0",       # Output projection
        "to_add_out",     # Cross-attention output
        "img_mlp.net.2",  # Image MLP layer
        "img_mod.1",      # Image modulation
        "txt_mlp.net.2",  # Text MLP layer
        "txt_mod.1",      # Text modulation
    ])
    
    # Which component to apply LoRA
    lora_base_model: str = "dit"  # "dit", "vae", "text_encoder"
    
    # Dropout for LoRA layers
    lora_dropout: float = 0.0
    
    # Initialize LoRA weights
    init_type: Literal["gaussian", "normal", "zero"] = "gaussian"
    
    # Existing LoRA to continue from
    resume_lora_path: Optional[str] = None


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    # Base path to dataset
    base_path: str = "./data"
    
    # Metadata file (CSV or JSONL)
    metadata_path: str = "./data/metadata.csv"
    
    # Column names in metadata
    image_column: str = "image"
    control_image_column: str = "control_image"
    prompt_column: str = "prompt"
    
    # Image processing
    max_pixels: int = 1048576  # 1024x1024
    min_pixels: Optional[int] = None
    
    # Data augmentation
    flip_p: float = 0.0  # Probability of horizontal flip
    
    # How many times to repeat the dataset per epoch
    repeat: int = 1
    
    # Number of workers for data loading
    num_workers: int = 8
    
    # Batch size
    train_batch_size: int = 1
    val_batch_size: int = 1


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    # Output
    output_dir: str = "./output/lora"
    experiment_name: Optional[str] = None
    
    # Training hyperparameters
    num_epochs: int = 5
    max_train_steps: Optional[int] = None
    
    # Optimizer
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"  # "constant", "linear", "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Batch & gradient
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True
    
    # Mixed precision training
    mixed_precision: Literal["no", "fp16", "bf16"] = "bfloat16"
    
    # Checkpointing
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 3
    
    # Validation
    validation: bool = True
    validation_steps: int = 100
    validation_prompt: str = "Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, seamlessly blending edges with surrounding areas, maintaining consistent isometric perspective, shadow direction, lighting, pixel density, and color harmony while preserving structural integrity and removing all border artifacts",
    num_validation_images: int = 4
    
    # Misc
    seed: int = 42
    logging_steps: int = 10
    report_to: str = "tensorboard"  # "tensorboard", "wandb", "none"
    enable_xformers: bool = True
    use_8bit_adam: bool = False
    
    # Components
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.experiment_name is None:
            from datetime import datetime
            self.experiment_name = f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Make paths absolute
        self.output_dir = str(Path(self.output_dir).resolve())
        self.dataset.base_path = str(Path(self.dataset.base_path).resolve())
        self.dataset.metadata_path = str(Path(self.dataset.metadata_path).resolve())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load from YAML file."""
        with open(path, "r") as f:
            data = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def save_yaml(self, path: str):
        """Save to YAML file."""
        conf = OmegaConf.create(self.to_dict())
        with open(path, "w") as f:
            OmegaConf.save(conf, f)
    
    def save_json(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    "default": TrainingConfig(
        num_epochs=5,
        learning_rate=1e-4,
        lora=LoRAConfig(rank=32),
    ),
    
    "fast": TrainingConfig(
        num_epochs=3,
        learning_rate=2e-4,
        lora=LoRAConfig(rank=16),
        gradient_accumulation_steps=8,
    ),
    
    "quality": TrainingConfig(
        num_epochs=10,
        learning_rate=5e-5,
        lora=LoRAConfig(rank=64),
        use_gradient_checkpointing=True,
    ),
    
    "low_vram": TrainingConfig(
        num_epochs=5,
        learning_rate=1e-4,
        lora=LoRAConfig(rank=16),
        model=ModelConfig(torch_dtype="float16"),
        gradient_accumulation_steps=8,
        enable_xformers=True,
    ),
}


def get_preset(name: str) -> TrainingConfig:
    """Get a preset configuration."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


# =============================================================================
# CLI Helpers
# =============================================================================

def add_config_args(parser):
    """Add common config arguments to argparse parser."""
    group = parser.add_argument_group("Training Configuration")
    
    # Training
    group.add_argument("--epochs", type=int, dest="num_epochs", help="Number of epochs")
    group.add_argument("--steps", type=int, dest="max_train_steps", help="Max training steps")
    group.add_argument("--lr", type=float, dest="learning_rate", help="Learning rate")
    group.add_argument("--batch-size", type=int, dest="train_batch_size", help="Batch size")
    group.add_argument("--gradient-accum", type=int, dest="gradient_accumulation_steps", help="Gradient accumulation")
    
    # LoRA
    group.add_argument("--lora-rank", type=int, dest="lora_rank", help="LoRA rank")
    group.add_argument("--lora-alpha", type=int, dest="lora_alpha", help="LoRA alpha")
    
    # Model
    group.add_argument("--model", type=str, dest="model_id", help="Model ID")
    group.add_argument("--dtype", type=str, dest="torch_dtype", help="Torch dtype (fp16/bf16/fp32)")
    
    # Output
    group.add_argument("--output", type=str, dest="output_dir", help="Output directory")
    group.add_argument("--config", type=str, help="Config file (YAML/JSON)")
    group.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Use preset config")
