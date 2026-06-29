"""
Main training script for LoRA fine-tuning Qwen-Image-Edit.

Usage:
    # From command line
    python -m training.train --config config.yaml
    
    # With overrides
    python -m training.train --model Qwen/Qwen-Image-Edit --epochs 10 --lora-rank 32
    
    # As a module
    from training import train_lora, TrainingConfig
    config = TrainingConfig(...)
    train_lora(config)
"""

import os
import sys
import json
import logging
import random
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import QwenImageEditPipeline
from diffusers.training_utils import cast_training_params
from tqdm import tqdm

from .config import TrainingConfig, LoRAConfig, DatasetConfig, ModelConfig
from .dataset import IsometricDataset, ValidationDataset, get_train_transform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = get_logger("training.train")


# =============================================================================
# Helper Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_noise_loss(
    model,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_images: torch.Tensor,
) -> torch.Tensor:
    """
    Compute diffusion noise prediction loss.
    
    This is the standard MSE loss for diffusion models:
    L = E[||noise_pred - noise||^2]
    """
    # Get model prediction
    model_pred = model(
        sample=latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        image=control_images,
    ).sample
    
    # MSE loss
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    
    return loss


def encode_prompt(prompt, text_encoder, tokenizer, device, batch_size=1):
    """Encode text prompt to hidden states."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
    
    return encoder_hidden_states


# =============================================================================
# Training Class
# =============================================================================

class LoRATrainer:
    """
    LoRA trainer for Qwen-Image-Edit.
    
    Handles:
        - Model setup with LoRA
        - Training loop
        - Checkpointing
        - Validation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = None
        self.pipeline = None
        self.vae = None
        self.text_encoder = None
        self.transformer = None
        self.optimizer = None
        self.train_dataloader = None
        self.global_step = 0
        self.current_epoch = 0
        
    def setup(self):
        """Initialize all components."""
        config = self.config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.report_to if config.report_to != "none" else None,
            project_dir=config.output_dir,
        )
        
        # Set seed
        set_seed(config.seed)
        
        # Make output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = Path(config.output_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info("=" * 60)
        logger.info("LoRA TRAINING SETUP")
        logger.info("=" * 60)
        logger.info(f"  Model:         {config.model.get_model_id()}")
        logger.info(f"  LoRA Rank:     {config.lora.rank}")
        logger.info(f"  LoRA Alpha:    {config.lora.alpha}")
        logger.info(f"  Learning Rate: {config.learning_rate}")
        logger.info(f"  Epochs:        {config.num_epochs}")
        logger.info(f"  Batch Size:    {config.dataset.train_batch_size}")
        logger.info(f"  Output:        {config.output_dir}")
        logger.info("=" * 60)
        
        # Load model
        self._load_model()
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup data
        self._setup_data()
        
        # Prepare with accelerator
        self._prepare_for_training()
        
        logger.info("Setup complete!")
        
    def _load_model(self):
        """Load Qwen-Image-Edit model."""
        config = self.config
        
        logger.info(f"Loading model: {config.model.get_model_id()}")
        
        # Load pipeline
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)
        
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            config.model.get_model_id(),
            torch_dtype=dtype,
        )
        
        # Move to accelerator device
        self.pipeline = self.pipeline.to(self.accelerator.device)
        
        # Get components
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.transformer = self.pipeline.transformer
        self.scheduler = self.pipeline.scheduler
        
        # Setup latent projection (VAE latent channels -> transformer expected channels)
        self._setup_latent_projection()
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
        
        # Freeze non-LoRA params
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        
        logger.info(f"  Model loaded to {self.accelerator.device}")
        
    def _setup_latent_projection(self):
        """Setup projection from VAE latent space to transformer expected channels."""
        vae_latent_channels = self.vae.config.latent_channels  # typically 16
        transformer_in_channels = self.transformer.config.in_channels  # typically 64
        
        logger.info(f"  VAE latent channels: {vae_latent_channels}")
        logger.info(f"  Transformer in_channels: {transformer_in_channels}")
        
        if vae_latent_channels != transformer_in_channels:
            self.latent_projection = torch.nn.Sequential(
                torch.nn.Conv2d(
                    vae_latent_channels,
                    transformer_in_channels,
                    kernel_size=1,
                    padding=0,
                ),
            )
            self.latent_projection.to(self.accelerator.device)
            logger.info(f"  Added projection: {vae_latent_channels} -> {transformer_in_channels} channels")
        else:
            self.latent_projection = None
            logger.info("  No projection needed (channels match)")
    
    def _setup_lora(self):
        """Setup LoRA on transformer."""
        from peft import LoraConfig, get_peft_model
        
        config = self.config
        lora_config = config.lora
        
        logger.info(f"Setting up LoRA with rank={lora_config.rank}, alpha={lora_config.alpha}")
        
        # Create PEFT config
        peft_config = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            init_lora_weights=lora_config.init_type,
        )
        
        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, peft_config)
        self.transformer.print_trainable_parameters()
        
        logger.info("LoRA setup complete!")
        
    def _setup_optimizer(self):
        """Setup optimizer."""
        config = self.config
        
        # Only optimize LoRA params
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        # Use AdamW
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        logger.info(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
        
    def _setup_data(self):
        """Setup training data."""
        config = self.config
        
        logger.info(f"Loading dataset from: {config.dataset.metadata_path}")
        
        # Get transform for converting PIL images to tensors
        train_transform = get_train_transform(config.dataset.max_pixels)
        
        # Create dataset
        self.dataset = IsometricDataset(
            base_path=config.dataset.base_path,
            metadata_path=config.dataset.metadata_path,
            image_column=config.dataset.image_column,
            control_image_column=config.dataset.control_image_column,
            prompt_column=config.dataset.prompt_column,
            max_pixels=config.dataset.max_pixels,
            repeat=config.dataset.repeat,
            transform=train_transform,
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=config.dataset.train_batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
        )
        
        logger.info(f"  Dataset size: {len(self.dataset)}")
        logger.info(f"  Batch size: {config.dataset.train_batch_size}")
        logger.info(f"  Steps per epoch: {len(self.train_dataloader)}")
        
    def _prepare_for_training(self):
        """Prepare models and optimizer for distributed training."""
        # Prepare with accelerator
        self.transformer, self.optimizer = self.accelerator.prepare(
            self.transformer, self.optimizer
        )
        
        # Prepare dataloader
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        
        # Recalculate total steps
        self.total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.num_update_steps_per_epoch = len(self.train_dataloader)
        
        logger.info(f"Total training steps: {self.total_steps}")
        
    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step."""
        config = self.config
        
        # Get images (batch, channels, height, width)
        images = batch["image"].to(self.accelerator.device)
        control_images = batch["control_image"].to(self.accelerator.device)
        prompts = batch["prompt"]
        
        # Get dtype from VAE
        vae_dtype = next(self.vae.parameters()).dtype
        
        # Convert to VAE's dtype (bfloat16 for Qwen)
        images = images.to(vae_dtype)
        control_images = control_images.to(vae_dtype)
        
        # Qwen VAE expects 5D tensor: (B, C, F, H, W) - add frame dimension
        if images.ndim == 4:
            images = images.unsqueeze(2)  # (B, C, 1, H, W)
        if control_images.ndim == 4:
            control_images = control_images.unsqueeze(2)  # (B, C, 1, H, W)
        
        # Convert images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Standard VAE scaling factor
        
        # Sample noise in same dtype as latents
        noise = torch.randn_like(latents)
        
        # Sample timesteps (DDPM schedulers typically use 1000 steps)
        num_train_timesteps = 1000
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        
        # Add noise to latents (FlowMatch interpolation)
        # x_t = (1 - t) * x_0 + t * epsilon, where t is normalized [0, 1]
        # Handle both 4D (B,C,H,W) and 5D (B,C,F,H,W) latent shapes
        t_normalized = timesteps.float() / num_train_timesteps
        if latents.ndim == 5:
            latents_for_flow = latents.squeeze(2)  # Remove frame dim -> (B, C, H, W)
            noise_for_flow = noise.squeeze(2)
            t_normalized = t_normalized.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        else:
            latents_for_flow = latents
            noise_for_flow = noise
            t_normalized = t_normalized.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_normalized) * latents_for_flow + t_normalized * noise_for_flow
        
        # Encode prompts
        encoder_hidden_states = encode_prompt(
            prompts,
            self.text_encoder,
            self.pipeline.tokenizer,
            self.accelerator.device,
        )
        
        # Predict noise (transformer expects 4D or 5D depending on model)
        if noisy_latents.ndim == 5:
            control_for_transformer = control_images.squeeze(2)  # (B, C, H, W)
        else:
            control_for_transformer = control_images
        
        # Reshape latents for QwenImageTransformer - expects (B, C, H, W) or patched format
        # Latents from VAE are (B, latent_channels, H, W)
        latent_shape = noisy_latents.shape  # (B, C, H, W)
        latent_channels = latent_shape[1]
        latent_h, latent_w = latent_shape[2], latent_shape[3]
        
        # QwenImageEdit model may expect hidden_states in (B, seq_len, C) or (B, H, W, C)
        # Flatten spatial dims to sequence: (B, H*W, C)
        hidden_states = noisy_latents.permute(0, 2, 3, 1)  # (B, H, W, C)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[3])  # (B, H*W, C)
        
        # Handle different API signatures - try with conditioning_image
        try:
            transformer_output = self.transformer(
                hidden_states=hidden_states,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                conditioning_image=control_for_transformer,
            )
        except TypeError:
            try:
                # Try without conditioning_image
                transformer_output = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                )
            except TypeError:
                # Fallback: try positional args with just hidden_states
                transformer_output = self.transformer(
                    hidden_states,
                    encoder_hidden_states,
                    timestep=timesteps,
                )
        
        # Support both old (.sample) and new API
        if hasattr(transformer_output, 'sample'):
            model_pred = transformer_output.sample
        elif hasattr(transformer_output, 'reshaped'):
            model_pred = transformer_output.reshaped
        else:
            model_pred = transformer_output
        
        # Reshape prediction back to (B, C, H, W) for loss computation
        if model_pred.ndim == 3:  # (B, seq, C)
            model_pred = model_pred.reshape(model_pred.shape[0], latent_channels, latent_h, latent_w)
        
        # Compute loss (all tensors now in same dtype and shape)
        loss = F.mse_loss(model_pred.float(), noise_for_flow.float(), reduction="mean")
        
        return loss
    
    def training_loop(self):
        """Main training loop."""
        config = self.config
        
        logger.info("Starting training loop...")
        
        progress_bar = tqdm(
            total=self.total_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            self.transformer.train()
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.transformer):
                    loss = self.training_step(batch)
                    
                    # Backward
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(),
                            config.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        logger.info(f"Step {self.global_step}/{self.total_steps} | Loss: {loss.item():.4f}")
                    
                    # Checkpointing
                    if self.global_step % config.checkpointing_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Validation
                if config.validation and self.global_step % config.validation_steps == 0 and self.global_step > 0:
                    self.run_validation()
            
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} complete!")
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint("final")
        
        logger.info("Training complete!")
        
    def run_validation(self):
        """Run validation and generate sample images."""
        config = self.config
        
        if not config.validation:
            return
        
        logger.info("Running validation...")
        
        self.transformer.eval()
        
        try:
            # Get validation samples
            val_dataset = ValidationDataset(
                base_path=config.dataset.base_path,
                metadata_path=config.dataset.metadata_path,
                num_samples=config.num_validation_images,
            )
            
            samples = val_dataset.get_samples()
            
            for i, sample in enumerate(samples):
                with torch.inference_mode():
                    output = self.pipeline(
                        prompt=sample["prompt"],
                        image=sample["control_image"],
                        num_inference_steps=14,
                        guidance_scale=3.0,
                        true_cfg_scale=2.0,
                    )
                
                # Save
                output_path = Path(config.output_dir) / "validation"
                output_path.mkdir(exist_ok=True)
                
                output.images[0].save(
                    output_path / f"val_step_{self.global_step}_sample_{i}.png"
                )
            
            logger.info(f"Validation samples saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            traceback.print_exc()
        
        self.transformer.train()
        
    def save_checkpoint(self, name: str):
        """Save LoRA checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        config = self.config
        
        checkpoint_dir = Path(config.output_dir) / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.transformer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict(),
        }
        
        torch.save(state, checkpoint_dir / "trainer_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        from peft import PeftModel
        
        # Load state
        state = torch.load(Path(checkpoint_path) / "trainer_state.pt")
        self.global_step = state["global_step"]
        self.current_epoch = state["epoch"]
        
        # Load optimizer state
        self.optimizer.load_state_dict(state["optimizer_state"])
        
        # Load LoRA weights
        self.transformer = PeftModel.from_pretrained(
            self.transformer,
            checkpoint_path,
        )
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def export_lora(self, output_path: str):
        """Export final LoRA weights in safetensors format."""
        from safetensors.torch import save_file
        
        # Get state dict
        state_dict = self.transformer.state_dict()
        
        # Save
        save_file(state_dict, output_path)
        
        logger.info(f"LoRA exported to: {output_path}")


# =============================================================================
# Main Training Function
# =============================================================================

def train_lora(config: Optional[TrainingConfig] = None, **kwargs) -> LoRATrainer:
    """
    Train LoRA on Qwen-Image-Edit.
    
    Args:
        config: TrainingConfig object (optional)
        **kwargs: Override config values
    
    Returns:
        LoRATrainer instance
    """
    # Merge kwargs into config
    if config is None:
        config = TrainingConfig(**kwargs)
    elif kwargs:
        # Override config values
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Setup and train
    trainer = LoRATrainer(config)
    trainer.setup()
    trainer.training_loop()
    
    # Export final LoRA
    final_path = Path(config.output_dir) / "pytorch_lora_weights.safetensors"
    trainer.export_lora(str(final_path))
    
    return trainer


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train LoRA on Qwen-Image-Edit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m training.train
  
  # Train with config file
  python -m training.train --config my_config.yaml
  
  # Train with overrides
  python -m training.train --epochs 10 --lora-rank 32 --lr 1e-4
  
  # Train with specific dataset
  python -m training.train --dataset-path ./my_dataset/metadata.csv --output ./output
        """
    )
    
    # Config file
    parser.add_argument("--config", type=str, help="Config file (YAML/JSON)")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, help="Max training steps")
    
    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    
    # Model args
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image-Edit", help="Model ID")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    
    # Data args
    parser.add_argument("--dataset-path", type=str, help="Dataset metadata path")
    parser.add_argument("--base-path", type=str, default="./data", help="Dataset base path")
    
    # Output args
    parser.add_argument("--output", type=str, default="./output/lora", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Other
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--validation", action="store_true", default=True, help="Run validation")
    parser.add_argument("--no-validation", dest="validation", action="store_false", help="Skip validation")
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        # Load from file
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = TrainingConfig.from_yaml(args.config)
        else:
            config = TrainingConfig.from_json(args.config)
    else:
        # Build from args
        model_config = ModelConfig(model_id=args.model)
        if args.dtype == "fp16":
            model_config.torch_dtype = "float16"
        elif args.dtype == "bf16":
            model_config.torch_dtype = "bfloat16"
        else:
            model_config.torch_dtype = "float32"
        
        lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
        
        # Auto-detect base_path from dataset-path if not explicitly provided
        if args.dataset_path:
            # base_path defaults to the directory containing the CSV
            base_path = args.base_path if args.base_path != "./data" else str(Path(args.dataset_path).parent)
        else:
            base_path = args.base_path
        
        dataset_config = DatasetConfig(
            base_path=base_path,
            metadata_path=args.dataset_path or f"{base_path}/metadata.csv",
        )
        
        # Create training config with mixed precision based on dtype
        from .config import TrainingConfig as TC
        
        train_cfg = {
            "num_epochs": args.epochs,
            "learning_rate": args.lr,
            "max_train_steps": args.max_steps,
            "output_dir": args.output,
            "seed": args.seed,
            "validation": args.validation,
            "mixed_precision": args.dtype if args.dtype != "fp32" else "no",
        }
        
        config = TrainingConfig(
            model=model_config,
            lora=lora_config,
            dataset=dataset_config,
            **train_cfg,
        )
    
    # Train
    try:
        trainer = train_lora(config)
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save emergency checkpoint
        trainer.save_checkpoint("emergency-interrupt")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
