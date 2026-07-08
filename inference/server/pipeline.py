"""Qwen Image Edit pipeline wrapper with LoRA support."""

import gc
import random
from pathlib import Path
from typing import Optional

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image


class QwenEditPipeline:
    """Wrapper for Qwen Image Edit pipeline with LoRA support."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen-Image-Edit",
        lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self.pipe: Optional[QwenImageEditPipeline] = None

    def load(self, lora_adapter_name: str = "isometric") -> None:
        """Load the base pipeline and optionally attach LoRA weights."""
        print(f"Loading base model: {self.base_model}")

        self.pipe = QwenImageEditPipeline.from_pretrained(
            self.base_model,
            torch_dtype=self.dtype,
        )

        if self.lora_path and Path(self.lora_path).exists():
            print(f"Loading LoRA from: {self.lora_path}")
            try:
                self.pipe.load_lora_weights(
                    self.lora_path,
                    adapter_name=lora_adapter_name,
                )
                self.pipe.set_adapters([lora_adapter_name], adapter_weights=[1.0])
                print(f"LoRA '{lora_adapter_name}' loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load LoRA: {e}")
                print("Continuing with base model only")

        self.pipe.to(self.device)
        print("Pipeline ready on", self.device)

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        true_cfg_scale: float = 2.0,
        steps: int = 14,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Run inference to edit the image based on prompt.

        Args:
            image: Input PIL Image
            prompt: Edit instruction
            negative_prompt: What to avoid
            true_cfg_scale: True CFG scale
            steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed (uses random if None)

        Returns:
            Edited PIL Image
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        gc.collect()
        torch.cuda.empty_cache()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Editing: prompt='{prompt[:50]}...' seed={seed} steps={steps}")

        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        return output.images[0]

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self.pipe is not None
