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
        lora_adapter_name: str = "isometric",
        lora_weight: float = 1.0,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.lora_adapter_name = lora_adapter_name
        self.lora_weight = lora_weight
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self.pipe: Optional[QwenImageEditPipeline] = None

    def load(self) -> None:
        """Load the base pipeline and attach LoRA weights. Raises if LoRA fails to load."""
        print(f"Loading base model: {self.base_model}")

        self.pipe = QwenImageEditPipeline.from_pretrained(
            self.base_model,
            torch_dtype=self.dtype,
        )

        if self.lora_path and Path(self.lora_path).exists():
            print(f"Loading LoRA from: {self.lora_path}")
            self.pipe.load_lora_weights(
                self.lora_path,
                adapter_name=self.lora_adapter_name,
            )
            self.pipe.set_adapters(
                [self.lora_adapter_name],
                adapter_weights=[self.lora_weight],
            )
            print(f"LoRA '{self.lora_adapter_name}' loaded successfully (weight={self.lora_weight})")
        else:
            raise RuntimeError(
                f"LoRA path is required but not found: {self.lora_path}. "
                "Please set LORA_PATH in .env to a valid LoRA weights directory."
            )

        # A40 has 44.4 GiB usable - tight for full Qwen-Image-Edit + LoRA.
        # Use sequential CPU offload (layer-by-layer) which is stable across requests
        # unlike enable_model_cpu_offload which can crash on cleanup.
        try:
            self.pipe.enable_sequential_cpu_offload()
            print("Sequential CPU offload enabled")
        except Exception as e:
            print(f"Sequential offload failed ({e}); falling back to .to(device)")
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
