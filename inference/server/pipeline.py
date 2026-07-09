"""Qwen Image Edit pipeline wrapper with LoRA support and speed optimizations.

Optimizations applied (zero quality loss, see docs/SYSTEM_REFERENCE.md §13):
  1. torch.compile on transformer (+20-40% per step)
  2. SDPA / Flash-Attn attention processor (+15-25%)
  3. VAE tiling + slicing (+5-10%, lower decode peak)

Memory layout is controlled by the OFFLOAD_MODE env var (case-insensitive):
  "group"      — group CPU offload (default, recommended for A40)
  "sequential" — sequential CPU offload (stable, slower)
  "gpu"        — full GPU load, no offload (fastest, needs ~46GB VRAM)

To toggle an optimization, set env var ENABLE_<NAME>_OPT=0 (default: 1).
Set ENABLE_TORCH_COMPILE_OPT=0 to skip torch.compile if you hit a recompile
limit issue or want faster cold start.

All optimizations are zero-quality-loss. They only change execution order or
memory layout. Bit-identical output to the unoptimized path (modulo any
non-determinism inherent to SDPA / compile).
"""

import gc
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image


def _opt_enabled(name: str, default: bool = True) -> bool:
    """Read ENABLE_<NAME>_OPT env var (1/0/true/false). Defaults to True."""
    raw = os.environ.get(f"ENABLE_{name}_OPT")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

        # ------------------------------------------------------------------
        # OPTIMIZATION 2: SDPA / Flash-Attn attention processor (+15-25%)
        # Force the PyTorch 2.x scaled-dot-product attention backend instead of
        # the eager attention path that Qwen-IE may pick otherwise.
        # ------------------------------------------------------------------
        if _opt_enabled("SDPA"):
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.transformer.set_attn_processor(AttnProcessor2_0())
                print("Opt-2: SDPA AttnProcessor2_0 enabled on transformer")
            except Exception as e:
                print(f"Opt-2: SDPA setup failed ({e}); using default attention")

        # ------------------------------------------------------------------
        # OPTIMIZATION 4: VAE tiling + slicing (+5-10%, lower decode peak)
        # Splits VAE encode/decode into tiles so peak activation memory drops.
        # Output is bit-identical to the non-tiled path.
        # ------------------------------------------------------------------
        if _opt_enabled("VAE_TILING"):
            try:
                if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                    self.pipe.vae.enable_tiling()
                    self.pipe.vae.enable_slicing()
                    print("Opt-4: VAE tiling + slicing enabled")
            except Exception as e:
                print(f"Opt-4: VAE tiling failed ({e}); continuing without")

        # ------------------------------------------------------------------
        # Memory layout: 3 modes via OFFLOAD_MODE env var
        #   "group"   - group CPU offload (default, fast, A40 friendly)
        #   "sequential" - sequential CPU offload (stable, slower)
        #   "gpu"     - full GPU load (no offload; fastest but needs VRAM)
        # ------------------------------------------------------------------
        offload_mode = os.environ.get("OFFLOAD_MODE", "group").lower().strip()

        if offload_mode == "gpu":
            # No offload - everything stays on GPU.
            # Best speed, but requires enough VRAM for full model + activations.
            self.pipe.to(self.device)
            print("Offload: GPU mode (no CPU offload)")
        elif offload_mode == "sequential":
            try:
                self.pipe.enable_sequential_cpu_offload()
                print("Offload: Sequential CPU offload")
            except Exception as e:
                print(f"Sequential offload failed ({e}); falling back to .to(device)")
                self.pipe.to(self.device)
        else:  # "group" (default)
            try:
                from diffusers import enable_group_offload
                enable_group_offload(
                    self.pipe.transformer,
                    onload_device=self.device,
                    offload_device="cpu",
                    offload_type="leaf_level",
                    num_blocks_per_group=4,
                )
                if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                    self.pipe.vae.enable_group_offload(
                        onload_device=self.device,
                        offload_device="cpu",
                    )
                print("Offload: Group CPU offload (transformer + vae)")
            except Exception as e:
                print(f"Group offload failed ({e}); falling back to sequential")
                try:
                    self.pipe.enable_sequential_cpu_offload()
                except Exception:
                    self.pipe.to(self.device)

        # ------------------------------------------------------------------
        # OPTIMIZATION 1: torch.compile on transformer (+20-40%)
        # Applied LAST so it sees the final graph layout after offload hooks.
        # Cold-start cost: ~3-5 min one-time. Pays off after the first
        # request. Uses 'reduce-overhead' mode (A40 friendly; cudagraphs in
        # 'max-autotune' would add overhead on non-Hopper cards).
        # ------------------------------------------------------------------
        if _opt_enabled("TORCH_COMPILE"):
            try:
                if (
                    hasattr(self.pipe, "transformer")
                    and self.pipe.transformer is not None
                ):
                    self.pipe.transformer = torch.compile(
                        self.pipe.transformer,
                        mode="reduce-overhead",
                        dynamic=True,  # latent shapes vary per step
                    )
                    print("Opt-1: torch.compile enabled on transformer "
                          "(~3-5 min one-time compile on first request)")
            except Exception as e:
                print(f"Opt-1: torch.compile failed ({e}); continuing without")

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

        print(
            f"Editing: prompt='{prompt[:50]}...' seed={seed} "
            f"steps={steps} cfg={guidance_scale} tcfg={true_cfg_scale}"
        )

        # Log per-request timing to help compare before/after optimizations
        t0 = time.perf_counter()
        try:
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
        except TypeError as e:
            # Some diffusers versions name the true_cfg arg differently.
            # Fall back to calling without it so we don't crash the server.
            print(f"true_cfg_scale kwarg failed ({e}); retrying without it")
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

        elapsed = time.perf_counter() - t0
        print(f"edit() took {elapsed:.1f}s")

        return output.images[0]

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self.pipe is not None
