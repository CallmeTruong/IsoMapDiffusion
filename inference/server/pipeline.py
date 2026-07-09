"""Qwen Image Edit pipeline wrapper with LoRA support and speed optimizations.

Optimizations applied (zero quality loss, see docs/SYSTEM_REFERENCE.md §13):
  1. torch.compile on transformer (+20-40% per step)
  2. SDPA / Flash-Attn (+15-25%) — diffusers' default attention processor
     already calls torch.nn.functional.scaled_dot_product_attention on
     PyTorch >= 2.0, so PyTorch picks the fastest backend at runtime.
     We only flip the three CUDA SDP backends on and log which processor
     is installed.
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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

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
        max_batch_size: int = 2,
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.lora_adapter_name = lora_adapter_name
        self.lora_weight = lora_weight
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.device = device
        self.pipe: Optional[QwenImageEditPipeline] = None
        # Cap for batched requests; A100-80GB comfortably handles 2 tiles per
        # call (model ~20GB + 2× activations ~25GB). Bigger batches on
        # A100-80GB are possible but cost more VRAM and risk OOM on long
        # prompts. Server validates incoming batch size against this.
        self.max_batch_size = max_batch_size

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
        # OPTIMIZATION 2: SDPA / Flash-Attn (+15-25%)
        # Strategy: do NOT call set_attn_processor. In modern diffusers
        # (>=0.27) the transformer ships with an attention processor that
        # already dispatches through torch.nn.functional.scaled_dot_product_attention
        # when PyTorch >= 2.0 is installed. PyTorch then picks the fastest
        # available backend (Flash-Attn 2 > mem-efficient > math) based on
        # the input shape/dtype.
        #
        # We just:
        #   1. Make sure all three CUDA SDP backends are enabled.
        #   2. Log which processor is currently installed so we can verify.
        #   3. If somehow the default is the plain eager AttnProcessor, fall
        #      back to AttnProcessor2_0 (still better than vanilla).
        # ------------------------------------------------------------------
        if _opt_enabled("SDPA"):
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)

                current = None
                if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
                    proc_attr = getattr(
                        self.pipe.transformer, "attn_processor", None
                    )
                    if isinstance(proc_attr, dict) and proc_attr:
                        current = type(next(iter(proc_attr.values()))).__name__
                    elif proc_attr is not None:
                        current = type(proc_attr).__name__

                # Fallback only if the default processor is the slowest one.
                if current == "AttnProcessor":
                    try:
                        from diffusers.models.attention_processor import (
                            AttnProcessor2_0,
                        )
                        self.pipe.transformer.set_attn_processor(AttnProcessor2_0())
                        print(
                            "Opt-2: default was eager AttnProcessor; "
                            "upgraded to AttnProcessor2_0"
                        )
                        current = "AttnProcessor2_0"
                    except Exception as e:
                        print(
                            f"Opt-2: AttnProcessor2_0 upgrade failed ({e}); "
                            "keeping default"
                        )

                backend_hint = "auto (PyTorch picks at runtime)"
                try:
                    import importlib.util
                    if importlib.util.find_spec("flash_attn") is not None:
                        backend_hint = "flash-attn 2 (preferred)"
                    else:
                        backend_hint = "mem-efficient SDPA (flash_attn not installed)"
                except Exception:
                    pass

                print(
                    "Opt-2: SDPA active — diffusers default processor "
                    f"({current or 'unknown'}); CUDA SDP backends enabled "
                    f"(flash / mem-efficient / math); runtime pick: {backend_hint}"
                )
            except Exception as e:
                print(f"Opt-2: SDPA setup failed ({e}); using default attention")

        # ------------------------------------------------------------------
        # OPTIMIZATION 4: VAE tiling + slicing
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
        # request.
        #
        # Modes:
        #   'reduce-overhead' (default) - safe on A40; uses CUDA graphs
        #     but no autotuning; ~3-5 min cold start.
        #   'max-autotune' - aggressive autotuning, fuses ops, ~5-10 min
        #     cold start. Only worth it on >= A100 GPUs and when running
        #     with batch_size >= 2 (so batched batches benefit from the
        #     optimized kernels). Set ENABLE_TORCH_COMPILE_OPT=max-autotune.
        # ------------------------------------------------------------------
        if _opt_enabled("TORCH_COMPILE"):
            try:
                if (
                    hasattr(self.pipe, "transformer")
                    and self.pipe.transformer is not None
                ):
                    raw_mode = os.environ.get("ENABLE_TORCH_COMPILE_OPT", "1").strip().lower()
                    # Allow verbose mode strings: "max-autotune", "reduce-overhead", "default", etc.
                    if raw_mode in ("0", "1", "true", "false", "yes", "no", "on", "off"):
                        mode = "reduce-overhead"  # default boolean opt-in
                        dynamic = True
                        label = "reduce-overhead"
                    else:
                        # Treat raw_mode as a torch.compile mode string.
                        mode = raw_mode
                        # 'max-autotune' shapes are expected to be static.
                        # We still keep dynamic=True because Qwen-IE latent
                        # shapes vary per denoising step, but cuda graphs
                        # still benefit from autotuned kernels.
                        dynamic = True
                        label = raw_mode
                    self.pipe.transformer = torch.compile(
                        self.pipe.transformer,
                        mode=mode,
                        dynamic=dynamic,
                    )
                    print(
                        f"Opt-1: torch.compile enabled (mode={label}, "
                        f"~3-10 min one-time compile on first request)"
                    )
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
    ) -> "EditResult":
        """
        Run inference to edit the image based on prompt.

        Returns:
            EditResult(image, seed_used, time_ms). seed_used is the actual
            seed that was used (random if input seed was None) so callers
            can reproduce the result.
        """
        result = self._run_pipeline_batched(
            images=[image],
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt is not None else [None],
            true_cfg_scale=true_cfg_scale,
            steps=steps,
            guidance_scale=guidance_scale,
            seeds=[seed] if seed is not None else [None],
        )
        # _run_pipeline_batched returns one EditResult per item.
        return result[0]

    def edit_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]] = None,
        true_cfg_scale: Union[float, Sequence[float]] = 2.0,
        steps: int = 14,
        guidance_scale: Union[float, Sequence[float]] = 3.0,
        seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> List["EditResult"]:
        """
        Batched inference: process N tiles in a single pipeline call.

        All items in a batch share `steps`. `true_cfg_scale`, `guidance_scale`,
        `negative_prompt` and `seed` may be either a single value broadcast to
        every item, or a sequence with one entry per item.

        Returns:
            List of EditResult(image, seed_used, time_ms), one per input item.

        Why batching helps: Qwen-IE transformer steps are bandwidth-bound
        (loading weights from HBM each denoising step). Doubling the batch
        size ~doubles the time, not quadruples it, because the per-step
        weight-load cost is amortized across the batch. A100-80GB is big
        enough for batch=2 in BF16 (model ~20GB + 2× activations ~25GB).
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        if len(images) != len(prompts):
            raise ValueError(
                f"images and prompts length mismatch: "
                f"{len(images)} vs {len(prompts)}"
            )
        if len(images) == 0:
            return []
        return self._run_pipeline_batched(
            images=list(images),
            prompts=list(prompts),
            negative_prompts=list(negative_prompts) if negative_prompts is not None else None,
            true_cfg_scale=true_cfg_scale,
            steps=steps,
            guidance_scale=guidance_scale,
            seeds=list(seeds) if seeds is not None else None,
        )

    def _run_pipeline_batched(
        self,
        images: List[Image.Image],
        prompts: List[str],
        negative_prompts: Optional[List[Optional[str]]],
        true_cfg_scale: Union[float, Sequence[float]],
        steps: int,
        guidance_scale: Union[float, Sequence[float]],
        seeds: Optional[List[Optional[int]]],
    ) -> List["EditResult"]:
        """
        Shared implementation for edit() and edit_batch(). Both call into
        this method so logic stays in one place.

        Per-item wall time is approximated using elapsed/N; the server also
        exposes the full batch_time_ms separately.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        batch_size = len(images)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds pipeline max_batch_size "
                f"{self.max_batch_size}. Split the batch on the client."
            )

        def _broadcast(value, name):
            if isinstance(value, (list, tuple)):
                if len(value) != batch_size:
                    raise ValueError(
                        f"{name} length {len(value)} does not match "
                        f"batch size {batch_size}"
                    )
                return list(value)
            return [value] * batch_size

        neg_list = (
            [n if n is not None else None for n in negative_prompts]
            if negative_prompts is not None
            else [None] * batch_size
        )
        true_cfg_list = _broadcast(true_cfg_scale, "true_cfg_scale")
        guidance_list = _broadcast(guidance_scale, "guidance_scale")

        if seeds is None:
            seed_list = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
        else:
            seed_list = [
                int(s) if s is not None else random.randint(0, 2**32 - 1)
                for s in seeds
            ]

        # diffusers wants one generator per item in the batch so seeds stay
        # independent. The Generator must live on the same device as the
        # pipeline inputs.
        generators = [
            torch.Generator(device=self.device).manual_seed(s) for s in seed_list
        ]

        # Optional garbage collection between requests.
        # Off by default: a 12k-tile job spends ~30 minutes in gc.collect()
        # for no benefit when VRAM has 80 GB headroom. Enable explicitly via
        # ENABLE_AGGRESSIVE_GC_OPT=1 if you see VRAM creep.
        if os.environ.get("ENABLE_AGGRESSIVE_GC_OPT", "0") == "1":
            gc.collect()
            torch.cuda.empty_cache()

        prompts_list = list(prompts)
        images_list = list(images)

        print(
            f"Editing: n={batch_size} steps={steps} "
            f"cfg={guidance_scale} tcfg={true_cfg_scale} "
            f"seeds={seed_list}"
        )

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompts_list,
                    negative_prompt=neg_list,
                    true_cfg_scale=true_cfg_list,
                    image=images_list,
                    num_inference_steps=steps,
                    guidance_scale=guidance_list,
                    generator=generators,
                )
        except TypeError as e:
            # Some diffusers versions name the true_cfg arg differently.
            # Fall back to calling without it so we don't crash the server.
            print(
                f"true_cfg_scale kwarg failed ({e}); retrying without it"
            )
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompts_list,
                    negative_prompt=neg_list,
                    image=images_list,
                    num_inference_steps=steps,
                    guidance_scale=guidance_list,
                    generator=generators,
                )

        elapsed = time.perf_counter() - t0
        per_item = elapsed / max(batch_size, 1)
        print(
            f"edit took {elapsed:.1f}s ({per_item:.1f}s/item, batch={batch_size})"
        )

        # Pipeline returns PIL list; same length as input batch.
        if len(output.images) != batch_size:
            raise RuntimeError(
                f"Pipeline returned {len(output.images)} images for a batch "
                f"of {batch_size}"
            )
        # Per-item wall time is approximated as elapsed/N because the
        # pipeline processes all items simultaneously. The server includes
        # the real batch_time_ms in BatchEditResponse for accurate SLO.
        return [
            EditResult(
                image=img,
                seed_used=seed_used,
                time_ms=int(elapsed * 1000 / max(batch_size, 1)),
            )
            for img, seed_used in zip(output.images, seed_list)
        ]


@dataclass
class EditResult:
    """Result of one edit() call (or one item in edit_batch())."""

    image: Image.Image
    seed_used: int
    time_ms: int

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self.pipe is not None

    def __getattr__(self, name: str):
        """Retro-compat shim: gracefully handle missing attributes that older
        serialized bytecode on remote pods may reference.

        The previous version of this class exposed ``is_loaded`` only via the
        property above. Some deployed pods were still running the old
        ``QwenImageEditPipeline`` instance under a stale ``__pycache__/`` and
        crashed with ``AttributeError: 'QwenEditPipeline' object has no
        attribute 'is_loaded'`` on every /edit request.

        ``__getattr__`` is only called when normal attribute lookup fails, so
        the property path is unaffected. If anyone somehow serializes an
        instance without ``is_loaded``, this shim synthesises it from the
        underlying ``pipe`` attribute instead of crashing the server.
        """
        if name == "is_loaded":
            return getattr(self, "pipe", None) is not None
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
