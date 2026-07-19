"""FastAPI inference server for Qwen Image Edit with LoRA."""

import asyncio
import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Load .env file from inference directory
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    print(f"Loaded environment from {_env_path}")
else:
    print(f"WARNING: No .env file found at {_env_path}")

from .models import (
    BatchEditItemResult,
    BatchEditRequest,
    BatchEditResponse,
    EditRequest,
    EditResponse,
    HealthResponse,
    ModelsResponse,
)
from .pipeline import QwenEditPipeline

# Server state
pipeline: QwenEditPipeline | None = None
model_name: str = "base"

# Load config from environment or defaults
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen-Image-Edit")
LORA_PATH = os.environ.get("LORA_PATH", "")
LORA_PATH = Path(LORA_PATH) if LORA_PATH else None
LORA_ADAPTER_NAME = os.environ.get("LORA_ADAPTER_NAME", "isometric")
LORA_WEIGHT = float(os.environ.get("LORA_WEIGHT", "1.0"))

# Require LoRA if LORA_PATH is specified but doesn't exist
if LORA_PATH is None or not str(LORA_PATH).strip():
    print("ERROR: LORA_PATH is not set. Please set LORA_PATH in .env file.")
    print(f"Hint: Download LoRA weights from Oxen and set LORA_PATH to the weights directory.")
    sys.exit(1)

if not LORA_PATH.exists():
    print(f"ERROR: LORA_PATH does not exist: {LORA_PATH}")
    print(f"Please download LoRA weights and set the correct LORA_PATH in .env")
    sys.exit(1)

DTYPE = os.environ.get("DTYPE", "bfloat16")
# Max items accepted in a single /edit/batch call. Cap at 2 for A100-80GB
# (matches pipeline.max_batch_size); larger caps will OOM on Qwen-IE
# activations and aren't a latency win on this model.
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "2"))

# FastAPI app
app = FastAPI(
    title="Isometric Image Edit API",
    description="Qwen Image Edit inference server with LoRA support",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    global pipeline, model_name

    print("=" * 60)
    print("Starting inference server...")
    print(f"  BASE_MODEL: {BASE_MODEL}")
    print(f"  LORA_PATH:  {LORA_PATH}")
    print(f"  LORA_ADAPTER_NAME: {LORA_ADAPTER_NAME}")
    print(f"  LORA_WEIGHT: {LORA_WEIGHT}")
    print(f"  DTYPE:      {DTYPE}")
    print("=" * 60)

    pipeline = QwenEditPipeline(
        base_model=BASE_MODEL,
        lora_path=str(LORA_PATH) if LORA_PATH else None,
        lora_adapter_name=LORA_ADAPTER_NAME,
        lora_weight=LORA_WEIGHT,
        dtype=DTYPE,
        max_batch_size=MAX_BATCH_SIZE,
    )

    try:
        pipeline.load()
        model_name = f"Qwen-IE + LoRA({LORA_PATH.name})"
        print(f"Model loaded: {model_name}")
    except Exception as e:
        print(f"ERROR: Failed to load model with LoRA: {e}")
        print("Server cannot start without valid LoRA weights.")
        sys.exit(1)


@app.post("/edit", response_model=EditResponse)
async def edit_image(req: EditRequest) -> EditResponse:
    """
    Edit an image based on a text prompt.

    The input image should be base64-encoded PNG with masked regions
    (outlined with red border) indicating what to generate.
    """
    global pipeline

    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server startup logs.",
        )

    try:
        # Decode image
        image_data = base64.b64decode(req.image_b64)
        # Default to PNG; clients can opt-in to JPEG by sending
        # image_format on the request. Pillow auto-detects the actual
        # format from magic bytes so this is purely advisory.
        input_image = Image.open(BytesIO(image_data)).convert("RGB")

        # Run inference off the event loop so concurrent requests can be
        # accepted (uvicorn will still process them sequentially through the
        # single pipeline, but it can interleave base64 encode/decode and
        # other I/O from the client side).
        result = await asyncio.to_thread(
            pipeline.edit,
            input_image,
            req.prompt,
            req.negative_prompt,
            req.true_cfg_scale,
            req.steps,
            req.guidance_scale,
            req.seed,
        )

        inference_time_ms = result.time_ms

        # Encode result. Default PNG (lossless). JPEG when client asks for
        # it; this trims the response payload by ~50% with no visible
        # loss because the model output is RGB and the output png is then
        # re-encoded by the client when stitching.
        response_fmt = (req.response_format or "PNG").upper()
        if response_fmt == "JPEG":
            buffer = BytesIO()
            result.image.save(buffer, format="JPEG", quality=92, optimize=False)
            mime_prefix = "image/jpeg"
        else:
            buffer = BytesIO()
            result.image.save(buffer, format="PNG", optimize=False)
            mime_prefix = "image/png"
        result_b64 = base64.b64encode(buffer.getvalue()).decode()

        return EditResponse(
            image_b64=result_b64,
            # Always return the actual seed used (random if request.seed
            # was None) so callers can reproduce the output.
            seed_used=result.seed_used,
            time_ms=inference_time_ms,
            image_format=response_fmt,
        )

    except Exception as e:
        import traceback
        print(f"ERROR in /edit: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit/batch", response_model=BatchEditResponse)
async def edit_batch(req: BatchEditRequest) -> BatchEditResponse:
    """
    Edit N tiles in a single pipeline call (batched).

    Use this for throughput: doubles effective tiles-per-hour vs /edit
    on A100-80GB at the cost of one extra VRAM-headroom. Items are
    processed in submission order; seeds are independent per item so
    results stay reproducible when client sets per-item seeds.

    All items share `steps`. `true_cfg_scale` and `guidance_scale` are
    taken from the top-level request and may be overridden per item.
    `negative_prompt` and `seed` are always per-item (use null to let
    the server pick a random seed).
    """
    global pipeline

    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server startup logs.",
        )

    if len(req.items) > pipeline.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size {len(req.items)} exceeds server cap "
                f"{pipeline.max_batch_size}. Split into smaller batches."
            ),
        )

    batch_start = time.time()
    try:
        # Decode all images first so the actual pipeline call is timed cleanly
        # (base64 decode time should not pollute the reported inference time).
        images: list[Image.Image] = []
        prompts: list[str] = []
        neg_prompts: list[Optional[str]] = []
        true_cfg_list: list[float] = []
        guidance_list: list[float] = []
        seed_list: list[Optional[int]] = []

        for it in req.items:
            image_data = base64.b64decode(it.image_b64)
            images.append(Image.open(BytesIO(image_data)).convert("RGB"))
            prompts.append(it.prompt)
            neg_prompts.append(it.negative_prompt)
            true_cfg_list.append(
                it.true_cfg_scale if it.true_cfg_scale is not None
                else req.true_cfg_scale
            )
            guidance_list.append(
                it.guidance_scale if it.guidance_scale is not None
                else req.guidance_scale
            )
            seed_list.append(it.seed)

        # Run the batched inference off the event loop so other requests
        # can be accepted while the GPU is busy.
        results = await asyncio.to_thread(
            pipeline.edit_batch,
            images,
            prompts,
            neg_prompts,
            true_cfg_list,
            req.steps,
            guidance_list,
            seed_list,
        )

        # Encode each result. We report:
        #   - batch_time_ms : real wall time including base64 of results
        #   - per_item time_ms : pipeline-reported per-item time from
        #       EditResult (approximated as elapsed/N). For SLO tracking,
        #       batch_time_ms is the truthful value.
        result_items: list[BatchEditItemResult] = []
        for item, res in zip(req.items, results):
            buffer = BytesIO()
            res.image.save(buffer, format="PNG")
            result_b64 = base64.b64encode(buffer.getvalue()).decode()
            result_items.append(
                BatchEditItemResult(
                    image_b64=result_b64,
                    seed_used=res.seed_used,
                    time_ms=res.time_ms,
                )
            )

        batch_time_ms = int((time.time() - batch_start) * 1000)
        per_item_avg_ms = batch_time_ms // max(len(result_items), 1)

        return BatchEditResponse(
            items=result_items,
            batch_time_ms=batch_time_ms,
            per_item_avg_ms=per_item_avg_ms,
        )

    except ValueError as e:
        # Client-side validation failed (mismatched lengths, too many items).
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check server health and model status."""
    global pipeline, model_name

    return HealthResponse(
        status="ok" if pipeline and pipeline.is_loaded else "degraded",
        model_loaded=pipeline is not None and pipeline.is_loaded,
        model_name=model_name if pipeline and pipeline.is_loaded else None,
        max_batch_size=(
            pipeline.max_batch_size
            if pipeline is not None and pipeline.is_loaded
            else MAX_BATCH_SIZE
        ),
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available models."""
    available = ["Qwen-IE base"]
    if LORA_PATH and LORA_PATH.exists():
        available.append(f"Qwen-IE + {LORA_PATH.name}")

    return ModelsResponse(
        available=available,
        default=available[-1] if available else "Qwen-IE base",
    )
