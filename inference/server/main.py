"""FastAPI inference server for Qwen Image Edit with LoRA."""

import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

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

from .models import EditRequest, EditResponse, HealthResponse, ModelsResponse
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
        input_image = Image.open(BytesIO(image_data)).convert("RGB")

        # Run inference
        start_time = time.time()
        result = pipeline.edit(
            image=input_image,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            true_cfg_scale=req.true_cfg_scale,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )

        inference_time_ms = int((time.time() - start_time) * 1000)

        # Encode result
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        result_b64 = base64.b64encode(buffer.getvalue()).decode()

        return EditResponse(
            image_b64=result_b64,
            seed_used=req.seed if req.seed is not None else 42,
            time_ms=inference_time_ms,
        )

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
