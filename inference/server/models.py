"""Pydantic models for API requests/responses."""

from pydantic import BaseModel, Field
from typing import Optional


class EditRequest(BaseModel):
    """Request body for /edit endpoint."""

    image_b64: str = Field(..., description="Base64-encoded PNG image")
    prompt: str = Field(..., description="Edit instruction prompt")
    negative_prompt: Optional[str] = Field(None, description="What to avoid")
    true_cfg_scale: float = Field(2.0, ge=0.0, le=10.0, description="True CFG scale")
    steps: int = Field(14, ge=1, le=100, description="Inference steps")
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class EditResponse(BaseModel):
    """Response from /edit endpoint."""

    image_b64: str = Field(..., description="Base64-encoded PNG result image")
    seed_used: int = Field(..., description="Seed that was used")
    time_ms: int = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")


class ModelsResponse(BaseModel):
    """Response from /models endpoint."""

    available: list[str] = Field(..., description="Available model names")
    default: str = Field(..., description="Default model name")
