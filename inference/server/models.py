"""Pydantic models for API requests/responses."""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class EditRequest(BaseModel):
    """Request body for /edit endpoint."""

    image_b64: str = Field(..., description="Base64-encoded PNG image")
    image_format: Optional[str] = Field(
        None, description="Source image format hint (PNG | JPEG)",
    )
    prompt: str = Field(..., description="Edit instruction prompt")
    negative_prompt: Optional[str] = Field(None, description="What to avoid")
    true_cfg_scale: float = Field(2.0, ge=0.0, le=10.0, description="True CFG scale")
    steps: int = Field(14, ge=1, le=100, description="Inference steps")
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    response_format: Optional[str] = Field(
        None, description="Preferred response image format (PNG | JPEG)",
    )


class EditResponse(BaseModel):
    """Response from /edit endpoint."""

    image_b64: str = Field(..., description="Base64-encoded result image")
    seed_used: int = Field(..., description="Seed that was used")
    time_ms: int = Field(..., description="Inference time in milliseconds")
    image_format: str = Field(
        "PNG", description="Format used to encode image_b64 (PNG | JPEG)",
    )


class BatchEditItem(BaseModel):
    """One tile inside a batched /edit/batch request."""

    image_b64: str = Field(..., description="Base64-encoded PNG image")
    image_format: Optional[str] = Field(
        None, description="Source image format hint (PNG | JPEG). "
                          "Server uses this only for decoding validation."
    )
    prompt: str = Field(..., description="Edit instruction prompt")
    negative_prompt: Optional[str] = Field(None, description="What to avoid")
    true_cfg_scale: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Per-item true CFG; falls back to top-level value if None",
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.0, le=20.0,
        description="Per-item guidance; falls back to top-level value if None",
    )
    seed: Optional[int] = Field(
        None, description="Per-item seed; server fills in a random one if None",
    )


class BatchEditRequest(BaseModel):
    """Request body for /edit/batch endpoint.

    All items share `steps`. `true_cfg_scale` and `guidance_scale` can be set
    once at the top level (broadcast to all items) and overridden per item.
    `negative_prompt` and `seed` are always per-item.
    """

    items: List[BatchEditItem] = Field(
        ..., min_length=1, max_length=8,
        description="Tiles to process in this batch (1..8)",
    )
    steps: int = Field(14, ge=1, le=100, description="Inference steps")
    true_cfg_scale: float = Field(
        2.0, ge=0.0, le=10.0, description="Default true CFG (broadcast)",
    )
    guidance_scale: float = Field(
        3.0, ge=0.0, le=20.0, description="Default guidance (broadcast)",
    )
    response_format: Optional[str] = Field(
        None, description="Preferred response image format (PNG | JPEG). "
                          "PNG is always sent; JPEG is an option to trim "
                          "the payload size on the wire.",
    )

    @model_validator(mode="after")
    def _check_size(self):
        if len(self.items) > 8:
            raise ValueError("Batch size cannot exceed 8 (server cap)")
        if len(self.items) < 1:
            raise ValueError("Batch must contain at least 1 item")
        return self


class BatchEditItemResult(BaseModel):
    """One tile's result inside a batched response."""

    image_b64: str = Field(..., description="Base64-encoded PNG result image")
    seed_used: int = Field(..., description="Seed that was used for this tile")
    time_ms: int = Field(..., description="Per-tile wall time in milliseconds")


class BatchEditResponse(BaseModel):
    """Response from /edit/batch endpoint."""

    items: List[BatchEditItemResult] = Field(
        ..., description="One result per input tile, same order as request",
    )
    batch_time_ms: int = Field(..., description="Total batch wall time")
    per_item_avg_ms: int = Field(
        ..., description="batch_time_ms / len(items), for SLO tracking",
    )


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    max_batch_size: int = Field(
        1, description="Maximum batch size accepted by /edit/batch",
    )


class ModelsResponse(BaseModel):
    """Response from /models endpoint."""

    available: list[str] = Field(..., description="Available model names")
    default: str = Field(..., description="Default model name")
