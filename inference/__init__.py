"""Inference package.

Exports client + config (khong export server app de tranh import nang).
"""
from inference.client import (
    GenerationClient,
    SyncGenerationClient,
    TemplateBuilder,
    InfillRegion,
    OmniTemplateBuilder,  # backward-compat alias
    TileTraversal,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
)
from inference.config import (
    get_inference_config,
    InferenceConfig,
    reset_config,
)

__all__ = [
    "GenerationClient",
    "SyncGenerationClient",
    "TemplateBuilder",
    "InfillRegion",
    "OmniTemplateBuilder",
    "TileTraversal",
    "scan_generated_set",
    "make_has_generation",
    "make_render_provider",
    "make_generation_provider",
    "crop_quadrant",
    "get_inference_config",
    "InferenceConfig",
    "reset_config",
]

# server.app import lazy (can install torch/uvicorn)
def __getattr__(name):
    if name == "app":
        from inference.server import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
