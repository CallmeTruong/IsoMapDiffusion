"""Inference package.

Exports client + config (khong export server app de tranh import nang).
"""
from inference.client import (
    GenerationClient,
    SyncGenerationClient,
    TemplateBuilder,
    InfillRegion,
    TemplatePlacement,
    validate_quadrant_selection,
    TileTraversal,
    QuadrantKVState,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
    Point,
    RectBounds,
    GenerationStep,
    RectanglePlan,
    create_rectangle_plan_from_tuples,
)
from inference.config import (
    get_inference_config,
    InferenceConfig,
    reset_config,
)

# `OmniTemplateBuilder` duoc giu nhu alias cho backward-compat (deprecated)
# Khong dua vao __all__ chinh de khuyen khich su dung TemplateBuilder.
from inference.client import OmniTemplateBuilder  # noqa: F401

__all__ = [
    # Generator
    "GenerationClient",
    "SyncGenerationClient",
    # Template
    "TemplateBuilder",
    "InfillRegion",
    "TemplatePlacement",
    "validate_quadrant_selection",
    # Traversal
    "TileTraversal",
    "QuadrantKVState",
    "scan_generated_set",
    "make_has_generation",
    "make_render_provider",
    "make_generation_provider",
    "crop_quadrant",
    # Plan
    "Point",
    "RectBounds",
    "GenerationStep",
    "RectanglePlan",
    "create_rectangle_plan_from_tuples",
    # Config
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