"""Inference client package."""

from .generator import GenerationClient, SyncGenerationClient
from .template import (
    TemplateBuilder,
    InfillRegion,
    TemplatePlacement,
    validate_quadrant_selection,
    TEMPLATE_SIZE,
    QUADRANT_SIZE,
    MAX_INFILL_AREA,
)
from .traversal import (
    TileTraversal,
    QuadrantKVState,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
    quadrant_iteration_order,
)
from .plan import (
    Point,
    RectBounds,
    GenerationStep,
    RectanglePlan,
    create_rectangle_plan_from_tuples,
)

# Backward-compat alias (legacy code referenced it; keep for safety)
OmniTemplateBuilder = TemplateBuilder

__all__ = [
    # Generator
    "GenerationClient",
    "SyncGenerationClient",
    # Template
    "TemplateBuilder",
    "OmniTemplateBuilder",  # legacy alias
    "InfillRegion",
    "TemplatePlacement",
    "validate_quadrant_selection",
    "TEMPLATE_SIZE",
    "QUADRANT_SIZE",
    "MAX_INFILL_AREA",
    # Traversal
    "TileTraversal",
    "QuadrantKVState",
    "scan_generated_set",
    "make_has_generation",
    "make_render_provider",
    "make_generation_provider",
    "crop_quadrant",
    "quadrant_iteration_order",
    # Plan
    "Point",
    "RectBounds",
    "GenerationStep",
    "RectanglePlan",
    "create_rectangle_plan_from_tuples",
]