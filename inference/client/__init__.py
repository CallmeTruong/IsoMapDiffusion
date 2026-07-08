"""Inference client package."""
from .generator import GenerationClient, SyncGenerationClient
from .template import (
    TemplateBuilder,
    InfillRegion,
    TemplatePlacement,
    OmniTemplateBuilder,  # backward-compat alias
    validate_quadrant_selection,
    TEMPLATE_SIZE,
    QUADRANT_SIZE,
    BORDER_COLOR,
    BORDER_WIDTH,
)
from .traversal import (
    TileTraversal,
    TileInfo,
    TileStatus,
    scan_generated_set,
    make_has_generation,
    make_render_provider,
    make_generation_provider,
    crop_quadrant,
    quadrant_iteration_order,
    build_neighbor_map,
    TILE_FILENAME_RE,
)

__all__ = [
    "GenerationClient",
    "SyncGenerationClient",
    "TemplateBuilder",
    "InfillRegion",
    "TemplatePlacement",
    "OmniTemplateBuilder",  # backward-compat
    "validate_quadrant_selection",
    "TEMPLATE_SIZE",
    "QUADRANT_SIZE",
    "BORDER_COLOR",
    "BORDER_WIDTH",
    "TileTraversal",
    "TileInfo",
    "TileStatus",
    "scan_generated_set",
    "make_has_generation",
    "make_render_provider",
    "make_generation_provider",
    "crop_quadrant",
    "quadrant_iteration_order",
    "build_neighbor_map",
    "TILE_FILENAME_RE",
]
