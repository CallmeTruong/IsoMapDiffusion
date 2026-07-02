"""
Global configuration constants for the isometric dataset pipeline.
"""

# Default prompt used throughout the system - MUST be consistent across all modules
DEFAULT_PROMPT = (
    "Fill in the outlined section with coherent pixels matching the <isometric pixel art> style, "
    "seamlessly blending edges with surrounding areas, maintaining consistent isometric perspective, "
    "shadow direction, lighting, pixel density, and color harmony while preserving structural integrity "
    "and removing all border artifacts"
)
