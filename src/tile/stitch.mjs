/**
 * tile/stitch.mjs — FACADE re-export from tile/stitch/*
 */

export { computeStride, stitchTestOffsets, computeLayout, defaultStride } from './stitch/layout.mjs';
export { createBaseCanvas, stitchTiles, stitchPair }                  from './stitch/compose.mjs';
export { annotateSeams, buildSeamSvg, rgbaStr }                        from './stitch/annotate.mjs';
export { savePng, stitchGrid }                                          from './stitch/io.mjs';
export { stitchWorld }                                                  from './stitch/world.mjs';
