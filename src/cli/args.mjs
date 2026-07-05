/**
 * cli/args.mjs — CLI argument parser + help text
 */

import { STITCH, TILE } from '../config.mjs';

/**
 * Parse argv into { positional, flags }.
 *   positional = ['3', '2']
 *   flags = { seed: 'sgn', output: './output' }
 */
export function parseCliArgs(argv) {
  const userArgs = argv.slice(2);
  const positional = [];
  const flags = {};
  for (const a of userArgs) {
    if (a.startsWith('--')) {
      const idx = a.indexOf('=');
      if (idx < 0) flags[a.slice(2)] = true;
      else flags[a.slice(2, idx)] = a.slice(idx + 1);
    } else {
      positional.push(a);
    }
  }
  return { positional, flags };
}

/** Help text */
export function showHelp() {
  console.log(`
isometric-style-convert pipeline CLI

Usage:
  node src/pipeline.mjs <command> [args] [options]

Commands:
  render <N> [M]                       Render N×M grid (default 2x2)
  render <start_qx> <start_qy> <N> [M] Render from offset
  test <qx> <qy>                      Render a single tile (debug)
  stitch <N> [M]                      Stitch N×M grid from output renders (DEBUG TOOL)
  stitch <start_qx> <start_qy> <N> [M] Stitch from offset (DEBUG TOOL)
  list                                List available tiles
  info                                Print config + paths

Options:
  --seed=sgn|nyc|lat,lng   Seed point (default: sgn)
  --output=<dir>           Output render dir (default: output/renders)
  --workers=<N>            Number of parallel workers (default: 1)
  --provider=<id>          Tile provider: google | cesium-ion (default: cesium-ion)
  --api-key=<key>          API key/token (default: from provider's env)
                           - google      → GOOGLE_KEY env
                           - cesium-ion  → CESIUM_ION_TOKEN env
                           (cesium-ion: register free at https://cesium.com/ion/tokens)
  --step=<0..1>            Camera move step (default: 0.5)
  --azimuth=<deg>          Camera azimuth (default: 180)
  --elevation=<deg>        Camera elevation (default: -45)
  --altitude=<m>           Camera altitude (default: ${TILE.altitude})
  --sse=<num>              Screen space error (default: 8)
  --fallback=true|false    Enable/disable 2D satellite fallback (default: env FALLBACK_ENABLED)
                           When ON, blank 3D tiles will automatically fetch a 2D image (ESRI) instead.
                           Set FALLBACK_ENABLED=true in env to enable by default.
  --allow-missing=true|false
                           Stitch: if a tile is missing, use a transparent placeholder
                           instead of throwing. Useful for partial renders.
  --debug-only             Stitch: stitch only, do not save annotation (default: ${STITCH.debugOnly !== false ? 'true' : 'false'})
                           (Use for quick seam checking; output is not a deliverable)
  --xaxis=auto|east|west|off
                           Stitch: X-axis sign mode. Default: auto (detect from seam).
                           east/west: force. off: skip detection.

Stitch is a DEBUG TOOL (not the primary output):
  - Stitch does NOT produce files for training/inference.
  - Stitch is ONLY for verifying that seams between 2 tiles align correctly.
  - If you need a large image for demo purposes, use stitch --debug-only=false --out=<path>.
  - The pipeline's primary output = individual tile PNGs in output/renders/.

Examples:
  node src/pipeline.mjs render 3 3 --seed=sgn                       # use default provider (cesium-ion)
  node src/pipeline.mjs render 3 3 --provider=cesium-ion --seed=sgn
  node src/pipeline.mjs render 3 3 --provider=google --seed=sgn     # use Google direct (requires GOOGLE_KEY)
  node src/pipeline.mjs render 0 0 3 3 --seed=sgn
  node src/pipeline.mjs render 3 3 --fallback=true --seed=sgn       # enable 2D fallback for blank 3D tiles
  node src/pipeline.mjs test 0 0
  node src/pipeline.mjs stitch 3 3 --debug-only              # quick stitch, no annotation
  node src/pipeline.mjs stitch 3 3 --xaxis=auto              # stitch + auto-detect X sign
  node src/pipeline.mjs stitch 3 3 --xaxis=east              # force east
  node src/pipeline.mjs stitch 3 3 --allow-missing=true      # stitch even with placeholders for missing tiles
  node src/pipeline.mjs list
  node src/pipeline.mjs info                                  # view providers + env status
`);
}