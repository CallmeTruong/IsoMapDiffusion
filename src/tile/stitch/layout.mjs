/**
 * tile/stitch/layout.mjs — Layout computation for stitch grids.
 *
 * Pure functions: no sharp/fs dependency, only arithmetic.
 * Exports: computeStride, stitchTestOffsets, computeLayout.
 */

import { TILE } from '../../config.mjs';

/**
 * Compute stride (pixel offset between two adjacent tiles) from render config.
 *
 * @param {Object} cfg
 * @param {number} cfg.tileSize       - Tile size in px (typically 1024)
 * @param {number} cfg.cameraMoveStep - Fraction of frustum camera shifts between two tiles.
 *                                      0.5 = 50% overlap (recommended)
 *                                      1.0 = 0% overlap (no seam check)
 * @returns {number} stride_px
 */
export function computeStride({ tileSize, cameraMoveStep }) {
  if (typeof tileSize !== 'number' || tileSize <= 0) {
    throw new Error(`computeStride: tileSize must be > 0, got ${tileSize}`);
  }
  if (typeof cameraMoveStep !== 'number' || cameraMoveStep <= 0) {
    throw new Error(`computeStride: cameraMoveStep must be > 0, got ${cameraMoveStep}`);
  }
  if (cameraMoveStep > 1) {
    throw new Error(
      `computeStride: cameraMoveStep=${cameraMoveStep} > 1 means camera skips tiles. Use ≤1.`
    );
  }
  return Math.round(tileSize * cameraMoveStep);
}

/**
 * Generate 4 offsets to test stitch: 0% / 25% / 50% / 75% overlap.
 * Useful for debugging which offset produces the best seam alignment.
 *
 * @param {number} tileSize - Tile size in px
 * @returns {number[]} Array of 4 offsets: [0, 0.25·tileSize, 0.5·tileSize, 0.75·tileSize]
 */
export function stitchTestOffsets(tileSize) {
  if (typeof tileSize !== 'number' || tileSize <= 0) {
    throw new Error(`stitchTestOffsets: tileSize must be > 0, got ${tileSize}`);
  }
  return [0, Math.round(tileSize * 0.25), Math.round(tileSize * 0.5), Math.round(tileSize * 0.75)];
}

/**
 * Compute layout (map size + per-tile pixel position) for an N×N or N×M grid.
 *
 * @param {Object} opts
 * @param {number} opts.gridSize - Grid N (square if gridCols/gridRows not provided)
 * @param {number} opts.tileSize - Tile size in px
 * @param {number} opts.stride   - Pixel offset between two adjacent tiles
 * @param {number} [opts.gridCols] - Number of columns (default = gridSize)
 * @param {number} [opts.gridRows] - Number of rows (default = gridSize)
 * @returns {{
 *   mapW: number, mapH: number,
 *   tileSize: number, stride: number, gridSize: number,
 *   positions: Array<{r,c,x,y}>
 * }}
 */
export function computeLayout({ gridSize, tileSize, stride, gridCols, gridRows }) {
  if (gridSize < 1) throw new Error(`computeLayout: gridSize must be ≥ 1, got ${gridSize}`);
  if (tileSize < 1) throw new Error(`computeLayout: tileSize must be ≥ 1, got ${tileSize}`);
  if (stride < 1)   throw new Error(`computeLayout: stride must be ≥ 1, got ${stride}`);

  const cols = gridCols ?? gridSize;
  const rows = gridRows ?? gridSize;

  const mapW = (cols - 1) * stride + tileSize;
  const mapH = (rows - 1) * stride + tileSize;

  const positions = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      positions.push({ r, c, x: c * stride, y: r * stride });
    }
  }

  return {
    mapW, mapH,
    tileSize, stride,
    gridSize, gridCols: cols, gridRows: rows,
    positions,
  };
}

/** Default stride from TILE config (convenience for callers). */
export function defaultStride() {
  return computeStride({ tileSize: TILE.sizePx, cameraMoveStep: TILE.cameraMoveStep });
}
