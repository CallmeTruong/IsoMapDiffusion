/**
 * tile/stitch/layout.mjs — Tính toán layout cho stitch grid
 *
 * Pure functions: không phụ thuộc sharp/fs, chỉ tính toán.
 * Gồm: computeStride, stitchTestOffsets, computeLayout.
 */

import { TILE } from '../../config.mjs';

/**
 * Tính stride (pixel offset giữa 2 tile kề) từ config render.
 *
 * @param {Object} cfg
 * @param {number} cfg.tileSize       - Kích thước mỗi tile (px), thường 1024
 * @param {number} cfg.cameraMoveStep - Fraction frustum camera dịch giữa 2 tile
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
 * Tạo 4 offset để test stitch: 0% / 25% / 50% / 75% overlap.
 * Dùng cho debug khi muốn xem offset nào cho seam khớp.
 *
 * @param {number} tileSize - Kích thước mỗi tile (px)
 * @returns {number[]} Mảng 4 offset: [0, 0.25·tileSize, 0.5·tileSize, 0.75·tileSize]
 */
export function stitchTestOffsets(tileSize) {
  if (typeof tileSize !== 'number' || tileSize <= 0) {
    throw new Error(`stitchTestOffsets: tileSize must be > 0, got ${tileSize}`);
  }
  return [0, Math.round(tileSize * 0.25), Math.round(tileSize * 0.5), Math.round(tileSize * 0.75)];
}

/**
 * Tính layout (kích thước map + vị trí pixel từng tile) cho grid N×N hoặc N×M.
 *
 * @param {Object} opts
 * @param {number} opts.gridSize - Grid N (vuông nếu không có gridCols/gridRows)
 * @param {number} opts.tileSize - Kích thước mỗi tile (px)
 * @param {number} opts.stride   - Pixel offset giữa 2 tile kề
 * @param {number} [opts.gridCols] - Số cột (mặc định = gridSize)
 * @param {number} [opts.gridRows] - Số hàng (mặc định = gridSize)
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

/** Default stride từ TILE config (cho caller tiện dùng) */
export function defaultStride() {
  return computeStride({ tileSize: TILE.sizePx, cameraMoveStep: TILE.cameraMoveStep });
}
