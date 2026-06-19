/**
 * tile/stitch/io.mjs — High-level helpers: stitchGrid (one-shot), savePng
 *
 * stitchGrid: stitch + save trong 1 call. Phù hợp cho scripts muốn ghép nhanh.
 * savePng: wrapper quanh sharp.png().toFile().
 */

import sharp from 'sharp';
import { stitchTiles } from './compose.mjs';
import { computeLayout } from './layout.mjs';
import { annotateSeams } from './annotate.mjs';

/**
 * Lưu sharp instance ra file PNG.
 *
 * @param {sharp.Sharp} img
 * @param {string} filepath
 * @returns {Promise<{size: number}>} Kích thước file (bytes)
 */
export async function savePng(img, filepath) {
  const info = await img.png().toFile(filepath);
  return { size: info.size };
}

/**
 * One-shot helper: stitch + (optional) annotate + save trong 1 call.
 *
 * @param {Object} opts
 * @param {Array<{r,c,png}>} opts.tiles
 * @param {number} opts.gridSize   - N (cho N×N)
 * @param {number} opts.tileSize   - tile px
 * @param {number} opts.stride     - pixel offset giữa 2 tile
 * @param {string} opts.outPath    - đường dẫn file PNG output
 * @param {boolean} [opts.annotate] - vẽ seam line đỏ
 * @param {boolean} [opts.flipX]   - lật ngang từng tile
 * @param {Object} [opts.background] - RGBA
 * @param {number} [opts.gridCols] - số cột (mặc định = gridSize)
 * @param {number} [opts.gridRows] - số hàng (mặc định = gridSize)
 * @returns {Promise<{outPath, size, mapW, mapH}>}
 */
export async function stitchGrid({
  tiles, gridSize, tileSize, stride, outPath,
  annotate, flipX, background, gridCols, gridRows,
}) {
  const cols = gridCols ?? gridSize;
  const rows = gridRows ?? gridSize;

  const base = await stitchTiles({
    tiles, gridSize, tileSize, stride, background,
    gridCols: cols, gridRows: rows, flipX,
  });

  const layout = computeLayout({ gridSize, tileSize, stride, gridCols: cols, gridRows: rows });

  let result = base;
  if (annotate) {
    // sharp instance chỉ composite 1 lần — toBuffer() trước rồi sharp(buffer) lại
    const baseBuf = await base.png().toBuffer();
    const baseCloned = sharp(baseBuf);
    result = await annotateSeams(baseCloned, layout);
  }

  const { size } = await savePng(result, outPath);
  return { outPath, size, mapW: layout.mapW, mapH: layout.mapH };
}
