/**
 * tile/stitch/io.mjs — High-level helpers: stitchGrid (one-shot), savePng.
 *
 * stitchGrid: stitch + save in one call. Convenient for scripts that want a quick grid stitch.
 * savePng: thin wrapper around sharp.png().toFile().
 */

import sharp from 'sharp';
import { stitchTiles } from './compose.mjs';
import { computeLayout } from './layout.mjs';
import { annotateSeams } from './annotate.mjs';

/**
 * Save a sharp instance to a PNG file.
 *
 * @param {sharp.Sharp} img
 * @param {string} filepath
 * @returns {Promise<{size: number}>} File size in bytes
 */
export async function savePng(img, filepath) {
  const info = await img.png().toFile(filepath);
  return { size: info.size };
}

/**
 * One-shot helper: stitch + (optional) annotate + save in one call.
 *
 * @param {Object} opts
 * @param {Array<{r,c,png}>} opts.tiles
 * @param {number} opts.gridSize   - N (for N×N)
 * @param {number} opts.tileSize   - tile size in px
 * @param {number} opts.stride     - pixel offset between two tiles
 * @param {string} opts.outPath    - output PNG file path
 * @param {boolean} [opts.annotate] - draw red seam line
 * @param {boolean} [opts.flipX]   - flip each tile horizontally
 * @param {Object} [opts.background] - RGBA
 * @param {number} [opts.gridCols] - number of columns (default = gridSize)
 * @param {number} [opts.gridRows] - number of rows (default = gridSize)
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
    // A sharp instance can only composite once — toBuffer() first then re-wrap in sharp(buffer)
    const baseBuf = await base.png().toBuffer();
    const baseCloned = sharp(baseBuf);
    result = await annotateSeams(baseCloned, layout);
  }

  const { size } = await savePng(result, outPath);
  return { outPath, size, mapW: layout.mapW, mapH: layout.mapH };
}
