import fs from 'fs';
import path from 'path';
import { TILE, DZI } from '../config.mjs';

export async function exportWorldLayoutPlan({ tiles, outJsonPath, padding }) {
  if (!Array.isArray(tiles) || tiles.length === 0) {
    throw new Error('exportWorldLayoutPlan: tiles must be a non-empty array');
  }

  const stepPx = TILE.sizePx * TILE.cameraMoveStep;
  const padPx = padding ?? DZI.paddingPx;

  // Parse qx, qy
  function parseQxy(tilePath) {
    const m = path.basename(tilePath).match(/tile_([+-]\d+)_([+-]\d+)/);
    if (!m) throw new Error(`Cannot parse qx/qy from: ${tilePath}`);
    return { qx: parseInt(m[1]), qy: parseInt(m[2]) };
  }

  // 1. Bounding box in grid space
  let minQx = Infinity, maxQx = -Infinity;
  let minQy = Infinity, maxQy = -Infinity;

  for (const t of tiles) {
    const { qx, qy } = parseQxy(t.path);
    if (qx < minQx) minQx = qx;
    if (qx > maxQx) maxQx = qx;
    if (qy < minQy) minQy = qy;
    if (qy > maxQy) maxQy = qy;
  }

  // 2. Canvas size
  const mapW = (maxQx - minQx + 1) * stepPx + 2 * padPx;
  const mapH = (maxQy - minQy + 1) * stepPx + 2 * padPx;

  const layoutPlan = { canvasWidth: mapW, canvasHeight: mapH, tiles: [] };

  // 3. Map qx/qy → canvas pixel
  for (const t of tiles) {
    if (!t.path) throw new Error(`Tile missing 'path'`);
    const { qx, qy } = parseQxy(t.path);

    const dx = (qx - minQx) * stepPx + padPx;
    const dy = (maxQy - qy) * stepPx + padPx;

    layoutPlan.tiles.push({ path: t.path, x: dx, y: dy });
  }

  fs.writeFileSync(outJsonPath, JSON.stringify(layoutPlan, null, 2), 'utf-8');
  console.log(`Export plan at: ${outJsonPath}`);
  console.log(`Canvas Size: ${mapW} x ${mapH} pixels`);

  return { outJsonPath, mapW, mapH, tileCount: layoutPlan.tiles.length };
}