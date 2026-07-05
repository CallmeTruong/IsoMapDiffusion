/**
 * tile/stitch/world.mjs — Stitch tile theo world coords (không cần grid vuông)
 *
 * stitchWorld: ghép tile thưa thớt (lat, lng) → 1 ảnh lớn đúng vị trí thế giới.
 * Khác stitchGrid: tile có thể rải rác, vị trí pixel tính từ lat/lng.
 *
 * Với cameraMoveStep=1.0, 1 tile = 200m×200m world, 1px = 200m/1024.
 */

import sharp from 'sharp';
import { TILE, GEO, STITCH } from '../../config.mjs';
import { savePng } from './io.mjs';
import { mPerDegLng, M_PER_DEG_LAT } from '../../utils/geo.mjs';

const DEFAULT_BG = STITCH.background;

/**
 * Stitch tile thưa thớt theo world coords (lat, lng).
 *
 * @param {Object} opts
 * @param {Array<{lat: number, lng: number, png: Buffer|string|sharp.Sharp}>} opts.tiles
 * @param {string} opts.outPath
 * @param {{r,g,b,alpha}} [opts.background]
 * @param {number} [opts.padding] - Padding px xung quanh map (default STITCH.worldPaddingPx)
 * @returns {Promise<{outPath, size, mapW, mapH, placements}>}
 */
export async function stitchWorld({ tiles, outPath, background, padding }) {
  if (!Array.isArray(tiles) || tiles.length === 0) {
    throw new Error('stitchWorld: tiles must be a non-empty array');
  }

  const bg = background ?? DEFAULT_BG;
  const _px = TILE.sizePx;                                // 1024
  const _m  = TILE.tileWorldSizeM;                        // world width mỗi tile
  const PX_PER_M = _px / _m;
  const padPx = padding ?? STITCH.worldPaddingPx;

  // 1. Tính bounding box world
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;
  for (const t of tiles) {
    if (t.lat < minLat) minLat = t.lat;
    if (t.lat > maxLat) maxLat = t.lat;
    if (t.lng < minLng) minLng = t.lng;
    if (t.lng > maxLng) maxLng = t.lng;
  }

  const mPerDegLngAtMid = mPerDegLng((minLat + maxLat) / 2);
  const widthM  = (maxLng - minLng) * mPerDegLngAtMid + _m;
  const heightM = (maxLat - minLat) * M_PER_DEG_LAT + _m;
  const mapW = Math.ceil(widthM * PX_PER_M) + 2 * padPx;
  const mapH = Math.ceil(heightM * PX_PER_M) + 2 * padPx;

  // 2. Tile → pixel offset (Y đảo vì screen Y hướng xuống, world North lên trên)
  const placements = [];
  const composites = [];
  for (const t of tiles) {
    const dx = Math.round((t.lng - minLng) * mPerDegLngAtMid * PX_PER_M) + padPx;
    const dy = Math.round((maxLat - t.lat) * M_PER_DEG_LAT * PX_PER_M) + padPx;
    placements.push({ lat: t.lat, lng: t.lng, dx, dy });
    composites.push({ input: t.png, left: dx, top: dy });
  }

  // 3. Tạo base canvas
  const baseBuf = Buffer.alloc(mapW * mapH * 4);
  for (let i = 0; i < baseBuf.length; i += 4) {
    baseBuf[i + 0] = bg.r;
    baseBuf[i + 1] = bg.g;
    baseBuf[i + 2] = bg.b;
    baseBuf[i + 3] = Math.round(bg.alpha * 255);
  }
  const base = sharp(baseBuf, { raw: { width: mapW, height: mapH, channels: 4 } });
  const result = base.composite(composites);
  const { size } = await savePng(result, outPath);
  return { outPath, size, mapW, mapH, placements };
}
