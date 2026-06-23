import fs from 'fs';
import { TILE, GEO } from '../config.mjs';

export async function exportWorldLayoutPlan({ tiles, outJsonPath, padding = 100 }) {
  if (!Array.isArray(tiles) || tiles.length === 0) {
    throw new Error('exportWorldLayoutPlan: tiles must be a non-empty array');
  }

  const _px = TILE.sizePx;                            // 1024
  const _m  = 200;                                    // world width
  const PX_PER_M = _px / _m;                          // 5.12 px/m
  const mPerDegLat = GEO.mPerDegLat;                  // 111111

  // 1. Calc bounding box
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;
  for (const t of tiles) {
    if (t.lat < minLat) minLat = t.lat;
    if (t.lat > maxLat) maxLat = t.lat;
    if (t.lng < minLng) minLng = t.lng;
    if (t.lng > maxLng) maxLng = t.lng;
  }

  const mPerDegLng = GEO.mPerDegLat * Math.cos((minLat + maxLat) / 2 * Math.PI / 180);
  const widthM  = (maxLng - minLng) * mPerDegLng + _m;
  const heightM = (maxLat - minLat) * mPerDegLat + _m;
  
  const mapW = Math.ceil(widthM * PX_PER_M) + 2 * padding;
  const mapH = Math.ceil(heightM * PX_PER_M) + 2 * padding;

  // 2. Layout
  const layoutPlan = {
    canvasWidth: mapW,
    canvasHeight: mapH,
    tiles: []
  };

  // 3. pixel offset for each tile
  for (const t of tiles) {
    if (!t.path) {
        throw new Error(`Tile at (${t.lat}, ${t.lng}) missing 'path'`);
    }

    const dx = Math.round((t.lng - minLng) * mPerDegLng * PX_PER_M) + padding;
    const dy = Math.round((maxLat - t.lat) * mPerDegLat * PX_PER_M) + padding;
    
    layoutPlan.tiles.push({
      path: t.path, 
      x: dx,
      y: dy
    });
  }

  // 4. Ghi file JSON xuống ổ cứng
  fs.writeFileSync(outJsonPath, JSON.stringify(layoutPlan, null, 2), 'utf-8');
  console.log(`Export plan at: ${outJsonPath}`);
  console.log(`Canvas Size: ${mapW} x ${mapH} pixels`);
  
  return { outJsonPath, mapW, mapH, tileCount: layoutPlan.tiles.length };
}

