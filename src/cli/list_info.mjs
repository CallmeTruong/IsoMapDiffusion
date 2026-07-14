import fs from 'fs';
import path from 'path';

import { TILE, TILE_SIZE_M, CELL_SIZE_M, PATHS, SEEDS, PROVIDERS, resolvePath } from '../config.mjs';
import { listTiles, tileBounds } from '../tile/tile_io.mjs';
import { listProviders, getProvider, resolveApiKey } from '../tile/provider/index.mjs';

export async function cmdList({ flags, projectRootDir }) {
  const outputDir = path.resolve(projectRootDir, flags.output ?? resolvePath('renders'));
  if (!fs.existsSync(outputDir)) {
    console.log(`[list] No output dir: ${outputDir}`);
    return;
  }
  const tiles = listTiles(outputDir);
  const bounds = tileBounds(tiles);
  console.log(`\n─── Tiles in ${outputDir} ───`);
  console.log(`Count: ${tiles.length}`);
  console.log(`Bounds: qx=[${bounds.minQx}..${bounds.maxQx}], qy=[${bounds.minQy}..${bounds.maxQy}]`);
  console.log(`Expected: ${bounds.expectedTiles}, missing: ${bounds.expectedTiles - bounds.count}`);
  for (const t of tiles.sort((a, b) => (a.qx - b.qx) || (a.qy - b.qy))) {
    const sizeKB = t.meta?.size_kb ?? '?';
    const lat = t.meta?.lat?.toFixed(6) ?? '?';
    const lng = t.meta?.lng?.toFixed(6) ?? '?';
    console.log(`  (${t.qx >= 0 ? '+' : ''}${t.qx},${t.qy >= 0 ? '+' : ''}${t.qy}) ${t.hash} ${sizeKB}KB  lat=${lat} lng=${lng}`);
  }
}

export function cmdInfo({ projectRootDir }) {
  console.log(`\n─── Pipeline Config ───`);
  console.log(`TILE.sizePx:         ${TILE.sizePx}`);
  console.log(`TILE.azimuth:        ${TILE.azimuth}°`);
  console.log(`TILE.elevation:      ${TILE.elevation}°`);
  console.log(`TILE.altitude:       ${TILE.altitude}m`);
  console.log(`TILE.cameraMoveStep: ${TILE.cameraMoveStep} (= ${Math.round(TILE.sizePx * TILE.cameraMoveStep)}px stride, ${TILE.sizePx * TILE.cameraMoveStep * (TILE_SIZE_M/TILE.sizePx)}m camera step)`);
  console.log(`TILE_SIZE_M:         ${TILE_SIZE_M}m`);
  console.log(`CELL_SIZE_M:         ${CELL_SIZE_M}m (= 1 tile)`);
  console.log(`Seed registry:       ${Object.keys(SEEDS).join(', ')}`);
  console.log(`\n─── Tile Providers ───`);
  console.log(`Default:             ${PROVIDERS.default}`);
  for (const p of listProviders()) {
    const key = resolveApiKey(p.id);
    const status = key ? '✓' : '✗';
    console.log(`  [${status}] ${p.id}: ${p.displayName} (env=${p.envVar})`);
  }
  console.log(`\n─── Legacy Env ───`);
  console.log(`GOOGLE_KEY env:      ${process.env.GOOGLE_KEY ? `...${process.env.GOOGLE_KEY.slice(-6)}` : 'NOT SET'}`);
  console.log(`Project root:        ${projectRootDir}`);
}
