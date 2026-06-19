import fs from 'fs';
import path from 'path';
import { DATASET_PATHS, SCORE_WEIGHTS, DIVERSITY_CONFIG } from './constants.mjs';

/**
 * Find render file cho (qx, qy).
 * Format: tile_{qx}_{qy}_{hash}.png or tile_+{qx}_+{qy}_{hash}.png
 */
export function findRenderFile(qx, qy) {
  if (!fs.existsSync(DATASET_PATHS.sourceRenders)) return null;
  const signQ = qx >= 0 ? '+' + qx : String(qx);
  const signY = qy >= 0 ? '+' + qy : String(qy);
  const prefix = `tile_${signQ}_${signY}_`;

  const files = fs.readdirSync(DATASET_PATHS.sourceRenders)
    .filter(f => f.startsWith(prefix) && f.endsWith('.png'));

  return files[0] ?? null;
}

/**
 * Load metadata JSON. Return array of tile objects.
 */
export async function loadAllMetaTiles() {
  const metaDir = DATASET_PATHS.sourceMeta;
  if (!fs.existsSync(metaDir)) {
    throw new Error(`Cant find meta dir: ${metaDir}`);
  }

  const files = fs.readdirSync(metaDir).filter(f => f.endsWith('.json'));
  const tiles = [];

  for (const f of files) {
    const metaPath = path.join(metaDir, f);
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    if (meta.qx === undefined || meta.qy === undefined) continue;

    tiles.push({
      qx:          meta.qx,
      qy:          meta.qy,
      hash:        meta.hash_sha256 ?? null,
      variance:    meta.variance    ?? 0,
      edgeDensity: meta.edgeDensity ?? 0,
      meanR:       meta.meanR       ?? 0,
      meanG:       meta.meanG       ?? 0,
      meanB:       meta.meanB       ?? 0,
      sizeKb:      meta.size_kb     ?? 0,
      isBlank:     meta.isBlank     ?? false,
      lat:         meta.lat         ?? null,
      lng:         meta.lng         ?? null,
      savedAt:     meta.saved_at    ?? null,
      renderFile:  findRenderFile(meta.qx, meta.qy),
      metaFile:    f,
    });
  }

  return tiles;
}

export function scoreTile(tile) {
  const { variance, edgeDensity, sizeKb } = SCORE_WEIGHTS;
  const v = Math.min((tile.variance    ?? 0) / 10000, 1);
  const e = Math.min((tile.edgeDensity ?? 0), 1);
  const s = Math.min((tile.sizeKb      ?? 0) / 2500, 1);
  return v * variance + e * edgeDensity + s * sizeKb;
}

/**
 * Filter
 */
export function filterQuality(tiles) {
  const { minVariance, minEdgeDensity } = DIVERSITY_CONFIG;
  return tiles.filter(t =>
    !t.isBlank &&
    t.variance    >= minVariance &&
    t.edgeDensity >= minEdgeDensity &&
    t.renderFile !== null
  );
}

/**
 * Quick stats
 */
export function summarizeTiles(tiles) {
  if (tiles.length === 0) {
    return { count: 0, variance: 0, edgeDensity: 0, sizeKb: 0 };
  }
  const sum = (key) => tiles.reduce((a, t) => a + (t[key] ?? 0), 0);
  return {
    count:       tiles.length,
    avgVariance: sum('variance')    / tiles.length,
    avgEdge:     sum('edgeDensity') / tiles.length,
    avgSizeKb:   sum('sizeKb')      / tiles.length,
    minVariance: Math.min(...tiles.map(t => t.variance ?? 0)),
    maxVariance: Math.max(...tiles.map(t => t.variance ?? 0)),
    blankCount:  tiles.filter(t => t.isBlank).length,
  };
}