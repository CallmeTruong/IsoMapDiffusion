import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DATASET_PROJECT_ROOT = path.resolve(__dirname, '..', '..');

function toAbs(p) {
  return path.isAbsolute(p) ? p : path.resolve(DATASET_PROJECT_ROOT, p);
}

// ─── Paths ──────────────────────────────────────────────────────────────────

export const DATASET_PATHS = {
  root:           toAbs('./output/dataset'),
  rawTiles:       toAbs('./output/dataset/raw_tiles'),
  aiGen:          toAbs('./output/dataset/ai_gen'),
  reportJson:     toAbs('./output/dataset/selection_report.json'),
  reportCsv:      toAbs('./output/dataset/selection_report.csv'),
  registry:       toAbs('./output/dataset/tile_registry.json'),
  zip:            toAbs('./output/dataset/ai_job.zip'),
  sourceRenders:  toAbs('./output/renders'),
  sourceMeta:     toAbs('./output/renders/meta'),
};

// ─── Scoring weights ────────────────────────────────────────────────────────
// Composite score = variance * w1 + edgeDensity_norm * w2 + sizeKb_norm * w3
// Tổng = 1.0

export const SCORE_WEIGHTS = {
  variance:       0.40,
  edgeDensity:    0.40,
  sizeKb:         0.20,
};

// ─── Diversity config ────────────────────────────────────────────────────────

export const DIVERSITY_CONFIG = {
  gridDivisions:    4,
  minTilesPerZone:  1,
  maxTilesPerZone:  3,
  colorBuckets:     6,
  minVariance:      2000,
  minEdgeDensity:   0.4,
};

// ─── Selection defaults ─────────────────────────────────────────────────────

export const DEFAULT_TARGET = 80;

// ─── Import rules ───────────────────────────────────────────────────────────

export const IMPORT_RULES = {
  filenamePattern: /^tile_(-?\d+)_(-?\d+)\.png$/i,
  minFileSizeKb:    20,
  acceptedExts:     ['.png', '.jpg', '.jpeg', '.webp'],
  sidecarSuffix:    '.png.json',
};


export const TILE_RENDER = {
  sizePx:         1024,
  cameraMoveStep: 0.5,
  cameraAzimuth:  180,
  cameraElev:     -45,
  cameraAlt:      2000,
};

// ─── Re-exports ─────────────────────────────────────────────────────────────

export { DATASET_PROJECT_ROOT as PROJECT_ROOT };