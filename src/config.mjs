/**
 * config.mjs — Central configuration for isometric pipeline
 *
 * Usage:
 *   import { TILE, PATHS, PROVIDERS, resolvePath } from './config.mjs';
 *
 * All sections support env-var overrides:
 *   e.g. TILE_TILEWAITMS=15000 node render_tiles.mjs
 */

import path from 'path';

// ─── Defaults registry ───────────────────────────────────────────────────

const DEFAULTS = {
  // ─── PATHS ───────────────────────────────────────────────────────────
  PATHS: {
    districts:        './districts',
    water:            './geo/water.geojson',
    infra:            './geo/infra.geojson',
    output:           './output',
    renders:          './output/renders',
    pocStitch:        './output/poc/stitch',
    db:               './output/quadrants.db',
    gridGeojson:      './output/final_grid.geojson',
    manifest:         './output/render_manifest.json',
    quadrantsGeojson: './output/quadrants.geojson',
    quadrantsMeta:    'meta',
    checkpoint:       'checkpoint.json',
    generationConfig: 'generation_config.json',
  },

  // ─── GRID ────────────────────────────────────────────────────────────
  GRID: {
    cellSizeKm:   0.1,
    minWaterM2:   250,
  },

  // ─── TILE (1 tile = 1 image 1024×1024) ────────────────────────────────
  TILE: {
    sizePx:           1024,
    azimuth:           180,
    elevation:         -45,
    altitude:          200,
    sse:              10,
    targetHeight:      0,

    // Settle / wait timing
    tileWaitMs:       12000,
    settlePollMs:     300,
    settleMaxMs:      4000,
    stableHits:       5,
    varianceThr:      250,

    // Blank/blurry thresholds (for analyzeCanvas in html.mjs)
    blankVarianceThr: 600,
    blankEdgeThr:     0.15,
    blankMeanThr:     [60, 110],
    blankSizeKb:      30,
    blankMeanRThr:    250,

    // Retry
    maxRetry:         3,

    // 2D Fallback (when 3D render produces blank/placeholder tile)
    fallback: {
      enabled:        true,
      provider:       'esri',
      maxRetries3D:  3,
      urlTemplate:    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      minZoom:        14,
      requestTimeoutMs: 8000,
      postProcess:    'desat-sepia',
      desatAmount:    0.35,
      sepiaAmount:    0.20,
      contrastBoost:  1.05,
    },

    // Worker session
    sessionMaxMs:     2.9 * 60 * 60 * 1000,
    protocolTimeout:   120_000,

    // Camera move fraction between tiles (0.5 = 50% overlap → stride = 512px)
    cameraMoveStep:    0.5,
    tileStep:         0.5,    // alias for cameraMoveStep (used in generation_config.json)

    // Hash prefix length in tile filename
    hashLength:        8,
  },

  // ─── RENDER pipeline ────────────────────────────────────────────────
  RENDER: {
    blankSizeKb:          30,
    protocolTimeoutMs:    120_000,
    sessionMaxMs:         10 * 60 * 1000,
    maxRetry:             2,

    // Sample grid for analyzeCanvas (WebGL readPixels)
    sampleGridSize:        20,     // 20×20 = 400 samples
    edgeGradThr:           30,

    // Stable frame detection (Cesium postRender)
    requiredStableFrames:   5,
    postRenderExtraFrames:  4,
    postRenderBufferMs:     30,

    // Cesium viewer
    targetFrameRate:      20,
    lightDirection:        { x: 0.5, y: -0.5, z: -0.7 },
  },

  // ─── STITCH defaults ────────────────────────────────────────────────
  STITCH: {
    background:       { r: 255, g: 255, b: 255, alpha: 1 },
    seamColor:         { r: 255, g:   0, b:   0, alpha: 1 },
    seamThicknessPx:   1,
    pairGapColor:      { r: 240, g: 240, b: 240, alpha: 1 },
    debugOnly:         true,
    xaxisAuto:         'auto',
    flipXByDefault:    null,
    outPathTemplate:   'stitch_{N}x{M}_offset{startQx}_{startQy}_{suffix}.png',
  },

  // ─── XAXIS sign detection ───────────────────────────────────────────
  XAXIS: {
    overlap:           512,    // sizePx * (1 - cameraMoveStep) = 1024 * 0.5
    sampleHeight:       1024,
    resizeWidth:         128,
    ratioThreshold:     1.2,
    highConfRatio:      2.0,
    mediumConfRatio:    1.5,
  },

  // ─── WORKERS ────────────────────────────────────────────────────────
  WORKERS: {
    default: 1,
    max:     4,
  },

  // ─── PROVIDERS (tile data source plugins) ──────────────────────────────
  PROVIDERS: {
    available: ['google', 'cesium-ion'],
    default:   'cesium-ion',

    cesiumIon: {
      googleAssetId:     2275207,
      validateUrl:      'https://api.cesium.com/v1/me',
      requestTimeoutMs: 10_000,
    },
  },

  // ─── GOOGLE (legacy — kept for GoogleProvider) ────────────────────
  GOOGLE: {
    rootJsonUrl:        'https://tile.googleapis.com/v1/3dtiles/root.json',
    requestTimeoutMs:   10_000,
  },

  // ─── TMP (worker profile + html paths) ─────────────────────────────
  TMP: {
    dirName:       'isometric-style-convert',
    htmlPrefix:    'tmp_cesium_w',
    profilePrefix: 'chrome_profile_w',
  },

  // ─── CHECKPOINT ────────────────────────────────────────────────────
  CHECKPOINT: {
    schemaVersion:   1,
    flushIntervalMs: 5000,
  },

  // ─── QUALITY thresholds ──────────────────────────────────────────────
  QUALITY: {
    useTileDefaults: true,
  },

  // ─── GEO constants ─────────────────────────────────────────────────
  GEO: {
    mPerDegLat: 111111,  // meters per 1° latitude (WGS84)
  },

  // ─── CESIUM viewer ────────────────────────────────────────────────
  CESIUM: {
    version:             '1.132',
    hiddenUi: [
      'cesium-viewer-toolbar',
      'cesium-viewer-animationContainer',
      'cesium-viewer-timelineContainer',
      'cesium-viewer-bottom',
      'cesium-credit-logoContainer',
      'cesium-credit-textContainer',
      'cesium-viewer-fullscreenContainer',
    ],
    showCreditsOnScreen: false,
  },

  // ─── SEEDS registry (named location presets) ─────────────────────────
  SEEDS: {
    sgn: { lat: 10.78465, lng: 106.70775, label: 'Saigon, District 1' },
    nyc: { lat: 40.7128,  lng: -74.0060,  label: 'New York City' },
  },
};

// ─── Env override helper ────────────────────────────────────────────────

function applyEnvOverrides(config) {
  for (const section of Object.keys(config)) {
    const sectionPrefix = section + '_';
    for (const field of Object.keys(config[section])) {
      const envKey = sectionPrefix + field.toUpperCase();
      const envVal = process.env[envKey];
      if (envVal === undefined) continue;

      const cur = config[section][field];
      try {
        if (typeof cur === 'number') {
          const n = Number(envVal);
          if (!isNaN(n)) config[section][field] = n;
        } else if (typeof cur === 'boolean') {
          config[section][field] = envVal.toLowerCase() === 'true';
        } else if (Array.isArray(cur)) {
          config[section][field] = envVal.split(',').map(s => s.trim());
        } else if (typeof cur === 'object' && cur !== null) {
          try { config[section][field] = JSON.parse(envVal); } catch { /* keep */ }
        } else {
          config[section][field] = envVal;
        }
      } catch { /* ignore parse errors */ }
    }
  }
}

// ─── Singleton config ─────────────────────────────────────────────────

// Project root = parent of src/
const _projectRoot = path.resolve(
  path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
  '..'
);

export const PROJECT_ROOT = _projectRoot;

let _config = null;
export function getConfig() {
  if (_config) return _config;
  _config = JSON.parse(JSON.stringify(DEFAULTS));
  applyEnvOverrides(_config);
  return _config;
}

const _cfg = getConfig();

// ─── Flat section exports (all read from resolved config) ────────────────

export const PATHS      = _cfg.PATHS;
export const GRID       = _cfg.GRID;
export const TILE       = _cfg.TILE;
export const RENDER     = _cfg.RENDER;
export const STITCH     = _cfg.STITCH;
export const XAXIS      = _cfg.XAXIS;
export const WORKERS    = _cfg.WORKERS;
export const GOOGLE     = _cfg.GOOGLE;
export const PROVIDERS  = _cfg.PROVIDERS;
export const TMP        = _cfg.TMP;
export const CHECKPOINT = _cfg.CHECKPOINT;
export const QUALITY    = _cfg.QUALITY;
export const GEO        = _cfg.GEO;
export const CESIUM     = _cfg.CESIUM;
export const SEEDS      = _cfg.SEEDS;

// Convenience: FALLBACK is just TILE.fallback (used by render_tiles.mjs)
export const FALLBACK   = TILE.fallback;

// For debugging / introspection
export const DEFAULTS_ALL = DEFAULTS;

// ─── Derived render constants ──────────────────────────────────────────

export const CELL_SIZE_M  = 200;                    // 200m world / 1 tile
export const QUADRANT_M   = 100;                    // half-tile (for overlap)

export const TILE_SIZE_M  = CELL_SIZE_M;            // 200m world
export const TILE_SIZE_PX = TILE.sizePx;            // 1024px

export const FRUSTUM_W    = TILE_SIZE_M;            // ortho frustum width (200m)

// Camera step: fraction of frustum per tile move
export const TILE_STEP_M  = TILE_SIZE_M * TILE.cameraMoveStep;   // 100m
export const TILE_STEP_PX = TILE_SIZE_PX * TILE.cameraMoveStep;   // 512px

export const M_PER_PX    = TILE_SIZE_M / TILE_SIZE_PX;           // ~0.195 m/px

// ─── Helpers ──────────────────────────────────────────────────────────

export function resolvePath(key) {
  const v = PATHS[key];
  if (typeof v !== 'string') throw new Error(`resolvePath: unknown key '${key}'`);
  return path.isAbsolute(v) ? v : path.resolve(_projectRoot, v);
}

export function getSeeds() {
  return { ...SEEDS };
}

/**
 * Parse CLI --flags from argv.
 * Returns: { flagKey: value } e.g. { provider: 'cesium-ion', workers: '4', fallback: true }
 */
export function parseArgs(argv = process.argv) {
  return Object.fromEntries(
    argv
      .slice(2)
      .filter(a => a.startsWith('--'))
      .map(a => {
        const idx = a.indexOf('=');
        if (idx === -1) return [a.slice(2), true];
        return [a.slice(2, idx), a.slice(idx + 1)];
      })
  );
}
