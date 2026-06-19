/**
 * config.mjs — Central configuration for isometric pipeline
 */

import { cpus } from 'os';
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
    quadrantsMeta:    'meta',                   // sub-folder for tile meta JSONs
    checkpoint:       'checkpoint.json',         // checkpoint filename
    generationConfig: 'generation_config.json',  // generation config filename
  },

  // ─── GRID ────────────────────────────────────────────────────────────
  GRID: {
    cellSizeKm:   0.1,
    minWaterM2:   250,
  },

  // ─── TILE (1 tile = 1 image 1024×1024) ────────────────────────────────
  TILE: {
    sizePx:           1024,
    azimuth:          180,
    elevation:        -45,
    altitude:         200,
    sse:              10,
    targetHeight:     0,

    // Settle / wait timing
    tileWaitMs:       12000,
    settlePollMs:     300,
    settleMaxMs:      4000,
    stableHits:       5,
    varianceThr:      250,

    // Blank/blurry thresholds (cho analyzeCanvas)
    blankVarianceThr: 800,
    blankEdgeThr:     0.15,
    blankMeanThr:     [60, 110],
    blankSizeKb:      30,
    blankMeanRThr:    250,    // > 250 = skydome / flat white

    // Retry
    maxRetry:         3,

    // Worker session
    sessionMaxMs:     2.9 * 60 * 60 * 1000,
    protocolTimeout:  120_000,

    // Camera step
    cameraMoveStep:   0.5,

    // Hash
    hashLength:       8,      // SHA-256 prefix length in filename
  },

  // ─── RENDER pipeline ────────────────────────────────────────────────
  RENDER: {
    // Worker pool
    blankSizeKb:        30,     // PNG size threshold (KB)
    protocolTimeoutMs:  120_000,
    sessionMaxMs:       10 * 60 * 1000,  // min per worker session
    maxRetry:           2,

    // Sample grid cho analyzeCanvas (WebGL readPixels)
    sampleGridSize:     20,     // 20×20 = 400 samples
    edgeGradThr:        30,     // gradient threshold per channel (0-255)

    // Stable frame detection (Cesium postRender)
    requiredStableFrames: 5,
    postRenderExtraFrames: 4,
    postRenderBufferMs:  30,

    // Cesium viewer
    targetFrameRate:    20,
    lightDirection:     { x: 0.5, y: -0.5, z: -0.7 },
  },

  // ─── STITCH defaults ────────────────────────────────────────────────
  STITCH: {
    // Default colors
    background:         { r: 255, g: 255, b: 255, alpha: 1 },
    seamColor:          { r: 255, g:   0, b:   0, alpha: 1 },
    seamThicknessPx:    1,
    pairGapColor:       { r: 240, g: 240, b: 240, alpha: 1 },

    // CLI defaults
    debugOnly:          true,
    xaxisAuto:          'auto',  // 'auto' | 'east' | 'west' | 'off'
    flipXByDefault:     null,    // null = auto-detect, 'east'/'west' = force

    // Output path template
    outPathTemplate:    'stitch_{N}x{M}_offset{startQx}_{startQy}_{suffix}.png',
  },

  // ─── XAXIS sign detection ───────────────────────────────────────────
  XAXIS: {
    overlap:            512,    // overlap px (= sizePx * (1 - cameraMoveStep))
    sampleHeight:       1024,
    resizeWidth:        128, 
    ratioThreshold:     1.2,    // ratio > 1.2 = east, < 1/1.2 = west
    highConfRatio:      2.0,    // ratio > N = high confidence
    mediumConfRatio:    1.5,    // ratio > N = medium confidence
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

    // Cesium Ion specifics
    cesiumIon: {
      // Google Photorealistic 3D Tiles on Cesium Ion
      googleAssetId:     2275207,
      validateUrl:      'https://api.cesium.com/v1/me',
      requestTimeoutMs: 10_000,
    },
  },

  // ─── GOOGLE API (legacy — change PROVIDERS plugin) ──────────────
  GOOGLE: {
    rootJsonUrl:        'https://tile.googleapis.com/v1/3dtiles/root.json',
    requestTimeoutMs:   10_000,
  },

  // ─── TMP (worker profile + html paths) ─────────────────────────────
  TMP: {
    dirName:           'isometric-style-convert',
    htmlPrefix:        'tmp_cesium_w',
    profilePrefix:     'chrome_profile_w',
  },

  // ─── CHECKPOINT ────────────────────────────────────────────────────
  CHECKPOINT: {
    schemaVersion:     1,
    flushIntervalMs:   5000,
  },

  // ─── QUALITY thresholds (for getThresholds helper) ─────────────────
  QUALITY: {
    useTileDefaults:    true,
  },

  // ─── GEO constants ─────────────────────────────────────────────────
  GEO: {
    mPerDegLat:        111111,  // meters per 1° latitude (constant)
  },

  // ─── CESIUM ────────────────────────────────────────────────────────
  CESIUM: {
    version:           '1.132',
    // UI elements
    hiddenUi:          [
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

  // ─── SEEDS registry (location points) ──────────────────────────────
  SEEDS: {
    sgn: { lat: 10.78465, lng: 106.70775, label: 'Saigon, District 1' },
    nyc: { lat: 40.7128,  lng: -74.0060,  label: 'New York City' },
    // Add more locations here
  },
};

// ─── Env override helper ────────────────────────────────────────────────

function applyEnvOverrides(config, projectRoot) {
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
          // Try JSON parse
          try { config[section][field] = JSON.parse(envVal); } catch { /* keep */ }
        } else {
          config[section][field] = envVal;
        }
      } catch { /* ignore parse errors */ }
    }
  }

}

// ─── Singleton export ──────────────────────────────────────────────────

// Project root = parent of src/
const PROJECT_ROOT_DEFAULT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');

/**
 * Public project root
 */
export const PROJECT_ROOT = PROJECT_ROOT_DEFAULT;

// Lazy: build config
let _config = null;
export function getConfig(projectRoot = PROJECT_ROOT_DEFAULT) {
  if (_config) return _config;
  _config = JSON.parse(JSON.stringify(DEFAULTS));
  applyEnvOverrides(_config, projectRoot);
  return _config;
}

const _cfg = getConfig();

// ─── Legacy flat exports (backward compat) ─────────────────────────────

export const PATHS = _cfg.PATHS;
export const GRID = _cfg.GRID;
export const TILE = _cfg.TILE;
export const RENDER = _cfg.RENDER;
export const STITCH = _cfg.STITCH;
export const XAXIS = _cfg.XAXIS;
export const WORKERS = _cfg.WORKERS;
export const GOOGLE = _cfg.GOOGLE;
export const PROVIDERS = _cfg.PROVIDERS;
export const BILLING = _cfg.BILLING;
export const TMP = _cfg.TMP;
export const CHECKPOINT = _cfg.CHECKPOINT;
export const QUALITY = _cfg.QUALITY;
export const GEO = _cfg.GEO;
export const CESIUM = _cfg.CESIUM;
export const SEEDS = _cfg.SEEDS;
export const DEFAULTS_ALL = DEFAULTS;

// ─── Derived constants (backward compat) ──────────────────────────────

export const CELL_SIZE_M     = 200;                            // 200m world / 1 tile
export const QUADRANT_M      = 100;                            // 100m

// Size 1 tile
export const TILE_SIZE_M     = CELL_SIZE_M;                    // 200m world / 1 tile
export const TILE_SIZE_PX    = TILE.sizePx;                    // 1024px

// Frustum width
export const FRUSTUM_W       = TILE_SIZE_M;                    // 200m

// Camera move (= 0.5 * 200m = 100m)
export const TILE_STEP_M     = TILE_SIZE_M * TILE.cameraMoveStep;

// Camera move pixel (= 0.5 * 1024 = 512px)
export const TILE_STEP_PX    = TILE_SIZE_PX * TILE.cameraMoveStep;

// Meters per pixel
export const M_PER_PX        = TILE_SIZE_M / TILE_SIZE_PX;     // ~0.195 m/px

// ─── Helpers ──────────────────────────────────────────────────────────

export function resolvePath(key) {
  const v = PATHS[key];
  if (typeof v !== 'string') {
    throw new Error(`resolvePath: unknown key '${key}'`);
  }
  return path.isAbsolute(v) ? v : path.resolve(PROJECT_ROOT_DEFAULT, v);
}

export function getSeeds() {
  return { ...SEEDS };
}

export function addSeed(key, { lat, lng, label }) {
  SEEDS[key] = { lat, lng, label };
}

// ─── Google API Keys ───────────────────────────────────────────────────

export function getGoogleKeys() {
  const env = process.env.GOOGLE_KEY ?? '';
  return env.split(',').map(k => k.trim()).filter(Boolean);
}

// ─── CLI argument parser (backward compat) ────────────────────────────

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
