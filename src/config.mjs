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
    cellsPerTile: 2,        // number of GRID cells per rendered TILE (matches cameraMoveStep=0.5 → 2×2 grid → 1 tile)
  },

  // ─── TILE (1 tile = 1 image 1024×1024) ────────────────────────────────
  TILE: {
    sizePx:           1024,
    azimuth:          180,
    elevation:        -45,
    altitude:         200,
    sse:              10,
    targetHeight:     0,

    // World-space footprint of a single tile (meters).
    // With cameraMoveStep=1.0, 1 tile ≈ 200m×200m; with 0.5, ≈ 100m×100m.
    tileWorldSizeM:   200,

    // Settle / wait timing
    tileWaitMs:       25000,
    settlePollMs:     300,
    settleMaxMs:      4000,
    stableHits:       5,
    varianceThr:      250,

    // Blank/blurry thresholds (for analyzeCanvas)
    blankVarianceThr: 600,
    blankEdgeThr:     0.15,
    blankMeanThr:     [60, 110],
    blankSizeKb:      30,
    blankMeanRThr:    250,

    // Retry
    maxRetry:         3,
    retryAttempts:    2,
    retryTimeoutMs:   45000,
    retrySessionMs:   120000,
    retryBackoffMs:   500,
    retryMaxRounds:   3,
    retryConcurrencyMax: 8,
    retryPassWaitBonusMs: 3000,
    retryPassWaitMs:   3000,
    retryPassCount:    3,
    retryPassExtraRounds: 2,
    retrySleepMs:      500,
    retrySleepTopMs:   1500,

    // 2D Fallback (for tiles where 3D render fails/blank)
    fallback: {
      enabled:        true,    // Enable 2D satellite fallback
      provider:      'esri',   // esri | mapbox | osm
      maxRetries3D:  3,       // Max 3D retries before fallback
      urlTemplate:   'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      minZoom:       17,       // zoom level used to compute slippy x/y for the fallback fetch (z17 ≈ 150m/tile, matches ~100m quadrants)
      defaultMinZoom: 14,      // fallback when caller doesn't provide minZoom
      minResponseBytes: 100,   // buf.length threshold to consider a fallback response valid
      requestTimeoutMs: 8000,  // fetch timeout for the fallback tile request
      userAgent:     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
      // Post-process look (see postProcess2D in fallback_2d.mjs) — defaults
      postProcess:    'desat-sepia', // 'desat-sepia' | 'none'
      desatAmount:    0.35,
      sepiaAmount:    0.20,
      contrastBoost:  1.05,
    },

    // Worker session
    sessionMaxMs:     2.9 * 60 * 60 * 1000,
    protocolTimeout:  120_000,
    createPageTimeoutMs: 60_000,
    waitFunctionTimeoutMs: 30_000,
    waitPollMs:      200,
    maxPageRecoveries: 3,
    sessionWarmupMs:  2000,

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
    tilesetWaitMs:      30_000,

    // Sample grid for analyzeCanvas (WebGL readPixels)
    sampleGridSize:     20,     // 20×20 = 400 samples
    edgeGradThr:        30,     // gradient threshold per channel (0-255)

    // Stable frame detection (Cesium postRender)
    requiredStableFrames: 5,
    postRenderExtraFrames: 4,
    postRenderBufferMs:  30,

    // Cesium viewer
    targetFrameRate:    20,
    lightDirection:     { x: 0.5, y: -0.5, z: -0.7 },

    // Blank-image detection (placeholders, corner sampling in html.mjs)
    placeholderCornerWhiteThr: 250,
    placeholderCornerDarkThr:  10,
    placeholderMeanThr:        [60, 110],
    placeholderVarianceThr:    600,
    placeholderEdgeThr:        0.15,
    googlePlaceholderBg:       { r: 245, g: 245, b: 245 },
    googlePlaceholderTol:      15,
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

    // Padding (px) around stitched maps (DZI, world stitch)
    paddingPx:          100,
    worldPaddingPx:     100,
  },

  // ─── DZI export ────────────────────────────────────────────────────
  DZI: {
    paddingPx:          100,
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
    mPerDegLat:          111111,         // meters per 1° latitude (constant)
    earthCircumferenceM: 40075016.686,   // WGS84 equatorial circumference (m)
    mercatorLatClipDeg:  85.05112878,    // Web Mercator latitude clip
    minZoom:             0,              // slippy tile zoom lower bound
    maxZoom:             22,             // slippy tile zoom upper bound
    degToRad:            Math.PI / 180,  // 1° in radians
    radToDeg:            180 / Math.PI,  // 1 rad in degrees
    twoPi:               2 * Math.PI,
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
export const FALLBACK = _cfg.TILE?.fallback ?? { enabled: false };
export const DEFAULTS_ALL = DEFAULTS;

// ─── Derived constants (backward compat) ──────────────────────────────
//
// These are computed at module-load from the live config and remain plain
// numbers so existing call-sites (`CELL_SIZE_M + offset`, `TILE_STEP_M * 2`)
// continue to work as before.
//
// If a JSON overlay or env override is applied *after* this module has been
// imported, the new values land in `getConfig().TILE.*`. Callers that need
// to react to runtime overrides should read `TILE.tileWorldSizeM` etc. directly.

export const CELL_SIZE_M     = TILE.tileWorldSizeM;               // 200m world / 1 tile
export const QUADRANT_M      = TILE.tileWorldSizeM / 2;           // 100m (half-tile when cameraMoveStep=0.5)
export const TILE_SIZE_M     = CELL_SIZE_M;                        // alias
export const TILE_SIZE_PX    = TILE.sizePx;                        // 1024px
export const FRUSTUM_W       = TILE_SIZE_M;                        // alias
export const TILE_STEP_M     = TILE.tileWorldSizeM * TILE.cameraMoveStep;
export const TILE_STEP_PX    = TILE.sizePx * TILE.cameraMoveStep;
export const M_PER_PX        = TILE.tileWorldSizeM / TILE.sizePx;  // ~0.195 m/px

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

// ─── JSON file overlay loader ─────────────────────────────────────────────
//
// Reads `output/generation_config.json` (or any other path) and merges its
// keys into the live config. Run this BEFORE starting the pipeline so
// `cmdInfo()` and every consumer see the overridden values.
//
// Load order (later overrides earlier):
//   1. DEFAULTS (in this file)
//   2. process.env (TILE_CAMERAMOVESTEP=0.6, etc.)
//   3. generation_config.json via loadConfigFromJson() + mergeJsonConfig()

import fs from 'fs';

/**
 * Read & parse a JSON config file. Returns null if the file is missing or
 * unreadable; callers should treat null as "no overlay".
 *
 * @param {string} jsonPath
 * @returns {object|null}
 */
export function loadConfigFromJson(jsonPath) {
  if (!jsonPath || !fs.existsSync(jsonPath)) return null;
  try {
    const raw = fs.readFileSync(jsonPath, 'utf8');
    return JSON.parse(raw);
  } catch (e) {
    console.warn(`[config] failed to parse ${jsonPath}: ${e.message}`);
    return null;
  }
}

/**
 * Merge a parsed generation_config.json object into the live config.
 * Maps the snake_case keys emitted by render/export.mjs to the camelCase
 * keys we use internally.
 *
 * @param {object} jsonCfg - parsed JSON
 * @returns {string[]} list of dotted keys that were overridden (for logging)
 */
export function mergeJsonConfig(jsonCfg) {
  if (!jsonCfg || typeof jsonCfg !== 'object') return [];
  const cfg = getConfig();
  const map = {
    'camera_azimuth_degrees':   ['TILE', 'azimuth'],
    'camera_elevation_degrees': ['TILE', 'elevation'],
    'camera_altitude_m':        ['TILE', 'altitude'],
    'camera_move_step':         ['TILE', 'cameraMoveStep'],
    'width_px':                 ['TILE', 'sizePx'],
    'height_px':                ['TILE', 'sizePx'],
    'view_height_meters':       ['TILE', 'tileWorldSizeM'],
    'tile_size_m':              ['TILE', 'tileWorldSizeM'],
    'tile_size_px':             ['TILE', 'sizePx'],
  };
  const applied = [];
  for (const [srcKey, [section, field]] of Object.entries(map)) {
    if (jsonCfg[srcKey] !== undefined) {
      cfg[section][field] = jsonCfg[srcKey];
      applied.push(`${section}.${field} <- ${srcKey}=${JSON.stringify(jsonCfg[srcKey])}`);
    }
  }
  return applied;
}
