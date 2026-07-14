// Central configuration for isometric pipeline

import { cpus } from 'os';
import path from 'path';



const DEFAULTS = {

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


  GRID: {
    cellSizeKm:   0.1,
    minWaterM2:   250,
  },


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

    // Blank/blurry thresholds (for analyzeCanvas)
    blankVarianceThr: 600,
    blankEdgeThr:     0.15,
    blankMeanThr:     [60, 110],
    blankSizeKb:      30,
    blankMeanRThr:    250,

    // Retry
    maxRetry:         3,


    fallback: {
      enabled:        true,
      maxRetries3D:  3,
      requestTimeoutMs: 8000,
      // Satellite/aerial providers queried by explicit BBOX (not slippy z/x/y).
      // ESRI first (best resolution), EOX Sentinel-2 last (global coverage).
      providers: [
        {
          name: 'esri',
          urlTemplate: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export'
            + '?bbox={minx},{miny},{maxx},{maxy}&bboxSR=3857&imageSR=3857&size={w},{h}&format=png32&f=image',
        },
        // EOX Sentinel-2 cloudless — global, gap-free (~10m/px native resolution)
        {
          name: 'eox-s2cloudless',
          urlTemplate: 'https://tiles.maps.eox.at/wms?service=WMS&version=1.1.1&request=GetMap'
            + '&layers=s2cloudless-2024_3857&styles=&bbox={minx},{miny},{maxx},{maxy}'
            + '&width={w}&height={h}&srs=EPSG:3857&format=image/jpeg',
          attribution: 'Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (CC BY-NC-SA 4.0, non-commercial only)',
        },
      ],

      postProcess:    'desat-sepia', // 'desat-sepia' | 'none'
      desatAmount:    0.35,
      sepiaAmount:    0.20,
      contrastBoost:  1.05,
    },

    sessionMaxMs:     2.9 * 60 * 60 * 1000,
    protocolTimeout:  120_000,

    cameraMoveStep:   0.5,
    hashLength:       8,
  },


  RENDER: {
    blankSizeKb:        30,
    protocolTimeoutMs:  120_000,
    sessionMaxMs:       10 * 60 * 1000,
    maxRetry:           2,
    sampleGridSize:     20,
    edgeGradThr:        30,
    requiredStableFrames: 5,
    postRenderExtraFrames: 4,
    postRenderBufferMs:  30,
    targetFrameRate:    20,
    lightDirection:     { x: 0.5, y: -0.5, z: -0.7 },
  },


  STITCH: {
    background:         { r: 255, g: 255, b: 255, alpha: 1 },
    seamColor:          { r: 255, g:   0, b:   0, alpha: 1 },
    seamThicknessPx:    1,
    pairGapColor:       { r: 240, g: 240, b: 240, alpha: 1 },
    debugOnly:          true,
    xaxisAuto:          'auto',  // 'auto' | 'east' | 'west' | 'off'
    flipXByDefault:     null,    // null = auto-detect, 'east'/'west' = force

    outPathTemplate:    'stitch_{N}x{M}_offset{startQx}_{startQy}_{suffix}.png',
  },
  XAXIS: {
    overlap:            512,
    sampleHeight:       1024,
    resizeWidth:        128,
    ratioThreshold:     1.2,
    highConfRatio:      2.0,
    mediumConfRatio:    1.5,
  },
  WORKERS: {
    default: 1,
    max:     4,
  },


  PROVIDERS: {
    available: ['google', 'cesium-ion'],
    default:   'cesium-ion',


    cesiumIon: {

      googleAssetId:     2275207,
      validateUrl:      'https://api.cesium.com/v1/me',
      requestTimeoutMs: 10_000,
    },
  },


  GOOGLE: {
    rootJsonUrl:        'https://tile.googleapis.com/v1/3dtiles/root.json',
    requestTimeoutMs:   10_000,
  },


  TMP: {
    dirName:           'isometric-style-convert',
    htmlPrefix:        'tmp_cesium_w',
    profilePrefix:     'chrome_profile_w',
  },


  CHECKPOINT: {
    schemaVersion:     1,
    flushIntervalMs:   5000,
  },


  QUALITY: {
    useTileDefaults:    true,
  },


  GEO: {
    mPerDegLat:        111111,
  },


  CESIUM: {
    version:           '1.132',

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


  SEEDS: {
    sgn: { lat: 10.78465, lng: 106.70775, label: 'Saigon, District 1' },
    nyc: { lat: 40.7128,  lng: -74.0060,  label: 'New York City' },

  },
};



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

const PROJECT_ROOT_DEFAULT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
export const PROJECT_ROOT = PROJECT_ROOT_DEFAULT;


let _config = null;
export function getConfig(projectRoot = PROJECT_ROOT_DEFAULT) {
  if (_config) return _config;
  _config = JSON.parse(JSON.stringify(DEFAULTS));
  applyEnvOverrides(_config, projectRoot);
  return _config;
}

const _cfg = getConfig();



export const PATHS = _cfg.PATHS;
export const GRID = _cfg.GRID;
export const TILE = _cfg.TILE;
export const RENDER = _cfg.RENDER;
export const STITCH = _cfg.STITCH;
export const XAXIS = _cfg.XAXIS;
export const WORKERS = _cfg.WORKERS;
export const GOOGLE = _cfg.GOOGLE;
export const PROVIDERS = _cfg.PROVIDERS;
export const TMP = _cfg.TMP;
export const CHECKPOINT = _cfg.CHECKPOINT;
export const QUALITY = _cfg.QUALITY;
export const GEO = _cfg.GEO;
export const CESIUM = _cfg.CESIUM;
export const SEEDS = _cfg.SEEDS;
export const FALLBACK = _cfg.TILE?.fallback ?? { enabled: false };

// Derived constants
export const CELL_SIZE_M     = 200;
export const QUADRANT_M      = 100;
export const TILE_SIZE_M     = CELL_SIZE_M;
export const TILE_SIZE_PX    = TILE.sizePx;
export const TILE_STEP_M     = TILE_SIZE_M * TILE.cameraMoveStep;
export const TILE_STEP_PX    = TILE_SIZE_PX * TILE.cameraMoveStep;
export const M_PER_PX        = TILE_SIZE_M / TILE_SIZE_PX;

export function resolvePath(key) {
  const v = PATHS[key];
  if (typeof v !== 'string') {
    throw new Error(`resolvePath: unknown key '${key}'`);
  }
  return path.isAbsolute(v) ? v : path.resolve(PROJECT_ROOT_DEFAULT, v);
}

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
