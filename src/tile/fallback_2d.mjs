import sharp from 'sharp';
import fs from 'node:fs';
import path from 'node:path';
import { tileIndexToLatLng } from './coords.mjs';
import { saveTile, signInt } from './tile_io.mjs';
import { isPlaceholderPng } from './placeholder_detect.mjs';
import {
  blendHorizontalEdge,
  blendVerticalEdge,
  blendCorner,
  computeStats,
  colorBalance,
  readNeighborSamples,
  FEATHER_PX,
} from './edge_blend.mjs';

// ─── Web Mercator (EPSG:3857) helpers ──────────────────────────────────────
//
// Providers are queried by explicit bounding box (WMS `GetMap` / ArcGIS
// `export`), NOT by slippy z/x/y tile index. This matters because our grid
// cells are ~100-200m apart (cfg.frustumW), while a single slippy tile at
// a fixed zoom can cover anywhere from a few hundred meters to several
// kilometers depending on the zoom level a provider happens to support.
// Snapping every request to the nearest z/x/y tile means many neighboring
// grid cells alias onto the exact same underlying tile — that's why widely
// different (qx,qy) tiles were coming back byte-identical (all landing on
// the same z14 WMTS cell, ~2.4km wide, while grid cells are ~100-200m
// apart). Requesting an explicit bbox sized to the tile's own footprint
// avoids this aliasing regardless of what zoom levels a provider supports.

const MERCATOR_R = 6378137; // meters, WGS84 sphere radius used by EPSG:3857
const MAX_MERC_LAT = 85.05112878;

function lngLatToMercator(lng, lat) {
  const clampedLat = Math.max(-MAX_MERC_LAT, Math.min(MAX_MERC_LAT, lat));
  const x = (lng * Math.PI / 180) * MERCATOR_R;
  const y = Math.log(Math.tan(Math.PI / 4 + (clampedLat * Math.PI / 360))) * MERCATOR_R;
  return { x, y };
}

/**
 * Bounding box (EPSG:3857 meters) centered on (lat,lng), sized to cover
 * `footprintM` meters of *ground* distance. Ground meters are converted to
 * Mercator meters via /cos(lat) to correct for the projection's latitude
 * scale factor — otherwise the bbox would be too narrow away from the
 * equator.
 */
function bboxMetersForFootprint(lat, lng, footprintM) {
  const { x: cx, y: cy } = lngLatToMercator(lng, lat);
  const scale = 1 / Math.cos(lat * Math.PI / 180);
  const half = (footprintM * scale) / 2;
  return { minx: cx - half, miny: cy - half, maxx: cx + half, maxy: cy + half };
}

/**
 * Render a fallback 2D tile for (qx, qy). Iterates through cfg.fallback.providers
 * in order until one succeeds. No API key needed — uses free public satellite
 * imagery servers only (ESRI World Imagery, EOX Sentinel-2 cloudless). Each
 * provider is queried by explicit bbox (see helpers above), sized to this
 * tile's own footprint (cfg.frustumW, or fb.footprintM override) — not by a
 * shared slippy z/x/y grid — so adjacent grid cells never alias onto the
 * same source tile.
 *
 * @param {{qx:number, qy:number}} tile
 * @param {number} seedLng
 * @param {number} seedLat
 * @param {object} cfg - full render cfg (must include cfg.fallback.providers)
 * @param {string} outputDir
 * @param {object} [parentPort] - worker parentPort (optional, for progress events)
 * @param {number} [workerId]
 * @returns {Promise<{ok:boolean, provider?:string, error?:string, filepath?:string, sizeKB?:number}>}
 */
export async function renderFallback2D(tile, seedLng, seedLat, cfg, outputDir, parentPort = null, workerId = 0) {
  const fb = cfg?.fallback;
  if (!fb || !fb.enabled) {
    return { ok: false, error: 'fallback disabled in cfg' };
  }
  const providers = Array.isArray(fb.providers) && fb.providers.length > 0
    ? fb.providers
    : [{
        name: 'esri',
        urlTemplate: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export'
          + '?bbox={minx},{miny},{maxx},{maxy}&bboxSR=3857&imageSR=3857&size={w},{h}&format=png32&f=image',
      }];

  const { lat, lng } = tileIndexToLatLng(tile.qx, tile.qy, seedLat, seedLng, cfg);
  const sizePx = cfg.sizePx ?? 1024;
  const footprintM = fb.footprintM ?? cfg.frustumW ?? 200;
  const { minx, miny, maxx, maxy } = bboxMetersForFootprint(lat, lng, footprintM);

  const errors = [];
  for (const provider of providers) {
    const url = provider.urlTemplate
      .replace('{minx}', String(minx))
      .replace('{miny}', String(miny))
      .replace('{maxx}', String(maxx))
      .replace('{maxy}', String(maxy))
      .replace('{w}', String(sizePx))
      .replace('{h}', String(sizePx));

    let buf;
    try {
      const res = await fetch(url, {
        signal: AbortSignal.timeout(fb.requestTimeoutMs ?? 8_000),
        headers: { 'User-Agent': 'isometric-style-convert/1.0' },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 100) throw new Error('Empty tile (likely ocean/pole)');
    } catch (e) {
      errors.push(`${provider.name}:${e.message}`);
      continue;
    }

    // Validate raw provider response — ESRI World Imagery returns the
    // Google "Map data not yet available" placeholder as a valid 200 OK
    // JPEG when no satellite coverage exists. Detect that here so we skip
    // to the next provider instead of saving the placeholder as the tile.
    if (await isPlaceholderPng(buf)) {
      errors.push(`${provider.name}:returned Google placeholder`);
      continue;
    }

    let processed;
    try {
      processed = await postProcess2D(buf, fb, cfg.sizePx ?? 1024);
    } catch (e) {
      errors.push(`${provider.name}:postProcess:${e.message}`);
      continue;
    }

    // ── Layer 1 (fallback_2d): color-match + 4-side/4-corner feathering
    //    against already-rendered 3D neighbors on disk. Without this, the 2D
    //    satellite tile stands out from surrounding 3D tiles in two ways:
    //      - color/contrast mismatch (3D post-process differs from 2D post-process)
    //      - hard seams at edges where 2D imagery doesn't align with the
    //        3D-rendered neighboring tiles.
    if (fb?.blendWithNeighbors !== false) {
      try {
        processed = await blend2DWith3DNeighbors(
          processed, outputDir, tile.qx, tile.qy
        );
      } catch (e) {
        // Non-fatal: keep the 2D tile as-is if blending fails.
        console.warn(`[fallback_2d] blend failed: ${e.message}`);
      }
    }

    const result = saveTile(processed, { qx: tile.qx, qy: tile.qy }, {
      lat, lng,
      fallback: provider.name,
      fallbackUrl: url,
      postProcess: fb.postProcess ?? 'desat-sepia',
      isBlank: false,
      attempt: 0,
      renderMs: 0,
      seedLat, seedLng,
    }, outputDir);

    parentPort?.postMessage({
      type: 'tile_done',
      workerId,
      qx: tile.qx, qy: tile.qy,
      fallback: provider.name,
    });

    return { ok: true, provider: provider.name, ...result };
  }

  return { ok: false, error: `all providers failed: ${errors.join(' | ')}` };
}

/**
 * Post-process 2D satellite tile to look closer to 3D isometric output.
 *
 * Pipeline (per pixel):
 *   1. Compute luminance L = 0.299*R + 0.587*G + 0.114*B
 *   2. Desaturate: pull each channel toward L by `desat` amount
 *   3. Sepia tint: warm shift (R*1.10, G*0.95, B*0.78) blended by `sepia` amount
 *   4. Contrast: (channel - 128) * contrast + 128
 *   5. Clamp to [0, 255]
 *
 * Output: 1024×1024 PNG buffer (resized to match TILE.sizePx).
 *
 * @param {Buffer} buf - input PNG/JPEG bytes from provider
 * @param {object} fbCfg - fallback config (desatAmount, sepiaAmount, contrastBoost, postProcess)
 * @param {number} tilePx - output tile size in px (default 1024)
 * @returns {Promise<Buffer>} processed PNG buffer
 */
export async function postProcess2D(buf, fbCfg, tilePx = 1024) {
  const skipPost = (fbCfg.postProcess ?? 'desat-sepia') === 'none';
  const out = await sharp(buf)
    .ensureAlpha()
    .resize(tilePx, tilePx, { fit: 'fill' })
    .png();

  if (skipPost) return out.toBuffer();

  const { data, info } = await out.raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;

  const desat    = clamp01(fbCfg.desatAmount ?? 0.35);
  const sepia    = clamp01(fbCfg.sepiaAmount ?? 0.20);
  const contrast = fbCfg.contrastBoost ?? 1.05;

  const outBuf = Buffer.alloc(data.length);

  for (let i = 0; i < data.length; i += channels) {
    const r = data[i], g = data[i + 1], b = data[i + 2];

    // Luminance (BT.601)
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    // Desaturate: each channel toward lum by `desat` factor
    let nr = r + (lum - r) * desat;
    let ng = g + (lum - g) * desat;
    let nb = b + (lum - b) * desat;

    // Sepia tint: warm shift
    nr = nr * (1 - sepia) + lum * 1.10 * sepia;
    ng = ng * (1 - sepia) + lum * 0.95 * sepia;
    nb = nb * (1 - sepia) + lum * 0.78 * sepia;

    // Contrast around midpoint 128
    nr = (nr - 128) * contrast + 128;
    ng = (ng - 128) * contrast + 128;
    nb = (nb - 128) * contrast + 128;

    outBuf[i]     = clamp255(nr);
    outBuf[i + 1] = clamp255(ng);
    outBuf[i + 2] = clamp255(nb);
    outBuf[i + 3] = data[i + 3];  // preserve alpha
  }

  return sharp(outBuf, { raw: { width, height, channels } })
    .png()
    .toBuffer();
}

// ============================================================================
// Layer 1 (fallback_2d): blend 2D tile against already-rendered 3D neighbors
// ============================================================================

/**
 * For a 2D fallback tile to look continuous with its 3D neighbors, we need:
 *   (a) color balance — shift the 2D tile's brightness AND contrast toward
 *       the 3D neighbors' so its overall tone matches; otherwise it stands
 *       out as obviously different (3D tends warmer and punchier than flat
 *       2D raw satellite imagery).
 *   (b) feather 4 sides + 4 corners with the 3D neighbors' edge strips and
 *       corner patches (using edge_blend.mjs helpers), so the seams where
 *       2D meets 3D are blended instead of hard-cut.
 *
 * If no 3D neighbors are found in any direction, this is a no-op (returns
 * the buffer unchanged). That's the typical case for the first few tiles
 * before the rest of the map has rendered.
 */
async function blend2DWith3DNeighbors(pngBuffer, rendersDir, qx, qy) {
  const neighbors = await findRendered3DNeighbors(rendersDir, qx, qy);
  if (neighbors.length === 0) return pngBuffer;

  // Decode to raw RGBA so we can mutate pixel-by-pixel.
  const { data, info } = await sharp(pngBuffer).ensureAlpha().raw()
    .toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  if (channels !== 4) {
    throw new Error(`expected RGBA, got ${channels} channels`);
  }
  const rgba = Buffer.from(data);

  // Step (a): color balance — match this tile's mean + contrast to the
  // weighted stats of its 3D neighbors. Gain is clamped inside colorBalance
  // so this is always safe to apply (no threshold gate needed): a tile
  // that already matches gets a near-identity transform, one that doesn't
  // gets pulled into line.
  const ownStats = computeStats(rgba);
  const neighStats = computeNeighborStats(neighbors);
  if (neighStats) {
    colorBalance(rgba, width, height, neighStats, ownStats, { strength: 0.85 });
  }

  // Step (b): feather 4 sides + 4 corners via edge_blend.mjs. We feather
  // the 2D tile against the 3D-rendered neighbor edges so the seam is
  // softly blended. Edge pull weight is 1 at the boundary and fades to 0
  // over FEATHER_PX.
  for (const n of neighbors) {
    if (n.dir.dqx === 0 && n.dir.dqy === 0) continue;
    if (n.dir.dqx !== 0 && n.dir.dqy !== 0) continue;
    if (n.dir.dqx === +1 && n.dir.dqy === 0) {
      blendHorizontalEdge(rgba, n.samples, width, 'left', 'west');
    } else if (n.dir.dqx === -1 && n.dir.dqy === 0) {
      blendHorizontalEdge(rgba, n.samples, width, 'right', 'east');
    } else if (n.dir.dqx === 0 && n.dir.dqy === +1) {
      blendVerticalEdge(rgba, n.samples, width, 'bottom', 'north');
    } else if (n.dir.dqx === 0 && n.dir.dqy === -1) {
      blendVerticalEdge(rgba, n.samples, width, 'top', 'south');
    }
  }
  for (const n of neighbors) {
    if (n.dir.dqx === 0 || n.dir.dqy === 0) continue;
    blendCorner(rgba, n.samples, width, n.dir);
  }

  // Re-encode to PNG.
  return sharp(rgba, { raw: { width, height, channels: 4 } })
    .png()
    .toBuffer();
}

/**
 * Find already-rendered 3D neighboring tiles on disk for (qx, qy).
 * "3D neighbor" = any tile file at the 8 adjacent positions (E, W, N, S,
 * NE, NW, SE, SW) whose meta file says it was rendered with provider 3D.
 *
 * We exclude 2D fallback tiles themselves, because blending a 2D tile
 * against a 2D tile doesn't fix the seam — the whole point is to align
 * the 2D fallbacks with the rest of the map (which is 3D).
 *
 * Returns array of { qx, qy, samples, dir, fallback } so callers can
 * distinguish but only need samples+dir for blending.
 */
async function findRendered3DNeighbors(rendersDir, qx, qy) {
  let files;
  try {
    files = fs.readdirSync(rendersDir);
  } catch {
    return [];
  }
  const metaDir = path.join(rendersDir, 'meta');

  const directions = [
    { name: 'east',  dqx: +1, dqy: 0 },
    { name: 'west',  dqx: -1, dqy: 0 },
    { name: 'north', dqx: 0, dqy: +1 },
    { name: 'south', dqx: 0, dqy: -1 },
    { name: 'NE',    dqx: +1, dqy: +1 },
    { name: 'NW',    dqx: -1, dqy: +1 },
    { name: 'SE',    dqx: +1, dqy: -1 },
    { name: 'SW',    dqx: -1, dqy: -1 },
  ];

  const out = [];
  for (const dir of directions) {
    const nqx = qx + dir.dqx;
    const nqy = qy + dir.dqy;
    // Find tile file at (nqx, nqy)
    const prefix = `tile_${signInt(nqx)}_${signInt(nqy)}_`;
    let pngFile = null;
    for (const f of files) {
      if (f.startsWith(prefix) && f.endsWith('.png')) {
        pngFile = f;
        break;
      }
    }
    if (!pngFile) continue;
    // Check meta: include only 3D-rendered tiles
    const metaFile = path.join(metaDir, `tile_${signInt(nqx)}_${signInt(nqy)}.json`);
    if (!fs.existsSync(metaFile)) continue;
    let is3D = false;
    try {
      const meta = JSON.parse(fs.readFileSync(metaFile, 'utf8'));
      // meta.fallback is set for 2D tiles; meta.altitude/cameraAzimuth indicates 3D
      if (!meta.fallback && (meta.altitude !== undefined || meta.cameraAzimuth !== undefined)) {
        is3D = true;
      }
    } catch { /* ignore */ }
    if (!is3D) continue;

    try {
      const samples = await readNeighborSamples(path.join(rendersDir, pngFile));
      out.push({ qx: nqx, qy: nqy, samples, dir });
    } catch { /* ignore */ }
  }
  return out;
}

/**
 * Weighted mean + stddev (all neighbors weighted equally — they're all
 * distance-1 adjacent) of the 3D neighbors' stats, for colorBalance's
 * `target`. Returns null if no neighbors.
 */
function computeNeighborStats(neighbors) {
  if (!neighbors || neighbors.length === 0) return null;
  let sR = 0, sG = 0, sB = 0, sdR = 0, sdG = 0, sdB = 0;
  for (const n of neighbors) {
    const s = computeStats(n.samples.raw);
    sR += s.meanR; sG += s.meanG; sB += s.meanB;
    sdR += s.stdR; sdG += s.stdG; sdB += s.stdB;
  }
  const n = neighbors.length;
  return {
    meanR: sR / n, meanG: sG / n, meanB: sB / n,
    stdR: sdR / n, stdG: sdG / n, stdB: sdB / n,
  };
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function clamp255(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}
