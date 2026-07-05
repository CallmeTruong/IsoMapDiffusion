import sharp from 'sharp';
import { TILE } from '../config.mjs';
import { tileIndexToLatLng } from './coords.mjs';
import { tileIndexToSlippyXYZ } from './coords_slippy.mjs';
import { saveTile } from './tile_io.mjs';

/**
 * Render a fallback 2D tile for (qx, qy). Pure fetch + post-process.
 *
 * @param {{qx:number, qy:number}} tile
 * @param {number} seedLng
 * @param {number} seedLat
 * @param {object} cfg - full render cfg (must include cfg.fallback)
 * @param {string} outputDir
 * @param {object} [parentPort] - worker parentPort (optional, for progress events)
 * @param {number} [workerId]
 * @returns {Promise<{ok:boolean, error?:string, filepath?:string, sizeKB?:number}>}
 */
export async function renderFallback2D(tile, seedLng, seedLat, cfg, outputDir, parentPort = null, workerId = 0) {
  const fb = cfg?.fallback;
  if (!fb || !fb.enabled) {
    return { ok: false, error: 'fallback disabled in cfg' };
  }
  if (!fb.urlTemplate) {
    return { ok: false, error: 'fallback.urlTemplate missing' };
  }

  const { lat, lng } = tileIndexToLatLng(tile.qx, tile.qy, seedLat, seedLng, cfg);
  const z = fb.minZoom ?? TILE.fallback.defaultMinZoom;
  const { x, y } = tileIndexToSlippyXYZ(lat, lng, z);

  const url = fb.urlTemplate
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y));

  let buf;
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(fb.requestTimeoutMs ?? TILE.fallback.requestTimeoutMs),
      headers: { 'User-Agent': TILE.fallback.userAgent },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    buf = Buffer.from(await res.arrayBuffer());
    if (buf.length < TILE.fallback.minResponseBytes) throw new Error('Empty tile (likely ocean/pole)');
  } catch (e) {
    parentPort?.postMessage({
      type: 'db_update',
      qx: tile.qx, qy: tile.qy,
      info: { workerId, error: `2D fallback fetch: ${e.message}` },
    });
    return { ok: false, error: `fetch: ${e.message}` };
  }

  let processed;
  try {
    processed = await postProcess2D(buf, fb, cfg.sizePx);
  } catch (e) {
    parentPort?.postMessage({
      type: 'db_update',
      qx: tile.qx, qy: tile.qy,
      info: { workerId, error: `2D fallback postProcess: ${e.message}` },
    });
    return { ok: false, error: `postProcess: ${e.message}` };
  }

  const result = saveTile(processed, { qx: tile.qx, qy: tile.qy }, {
    lat, lng,
    fallback: fb.provider ?? 'esri',
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
    fallback: fb.provider ?? 'esri',
  });

  return { ok: true, ...result };
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
 * @param {number} tilePx - output tile size in px
 * @returns {Promise<Buffer>} processed PNG buffer
 */
export async function postProcess2D(buf, fbCfg, tilePx) {
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

    // Desaturate: each channel → lum by `desat` factor
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

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function clamp255(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}