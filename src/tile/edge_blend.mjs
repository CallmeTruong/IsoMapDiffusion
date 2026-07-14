/**
 * edge_blend.mjs - Layer 1: 4-sided + 4-corner feathering helper for tile seams.
 *
 * Used by fallback_2d.mjs to blend a 2D satellite fallback tile with its
 * already-rendered 3D neighbors (color balance + edge/corner feathering).
 *
 * Public API:
 *   - extractHorizontalEdgeStrip(neighbor, side)   left/right vertical strip
 *   - extractVerticalEdgeStrip(neighbor, side)     top/bottom horizontal strip
 *   - extractCorner(neighbor, dir, sizePx=256)
 *   - blendHorizontalEdge(rgba, neighbor, width, neighborSide, place)
 *   - blendVerticalEdge(rgba, neighbor, width, neighborSide, place)
 *   - blendCorner(rgba, neighbor, width, dir, cornerRadiusPx=256)
 *   - computeStats(rgba, stride=16)               mean + stddev per channel
 *   - colorBalance(rgba, w, h, target, source, opts) mean+contrast match (gain-clamped)
 *   - readNeighborSamples(filePath)
 *   - constants: EDGE_STRIP_PX=256, CORNER_STRIP_PX=256, FEATHER_PX=384
 *
 * Conventions:
 *   - rgba buffer: width*height*4 bytes, channels=4 always (RGBA).
 *   - tile size = TILE.sizePx (1024 by default).
 *
 * Caller mapping:
 *   East neighbor (dqx=+1): blendHorizontalEdge(..., neighborSide='left',  place='west')
 *   West neighbor (dqx=-1): blendHorizontalEdge(..., neighborSide='right', place='east')
 *   North neighbor (dqy=+1): blendVerticalEdge(..., neighborSide='bottom', place='north')
 *   South neighbor (dqy=-1): blendVerticalEdge(..., neighborSide='top',    place='south')
 *   Diagonal neighbor: blendCorner(rgba, neighbor, w, dir)
 */

import sharp from 'sharp';
import fs from 'node:fs';

export const EDGE_STRIP_PX = 256;
export const CORNER_STRIP_PX = 256;
export const FEATHER_PX = 384;

function clamp255(v) { return Math.max(0, Math.min(255, v)); }


// ============================================================================
// Strip / corner extraction
// ============================================================================

/**
 * Extract a vertical edge strip from neighbor: W=EDGE_STRIP_PX pixels wide, full height.
 * side='left' -> cols [0, W); side='right' -> cols [W-W, W).
 * Returns { strip, stripW, stripH, channels }.
 */
export function extractHorizontalEdgeStrip(neighbor, side) {
  const { raw, width, height, channels } = neighbor;
  const strip = Buffer.alloc(EDGE_STRIP_PX * height * channels);
  const srcColStart = side === 'left' ? 0 : width - EDGE_STRIP_PX;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < EDGE_STRIP_PX; x++) {
      const si = (y * width + (srcColStart + x)) * channels;
      const di = (y * EDGE_STRIP_PX + x) * channels;
      strip[di]     = raw[si];
      strip[di + 1] = raw[si + 1];
      strip[di + 2] = raw[si + 2];
      strip[di + 3] = raw[si + 3];
    }
  }
  return { strip, stripW: EDGE_STRIP_PX, stripH: height, channels };
}

/**
 * Extract a horizontal edge strip from neighbor: full width, H=EDGE_STRIP_PX tall.
 * side='top' -> rows [0, H); side='bottom' -> rows [H-H, H).
 */
export function extractVerticalEdgeStrip(neighbor, side) {
  const { raw, width, height, channels } = neighbor;
  const strip = Buffer.alloc(width * EDGE_STRIP_PX * channels);
  const srcRowStart = side === 'top' ? 0 : height - EDGE_STRIP_PX;

  for (let y = 0; y < EDGE_STRIP_PX; y++) {
    for (let x = 0; x < width; x++) {
      const si = ((srcRowStart + y) * width + x) * channels;
      const di = (y * width + x) * channels;
      strip[di]     = raw[si];
      strip[di + 1] = raw[si + 1];
      strip[di + 2] = raw[si + 2];
      strip[di + 3] = raw[si + 3];
    }
  }
  return { strip, stripW: width, stripH: EDGE_STRIP_PX, channels };
}

/**
 * Extract a square corner from neighbor that faces the missing tile.
 * Convention: missing tile at (qx,qy), neighbor at (qx+dir.dqx, qy+dir.dqy).
 * The neighbor's CORNER that touches the missing tile is OPPOSITE to dir.
 * (e.g. neighbor NE of missing -> neighbor's SW corner faces missing)
 */
export function extractCorner(neighbor, dir, sizePx = CORNER_STRIP_PX) {
  const { raw, width, height, channels } = neighbor;
  const corner = Buffer.alloc(sizePx * sizePx * channels);

  let rowStart, colStart;
  if (dir.dqx === +1 && dir.dqy === +1) {
    // neighbor NE of missing -> neighbor's SW corner -> row:bottom, col:left
    rowStart = height - sizePx; colStart = 0;
  } else if (dir.dqx === +1 && dir.dqy === -1) {
    // neighbor SE of missing -> neighbor's NW corner -> row:top, col:left
    rowStart = 0; colStart = 0;
  } else if (dir.dqx === -1 && dir.dqy === +1) {
    // neighbor NW of missing -> neighbor's SE corner -> row:bottom, col:right
    rowStart = height - sizePx; colStart = width - sizePx;
  } else if (dir.dqx === -1 && dir.dqy === -1) {
    // neighbor SW of missing -> neighbor's NE corner -> row:top, col:right
    rowStart = 0; colStart = width - sizePx;
  } else {
    throw new Error('extractCorner: dir must be diagonal, got dqx=' + dir.dqx + ' dqy=' + dir.dqy);
  }

  for (let y = 0; y < sizePx; y++) {
    for (let x = 0; x < sizePx; x++) {
      const si = ((rowStart + y) * width + (colStart + x)) * channels;
      const di = (y * sizePx + x) * channels;
      corner[di]     = raw[si];
      corner[di + 1] = raw[si + 1];
      corner[di + 2] = raw[si + 2];
      corner[di + 3] = raw[si + 3];
    }
  }
  return { corner, cornerW: sizePx, cornerH: sizePx, channels };
}


// ============================================================================
// Mask helpers (Float32Array of length tileDim)
// ============================================================================

/**
 * Horizontal mask of length `width`: ramps to 1 at x=0 AND x=width-1 over FEATHER_PX.
 */
export function buildHorizontalMask(width, featherPx = FEATHER_PX) {
  const m = new Float32Array(width);
  for (let x = 0; x < featherPx; x++) m[x] = x / featherPx;
  for (let x = 0; x < featherPx; x++) m[width - 1 - x] = x / featherPx;
  for (let x = featherPx; x < width - featherPx; x++) m[x] = 1.0;
  return m;
}

/**
 * Vertical mask of length `height`: ramps to 1 at y=0 AND y=height-1.
 */
export function buildVerticalMask(height, featherPx = FEATHER_PX) {
  const m = new Float32Array(height);
  for (let y = 0; y < featherPx; y++) m[y] = y / featherPx;
  for (let y = 0; y < featherPx; y++) m[height - 1 - y] = y / featherPx;
  for (let y = featherPx; y < height - featherPx; y++) m[y] = 1.0;
  return m;
}


// ============================================================================
// Blenders
// ============================================================================

/**
 * Blend a vertical edge strip (W px wide, full height) into the matching
 * horizontal edge of rgba. neighborSide = side of neighbor we pull FROM.
 * place = which side of the rgba tile we push INTO.
 *
 *   E neighbor  -> neighborSide='left',  place='west'
 *   W neighbor  -> neighborSide='right', place='east'
 */
export function blendHorizontalEdge(rgba, neighbor, width, neighborSide, place) {
  const { strip, stripW, stripH } = extractHorizontalEdgeStrip(neighbor, neighborSide);
  const height = rgba.length / (width * 4);

  let xMapper, blend;
  if (place === 'west') {
    xMapper = (sx) => Math.floor(sx * width / stripW);
    blend = (x) => (x < FEATHER_PX ? 1 - x / FEATHER_PX : 0);
  } else if (place === 'east') {
    xMapper = (sx) => width - 1 - Math.floor(sx * width / stripW);
    blend = (x) => (x >= width - FEATHER_PX ? (x - (width - FEATHER_PX)) / FEATHER_PX : 0);
  } else {
    throw new Error("blendHorizontalEdge: place must be 'west' or 'east', got " + place);
  }

  for (let y = 0; y < height; y++) {
    const sy = Math.floor(y * stripH / height);
    for (let sx = 0; sx < stripW; sx++) {
      const dx = xMapper(sx);
      if (dx < 0 || dx >= width) continue;
      const w = blend(dx);
      if (w <= 0) continue;
      const si = (sy * stripW + sx) * 4;
      const di = (y * width + dx) * 4;
      rgba[di]     = clamp255(rgba[di]     * (1 - w) + strip[si]     * w);
      rgba[di + 1] = clamp255(rgba[di + 1] * (1 - w) + strip[si + 1] * w);
      rgba[di + 2] = clamp255(rgba[di + 2] * (1 - w) + strip[si + 2] * w);
    }
  }
}

/**
 * Blend a horizontal edge strip (H px tall, full width) into the matching
 * vertical edge of rgba. neighborSide = side of neighbor we pull FROM.
 * place = which side of the rgba tile we push INTO.
 *
 *   N neighbor (dqy=+1) -> neighborSide='bottom', place='north'
 *   S neighbor (dqy=-1) -> neighborSide='top',    place='south'
 */
export function blendVerticalEdge(rgba, neighbor, width, neighborSide, place) {
  const { strip, stripW, stripH } = extractVerticalEdgeStrip(neighbor, neighborSide);
  const height = rgba.length / (width * 4);

  let yMapper, blend;
  if (place === 'north') {
    yMapper = (sy) => Math.floor(sy * height / stripH);
    blend = (y) => (y < FEATHER_PX ? 1 - y / FEATHER_PX : 0);
  } else if (place === 'south') {
    yMapper = (sy) => height - 1 - Math.floor(sy * height / stripH);
    blend = (y) => (y >= height - FEATHER_PX ? (y - (height - FEATHER_PX)) / FEATHER_PX : 0);
  } else {
    throw new Error("blendVerticalEdge: place must be 'north' or 'south', got " + place);
  }

  for (let sy = 0; sy < stripH; sy++) {
    const dy = yMapper(sy);
    if (dy < 0 || dy >= height) continue;
    const w = blend(dy);
    if (w <= 0) continue;
    for (let sx = 0; sx < stripW; sx++) {
      const si = (sy * stripW + sx) * 4;
      const di = (dy * width + sx) * 4;
      rgba[di]     = clamp255(rgba[di]     * (1 - w) + strip[si]     * w);
      rgba[di + 1] = clamp255(rgba[di + 1] * (1 - w) + strip[si + 1] * w);
      rgba[di + 2] = clamp255(rgba[di + 2] * (1 - w) + strip[si + 2] * w);
    }
  }
}

/**
 * Blend a diagonal corner into the matching corner of rgba with a radial mask.
 *
 * dir is diagonal ({dqx, dqy}). Target corner of rgba determined by sign:
 *   neighbor NE (+1,+1) -> rgba NE corner
 *   neighbor SE (+1,-1) -> rgba SE corner
 *   neighbor NW (-1,+1) -> rgba NW corner
 *   neighbor SW (-1,-1) -> rgba SW corner
 *
 * Radial mask: weight = max(|dx|/R, |dy|/R) clamped 0..1, fading from 1 at
 * the corner to 0 at radius R (default 256). Past R, mid-tile mean dominates.
 */
export function blendCorner(rgba, neighbor, width, dir, cornerRadiusPx = CORNER_STRIP_PX) {
  const { corner, cornerW, cornerH } = extractCorner(neighbor, dir, cornerRadiusPx);
  const height = rgba.length / (width * 4);

  let cx, cy;
  if (dir.dqx === +1 && dir.dqy === +1)      { cx = width - 1;  cy = 0; }
  else if (dir.dqx === +1 && dir.dqy === -1) { cx = width - 1;  cy = height - 1; }
  else if (dir.dqx === -1 && dir.dqy === +1) { cx = 0;          cy = 0; }
  else if (dir.dqx === -1 && dir.dqy === -1) { cx = 0;          cy = height - 1; }
  else throw new Error('blendCorner: dir must be diagonal, got dqx=' + dir.dqx + ' dqy=' + dir.dqy);

  function weight(dx, dy) {
    const ax = Math.abs(dx) / cornerRadiusPx;
    const ay = Math.abs(dy) / cornerRadiusPx;
    const r = Math.max(ax, ay);
    return r >= 1 ? 0 : 1 - r;
  }

  for (let sy = 0; sy < cornerH; sy++) {
    for (let sx = 0; sx < cornerW; sx++) {
      let dx, dy;
      if (cx === 0) dx =  sx;     else dx = -sx;
      if (cy === 0) dy =  sy;     else dy = -sy;

      const tx = cx + dx;
      const ty = cy + dy;
      if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

      const w = weight(dx, dy);
      if (w <= 0) continue;

      const si = (sy * cornerW + sx) * 4;
      const di = (ty * width + tx) * 4;
      rgba[di]     = clamp255(rgba[di]     * (1 - w) + corner[si]     * w);
      rgba[di + 1] = clamp255(rgba[di + 1] * (1 - w) + corner[si + 1] * w);
      rgba[di + 2] = clamp255(rgba[di + 2] * (1 - w) + corner[si + 2] * w);
    }
  }
}


// ============================================================================
// Color balance: match a tile's brightness AND contrast to a reference,
// per RGB channel. Used by fallback_2d.mjs so a 2D satellite tile matches
// the tone of its surrounding 3D-rendered neighbors.
// ============================================================================

/**
 * Sampled mean + standard deviation per RGB channel (sampled at `stride`
 * pixels for speed — full-res stats aren't needed for a global tone match).
 * Returns { meanR, meanG, meanB, stdR, stdG, stdB, n }. std is floored to 1
 * so it's always safe to use as a gain divisor.
 */
export function computeStats(rgba, stride = 16) {
  let n = 0, sumR = 0, sumG = 0, sumB = 0;
  for (let i = 0; i < rgba.length; i += 4 * stride) {
    sumR += rgba[i]; sumG += rgba[i + 1]; sumB += rgba[i + 2];
    n++;
  }
  n = Math.max(n, 1);
  const meanR = sumR / n, meanG = sumG / n, meanB = sumB / n;

  let varR = 0, varG = 0, varB = 0;
  for (let i = 0; i < rgba.length; i += 4 * stride) {
    varR += (rgba[i]     - meanR) ** 2;
    varG += (rgba[i + 1] - meanG) ** 2;
    varB += (rgba[i + 2] - meanB) ** 2;
  }
  return {
    meanR, meanG, meanB,
    stdR: Math.max(Math.sqrt(varR / n), 1),
    stdG: Math.max(Math.sqrt(varG / n), 1),
    stdB: Math.max(Math.sqrt(varB / n), 1),
    n,
  };
}

/**
 * Shift `rgba` so its per-channel mean AND contrast (stddev) match `target`
 * (e.g. the weighted stats of the surrounding 3D neighbors), starting from
 * `source` (this tile's own stats). This is a linear (Reinhard-style) color
 * transfer:
 *
 *   out = (in - srcMean) * gain + tgtMean,   gain = tgtStd / srcStd
 *
 * Matching contrast in addition to brightness is what makes the result look
 * "natural" rather than just tinted — a flat 2D satellite tile plopped next
 * to punchier 3D renders reads as a soft patch even after a brightness-only
 * shift. Gain is clamped to [gainMin, gainMax] to avoid amplifying noise or
 * banding on tiles with very low source variance (e.g. flat water/desert),
 * and `strength` (0..1) lets callers dial back how much of the correction
 * is actually applied.
 */
export function colorBalance(rgba, width, height, target, source, opts = {}) {
  const gainMin = opts.gainMin ?? 0.85;
  const gainMax = opts.gainMax ?? 1.15;
  const strength = opts.strength ?? 1.0;
  const clampGain = v => Math.max(gainMin, Math.min(gainMax, v));

  const gainR = clampGain(target.stdR / source.stdR);
  const gainG = clampGain(target.stdG / source.stdG);
  const gainB = clampGain(target.stdB / source.stdB);

  for (let i = 0; i < rgba.length; i += 4) {
    const nr = (rgba[i]     - source.meanR) * gainR + target.meanR;
    const ng = (rgba[i + 1] - source.meanG) * gainG + target.meanG;
    const nb = (rgba[i + 2] - source.meanB) * gainB + target.meanB;
    rgba[i]     = clamp255(rgba[i]     + (nr - rgba[i])     * strength);
    rgba[i + 1] = clamp255(rgba[i + 1] + (ng - rgba[i + 1]) * strength);
    rgba[i + 2] = clamp255(rgba[i + 2] + (nb - rgba[i + 2]) * strength);
  }
}


// ============================================================================
// Read a tile from disk into the {raw, width, height, channels} sample object.
// ============================================================================

/**
 * Read tile PNG into raw buffer + metadata. Channels forced to 4 via ensureAlpha.
 */
export async function readNeighborSamples(filePath) {
  const buf = fs.readFileSync(filePath);
  const { data, info } = await sharp(buf).ensureAlpha().raw()
    .toBuffer({ resolveWithObject: true });
  return { raw: data, width: info.width, height: info.height, channels: info.channels };
}
