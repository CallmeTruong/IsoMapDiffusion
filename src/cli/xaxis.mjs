import { TILE } from '../config.mjs';
import { findTile } from '../tile/tile_io.mjs';
import { detectXAxisSignFromPairs } from '../tile/xaxis_detect.mjs';

/**
 * @param {Object} opts
 * @param {string} opts.renderDir
 * @param {number} opts.startQx
 * @param {number} opts.startQy
 * @param {number} opts.N
 * @returns {Promise<'east'|'west'|'ambiguous'>}
 */
export async function runXaxisDetect({ renderDir, startQx, startQy, N }) {
  const TILE_SIZE_PX = TILE.sizePx;
  const STRIDE = Math.round(TILE_SIZE_PX * TILE.cameraMoveStep);
  const OVERLAP = TILE_SIZE_PX - STRIDE;

  if (N < 2) {
    console.log(`[xaxis] Grid only has ${N} column(s), cannot detect sign (need ≥ 2 columns)`);
    return 'ambiguous';
  }

  // Get horizontally adjacent tile pairs (first row)
  const pairs = [];
  for (let c = 0; c < N - 1; c++) {
    const qxL = startQx + c;
    const qxR = startQx + c + 1;
    const qy = startQy;
    const tileL = findTile(renderDir, qxL, qy);
    const tileR = findTile(renderDir, qxR, qy);
    if (!tileL || !tileR) continue;
    pairs.push({ left: tileL.filepath, right: tileR.filepath });
  }
  if (pairs.length === 0) {
    console.log(`[xaxis] No tile pairs found for detection`);
    return 'ambiguous';
  }

  console.log(`[xaxis] Analyzing ${pairs.length} tile pair(s) (overlap=${OVERLAP}px)...`);

  const result = await detectXAxisSignFromPairs({
    pairs, tileSize: TILE_SIZE_PX, overlap: OVERLAP,
  });

  const conf = (() => {
    if (result.sign === 'ambiguous') return 'low';
    const confCounts = { high: 0, medium: 0, low: 0 };
    for (const d of result.details) confCounts[d.confidence]++;
    if (confCounts.high >= pairs.length / 2) return 'high';
    if (confCounts.medium + confCounts.high >= pairs.length / 2) return 'medium';
    return 'low';
  })();

  console.log(`[xaxis] Result:`);
  console.log(`         sign       = ${result.sign.toUpperCase()}`);
  console.log(`         confidence = ${conf}`);
  console.log(`         avgRatio   = ${result.avgRatio.toFixed(3)}  (>1 = east, <1 = west)`);
  console.log(`         votes      = east:${result.eastVotes} west:${result.details.length - result.eastVotes}`);
  for (let i = 0; i < result.details.length; i++) {
    const d = result.details[i];
    console.log(`         pair ${i + 1}: ratio=${d.ratio.toFixed(3)} mse_n=${d.mse_normal.toFixed(1)} mse_f=${d.mse_flipped.toFixed(1)} → ${d.sign}`);
  }

  if (result.sign === 'ambiguous') {
    console.log(`[xaxis] ⚠ Could not determine sign with confidence — will NOT flip (keeping east as default). Possible causes:`);
    console.log(`         - Tiles are too empty (both blank) — try re-rendering with richer content`);
    console.log(`         - Camera misaligned — check azimuth/elevation settings`);
    console.log(`         - Use --xaxis=west or --xaxis=east to force the sign manually`);
  } else if (result.sign === 'west') {
    console.log(`[xaxis] → Will apply horizontal FLIP to all tiles during stitch (sign=WEST)`);
    console.log(`         Note: if result looks wrong, review tileIndexToLatLng in coords.mjs`);
  } else {
    console.log(`[xaxis] → Stitching without flip (sign=EAST, no flip applied)`);
  }

  return result.sign;
}