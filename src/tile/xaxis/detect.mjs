import sharp from 'sharp';
import { XAXIS, TILE } from '../../config.mjs';
import { mseRaw, mirrorHorizontal } from './compare.mjs';

const DEFAULT_OVERLAP        = XAXIS.overlap;
const DEFAULT_SAMPLE_HEIGHT  = XAXIS.sampleHeight;
const DEFAULT_RESIZE_WIDTH   = XAXIS.resizeWidth;
const DEFAULT_RATIO_THRESHOLD = XAXIS.ratioThreshold;
const HIGH_CONF_RATIO        = XAXIS.highConfRatio;
const MEDIUM_CONF_RATIO      = XAXIS.mediumConfRatio;
const DEFAULT_TILE_SIZE      = TILE.sizePx;


async function extractOverlapPair(tileA, tileB, opts = {}) {
  const tileSize = opts.tileSize ?? DEFAULT_TILE_SIZE;
  const overlap  = opts.overlap ?? DEFAULT_OVERLAP;
  const resizeW  = opts.resizeWidth ?? DEFAULT_RESIZE_WIDTH;
  const sampleH  = opts.sampleHeight ?? DEFAULT_SAMPLE_HEIGHT;

  if (overlap < 1 || overlap > tileSize) {
    throw new Error(`extractOverlapPair: overlap=${overlap} must be in [1, ${tileSize}]`);
  }

  const leftExtract = await sharp(tileA)
    .extract({ left: tileSize - overlap, top: 0, width: overlap, height: sampleH })
    .resize({ width: resizeW })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rightExtract = await sharp(tileB)
    .extract({ left: 0, top: 0, width: overlap, height: sampleH })
    .resize({ width: resizeW })
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    left:  leftExtract.data,
    right: rightExtract.data,
    width:  leftExtract.info.width,
    height: leftExtract.info.height,
  };
}


export async function detectXAxisSign({ tileLeft, tileRight, tileSize = DEFAULT_TILE_SIZE, overlap = DEFAULT_OVERLAP }) {
  const pair = await extractOverlapPair(tileLeft, tileRight, { tileSize, overlap });
  const { left, right, width, height } = pair;

  // MSE 1
  const mseNormal = mseRaw(left, right, width, height);

  // MSE 2
  const rightMirrored = mirrorHorizontal(right, width, height);
  const mseFlipped = mseRaw(left, rightMirrored, width, height);

  // Ratio
  const ratio = mseNormal / Math.max(mseFlipped, 1e-6);

  let sign, confidence;
  const T = DEFAULT_RATIO_THRESHOLD;
  if (ratio > T) {
    sign = 'east';
    confidence = ratio > HIGH_CONF_RATIO ? 'high' : ratio > MEDIUM_CONF_RATIO ? 'medium' : 'low';
  } else if (ratio < 1 / T) {
    sign = 'west';
    confidence = ratio < 1 / HIGH_CONF_RATIO ? 'high' : ratio < 1 / MEDIUM_CONF_RATIO ? 'medium' : 'low';
  } else {
    sign = 'ambiguous';
    confidence = 'low';
  }

  return { sign, mse_normal: mseNormal, mse_flipped: mseFlipped, ratio, confidence };
}


export async function detectXAxisSignFromPairs({ pairs, tileSize = DEFAULT_TILE_SIZE, overlap = DEFAULT_OVERLAP }) {
  const details = [];
  for (const p of pairs) {
    const r = await detectXAxisSign({
      tileLeft: p.left, tileRight: p.right, tileSize, overlap,
    });
    details.push(r);
  }

  let eastVotes = 0, westVotes = 0, ambiguousVotes = 0;
  let sumRatio = 0;
  for (const d of details) {
    sumRatio += d.ratio;
    if (d.sign === 'east') eastVotes++;
    else if (d.sign === 'west') westVotes++;
    else ambiguousVotes++;
  }
  const avgRatio = details.length > 0 ? sumRatio / details.length : 1.0;

  let sign;
  if (eastVotes > westVotes && eastVotes > ambiguousVotes) sign = 'east';
  else if (westVotes > eastVotes && westVotes > ambiguousVotes) sign = 'west';
  else sign = 'ambiguous';

  return {
    sign,
    pairsChecked: details.length,
    avgRatio,
    eastVotes,
    westVotes,
    details,
  };
}
