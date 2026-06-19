import { scoreTile, filterQuality } from './analyze.mjs';
import { DIVERSITY_CONFIG, DEFAULT_TARGET } from './constants.mjs';


function bucketIndex(value, minVal, maxVal, numBuckets) {
  if (maxVal === minVal) return 0;
  const ratio = (value - minVal) / (maxVal - minVal);
  return Math.min(Math.floor(ratio * numBuckets), numBuckets - 1);
}

/**
 * Main selection algorithm.
 */
export function selectDiverseTiles(tiles, options = {}) {
  const target = options.target ?? DEFAULT_TARGET;
  const cfg = DIVERSITY_CONFIG;

  // 1. Filter quality
  const qualityTiles = filterQuality(tiles);
  if (qualityTiles.length === 0) return [];

  // 2. Compute score
  const scored = qualityTiles.map(t => ({ ...t, score: scoreTile(t) }));

  // 3. Compute grid bounds
  const xs = scored.map(t => t.qx);
  const ys = scored.map(t => t.qy);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  // 4. Geographic zones
  const zoneW = Math.max(1, Math.ceil((maxX - minX + 1) / cfg.gridDivisions));
  const zoneH = Math.max(1, Math.ceil((maxY - minY + 1) / cfg.gridDivisions));

  const zones = new Map();
  for (const t of scored) {
    const zx = Math.floor((t.qx - minX) / zoneW);
    const zy = Math.floor((t.qy - minY) / zoneH);
    const key = `${zx}_${zy}`;
    if (!zones.has(key)) zones.set(key, []);
    zones.get(key).push({ ...t, zoneKey: key });
  }

  // 5. Pick top-K from each zone
  const picked = [];
  for (const [zoneKey, zoneTiles] of zones) {
    zoneTiles.sort((a, b) => b.score - a.score);
    const take = Math.min(cfg.maxTilesPerZone, zoneTiles.length);
    picked.push(...zoneTiles.slice(0, take));
  }

  // 6. top score
  if (picked.length < target) {
    const pickedKeys = new Set(picked.map(t => `${t.qx}_${t.qy}`));
    const remaining = scored
      .filter(t => !pickedKeys.has(`${t.qx}_${t.qy}`))
      .sort((a, b) => b.score - a.score);
    picked.push(...remaining.slice(0, target - picked.length));
  }

  // 7. Trim
  let selected = picked.slice(0, target);

  // 8. Color bucket — re-rank
  selected = rerankByColor(selected, cfg.colorBuckets);

  return selected;
}

function rerankByColor(tiles, numBuckets) {
  const withBucket = tiles.map(t => ({
    ...t,
    colorBucket: bucketIndex(t.meanR ?? 128, 0, 255, numBuckets),
  }));

  withBucket.sort((a, b) => {
    if (a.colorBucket !== b.colorBucket) return a.colorBucket - b.colorBucket;
    return b.score - a.score;
  });

  return withBucket;
}


export function summarizeSelection(selected, allTiles) {
  if (selected.length === 0) {
    return { count: 0, zones: 0, colorBuckets: new Set() };
  }
  return {
    count:        selected.length,
    totalInPool:  allTiles.length,
    zones:        new Set(selected.map(t => t.zoneKey)).size,
    colorBuckets: new Set(selected.map(t => t.colorBucket)),
    avgScore:     selected.reduce((a, t) => a + t.score, 0) / selected.length,
    minScore:     Math.min(...selected.map(t => t.score)),
    maxScore:     Math.max(...selected.map(t => t.score)),
  };
}