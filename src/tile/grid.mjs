import fs from 'fs';
import path from 'path';
import { latLngToQxy } from './coords.mjs';
import { CELL_SIZE_M } from '../config.mjs';

export function computeSeedPoint(manifest, outputDir = null) {
  if (outputDir) {
    const seedFile = path.join(outputDir, 'seed.json');
    if (fs.existsSync(seedFile)) {
      try {
        const s = JSON.parse(fs.readFileSync(seedFile, 'utf8'));
        if (typeof s.seed_lat === 'number' && typeof s.seed_lng === 'number') {
          return { seedLat: s.seed_lat, seedLng: s.seed_lng, source: 'seed.json' };
        }
      } catch { /* fall through */ }
    }
  }

  if (!manifest || manifest.length === 0) {
    throw new Error('computeSeedPoint: empty manifest and no seed.json');
  }
  const lats = manifest.map(c => c.centroid_lat);
  const lngs = manifest.map(c => c.centroid_lng);
  return {
    seedLat: (Math.min(...lats) + Math.max(...lats)) / 2,
    seedLng: (Math.min(...lngs) + Math.max(...lngs)) / 2,
    source: 'manifest-fallback',
  };
}


export function cellsToQuadrants(manifest, seedLat, seedLng, frustumW, tileSizePx) {
  const tileStepM = CELL_SIZE_M;  // 200m

  return manifest.map(c => {
    const { qx, qy } = latLngToQxy(c.centroid_lat, c.centroid_lng, seedLat, seedLng, tileStepM);
    return { ...c, qx, qy };
  });
}

export function computeTiles(cellQxy) {
  const tileSet = new Set();
  const quadrantStatus = new Map();

  for (const c of cellQxy) {
    const key = `${c.qx},${c.qy}`;
    const existing = quadrantStatus.get(key);
    if (!existing || (existing === 'SKIP' && c.status !== 'SKIP') ||
        (existing !== 'INFRA' && c.status === 'INFRA')) {
      quadrantStatus.set(key, c.status);
    }

    tileSet.add(key);
  }

  const tiles = Array.from(tileSet).map(t => {
    const [qx, qy] = t.split(',').map(Number);
    return { qx, qy };
  }).sort((a, b) => (a.qx - b.qx) || (a.qy - b.qy));

  return { tiles, quadrantStatus };
}

export function getQuadrantBounds(quadrantStatus) {
  let minQx = Infinity, maxQx = -Infinity;
  let minQy = Infinity, maxQy = -Infinity;

  for (const key of quadrantStatus.keys()) {
    const [qx, qy] = key.split(',').map(Number);
    minQx = Math.min(minQx, qx);
    maxQx = Math.max(maxQx, qx);
    minQy = Math.min(minQy, qy);
    maxQy = Math.max(maxQy, qy);
  }

  return { minQx, maxQx, minQy, maxQy };
}
