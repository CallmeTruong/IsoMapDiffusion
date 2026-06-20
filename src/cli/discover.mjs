import fs from 'fs';
import path from 'path';

import { listTiles } from '../tile/tile_io.mjs';

export function scanRenderDirs(outputBase) {
  if (!fs.existsSync(outputBase)) return [];
  const dirs = [];
  for (const e of fs.readdirSync(outputBase, { withFileTypes: true })) {
    if (!e.isDirectory()) continue;
    const sub = path.join(outputBase, e.name);
    const files = fs.readdirSync(sub);
    if (files.some(f => f.startsWith('tile_') && f.endsWith('.png'))) {
      dirs.push(sub);
    }
  }

  dirs.sort((a, b) => {
    const count = d => fs.readdirSync(d).filter(f => f.startsWith('tile_') && f.endsWith('.png')).length;
    return count(b) - count(a);
  });
  return dirs;
}

export function dirHasGrid(dir, startQx, startQy, N, M) {
  const files = fs.readdirSync(dir);
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      const qx = startQx + c;
      const qy = startQy + r;
      const prefix = `tile_${qx >= 0 ? '+' + qx : qx}_${qy >= 0 ? '+' + qy : qy}_`;
      if (!files.some(f => f.startsWith(prefix) && f.endsWith('.png'))) return false;
    }
  }
  return true;
}

export function findBestContiguousGrid(dir, wantN, wantM) {
  const tiles = listTiles(dir);
  if (tiles.length === 0) return null;
  const tileSet = new Set(tiles.map(t => `${t.qx},${t.qy}`));

  let best = null;
  for (const t of tiles) {
    for (let tryM = wantM; tryM >= 1; tryM--) {
      for (let tryN = wantN; tryN >= 1; tryN--) {
        let ok = true;
        for (let r = 0; r < tryM && ok; r++) {
          for (let c = 0; c < tryN && ok; c++) {
            if (!tileSet.has(`${t.qx + c},${t.qy + r}`)) ok = false;
          }
        }
        if (ok) {
          const count = tryN * tryM;
          if (!best || count > best.count) {
            best = { dir, qx0: t.qx, qy0: t.qy, n: tryN, m: tryM, count };
          }
          break;
        }
      }
    }
  }
  return best;
}
