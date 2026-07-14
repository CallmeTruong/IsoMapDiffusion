import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { TILE } from '../config.mjs';

const HASH_LEN = TILE.hashLength;



export function signInt(n) {
  return n >= 0 ? `+${n}` : `${n}`;
}

// Regex: tile_[+-]\d+_[+-]\d+_<hash>.png
export const TILE_FILE_RE = new RegExp(`^tile_[+-]\\d+_[+-]\\d+_([a-f0-9]{${HASH_LEN}})\\.png$`);

// Regex: tile_[+-]\d+_[+-]\d+.json
export const TILE_META_RE = /^tile_[+-]\d+_[+-]\d+\.json$/;

export function parseTileFilename(name) {
  const m = name.match(TILE_FILE_RE);
  if (!m) return null;
  const noPrefix = name.slice('tile_'.length, -'.png'.length);
  const lastUnderscore = noPrefix.lastIndexOf('_');
  const qxStr = noPrefix.slice(0, noPrefix.indexOf('_'));
  const qyStr = noPrefix.slice(noPrefix.indexOf('_') + 1, lastUnderscore);
  return {
    qx: Number(qxStr),
    qy: Number(qyStr),
    hash: m[1],
  };
}


export function tileFilename(qx, qy, hash) {
  return `tile_${signInt(qx)}_${signInt(qy)}_${hash}.png`;
}


export function tileMetaFilename(qx, qy) {
  return `tile_${signInt(qx)}_${signInt(qy)}.json`;
}


export function computeHash(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex').slice(0, HASH_LEN);
}


export function tileMetaPath(outputDir, qx, qy) {
  return path.join(outputDir, 'meta', tileMetaFilename(qx, qy));
}

export function tileDeletedMarkerPath(outputDir, qx, qy) {
  return path.join(outputDir, `.deleted_tile_${signInt(qx)}_${signInt(qy)}`);
}

function safeUnlink(fp) {
  try { fs.unlinkSync(fp); } catch { /* ignore */ }
}

export function saveTile(pngBuf, tileInfo, meta, outputDir) {
  const { qx, qy } = tileInfo;
  const hash = computeHash(pngBuf);
  const filename = tileFilename(qx, qy, hash);
  const filepath = path.join(outputDir, filename);
  const metaPath = tileMetaPath(outputDir, qx, qy);

  fs.writeFileSync(filepath, pngBuf);

  const fullMeta = {
    id: `tile_${signInt(qx)}_${signInt(qy)}_${hash}`,
    qx, qy,
    hash_sha256: hash,
    size_kb: +(pngBuf.length / 1024).toFixed(2),
    saved_at: new Date().toISOString(),
    ...meta,
  };

  fs.mkdirSync(path.dirname(metaPath), { recursive: true });
  fs.writeFileSync(metaPath, JSON.stringify(fullMeta, null, 2));

  return { filename, hash, filepath, metaPath, sizeKB: fullMeta.size_kb, meta: fullMeta };
}

export function findTile(outputDir, qx, qy) {
  const prefix = `tile_${signInt(qx)}_${signInt(qy)}_`;
  let match = null;
  try {
    const files = fs.readdirSync(outputDir);
    for (const f of files) {
      if (f.startsWith(prefix) && TILE_FILE_RE.test(f)) {
        match = f;
        break;
      }
    }
  } catch {
    return null;
  }
  if (!match) return null;

  const filepath = path.join(outputDir, match);
  const { hash } = parseTileFilename(match);

  let meta = null;
  const metaPath = tileMetaPath(outputDir, qx, qy);
  if (fs.existsSync(metaPath)) {
    try {
      meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    } catch { /* corrupt meta */ }
  }

  return {
    qx, qy,
    filename: match,
    filepath,
    hash,
    meta,
  };
}

export function deleteTileFiles(outputDir, qx, qy) {
  const prefix = `tile_${signInt(qx)}_${signInt(qy)}_`;
  try {
    for (const f of fs.readdirSync(outputDir)) {
      if (f.startsWith(prefix) && TILE_FILE_RE.test(f)) {
        safeUnlink(path.join(outputDir, f));
      }
    }
  } catch { /* dir missing */ }
  safeUnlink(tileMetaPath(outputDir, qx, qy));
}

export function markTileDeleted(outputDir, qx, qy) {
  const fp = tileDeletedMarkerPath(outputDir, qx, qy);
  try {
    fs.writeFileSync(fp, JSON.stringify({
      qx, qy,
      markedAt: new Date().toISOString(),
    }));
  } catch { /* best effort */ }
}


export function clearTileDeletedMarker(outputDir, qx, qy) {
  safeUnlink(tileDeletedMarkerPath(outputDir, qx, qy));
}


export function deleteTileWithMarker(outputDir, qx, qy) {
  markTileDeleted(outputDir, qx, qy);
  deleteTileFiles(outputDir, qx, qy);
}



export function listTiles(outputDir) {
  let files = [];
  try {
    files = fs.readdirSync(outputDir);
  } catch {
    return [];
  }

  const result = [];
  for (const f of files) {
    const parsed = parseTileFilename(f);
    if (!parsed) continue;
    const filepath = path.join(outputDir, f);
    const metaPath = tileMetaPath(outputDir, parsed.qx, parsed.qy);
    let meta = null;
    if (fs.existsSync(metaPath)) {
      try {
        meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
      } catch { /* ignore */ }
    }
    result.push({
      qx: parsed.qx,
      qy: parsed.qy,
      hash: parsed.hash,
      filename: f,
      filepath,
      meta,
    });
  }
  return result;
}

export function tileBounds(tiles) {
  if (tiles.length === 0) {
    return { minQx: 0, maxQx: 0, minQy: 0, maxQy: 0, count: 0, expectedTiles: 0 };
  }
  let minQx = Infinity, maxQx = -Infinity;
  let minQy = Infinity, maxQy = -Infinity;
  for (const t of tiles) {
    if (t.qx < minQx) minQx = t.qx;
    if (t.qx > maxQx) maxQx = t.qx;
    if (t.qy < minQy) minQy = t.qy;
    if (t.qy > maxQy) maxQy = t.qy;
  }
  const count = (maxQx - minQx + 1) * (maxQy - minQy + 1);
  return { minQx, maxQx, minQy, maxQy, count: tiles.length, expectedTiles: count };
}
