import fs from 'fs';
import path from 'path';
import { TILE } from '../config.mjs';
import {
  findTile,
  tileMetaPath,
  tileDeletedMarkerPath,
} from './tile_io.mjs';


export function getThresholds(overrides = {}) {
  return {
    blankSizeKb:      TILE.blankSizeKb,
    blankVarianceThr: TILE.blankVarianceThr,
    blankEdgeThr:     TILE.blankEdgeThr,
    ...overrides,
  };
}



function isMetaBad(meta, thr) {
  if (!meta) return false;
  if (meta.isBlank === true) return true;
  if (typeof meta.variance === 'number' && meta.variance < thr.blankVarianceThr) return true;
  if (meta.meanR !== undefined && meta.meanR > 250) return true;
  if (typeof meta.edgeDensity === 'number' && meta.edgeDensity < thr.blankEdgeThr) return true;
  return false;
}

function safeUnlink(fp) {
  try { fs.unlinkSync(fp); } catch { /* ignore */ }
}



export {
  markTileDeleted,
  clearTileDeletedMarker as clearDeletionMarker,
  deleteTileFiles as deleteTile,
  tileDeletedMarkerPath as deletedMarkerPath,
  tileMetaPath,
} from './tile_io.mjs';



export function isTileValid(outputDir, qx, qy, thr = null) {
  const t = thr || getThresholds();

  const tile = findTile(outputDir, qx, qy);

  if (!tile) return false;

  // Check size
  let stat;
  try {
    stat = fs.statSync(tile.filepath);
  } catch {
    return false;
  }
  if (stat.size / 1024 < t.blankSizeKb) {
    safeUnlink(tile.filepath);
    safeUnlink(tileMetaPath(outputDir, qx, qy));
    return false;
  }

  // Check meta
  if (tile.meta) {
    if (isMetaBad(tile.meta, t)) {
      safeUnlink(tile.filepath);
      safeUnlink(tileMetaPath(outputDir, qx, qy));
      return false;
    }
  }

  return true;
}

export function getTileInvalidReason(outputDir, qx, qy, thr = null, wasRenderedBefore = false) {
  const t = thr || getThresholds();

  // 1. Check marker `.deleted_tile_*`
  if (fs.existsSync(tileDeletedMarkerPath(outputDir, qx, qy))) {
    return 'user-deleted';
  }

  const tile = findTile(outputDir, qx, qy);

  // 2. File not exist
  if (!tile) {
    return wasRenderedBefore ? 'user-deleted' : 'missing';
  }

  // 3. CORRUPT
  let stat;
  try {
    stat = fs.statSync(tile.filepath);
  } catch {
    return wasRenderedBefore ? 'user-deleted' : 'missing';
  }
  if (stat.size / 1024 < t.blankSizeKb) {
    return 'corrupt';
  }

  // 4.BLURRY
  if (tile.meta && isMetaBad(tile.meta, t)) return 'blurry';

  return null;
}

// Priority order: user-deleted > missing > corrupt > blurry


export function filterPendingTiles(tiles, outputDir, blankSizeKb = null) {
  if (!fs.existsSync(outputDir)) return tiles.slice();
  if (tiles.length === 0) return [];

  const thr = blankSizeKb != null
    ? { ...getThresholds(), blankSizeKb }
    : getThresholds();

  return tiles.filter(tile => !isTileValid(outputDir, tile.qx, tile.qy, thr));
}


export function filterAndTagPending(tiles, outputDir, blankSizeKb = null, doneSet = null) {
  if (!fs.existsSync(outputDir)) {
    return tiles.map(t => ({ ...t, _invalidReason: 'missing' }));
  }
  if (tiles.length === 0) return [];

  const thr = blankSizeKb != null
    ? { ...getThresholds(), blankSizeKb }
    : getThresholds();

  return tiles
    .map(t => {
      const wasRenderedBefore = doneSet?.has(`${t.qx},${t.qy}`) ?? false;
      const reason = getTileInvalidReason(outputDir, t.qx, t.qy, thr, wasRenderedBefore);
      return reason ? { ...t, _invalidReason: reason } : null;
    })
    .filter(Boolean);
}


const PRIORITY_ORDER = { 'user-deleted': 0, missing: 1, corrupt: 2, blurry: 3 };


export function sortPendingByPriority(pending, outputDir, blankSizeKb = null, doneSet = null) {
  if (pending.length === 0) return [];

  // Tag reason
  const tagged = pending.map(t => {
    if (t._invalidReason) return t;
    const wasRenderedBefore = doneSet?.has(`${t.qx},${t.qy}`) ?? false;
    const r = getTileInvalidReason(outputDir, t.qx, t.qy, null, wasRenderedBefore);
    return { ...t, _invalidReason: r || 'unknown' };
  });

  return tagged.sort((a, b) => {
    const pa = PRIORITY_ORDER[a._invalidReason] ?? 99;
    const pb = PRIORITY_ORDER[b._invalidReason] ?? 99;
    if (pa !== pb) return pa - pb;
    return (a.qx - b.qx) || (a.qy - b.qy);
  });
}

export function isTileFullyRendered(tile, outputDir, blankSizeKb = null) {
  if (!fs.existsSync(outputDir)) return false;
  const thr = blankSizeKb != null
    ? { ...getThresholds(), blankSizeKb }
    : getThresholds();
  return isTileValid(outputDir, tile.qx, tile.qy, thr);
}
