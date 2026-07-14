import fs from 'fs';
import path from 'path';

import { TILE, STITCH, PATHS, resolvePath } from '../config.mjs';
import { stitchGrid } from '../tile/stitch.mjs';
import { scanRenderDirs, dirHasGrid, findBestContiguousGrid } from './discover.mjs';
import { runXaxisDetect } from './xaxis.mjs';

function parseStitchArgs(positional) {
  let startQx = 0, startQy = 0, N = 2, M = 2;
  if (positional.length >= 4) [startQx, startQy, N, M] = positional.map(Number);
  else if (positional.length >= 2) [N, M] = positional.map(Number);
  return { startQx, startQy, N, M };
}

/** Parse --xaxis flag */
function parseXaxisMode(flagValue) {
  if (typeof flagValue !== 'string') return 'east';
  const v = flagValue.toLowerCase();
  if (v === 'east') return 'east';
  if (v === 'west') return 'west';
  if (v === 'auto' || v === 'detect') return 'auto';
  if (v === 'false' || v === 'off' || v === 'no') return 'east';
  return 'east';
}

function findTileFile(renderDir, qx, qy) {
  const prefix = `tile_${qx >= 0 ? '+' + qx : qx}_${qy >= 0 ? '+' + qy : qy}_`;
  for (const f of fs.readdirSync(renderDir)) {
    if (f.startsWith(prefix) && f.endsWith('.png')) return path.join(renderDir, f);
  }
  return null;
}

function buildGridTileList(renderDir, startQx, startQy, N, M, allowMissing = false) {
  const tiles = [];
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      const qx = startQx + c;
      const qy = startQy + r;
      const png = findTileFile(renderDir, qx, qy);
      if (!png) {
        if (allowMissing) {
          console.warn(`[stitch] tile (${qx},${qy}) missing — substituting transparent placeholder`);
          tiles.push({ r, c, png: TRANSPARENT_1x1_PNG, qx, qy, missing: true });
          continue;
        }
        throw new Error(`Missing tile (${qx}, ${qy}) in ${renderDir}`);
      }
      tiles.push({ r, c, png, qx, qy });
    }
  }
  return tiles;
}

const TRANSPARENT_1x1_PNG = Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489' +
  '0000000d49444154789c63000100000005000100' +
  '0d0a2db40000000049454e44ae426082',
  'hex'
);


export async function cmdStitch({ positional, flags, projectRootDir }) {
  let { startQx, startQy, N, M } = parseStitchArgs(positional);

  // ── DEBUG-ONLY MODE ────────────────────────────────────────────────────
  const debugOnly = flags['debug-only'] !== 'false' && STITCH.debugOnly !== false;
  const annotate = !debugOnly;

  // ── ALLOW MISSING (for partial grids when fallback disabled) ───────────
  const allowMissing = flags['allow-missing'] === 'true' || flags['allow-missing'] === '1';

  // ── X-AXIS MODE ───────────────────────────────────────────────────────
  const xaxisMode = parseXaxisMode(flags['xaxis']);

  // ── Resolve renderDir ─────────────────────────────────────────────────
  const outputBase = path.resolve(projectRootDir, resolvePath('output'));
  const candidateDirs = [];

  if (flags.output) {
    candidateDirs.push(path.join(projectRootDir, flags.output));
  } else {
    candidateDirs.push(...scanRenderDirs(outputBase));
  }

  let renderDir = null;
  for (const dir of candidateDirs) {
    if (dirHasGrid(dir, startQx, startQy, N, M)) {
      renderDir = dir;
      break;
    }
  }

  if (!renderDir && candidateDirs.length > 0) {
    console.log(`[stitch] No dir has full ${N}x${M} grid at offset (${startQx},${startQy}).`);
    console.log(`[stitch] Available dirs: ${candidateDirs.map(d => path.basename(d)).join(', ')}`);
    const best = findBestContiguousGrid(candidateDirs[0], N, M);
    if (best) {
      console.log(`[stitch] Best grid: ${best.n}x${best.m} at offset (${best.qx0},${best.qy0}) from ${path.basename(best.dir)} (${best.count} tiles)`);
      renderDir = best.dir;
      N = best.n; M = best.m; startQx = best.qx0; startQy = best.qy0;
    }
  }

  if (!renderDir) throw new Error(`No render dir with tiles found in ${outputBase}`);

  console.log(`[stitch] Using render dir: ${path.relative(projectRootDir, renderDir)}`);
  console.log(`[stitch] Mode: ${debugOnly ? 'DEBUG-ONLY (no annotation, output to output/poc/stitch/)' : 'FULL (with seam annotation)'}`);

  // ── X-AXIS DETECT & APPLY ────────────────────────────────────────────
  let detectedSign = null;
  if (xaxisMode === 'auto') {
    detectedSign = await runXaxisDetect({ renderDir, startQx, startQy, N });
    if (detectedSign === 'west') {
      console.log(`[stitch] ⚠ Auto-detect: sign=WEST (may be false positive due to image content)`);
      console.log(`         Default: NO flip. If you are certain a flip is needed, use --xaxis=west.`);
    } else if (detectedSign === 'east') {
      console.log(`[stitch] ✓ Auto-detect: sign=EAST (matches convention, NO flip)`);
    } else {
      console.log(`[stitch] ? Auto-detect: AMBIGUOUS → default NO flip`);
    }
  } else if (xaxisMode === 'east') {
    console.log(`[stitch] X-axis: EAST (default, NO FLIP — correct per qx→East convention)`);
  } else if (xaxisMode === 'west') {
    console.log(`[stitch] X-axis: WEST (forced, APPLY horizontal flip to all tiles)`);
  }

  const applyFlip = (xaxisMode === 'west');  // only flip when user explicitly passes --xaxis=west
  if (applyFlip) {
    console.log(`[stitch] X-axis: APPLY horizontal flip to all tiles (sign=WEST)`);
  }

  // ── Build tile list + stitch ─────────────────────────────────────────
  const tiles = buildGridTileList(renderDir, startQx, startQy, N, M, allowMissing);
  const TILE_SIZE_PX = TILE.sizePx;
  const STRIDE = Math.round(TILE_SIZE_PX * TILE.cameraMoveStep);

  const outDir = path.resolve(projectRootDir, resolvePath('pocStitch'));
  fs.mkdirSync(outDir, { recursive: true });
  const suffix = debugOnly ? 'debug' : 'annotated';
  const outPath = path.join(outDir, `stitch_${N}x${M}_offset${startQx}_${startQy}_${suffix}.png`);

  console.log(`[stitch] ${N}x${M} grid, stride=${STRIDE}px, tileSize=${TILE_SIZE_PX}px, flipX=${applyFlip}`);

  const result = await stitchGrid({
    tiles,
    gridSize: N,
    gridCols: N,
    gridRows: M,
    tileSize: TILE_SIZE_PX,
    stride: STRIDE,
    outPath,
    annotate,
    flipX: applyFlip,
    background: STITCH.background,
  });

  console.log(`[stitch] Saved: ${result.outPath}`);
  console.log(`[stitch] Size: ${(result.size/1024).toFixed(0)}KB, Map: ${result.mapW}x${result.mapH}`);
  return result;
}