import fs from 'fs';
import os from 'os';
import path from 'path';
import { Worker } from 'worker_threads';

import { TILE, TILE_SIZE_M, CELL_SIZE_M, WORKERS, PATHS, GOOGLE, TMP, CHECKPOINT, PROVIDERS, resolvePath } from './config.mjs';
import { getGoogleKeys, parseArgs } from './config.mjs';
import {
  projectRoot, loadEnv, resolveNumWorkers, buildRenderCfg, RENDER_DEFAULTS,
  resolveCredentials, getProvider, resolveApiKey,
  validateAllGoogleKeys, chunkArray, exportRenderOutputs,
} from './render/index.mjs';
import {
  computeSeedPoint, cellsToQuadrants, computeTiles,
  filterPendingTiles, sortPendingByPriority,
  initDB, insertOrIgnoreQuadrant, updateQuadrantAttempt,
  loadCheckpoint, saveCheckpoint, CheckpointTracker, isTileFullyRendered,
} from './tile/index.mjs';

// ─── Setup ────────────────────────────────────────────────────────────────────

const __dirname = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ''));
const PROJECT_ROOT = projectRoot(import.meta.url);
loadEnv(PROJECT_ROOT);

const ARGS = parseArgs();
const PROVIDER_ID = ARGS.provider ?? PROVIDERS.default;
const PROVIDER = getProvider(PROVIDER_ID);
const API_KEY = resolveApiKey(PROVIDER_ID, ARGS['api-key']);
const MANIFEST_FILE = ARGS.manifest ?? resolvePath('manifest');
const OUTPUT_DIR    = ARGS.output   ?? resolvePath('renders');
const DB_PATH       = ARGS.db       ?? resolvePath('db');

const NUM_WORKERS = resolveNumWorkers(ARGS.workers);
const RENDER_CFG  = buildRenderCfg(ARGS);

const SESSION_MAX_MS   = TILE.sessionMaxMs;
const PROTOCOL_TIMEOUT = TILE.protocolTimeout;
const BLANK_SIZE_KB    = Number(ARGS.blanksize ?? TILE.blankSizeKb);
const MAX_RETRY        = Number(ARGS.retry     ?? TILE.maxRetry);
const CHECKPOINT_PATH  = path.join(OUTPUT_DIR, PATHS.checkpoint);

// ─── Temp dir ─────────────────────────────────────────────────────────────────

const TMP_DIR = path.join(os.tmpdir(), TMP.dirName);
fs.mkdirSync(TMP_DIR, { recursive: true });
const tmpHtml   = id => path.join(TMP_DIR, `${TMP.htmlPrefix}${id}.html`);
const userData  = id => path.join(TMP_DIR, `${TMP.profilePrefix}${id}`);

function cleanupTemp() {
  try {
    fs.readdirSync(TMP_DIR)
      .filter(f => f.startsWith(TMP.profilePrefix) || f.startsWith(TMP.htmlPrefix))
      .forEach(f => fs.rmSync(path.join(TMP_DIR, f), { recursive: true, force: true }));
  } catch {}
}

// ─── Validate inputs ──────────────────────────────────────────────────────────

if (!API_KEY) {
  console.error(`Need credentials for provider '${PROVIDER_ID}'.`);
  console.error(`Set in .env: ${PROVIDER.envVar}=... or flag --api-key=...`);
  process.exit(1);
}
if (!fs.existsSync(MANIFEST_FILE)) {
  console.error('Not found ' + MANIFEST_FILE);
  process.exit(1);
}

const validKeys = await PROVIDER.validateKey(API_KEY) ? [API_KEY] : [];
if (validKeys.length === 0) {
  console.error(`\n${PROVIDER.envVar} Not valid (provider: ${PROVIDER_ID}). Please check key/token/permission.`);
  process.exit(1);
}
console.log(`  ${PROVIDER.displayName}: OK (${PROVIDER.envVar}=...${API_KEY.slice(-6)})\n`);

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
cleanupTemp();
try {
  const legacyPrefixes = [TMP.profilePrefix, TMP.htmlPrefix];
  fs.readdirSync('.')
    .filter(f => legacyPrefixes.some(p => f.startsWith(p)))
    .forEach(f => { fs.rmSync(f, { recursive: true, force: true }); console.log('clean old file: ' + f); });
} catch {}

// ─── Load manifest + grid ─────────────────────────────────────────────────────

const manifest = JSON.parse(fs.readFileSync(MANIFEST_FILE, 'utf8'));
const seedInfo = computeSeedPoint(manifest, path.dirname(MANIFEST_FILE));
const { seedLat, seedLng } = seedInfo;
console.log(`Seed point: (${seedLat.toFixed(6)}, ${seedLng.toFixed(6)}) [source: ${seedInfo.source}]`);

const cellQxy = cellsToQuadrants(manifest, seedLat, seedLng, RENDER_CFG.frustumW, TILE.sizePx);
const { tiles, quadrantStatus } = computeTiles(cellQxy);

console.log(`\n─── Quadrant Grid ───`);
console.log(`Cells in manifest: ${manifest.length}`);
console.log(`Unique quadrants: ${quadrantStatus.size}`);
console.log(`Tiles to render: ${tiles.length}`);

const statusCounts = { LAND: 0, INFRA: 0, SKIP: 0 };
for (const status of quadrantStatus.values()) {
  statusCounts[status] = (statusCounts[status] || 0) + 1;
}
console.log('Quadrant status:', statusCounts);

// ─── SQLite DB ────────────────────────────────────────────────────────────────

console.log('\n─── SQLite DB ───');
const dbInit = await initDB(DB_PATH);
if (dbInit) {
  console.log('DB: ' + DB_PATH);
  for (const [key, status] of quadrantStatus) {
    const [qx, qy] = key.split(',').map(Number);
    await insertOrIgnoreQuadrant(qx, qy, status);
  }
  console.log('Inserted quadrants into DB');
}

// ─── Filter pending + checkpoint ──────────────────────────────────────────────

const checkpoint = loadCheckpoint(CHECKPOINT_PATH);
let doneSet = null;
if (checkpoint && Array.isArray(checkpoint.doneTiles)) {
  doneSet = new Set(checkpoint.doneTiles.map(t => `${t.qx},${t.qy}`));
}

let pending = filterPendingTiles(tiles, OUTPUT_DIR, BLANK_SIZE_KB, doneSet);

if (doneSet) {
  const before = pending.length;
  const workingDoneSet = new Set(doneSet);
  pending = pending.filter(t => {
    if (!workingDoneSet.has(`${t.qx},${t.qy}`)) return true;
    if (isTileFullyRendered(t, OUTPUT_DIR, BLANK_SIZE_KB)) return false;
    workingDoneSet.delete(`${t.qx},${t.qy}`);
    doneSet.delete(`${t.qx},${t.qy}`);
    return true;
  });
  const remaining = checkpoint.doneTiles.filter(t => workingDoneSet.has(`${t.qx},${t.qy}`));
  if (remaining.length !== checkpoint.doneTiles.length) {
    saveCheckpoint(CHECKPOINT_PATH, remaining);
    console.log(`Checkpoint: invalidated ${checkpoint.doneTiles.length - remaining.length} tiles (file missing/corrupt)`);
  }
  console.log(`Checkpoint: skip ${before - pending.length} tiles complete`);
}

pending = sortPendingByPriority(pending, OUTPUT_DIR, BLANK_SIZE_KB, doneSet);

if (pending.length > 0) {
  const counts = { missing: 0, corrupt: 0, blurry: 0, unknown: 0 };
  for (const t of pending) counts[t._invalidReason] = (counts[t._invalidReason] || 0) + 1;
  console.log(`Priority: missing=${counts.missing} corrupt=${counts.corrupt} blurry=${counts.blurry}`);
}
console.log('FRUSTUM_W:', TILE_SIZE_M, 'CELL:', CELL_SIZE_M);

console.log('\n─── Render Config ───');
console.log('Tiles to render:', pending.length, '/', tiles.length);
console.log('Workers:', NUM_WORKERS, `(cpus=${os.cpus().length}, config.default=${WORKERS.default}, max=${WORKERS.max})`);
console.log(`Provider: ${PROVIDER.displayName} (${validKeys.length} key(s) valid)`);

if (pending.length === 0) {
  console.log('All tiles done.');
  process.exit(0);
}

const estSecsPerTile = (RENDER_CFG.tileWaitMs * 0.6 + RENDER_CFG.settleMaxMs * 0.5) / 1000 + 0.5;
const estTotal = Math.round(pending.length * estSecsPerTile / NUM_WORKERS);
console.log(`ETA ~ ${Math.floor(estTotal/60)}m${estTotal%60}s\n`);

// ─── Dispatch workers ─────────────────────────────────────────────────────────

const chunks = chunkArray(pending, NUM_WORKERS);
const workerTemplate = {
  provider: PROVIDER_ID,
  apiKey: validKeys[0],
  outputDir: OUTPUT_DIR,
  blankSizeKb: BLANK_SIZE_KB,
  maxRetry: MAX_RETRY,
  cfg: RENDER_CFG,
  seedLng, seedLat,
  protocolTimeout: PROTOCOL_TIMEOUT,
  sessionMaxMs: SESSION_MAX_MS,
  checkpointPath: CHECKPOINT_PATH,
};

const workerStats = Array.from({ length: chunks.length }, () => ({
  done: 0, fail: 0, retry: 0, sessions: 0,
}));

const startTime = Date.now();
const checkpointTracker = new CheckpointTracker(CHECKPOINT_PATH, CHECKPOINT.flushIntervalMs);

function printStatus() {
  const totalDone  = workerStats.reduce((s, w) => s + w.done, 0);
  const totalFail  = workerStats.reduce((s, w) => s + w.fail, 0);
  const totalRetry = workerStats.reduce((s, w) => s + w.retry, 0);
  const elapsed    = (Date.now() - startTime) / 1000;
  const rate       = totalDone / (elapsed || 1);
  const eta        = Math.round((pending.length - totalDone - totalFail) / (rate || 0.01));
  process.stdout.write(
    `\r[${workerStats.map((w, i) => `W${i}:${w.done}`).join(' ')}]` +
    `  tiles=${totalDone}/${pending.length}  fail=${totalFail}  retry=${totalRetry}` +
    `  ${rate.toFixed(1)}/s  ETA ${Math.floor(eta/60)}m${eta%60}s`
  );
}

const mainCleanup = () => {
  cleanupTemp();
  try { checkpointTracker.close(); } catch {}
};
process.on('SIGINT',  () => { console.log('\nInterrupted.'); mainCleanup(); process.exit(1); });
process.on('SIGTERM', () => { mainCleanup(); process.exit(1); });

const workerPromises = chunks.map((chunk, i) => new Promise((resolve, reject) => {
  const cfg = { ...workerTemplate, tiles: chunk, workerId: i, tmpHtmlPath: tmpHtml(i), userDataDir: userData(i) };
  const worker = new Worker(new URL('./tile/worker_entry.mjs', import.meta.url), { workerData: cfg });

  worker.on('message', async msg => {
    if (msg.type === 'progress') {
      const m = msg.msg.match(/(\d+)\/(\d+) tiles done.*?(\d+) fail.*?(\d+) retry/);
      if (m) {
        workerStats[i].done  = parseInt(m[1]);
        workerStats[i].fail  = parseInt(m[3]);
        workerStats[i].retry = parseInt(m[4]);
        printStatus();
      }
    } else if (msg.type === 'tile_done') {
      checkpointTracker.markDone(msg.qx, msg.qy);
    } else if (msg.type === 'db_update') {
      if (dbInit) {
        try { await updateQuadrantAttempt(msg.qx, msg.qy, msg.info); }
        catch (e) { console.warn(`[db] update failed for (${msg.qx},${msg.qy}): ${e.message}`); }
      }
    } else if (msg.type === 'done') {
      workerStats[i].done     = msg.stats.doneCount;
      workerStats[i].fail     = msg.stats.failCount;
      workerStats[i].retry    = msg.stats.retryCount;
      workerStats[i].sessions = msg.stats.sessionCount;
      printStatus();
      resolve(msg.stats);
    } else if (msg.type === 'error') {
      console.error(`\nWorker ${i} error:`, msg.error);
      reject(new Error(msg.error));
    }
  });
  worker.on('error', reject);
  worker.on('exit', code => {
    if (code !== 0) reject(new Error(`Worker ${i} exited with code ${code}`));
  });
}));

let allStats;
try {
  allStats = await Promise.all(workerPromises);
} catch (e) {
  console.error('\nWorker crashed:', e.message);
}

mainCleanup();

const totalDone  = allStats?.reduce((s, w) => s + (w?.doneCount  ?? 0), 0) ?? 0;
const totalFail  = allStats?.reduce((s, w) => s + (w?.failCount  ?? 0), 0) ?? 0;
const totalRetry = allStats?.reduce((s, w) => s + (w?.retryCount ?? 0), 0) ?? 0;
const totalSess  = allStats?.reduce((s, w) => s + (w?.sessionCount ?? 0), 0) ?? 0;
const elapsed    = ((Date.now() - startTime) / 1000).toFixed(1);

console.log('\n\n=== Summary ===');
console.log('  Tiles rendered: ' + totalDone);
console.log('  Failed: ' + totalFail);
console.log('  Retried: ' + totalRetry);
console.log('  Elapsed: ' + Math.floor(elapsed/60) + 'm' + Math.round(elapsed%60) + 's');
console.log('  Rate: ' + (totalDone / elapsed).toFixed(2) + ' tiles/s');

if (totalFail > 0) console.log('\n' + totalFail + ' tiles failed — check render_errors_w*.log');

// ─── Export outputs ──────────────────────────────────────────────────────────

console.log('\n─── Generate quadrants.geojson + generation_config.json ───');
const { geojsonPath, configPath } = exportRenderOutputs({
  quadrantStatus, outputDir: OUTPUT_DIR, seedLat, seedLng, cfg: RENDER_CFG,
  blankSizeKb: BLANK_SIZE_KB, manifestFile: MANIFEST_FILE, projectRoot: PROJECT_ROOT,
  sessionCount: totalSess,
});
console.log('Saved:', geojsonPath);
console.log('Saved:', configPath);
console.log('\n─── Export render complete ───');
console.log('  Output dir: ' + OUTPUT_DIR);
console.log('  DB: ' + DB_PATH);
console.log('  GeoJSON: ' + geojsonPath);
console.log('  Config: ' + configPath + '\n');
