import fs from 'fs';
import os from 'os';
import path from 'path';

import { Worker } from 'worker_threads';

import { TILE, WORKERS, PATHS, TMP, PROVIDERS, FALLBACK, resolvePath } from '../config.mjs';
import { buildRenderCfg, RENDER_DEFAULTS, resolveCredentials } from '../render/env.mjs';
import { chunkArray } from '../render/dispatch.mjs';
import { tileIndexToLatLng } from '../tile/coords.mjs';
import { getSeed } from './seed.mjs';

export async function runRender({ tiles, seed, outputDir, cfg, workers = 1, provider, apiKey }) {
  const { providerId, apiKey: resolvedKey, provider: prov } = resolveCredentials({
    provider, 'api-key': apiKey,
  });

  if (!resolvedKey) {
    throw new Error(
      `Need credentials for provider '${providerId}'. ` +
      `Set in .env: ${prov.envVar}=... or flag --api-key=...`
    );
  }

  console.log(`  Provider: ${prov.displayName} (${prov.envVar}=...${resolvedKey.slice(-6)})`);
  const valid = await prov.validateKey(resolvedKey);
  if (!valid) {
    throw new Error(
      `Invalid ${prov.envVar} for provider '${providerId}'. ` +
      `Please check key/token permission.`
    );
  }

  fs.mkdirSync(outputDir, { recursive: true });
  for (const f of fs.readdirSync(outputDir)) {
    if (f.endsWith('.png')) fs.unlinkSync(path.join(outputDir, f));
  }
  const metaDir = path.join(outputDir, 'meta');
  if (fs.existsSync(metaDir)) {
    for (const f of fs.readdirSync(metaDir)) fs.unlinkSync(path.join(metaDir, f));
  }

  const chunks = workers > 1
    ? Array.from({ length: workers }, (_, i) => tiles.filter((_, j) => j % workers === i))
    : [tiles];

  const startTime = Date.now();

  const stats = await Promise.all(chunks.map((chunk, i) => new Promise((resolve, reject) => {
    const tmpHtml = path.join(os.tmpdir(), `${TMP.dirName}_w${i}_${Date.now()}.html`);
    const userData = path.join(os.tmpdir(), `${TMP.dirName}_w${i}_${Date.now()}`);
    const worker = new Worker(new URL('../tile/worker_entry.mjs', import.meta.url), {
      workerData: {
        tiles: chunk,
        workerId: i,
        provider: providerId,
        apiKey: resolvedKey,
        outputDir,
        blankSizeKb: RENDER_DEFAULTS.blankSizeKb,
        maxRetry:    RENDER_DEFAULTS.maxRetry,
        cfg,
        seedLng: seed.lng,
        seedLat: seed.lat,
        protocolTimeout: RENDER_DEFAULTS.protocolTimeout,
        sessionMaxMs:    RENDER_DEFAULTS.sessionMaxMs,
        tmpHtmlPath: tmpHtml,
        userDataDir: userData,
        checkpointPath: null,
      },
    });
    worker.on('message', m => {
      if (m.type === 'progress') console.log(`[w${i}] ${m.msg}`);
      else if (m.type === 'tile_done') console.log(`[w${i}] done (${m.qx},${m.qy})`);
      else if (m.type === 'done') resolve(m.stats);
      else if (m.type === 'error') {
        console.error(`[w${i}] error: ${m.error}`);
        reject(new Error(m.error));
      }
    });
    worker.on('error', reject);
    worker.on('exit', code => {
      if (code !== 0) reject(new Error(`Worker ${i} exit code ${code}`));
    });
  })));

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const totalDone = stats.reduce((s, w) => s + (w?.doneCount ?? 0), 0);
  const totalFail = stats.reduce((s, w) => s + (w?.failCount ?? 0), 0);
  console.log(`\n[render] ${totalDone}/${tiles.length} tiles in ${elapsed}s (${totalFail} fail)`);
  return { totalDone, totalFail, elapsed };
}

function parseRenderArgs(positional) {
  let startQx = 0, startQy = 0, N = 2, M = 2;
  if (positional.length >= 4) [startQx, startQy, N, M] = positional.map(Number);
  else if (positional.length >= 2) [N, M] = positional.map(Number);
  else if (positional.length === 1) N = Number(positional[0]);
  return { startQx, startQy, N, M };
}

export async function cmdRender({ positional, flags, projectRoot, projectRootDir }) {
  const { startQx, startQy, N, M } = parseRenderArgs(positional);
  const seed = getSeed(flags.seed);
  const cfg = buildRenderCfg(flags);
  cfg.fallback = FALLBACK;
  if (flags.fallback === 'true' || flags.fallback === '1') cfg.fallback.enabled = true;
  else if (flags.fallback === 'false' || flags.fallback === '0') cfg.fallback.enabled = false;

  const outputDir = path.resolve(projectRootDir, flags.output ?? resolvePath('renders'));
  const workers = Number(flags.workers ?? WORKERS.default);
  const provider = flags.provider;
  const apiKey = flags['api-key'];

  console.log(`\n─── Render ${N}×${M} (offset ${startQx},${startQy}) ───`);
  console.log(`Seed: (${seed.lat}, ${seed.lng}) [${seed.label}]`);
  console.log(`Output: ${outputDir}`);
  console.log(`Workers: ${workers}, step: ${cfg.cameraMoveStep}`);
  console.log(`Fallback 2D: ${cfg.fallback.enabled ? `ON (${cfg.fallback.provider}, ${cfg.fallback.postProcess})` : 'OFF'}\n`);

  const tiles = [];
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      tiles.push({ qx: startQx + c, qy: startQy + r });
    }
  }
  for (const t of tiles) {
    const ll = tileIndexToLatLng(t.qx, t.qy, seed.lat, seed.lng, cfg);
    console.log(`  tile (${t.qx},${t.qy}): lat=${ll.lat.toFixed(6)}, lng=${ll.lng.toFixed(6)}`);
  }

  await runRender({ tiles, seed, outputDir, cfg, workers, provider, apiKey });
}

export async function cmdTest({ positional, flags, projectRootDir }) {
  if (positional.length < 2) throw new Error('Usage: test <qx> <qy>');
  const [qx, qy] = positional.map(Number);
  const seed = getSeed(flags.seed);
  const cfg = buildRenderCfg(flags);
  const outputDir = path.resolve(projectRootDir, flags.output ?? resolvePath('renders'));

  console.log(`\n─── Test 1 tile (${qx}, ${qy}) ───`);
  const ll = tileIndexToLatLng(qx, qy, seed.lat, seed.lng, cfg);
  console.log(`  Camera at lat=${ll.lat.toFixed(6)}, lng=${ll.lng.toFixed(6)}\n`);

  await runRender({
    tiles: [{ qx, qy }], seed, outputDir, cfg, workers: 1,
    provider: flags.provider, apiKey: flags['api-key'],
  });
}
