import fs from 'fs';
import os from 'os';
import path from 'path';

import { Worker } from 'worker_threads';

import { TILE, WORKERS, PATHS, TMP, PROVIDERS, FALLBACK, resolvePath } from '../config.mjs';
import { buildRenderCfg, RENDER_DEFAULTS, resolveCredentials } from '../render/env.mjs';
import { chunkArray } from '../render/dispatch.mjs';
import { tileIndexToLatLng } from '../tile/coords.mjs';
import { getSeed } from './seed.mjs';
import { tileFilename } from '../tile/tile_io.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));

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

  // ─── Auto-retry missing/failed tiles ─────────────────────────────────────────
  await retryMissingTiles(tiles, outputDir, cfg, seed);
}

// ─── Smart retry for missing/failed tiles ──────────────────────────────────────

async function retryMissingTiles(originalTiles, outputDir, cfg, seed) {
  const blankSizeKb = RENDER_DEFAULTS.blankSizeKb;
  const missingTiles = [];

  // Find tiles that weren't rendered successfully
  for (const tile of originalTiles) {
    let hasValid = false;
    try {
      const files = fs.readdirSync(outputDir);
      for (const f of files) {
        if (f.includes(`_${tile.qx}_${tile.qy}_`)) {
          const stat = fs.statSync(path.join(outputDir, f));
          if (stat.size >= blankSizeKb * 1024) {
            hasValid = true;
            break;
          }
        }
      }
    } catch {}
    
    if (!hasValid) {
      missingTiles.push(tile);
    }
  }

  if (missingTiles.length === 0) {
    console.log('\n[retry] All tiles rendered successfully');
    return;
  }

  console.log(`\n[retry] Found ${missingTiles.length} missing/invalid tiles, retrying...`);

  // Retry with more patience (longer wait, more retries)
  const retryCfg = { ...cfg, tileWaitMs: cfg.tileWaitMs + 3000 };
  
  for (const tile of missingTiles) {
    let success = false;
    
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const ll = tileIndexToLatLng(tile.qx, tile.qy, seed.lat, seed.lng, cfg);
        const tmpHtml = path.join(os.tmpdir(), `retry_w0_${Date.now()}.html`);
        const userData = path.join(os.tmpdir(), `retry_w0_${Date.now()}`);
        
        const worker = new Promise((resolve, reject) => {
          const w = new Worker(new URL('../tile/worker_entry.mjs', import.meta.url), {
            workerData: {
              tiles: [tile],
              workerId: 99,
              provider: 'cesium-ion',
              apiKey: process.env.CESIUM_TOKEN || process.env.CESIUM_ION_TOKEN,
              outputDir,
              blankSizeKb,
              maxRetry: 2,
              cfg: retryCfg,
              seedLng: seed.lng,
              seedLat: seed.lat,
              protocolTimeout: 120000,
              sessionMaxMs: 300000,
              tmpHtmlPath: tmpHtml,
              userDataDir: userData,
              checkpointPath: null,
            },
          });

          w.on('message', m => {
            if (m.type === 'tile_done') resolve({ ok: true });
            else if (m.type === 'done') resolve({ ok: m.stats?.doneCount > 0 });
            else if (m.type === 'error') resolve({ ok: false, error: m.error });
          });
          w.on('error', e => resolve({ ok: false, error: e.message }));
        });

        const result = await Promise.race([
          worker,
          sleep(120000).then(() => ({ ok: false, error: 'timeout' }))
        ]);

        if (result.ok) {
          console.log(`[retry] (${tile.qx},${tile.qy}) ✓`);
          success = true;
          break;
        } else {
          console.log(`[retry] (${tile.qx},${tile.qy}) attempt ${attempt + 1} failed: ${result.error}`);
          await sleep(2000); // Wait before retry
        }
      } catch (e) {
        console.log(`[retry] (${tile.qx},${tile.qy}) error: ${e.message}`);
      }
    }

    if (!success) {
      // Final fallback: 2D satellite tile
      if (cfg.fallback?.enabled) {
        console.log(`[retry] (${tile.qx},${tile.qy}) using 2D fallback...`);
        try {
          const { renderFallback2D } = await import('../tile/fallback_2d.mjs');
          const r = await renderFallback2D(tile, seed.lng, seed.lat, cfg, outputDir, null, 99);
          if (r.ok) {
            console.log(`[retry] (${tile.qx},${tile.qy}) 2D fallback ✓`);
            success = true;
          }
        } catch (e) {
          console.log(`[retry] (${tile.qx},${tile.qy}) 2D fallback failed: ${e.message}`);
        }
      }
    }

    if (!success) {
      console.log(`[retry] (${tile.qx},${tile.qy}) ✗ FAILED after all retries`);
    }
  }

  // Summary
  let stillMissing = 0;
  for (const tile of originalTiles) {
    let hasValid = false;
    try {
      const files = fs.readdirSync(outputDir);
      for (const f of files) {
        if (f.includes(`_${tile.qx}_${tile.qy}_`)) {
          const stat = fs.statSync(path.join(outputDir, f));
          if (stat.size >= blankSizeKb * 1024) {
            hasValid = true;
            break;
          }
        }
      }
    } catch {}
    if (!hasValid) stillMissing++;
  }

  console.log(`\n[retry] Complete. ${originalTiles.length - stillMissing}/${originalTiles.length} tiles rendered`);
  if (stillMissing > 0) {
    console.log(`[retry] Warning: ${stillMissing} tiles still missing`);
  }
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

  // Auto-retry for test
  await retryMissingTiles([{ qx, qy }], outputDir, cfg, seed);
}
