/**
 * worker.mjs — Worker thread logic for tile rendering
 *
 * Output format (xem src/tile/tile_io.mjs):
 *   <outputDir>/tile_<qx>_<qy>_<hash>.png
 *   <outputDir>/meta/tile_<qx>_<qy>.json
 */

import fs from 'fs';
import path from 'path';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { makeCesiumHTML } from './html.mjs';
import { tileIndexToLatLng } from './coords.mjs';
import { filterPendingTiles, sortPendingByPriority } from './quality.mjs';
import { saveTile, clearTileDeletedMarker } from './tile_io.mjs';
import { getProvider } from './provider/index.mjs';
import { renderFallback2D } from './fallback_2d.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));

function createCleanup(tmpHtml, userDataDir) {
  return () => {
    try { if (fs.existsSync(tmpHtml)) fs.unlinkSync(tmpHtml); } catch {}
    try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch {}
  };
}

/**
 * Render one tile and save as 1 file (1024×1024).
 */
async function renderOneTile(page, tile, seedLng, seedLat, cfg, outputDir, blankSizeKb, maxRetry, parentPort, workerId) {

  const { lat, lng } = tileIndexToLatLng(tile.qx, tile.qy, seedLat, seedLng, cfg);

  const fbEnabled = !!cfg?.fallback?.enabled;
  const effectiveMaxRetry = fbEnabled
    ? Math.min(maxRetry, cfg.fallback.maxRetries3D ?? 1)
    : maxRetry;

  for (let attempt = 0; attempt < effectiveMaxRetry; attempt++) {
    const wait = attempt * cfg.tileWaitMs;
    const t0 = Date.now();
    try {
      const tileDataUrl = await page.evaluate(
        (lng, lat, w) => window.renderTile(lng, lat, w),
        lng, lat, wait
      );

      const analysis = await page.evaluate(() => window.analyzeCanvas());
      if (analysis.isBlank) {
        if (attempt < effectiveMaxRetry - 1) continue;
        if (fbEnabled) {
          return { ok: false, isBlank: true, error: 'Blank tile (will fallback)', retries: attempt };
        }
        throw new Error('Blank tile');
      }

      const renderMs = Date.now() - t0;
      const buf = Buffer.from(tileDataUrl.split(',')[1], 'base64');

      const result = saveTile(buf, { qx: tile.qx, qy: tile.qy }, {
        lat, lng,
        cameraAzimuth: cfg.azimuth,
        cameraElevation: cfg.elevation,
        cameraAltitude: cfg.altitude,
        variance:    analysis.variance,
        meanR:       analysis.meanR,
        meanG:       analysis.meanG,
        meanB:       analysis.meanB,
        edgeDensity: analysis.edgeDensity,
        isBlank:     false,
        attempt,
        renderMs,
        seedLat, seedLng,
      }, outputDir);

      // Mark quadrant boundaries (single cell) for downstream stitching
      parentPort?.postMessage({
        type: 'db_update', qx: tile.qx, qy: tile.qy,
        info: { workerId, variance: analysis.variance, renderMs },
      });

      if (result.sizeKB >= blankSizeKb) {
        clearTileDeletedMarker(outputDir, tile.qx, tile.qy);
      } else {
        try { fs.unlinkSync(result.filepath); } catch {}
        try { fs.unlinkSync(result.metaPath); } catch {}
        throw new Error('Tile size too small');
      }

      return { ok: true, retries: attempt, ...result, variance: analysis.variance, renderMs };
    } catch (e) {
      if (attempt === effectiveMaxRetry - 1) {
        parentPort?.postMessage({
          type: 'db_update',
          qx: tile.qx, qy: tile.qy,
          info: { workerId, error: e.message },
        });
        return { ok: false, error: e.message, retries: attempt };
      }
    }
  }
}

export async function runWorker(workerData) {
  const {
    tiles, workerId, provider: providerId, apiKey,
    outputDir, blankSizeKb, maxRetry,
    cfg, seedLng, seedLat,
    protocolTimeout, sessionMaxMs,
    tmpHtmlPath, userDataDir,
    checkpointPath,
  } = workerData;

  // Provider plugin
  const provider = getProvider(providerId);
  const tokenData = provider.getCesiumToken(apiKey);
  const tilesetJs = provider.buildTilesetJs(tokenData, cfg);
  const ionReset = providerId !== 'cesium-ion';

  const errLog  = `render_errors_w${workerId}.log`;
  const tmpHtml = tmpHtmlPath ?? path.resolve(`./tmp_cesium_w${workerId}.html`);
  const userData = userDataDir ?? path.resolve(`./chrome_profile_w${workerId}`);
  const cleanup  = createCleanup(tmpHtml, userData);

  process.on('exit',    cleanup);
  process.on('SIGINT',  () => { cleanup(); process.exit(1); });
  process.on('SIGTERM', () => { cleanup(); process.exit(1); });
  process.on('uncaughtException', e => {
    parentPort?.postMessage({ type: 'error', workerId, error: e.message });
    cleanup();
    process.exit(1);
  });

  const report = msg => parentPort.postMessage({ type: 'progress', workerId, msg });
  const done   = stats => parentPort.postMessage({ type: 'done', workerId, stats });

  const { default: puppeteer } = await import('puppeteer');

  const browser = await puppeteer.launch({
    headless: true,
    protocolTimeout,
    userDataDir: userData,
    args: [
      '--no-sandbox', '--disable-web-security',
      '--enable-webgl', '--use-gl=angle', '--use-angle=d3d11',
      '--ignore-gpu-blocklist', '--disable-dev-shm-usage',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
      '--disable-extensions', '--disable-background-networking',
      '--disable-default-apps', '--disable-sync', '--disable-translate',
      '--metrics-recording-only', '--mute-audio', '--no-first-run',
      '--safebrowsing-disable-auto-update', '--hide-scrollbars',
      '--disable-backgrounding-occluded-windows', '--disable-hang-monitor',
      '--disable-prompt-on-repost', '--disable-popup-blocking',
      `--window-size=${cfg.sizePx},${cfg.sizePx}`,
    ],
  });

  fs.mkdirSync(userData, { recursive: true });
  const htmlArgs = { tilesetJs, ionReset };
  fs.writeFileSync(tmpHtml, makeCesiumHTML(htmlArgs, cfg), 'utf8');
  const fileUrl = 'file://' + tmpHtml.replace(/\\/g, '/');

  let sessionStart = Date.now();
  let sessionCount = 1;

  async function createPage() {
    const p = await browser.newPage();
    p.on('console', m => { if (m.type() === 'error') report('[Browser] ' + m.text()); });
    await p.setViewport({ width: cfg.sizePx, height: cfg.sizePx, deviceScaleFactor: 1 });
    await p.goto(fileUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await p.waitForFunction('typeof window.renderTile === "function"', { timeout: 30_000 });
    await p.waitForFunction('window.isSessionOk()', { timeout: 30_000, polling: 200 });
    await sleep(2000);
    return p;
  }

  let page = await createPage();
  report('Session ready');

  let pending = filterPendingTiles(tiles, outputDir, blankSizeKb);
  let doneSet = null;
  try {
    if (checkpointPath && fs.existsSync(checkpointPath)) {
      const cp = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));
      if (Array.isArray(cp.doneTiles)) {
        doneSet = new Set(cp.doneTiles.map(t => `${t.qx},${t.qy}`));
      }
    }
  } catch { /* ignore */ }
  pending = sortPendingByPriority(pending, outputDir, blankSizeKb, doneSet);

  let doneCount = 0, failCount = 0, retryCount = 0;
  const failed = [];
  const blankTiles = [];   // tiles where 3D render returned blank → fallback candidate
  let errLogBuffer = '';
  let pageReady = true;

  for (const tile of pending) {
    if (!pageReady) {
      pending.push(tile);
      continue;
    }

    if (Date.now() - sessionStart >= sessionMaxMs && pageReady) {
      report('Restarting page (full GPU reset)...');
      pageReady = false;
      try { await page.close(); } catch { /* ignore */ }
      page = await createPage();
      sessionCount++;
      sessionStart = Date.now();
      pageReady = true;
      report(`Session #${sessionCount} ready`);
    }

    const result = await renderOneTile(page, tile, seedLng, seedLat, cfg, outputDir, blankSizeKb, maxRetry, parentPort, workerId);

    if (result.ok) {
      doneCount++;
      retryCount += result.retries;
      parentPort?.postMessage({
        type: 'tile_done', workerId,
        qx: tile.qx, qy: tile.qy,
        variance: result.variance, renderMs: result.renderMs,
      });
    } else if (result.isBlank) {
      blankTiles.push(tile);
      retryCount += result.retries;
      errLogBuffer += `[BLANK] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
      try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
    } else {
      failCount++;
      retryCount += result.retries;
      failed.push(tile);
      errLogBuffer += `[MAIN] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
      try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
    }

    report(`${doneCount}/${pending.length} tiles done, ${failCount} fail, ${retryCount} retry, ${blankTiles.length} blank`);
  }

  if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);

  if (failed.length > 0) {
    report(`Retry pass: ${failed.length} tiles`);
    errLogBuffer = '';
    for (const tile of failed) {
      if (!pageReady) {
        failed.push(tile);  // push back; will retry on next pass
        continue;
      }
      if (Date.now() - sessionStart >= sessionMaxMs && pageReady) {
        pageReady = false;
        try { await page.close(); } catch { /* ignore */ }
        page = await createPage();
        sessionCount++;
        sessionStart = Date.now();
        pageReady = true;
      }

      const result = await renderOneTile(page, tile, seedLng, seedLat, cfg, outputDir, blankSizeKb, maxRetry, parentPort, workerId);
      if (result.ok) {
        doneCount++;
        failCount--;
        parentPort?.postMessage({ type: 'tile_done', workerId, qx: tile.qx, qy: tile.qy });
      } else if (result.isBlank) {
        blankTiles.push(tile);
        failCount--;
        errLogBuffer += `[BLANK-RETRY] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
        try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
      } else {
        errLogBuffer += `[RETRY] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
        try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
      }
    }
    if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);
  }

  // ─── 2D fallback pass ───────────────────────────────────────────────
  // For tiles where 3D render returned blank, fetch a 2D satellite tile.
  // The fallback path runs in pure Node (no Puppeteer), so no pageReady needed.
  if (blankTiles.length > 0 && cfg?.fallback?.enabled) {
    report(`2D fallback pass: ${blankTiles.length} blank tiles → ${cfg.fallback.provider ?? 'esri'}`);
    let fbOk = 0, fbFail = 0;
    for (const tile of blankTiles) {
      const r = await renderFallback2D(tile, seedLng, seedLat, cfg, outputDir, parentPort, workerId);
      if (r.ok) {
        doneCount++;
        fbOk++;
        retryCount += r.retries ?? 0;
      } else {
        failCount++;
        fbFail++;
        errLogBuffer += `[FALLBACK] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${r.error}\n`;
        try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
      }
    }
    report(`2D fallback done: ${fbOk} ok, ${fbFail} fail`);
  } else if (blankTiles.length > 0) {
    // Fallback disabled — blank tiles count as failures (preserves old behavior)
    failCount += blankTiles.length;
    report(`2D fallback disabled: ${blankTiles.length} blank tiles counted as fail`);
  }

  if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);

  await browser.close();
  cleanup();
  done({ doneCount, failCount, retryCount, sessionCount });
}

export function startWorkerIfWorkerThread() {
  if (!isMainThread) {
    runWorker(workerData).catch(e => {
      parentPort?.postMessage({ type: 'error', workerId: workerData.workerId, error: e.message });
      process.exit(1);
    });
  }
}