/**
 * worker.mjs — Worker thread logic for tile rendering
 *
 * Output format (see src/tile/tile_io.mjs):
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

// Errors that mean the browser tab/session is dead and unusable — retrying
// on the same `page` will just keep failing instantly for every remaining
// tile. These must trigger a page recreation, not a normal tile-level retry.
function isFatalPageError(msg) {
  if (!msg) return false;
  return /Target closed|detached Frame|Protocol error|Session closed|Connection closed/i.test(msg);
}

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
    checkpointPath, chromePath,
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

  let browser = null;
  // Always close the browser cleanly — even on any exception inside the main
  // render loop, the retry pass, or `attemptFallbackNow`. Without this,
  // a worker killed mid-render (or one that throws) leaves a Chromium child
  // process with a blank top-level window pinned in DWM/the taskbar.
  const closeBrowserSafely = async () => {
    if (!browser || browser._closed) return;
    try {
      await Promise.race([
        browser.close(),
        new Promise((_, rej) => setTimeout(() => rej(new Error('browser.close timeout')), 3000)),
      ]);
    } catch (e) {
      report(`[WARN] browser.close failed, killing process: ${e.message}`);
      try {
        const proc = browser.process?.();
        if (proc && proc.exitCode === null) proc.kill('SIGKILL');
      } catch { /* nothing left to do */ }
    }
  };

  // Graceful shutdown flag - when true, finish current tile but don't start new ones
  let draining = false;
  let currentTile = null;

  // Listen for messages from main thread (e.g., flush_checkpoint on shutdown)
  parentPort?.on('message', msg => {
    if (msg?.type === 'flush_checkpoint') {
      console.log(`[Worker ${workerId}] Draining, finishing current tile...`);
      draining = true;
    }
  });

  const { default: puppeteer } = await import('puppeteer');

  // Force `headless: 'shell'` regardless of what the main thread passed.
  //
  // Why: on Windows, `headless: true` (boolean) and `headless: 'new'` both end
  // up creating a real top-level OS window in DWM/the taskbar when WebGL is
  // enabled (ANGLE/d3d11 path used by Cesium). That window stays open as a
  // blank white tab if `browser.close()` is skipped — which happens whenever
  // the worker thread is killed by the parent (SIGTERM/timeout) before
  // reaching the closing await at the end of runWorker.
  //
  // `headless: 'shell'` is the modern Chromium implementation that draws to
  // an offscreen surface. No OS window, no taskbar entry, no leftover blank
  // tab. It still supports WebGL via ANGLE/SwiftShader, which is what we
  // need for Cesium. The redundant hidden-window flags below are belt &
  // suspenders for the rare case where shell falls back to a non-shell mode.
  const launchOpts = {
    headless: 'shell',
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
      '--no-startup-window',
      '--no-default-browser-check',
      '--noerrdialogs',
      '--disable-session-crashed-bubble',
      '--disable-infobars',
      `--window-size=${cfg.sizePx},${cfg.sizePx}`,
    ],
  };
  if (chromePath) launchOpts.executablePath = chromePath;

  fs.mkdirSync(userData, { recursive: true });
  const htmlArgs = { tilesetJs, ionReset };
  fs.writeFileSync(tmpHtml, makeCesiumHTML(htmlArgs, cfg), 'utf8');
  const fileUrl = 'file://' + tmpHtml.replace(/\\/g, '/');

  let sessionStart = Date.now();
  let sessionCount = 1;

  // ─── Render loop ──────────────────────────────────────────────────────────

  try {
    browser = await puppeteer.launch(launchOpts);

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

    if (cfg?.fallback?.enabled && (!Array.isArray(cfg.fallback.providers) || cfg.fallback.providers.length === 0)) {
      report('[WARN] fallback.enabled=true but fallback.providers[] is empty — ' +
             'blank tiles will NOT be recoverable via 2D fallback until this is set in config.');
    }

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
    const blankUnrecovered = [];
    let errLogBuffer = '';
    let pageReady = true;
    // Tiles deferred because the page was not ready — processed after recovery.
    // NEVER push into `pending` while iterating it (causes infinite growth →
    // RangeError: Invalid array length at Array.push in worker.mjs:289).
    const pageNotReadyQueue = [];

    const MAX_PAGE_RECOVERIES = 3;

    async function attemptFallbackNow(tile) {
      if (!cfg?.fallback?.enabled) {
        failCount++;
        blankUnrecovered.push(tile);
        report(`${doneCount}/${pending.length} tiles done, ${failCount} fail, ${retryCount} retry`);
        return;
      }
      const r = await renderFallback2D(tile, seedLng, seedLat, cfg, outputDir, parentPort, workerId);
      if (r.ok) {
        doneCount++;
        retryCount += r.retries ?? 0;
        parentPort?.postMessage({ type: 'tile_done', workerId, qx: tile.qx, qy: tile.qy });
        report(`${doneCount}/${pending.length} tiles done, ${failCount} fail, ${retryCount} retry`);
        return;
      }

      // 2D fallback failed (all providers exhausted) — no further synthesis
      // is attempted. The tile is logged and left unrecovered.
      errLogBuffer += `[FALLBACK] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${r.error}\n`;
      try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}

      failCount++;
      blankUnrecovered.push(tile);
      report(`${doneCount}/${pending.length} tiles done, ${failCount} fail, ${retryCount} retry`);
    }

    for (const tile of pending) {
      if (!pageReady) {
        // Do NOT push back into `pending` — that mutates the array we are
        // iterating and causes unbounded growth → RangeError.
        pageNotReadyQueue.push(tile);
        continue;
      }

      // Drain tiles that were deferred while the page was recovering.
      while (pageReady && pageNotReadyQueue.length > 0) {
        pending.push(pageNotReadyQueue.shift());
      }

      if (draining && currentTile === null) {
        currentTile = tile;
        console.log(`[Worker ${workerId}] Draining: finishing tile (${tile.qx},${tile.qy}) then stopping`);
      }
      if (draining && currentTile !== tile) {
        console.log(`[Worker ${workerId}] Skipping tile (${tile.qx},${tile.qy}) due to drain`);
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

      let result;
      for (let recovery = 0; ; recovery++) {
        result = await renderOneTile(page, tile, seedLng, seedLat, cfg, outputDir, blankSizeKb, maxRetry, parentPort, workerId);

        const fatal = !result.ok && !result.isBlank && isFatalPageError(result.error);
        if (!fatal || recovery >= MAX_PAGE_RECOVERIES) break;

        report(`[WARN] Page died (${result.error}) on tile (${tile.qx},${tile.qy}), recreating session (recovery ${recovery + 1}/${MAX_PAGE_RECOVERIES})...`);
        pageReady = false;
        try { await page.close(); } catch { /* ignore */ }
        try {
          page = await createPage();
          sessionCount++;
          sessionStart = Date.now();
          pageReady = true;
        } catch (e) {
          report(`[WARN] Failed to recreate page: ${e.message}`);
          pageReady = false;
          break;
        }
      }

      // If draining, finalize and break out of the loop — `finally` still
      // closes the browser.
      if (draining && currentTile === tile) {
        console.log(`[Worker ${workerId}] Tile (${tile.qx},${tile.qy}) done, exiting gracefully`);
        if (result.ok) {
          doneCount++;
          parentPort?.postMessage({
            type: 'tile_done', workerId,
            qx: tile.qx, qy: tile.qy,
            variance: result.variance, renderMs: result.renderMs,
          });
        }
        done({ doneCount, failCount, retryCount, sessionCount });
        break;
      }

      if (result.ok) {
        doneCount++;
        retryCount += result.retries;
        parentPort?.postMessage({
          type: 'tile_done', workerId,
          qx: tile.qx, qy: tile.qy,
          variance: result.variance, renderMs: result.renderMs,
        });
      } else if (result.isBlank) {
        retryCount += result.retries;
        errLogBuffer += `[BLANK] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
        try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
        await attemptFallbackNow(tile);
        continue;
      } else {
        failCount++;
        retryCount += result.retries;
        failed.push(tile);
        errLogBuffer += `[MAIN] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
        try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
      }

      report(`${doneCount}/${pending.length} tiles done, ${failCount} fail, ${retryCount} retry`);
    }

    if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);

    // Retry pass for tiles that failed without being marked blank
    if (failed.length > 0 && !draining) {
      report(`Retry pass: ${failed.length} tiles`);
      errLogBuffer = '';
      const retryQueue = [];
      for (const tile of failed) {
        if (!pageReady) {
          // Same fix: never push back into the array we are iterating.
          retryQueue.push(tile);
          continue;
        }
        // Drain any deferred retry tiles when the page recovers.
        while (pageReady && retryQueue.length > 0) {
          failed.push(retryQueue.shift());
        }
        if (Date.now() - sessionStart >= sessionMaxMs && pageReady) {
          pageReady = false;
          try { await page.close(); } catch { /* ignore */ }
          page = await createPage();
          sessionCount++;
          sessionStart = Date.now();
          pageReady = true;
        }

        let result;
        for (let recovery = 0; ; recovery++) {
          result = await renderOneTile(page, tile, seedLng, seedLat, cfg, outputDir, blankSizeKb, maxRetry, parentPort, workerId);

          const fatal = !result.ok && !result.isBlank && isFatalPageError(result.error);
          if (!fatal || recovery >= MAX_PAGE_RECOVERIES) break;

          report(`[WARN] Page died (${result.error}) on retry tile (${tile.qx},${tile.qy}), recreating session (recovery ${recovery + 1}/${MAX_PAGE_RECOVERIES})...`);
          pageReady = false;
          try { await page.close(); } catch { /* ignore */ }
          try {
            page = await createPage();
            sessionCount++;
            sessionStart = Date.now();
            pageReady = true;
          } catch (e) {
            report(`[WARN] Failed to recreate page: ${e.message}`);
            pageReady = false;
            break;
          }
        }

        if (result.ok) {
          doneCount++;
          failCount--;
          parentPort?.postMessage({ type: 'tile_done', workerId, qx: tile.qx, qy: tile.qy });
        } else if (result.isBlank) {
          failCount--;
          errLogBuffer += `[BLANK-RETRY] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
          try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
          await attemptFallbackNow(tile);
        } else {
          errLogBuffer += `[RETRY] ${new Date().toISOString()} tile(${tile.qx},${tile.qy}): ${result.error}\n`;
          try { fs.appendFileSync(errLog, errLogBuffer); errLogBuffer = ''; } catch {}
        }
      }
      if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);
    }

    if (blankUnrecovered.length > 0) {
      report(`${blankUnrecovered.length} blank tiles could not be recovered via fallback`);
    }

    if (errLogBuffer) fs.appendFileSync(errLog, errLogBuffer);

    done({ doneCount, failCount, retryCount, sessionCount });
  } finally {
    // GUARANTEED cleanup — runs on normal exit, exception, or abrupt
    // process.exit from the parent. Kills Chromium first (the source of the
    // blank-window symptom), then deletes the temp profile directory.
    await closeBrowserSafely();
    try { cleanup(); } catch {}
  }
}

export function startWorkerIfWorkerThread() {
  if (!isMainThread) {
    runWorker(workerData).catch(e => {
      parentPort?.postMessage({ type: 'error', workerId: workerData.workerId, error: e.message });
      process.exit(1);
    });
  }
}
