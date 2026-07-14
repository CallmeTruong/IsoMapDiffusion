import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { TILE, TILE_SIZE_M, QUADRANT_M, WORKERS, CHECKPOINT, PATHS, PROVIDERS } from '../config.mjs';
import { parseArgs } from '../config.mjs';
import { resolveApiKey, getProvider } from '../tile/provider/index.mjs';

export { parseArgs };


export function projectRoot(importMetaUrl) {
  const __dirname = path.dirname(fileURLToPath(importMetaUrl));
  return path.resolve(__dirname, '..');
}


export function loadEnv(projectRootDir) {
  const envPath = path.join(projectRootDir, '.env');
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
    const stripped = line.trim().replace(/\r$/, '');
    if (!stripped || stripped.startsWith('#')) continue;
    const idx = stripped.indexOf('=');
    if (idx < 0) continue;
    const key = stripped.slice(0, idx).trim();
    const val = stripped.slice(idx + 1).trim();
    if (key && val) process.env[key] = val;
  }
}

export function resolveNumWorkers(workersArg) {
  const cpu = os.cpus().length;
  const desired = Number(workersArg ?? Math.min(WORKERS.default, cpu));
  return Math.max(1, Math.min(desired, Math.min(WORKERS.max, cpu)));
}


export function buildRenderCfg(args = {}) {
  return {
    sizePx:         TILE.sizePx,
    azimuth:        Number(args.azimuth   ?? args.heading  ?? TILE.azimuth),
    elevation:      Number(args.elevation ?? args.pitch    ?? TILE.elevation),
    altitude:       Number(args.altitude  ?? TILE.altitude),
    frustumW:       TILE_SIZE_M,
    tileStep:       TILE.tileStep,
    cameraMoveStep: TILE.cameraMoveStep,
    quadrantM:      QUADRANT_M,
    targetHeight:   TILE.targetHeight ?? 0,
    tileWaitMs:     Number(args.tilewait  ?? TILE.tileWaitMs),
    settlePollMs:   Number(args.settle    ?? TILE.settlePollMs),
    settleMaxMs:    Number(args.settlemax ?? TILE.settleMaxMs),
    stableHits:     Number(args.stable    ?? TILE.stableHits),
    varianceThr:    Number(args.variance  ?? TILE.varianceThr),
    sse:            Number(args.sse       ?? TILE.sse),
  };
}


export function validateBatchOpts(opts, projectRootDir) {
  const errors = [];
  if (!opts.manifest) errors.push('--manifest is required');
  if (!opts.output)   errors.push('--output is required');
  if (!resolveApiKey(opts.provider, opts['api-key'])) {
    const pid = opts.provider ?? PROVIDERS.default;
    const prov = getProvider(pid);
    errors.push(`${prov.envVar} env or --api-key is required (provider: ${pid})`);
  }
  if (!fs.existsSync(path.resolve(projectRootDir, opts.manifest ?? ''))) {
    errors.push(`manifest not found: ${opts.manifest}`);
  }
  return errors;
}


export function resolveCredentials(opts = {}) {
  const providerId = opts.provider ?? PROVIDERS.default;
  const provider = getProvider(providerId);
  const apiKey = resolveApiKey(providerId, opts['api-key']);
  return { providerId, apiKey, provider };
}

export const RENDER_DEFAULTS = {
  blankSizeKb:    TILE.blankSizeKb,
  maxRetry:       TILE.maxRetry,
  protocolTimeout: TILE.protocolTimeout,
  sessionMaxMs:   TILE.sessionMaxMs,
  checkpointFlushMs: CHECKPOINT.flushIntervalMs,
};
