/**
 * render/index.mjs — Re-export facade cho render/*
 */

export {
  projectRoot, loadEnv, parseArgs,
  resolveNumWorkers, buildRenderCfg, resolveCredentials,
  validateBatchOpts, RENDER_DEFAULTS,
} from './env.mjs';

export { validateKey, validateAllKeys, validateGoogleKey, validateAllGoogleKeys, resolveApiKey, getProvider } from './keys.mjs';
export { chunkArray, dispatchWorkers } from './dispatch.mjs';
export {
  buildQuadrantsGeojson, buildGenerationConfig, exportRenderOutputs,
} from './export.mjs';
