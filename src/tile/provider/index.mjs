import { GoogleProvider } from './google.mjs';
import { CesiumIonProvider } from './cesium-ion.mjs';
import { PROVIDERS } from '../../config.mjs';

const providers = new Map();
providers.set('google',     new GoogleProvider());
providers.set('cesium-ion', new CesiumIonProvider());

export function getProvider(id) {
  const pid = id ?? PROVIDERS.default;
  const p = providers.get(pid);
  if (!p) {
    throw new Error(
      `Unknown provider '${pid}'. Available: ${[...providers.keys()].join(', ')}`
    );
  }
  return p;
}

export function listProviders() {
  return [...providers.values()].map(p => ({
    id: p.id,
    envVar: p.envVar,
    displayName: p.displayName,
  }));
}


export function resolveApiKey(providerId, explicitKey) {
  if (explicitKey) return explicitKey;
  const p = getProvider(providerId);
  if (process.env[p.envVar]) return process.env[p.envVar];
  if (process.env.GOOGLE_KEY) return process.env.GOOGLE_KEY;
  return null;
}

export { TileProvider } from './base.mjs';
export { GoogleProvider } from './google.mjs';
export { CesiumIonProvider } from './cesium-ion.mjs';
