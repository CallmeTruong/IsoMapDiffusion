import { getProvider, resolveApiKey } from '../tile/provider/index.mjs';
import { PROVIDERS } from '../config.mjs';

export async function validateKey(key, providerId = PROVIDERS.default) {
  const provider = getProvider(providerId);
  return provider.validateKey(key);
}

export async function validateAllKeys(keys, providerId = PROVIDERS.default) {
  const provider = getProvider(providerId);
  console.log(`\n─── Validate ${provider.displayName} credentials ───`);
  const results = await Promise.all(keys.map(async key => {
    const ok = await provider.validateKey(key);
    const masked = key.length > 6 ? `...${key.slice(-6)}` : '***';
    console.log(`  ${provider.envVar}=${masked}: ${ok ? 'OK' : 'FAIL'}`);
    return ok ? key : null;
  }));
  return results.filter(Boolean);
}


export async function validateGoogleKey(key) {
  return validateKey(key, 'google');
}

export async function validateAllGoogleKeys(keys) {
  return validateAllKeys(keys, 'google');
}

export { resolveApiKey, getProvider };
