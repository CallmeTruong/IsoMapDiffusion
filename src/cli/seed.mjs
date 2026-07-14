import { SEEDS } from '../config.mjs';

export function getSeed(seedArg) {
  if (SEEDS[seedArg]) {
    return { ...SEEDS[seedArg], key: seedArg };
  }
  if (seedArg && seedArg.includes(',')) {
    const [lat, lng] = seedArg.split(',').map(Number);
    return { lat, lng, label: `Custom (${lat}, ${lng})`, key: 'custom' };
  }
  const firstKey = Object.keys(SEEDS)[0] ?? 'sgn';
  return { ...(SEEDS[firstKey] ?? SEEDS.sgn), key: firstKey };
}
