import fs from 'fs';
import path from 'path';
import { DATASET_PATHS } from './constants.mjs';

/**
 * Read current registry.
 */
export function loadRegistry() {
  if (!fs.existsSync(DATASET_PATHS.registry)) {
    return {
      schema_version: 1,
      generated_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      total: 0,
      tiles: [],
    };
  }
  return JSON.parse(fs.readFileSync(DATASET_PATHS.registry, 'utf8'));
}

/**
 * Save registry.
 */
export function saveRegistry(registry) {
  registry.updated_at = new Date().toISOString();
  registry.total = registry.tiles.length;
  fs.mkdirSync(DATASET_PATHS.root, { recursive: true });
  fs.writeFileSync(DATASET_PATHS.registry, JSON.stringify(registry, null, 2));
  return registry;
}

/**
 * Find tile in registry.
 */
export function findInRegistry(registry, qx, qy) {
  return registry.tiles.find(t => t.qx === qx && t.qy === qy) ?? null;
}

/**
 * Upsert 1 tile to registry.
 */
export function upsertRegistry(registry, entry) {
  const idx = registry.tiles.findIndex(t => t.qx === entry.qx && t.qy === entry.qy);
  if (idx >= 0) {
    registry.tiles[idx] = { ...registry.tiles[idx], ...entry };
  } else {
    registry.tiles.push(entry);
  }
  return registry;
}

/**
 * Build registry from raw_tiles + ai_gen.
 */
export function buildRegistry(paths = DATASET_PATHS) {
  const registry = loadRegistry();

  // Scan raw_tiles/
  const rawFiles = fs.existsSync(paths.rawTiles)
    ? fs.readdirSync(paths.rawTiles).filter(f => /^tile_-?\d+_-?\d+\.png$/i.test(f))
    : [];

  // Scan ai_gen/
  const aiFiles = fs.existsSync(paths.aiGen)
    ? fs.readdirSync(paths.aiGen).filter(f => /^tile_-?\d+_-?\d+\.png$/i.test(f))
    : [];

  const aiMap = new Map();
  for (const f of aiFiles) {
    const m = f.match(/^tile_(-?\d+)_(-?\d+)\.png$/);
    if (m) aiMap.set(`${m[1]}_${m[2]}`, f);
  }

  for (const f of rawFiles) {
    const m = f.match(/^tile_(-?\d+)_(-?\d+)\.png$/);
    if (!m) continue;
    const qx = parseInt(m[1]);
    const qy = parseInt(m[2]);
    const aiFile = aiMap.get(`${qx}_${qy}`);

    const signQ = qx >= 0 ? '+' + qx : String(qx);
    const signY = qy >= 0 ? '+' + qy : String(qy);
    const possibleRaw = fs.readdirSync(paths.sourceRenders)
      .filter(name => name.startsWith(`tile_${signQ}_${signY}_`) && name.endsWith('.png'));
    const rawRenderAbs = possibleRaw[0]
      ? path.join(paths.sourceRenders, possibleRaw[0]).replace(/\\/g, '/')
      : null;

    upsertRegistry(registry, {
      qx, qy,
      raw_file: rawRenderAbs,
      ai_file:  aiFile ? path.join(paths.aiGen, aiFile).replace(/\\/g, '/') : null,
      status:   aiFile ? 'imported' : 'pending',
    });
  }

  for (const [key, f] of aiMap) {
    const [qxStr, qyStr] = key.split('_');
    const qx = parseInt(qxStr);
    const qy = parseInt(qyStr);
    if (!findInRegistry(registry, qx, qy)) {
      upsertRegistry(registry, {
        qx, qy,
        raw_file: null,
        ai_file:  path.join(paths.aiGen, f).replace(/\\/g, '/'),
        status:   'imported',
      });
    }
  }

  return saveRegistry(registry);
}

/**
 * Quick stats from registry.
 */
export function registryStats(registry) {
  const counts = { pending: 0, imported: 0, verified: 0, missing: 0 };
  for (const t of registry.tiles) {
    counts[t.status] = (counts[t.status] ?? 0) + 1;
  }
  return {
    total:     registry.tiles.length,
    by_status: counts,
  };
}