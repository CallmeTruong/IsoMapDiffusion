import fs from 'fs';
import path from 'path';
import { DATASET_PATHS, IMPORT_RULES } from './constants.mjs';
import { upsertRegistry, saveRegistry, loadRegistry, findInRegistry } from './registry.mjs';

function parseAiTileName(filename) {
  const match = filename.match(IMPORT_RULES.filenamePattern);
  if (!match) return null;
  return { qx: parseInt(match[1]), qy: parseInt(match[2]) };
}

/**
 * Read raw metadata from output/renders/meta/.
 */
function readRawMeta(qx, qy) {
  const signQ = qx >= 0 ? '+' + qx : String(qx);
  const signY = qy >= 0 ? '+' + qy : String(qy);
  const metaPath = path.join(DATASET_PATHS.sourceMeta, `tile_${signQ}_${signY}.json`);
  if (!fs.existsSync(metaPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  } catch {
    return null;
  }
}

/**
 * find raw render file
 */
function findRawRender(qx, qy) {
  const signQ = qx >= 0 ? '+' + qx : String(qx);
  const signY = qy >= 0 ? '+' + qy : String(qy);
  const prefix = `tile_${signQ}_${signY}_`;
  if (!fs.existsSync(DATASET_PATHS.sourceRenders)) return null;
  const files = fs.readdirSync(DATASET_PATHS.sourceRenders)
    .filter(f => f.startsWith(prefix) && f.endsWith('.png'));
  return files[0] ?? null;
}

/**
 * Build sidecar JSON for AI gen tile.
 */
function buildSidecar(qx, qy, aiPath) {
  const rawMeta = readRawMeta(qx, qy);
  const rawFile = findRawRender(qx, qy);

  return {
    schema_version: 1,
    id: `tile_${qx}_${qy}`,
    qx, qy,
    imported_at: new Date().toISOString(),
    source: {
      raw_file:  rawFile ? path.join('renders', rawFile).replace(/\\/g, '/') : null,
      raw_meta:  rawMeta,
    },
    ai_gen: {
      ai_file: aiPath,
      provider: null,
      prompt:   null,
      model:    null,
      timestamp: null,
    },
    mapping: {
      cell_id_lookup: `cell_${qx + 1}_${qy + 1}_...`,
    },
  };
}

/**
 * Main import function.
 */
export async function importAiGenFiles(inputDir, destDir = DATASET_PATHS.aiGen) {
  if (!fs.existsSync(inputDir)) {
    throw new Error(`Input directory not exist: ${inputDir}`);
  }

  fs.mkdirSync(destDir, { recursive: true });

  const allFiles = fs.readdirSync(inputDir);
  const imageFiles = allFiles.filter(f =>
    IMPORT_RULES.acceptedExts.some(ext => f.toLowerCase().endsWith(ext))
  );

  const registry = loadRegistry();
  const imported = [];
  const skipped = [];
  const errors = [];

  for (const file of imageFiles) {
    const parsed = parseAiTileName(file);
    if (!parsed) {
      skipped.push(`${file}: wrong format tile_{qx}_{qy}.png`);
      continue;
    }

    const { qx, qy } = parsed;
    const srcPath = path.join(inputDir, file);
    const dstPath = path.join(destDir, `tile_${qx}_${qy}.png`);

    // Check file size
    const stat = fs.statSync(srcPath);
    if (stat.size / 1024 < IMPORT_RULES.minFileSizeKb) {
      skipped.push(`${file}: file too small (${(stat.size/1024).toFixed(1)}KB)`);
      continue;
    }

    try {
      // Copy
      fs.copyFileSync(srcPath, dstPath);

      // make sidecar
      const sidecar = buildSidecar(qx, qy, path.basename(dstPath));
      fs.writeFileSync(`${dstPath}.json`, JSON.stringify(sidecar, null, 2));

      // Update registry
      const aiFileAbs = path.resolve(dstPath).replace(/\\/g, '/');
      upsertRegistry(registry, {
        qx, qy,
        raw_file: sidecar.source.raw_file ? path.resolve(DATASET_PATHS.sourceRenders, '..', sidecar.source.raw_file).replace(/\\/g, '/') : null,
        ai_file:  aiFileAbs,
        status:   'imported',
        imported_at: sidecar.imported_at,
      });

      imported.push(`tile_${qx}_${qy}.png`);
    } catch (e) {
      errors.push(`${file}: ${e.message}`);
    }
  }

  saveRegistry(registry);

  return { imported: imported.length, skipped, errors, imported_list: imported };
}

/**
 * Update sidecar metadata
 */
export async function updateAiMeta(qx, qy, fields) {
  const sidecarPath = path.join(DATASET_PATHS.aiGen, `tile_${qx}_${qy}.png.json`);
  if (!fs.existsSync(sidecarPath)) {
    throw new Error(`Sidecar not exist: ${sidecarPath}`);
  }

  const sidecar = JSON.parse(fs.readFileSync(sidecarPath, 'utf8'));
  Object.assign(sidecar.ai_gen, fields);
  sidecar.ai_gen.timestamp = sidecar.ai_gen.timestamp ?? new Date().toISOString();

  fs.writeFileSync(sidecarPath, JSON.stringify(sidecar, null, 2));
  return sidecar;
}