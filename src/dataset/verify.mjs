import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { DATASET_PATHS, TILE_RENDER } from './constants.mjs';
import { loadRegistry } from './registry.mjs';


function readRawMetaLocal(qx, qy) {
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


export async function verifyDataset() {
  const registry = loadRegistry();
  const report = {
    verified_at: new Date().toISOString(),
    total: registry.tiles.length,
    issues: [],
    stats: {
      raw_ok: 0, raw_missing: 0,
      ai_ok:  0, ai_missing: 0,
      sidecar_ok: 0, sidecar_missing: 0,
      dim_mismatch: 0,
      mapped_to_grid: 0,
    },
    tiles: [],
  };

  // Load render_manifest.json
  let manifest = null;
  const manifestPath = path.join(DATASET_PATHS.root.replace(/dataset$/, ''), 'render_manifest.json');
  if (fs.existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    } catch { /* ignore */ }
  }

  for (const entry of registry.tiles) {
    const tileReport = {
      qx: entry.qx,
      qy: entry.qy,
      raw_ok: false,
      ai_ok: false,
      sidecar_ok: false,
      dim_match: null,
      manifest_cell_id: null,
      issues: [],
    };

    // 1. Resolve raw_file path
    let rawAbs = null;
    if (entry.raw_file) {
      const candidate1 = path.resolve(DATASET_PATHS.root, '..', '..', entry.raw_file);
      const candidate2 = path.resolve(DATASET_PATHS.root, '..', entry.raw_file);
      if (fs.existsSync(candidate1)) {
        rawAbs = candidate1;
      } else if (fs.existsSync(candidate2)) {
        rawAbs = candidate2;
      }
    }
    if (rawAbs) {
      tileReport.raw_ok = true;
      report.stats.raw_ok++;
    } else {
      tileReport.issues.push('raw_missing');
      report.stats.raw_missing++;
    }

    // 2. Resolve ai_file path
    let aiAbs = null;
    if (entry.ai_file) {
      const candidate1 = path.resolve(DATASET_PATHS.root, '..', '..', entry.ai_file);
      const candidate2 = path.resolve(DATASET_PATHS.root, entry.ai_file);
      if (fs.existsSync(candidate1)) {
        aiAbs = candidate1;
      } else if (fs.existsSync(candidate2)) {
        aiAbs = candidate2;
      }
    }
    if (aiAbs) {
      tileReport.ai_ok = true;
      report.stats.ai_ok++;
    } else {
      tileReport.issues.push('ai_missing');
      report.stats.ai_missing++;
    }

    // 3. Check sidecar
    if (entry.ai_file) {
      const sidecarPath = path.join(DATASET_PATHS.aiGen, `tile_${entry.qx}_${entry.qy}.png.json`);
      if (fs.existsSync(sidecarPath)) {
        try {
          const sidecar = JSON.parse(fs.readFileSync(sidecarPath, 'utf8'));
          if (sidecar.qx === entry.qx && sidecar.qy === entry.qy) {
            tileReport.sidecar_ok = true;
            report.stats.sidecar_ok++;
          } else {
            tileReport.issues.push('sidecar_qx_qy_mismatch');
          }
        } catch {
          tileReport.issues.push('sidecar_parse_error');
        }
      } else {
        tileReport.issues.push('sidecar_missing');
        report.stats.sidecar_missing++;
      }
    }

    // 4. Check dimensions
    if (tileReport.raw_ok && tileReport.ai_ok) {
      try {
        const [rawMeta, aiMeta] = await Promise.all([
          sharp(rawAbs).metadata(),
          sharp(aiAbs).metadata(),
        ]);
        const rawW = rawMeta.width;
        const aiW = aiMeta.width;
        tileReport.dim_match = (rawW === aiW && rawMeta.height === aiMeta.height);
        if (!tileReport.dim_match) {
          tileReport.issues.push(`dim_mismatch: raw=${rawW}x${rawMeta.height} ai=${aiW}x${aiMeta.height}`);
          report.stats.dim_mismatch++;
        }
      } catch (e) {
        tileReport.issues.push(`dim_check_error: ${e.message}`);
      }
    }

    // 5. Map to render_manifest.json
    if (manifest && Array.isArray(manifest)) {
      const cell = manifest.find(c => c.col === entry.qx + 1 && c.row === entry.qy + 1);
      if (cell) {
        tileReport.manifest_cell_id = cell.cell_id;
        report.stats.mapped_to_grid++;
      }
    }

    if (tileReport.issues.length > 0) {
      report.issues.push({ qx: entry.qx, qy: entry.qy, issues: tileReport.issues });
    }

    report.tiles.push(tileReport);
  }

  return report;
}

/**
 * Print verification report
 */
export function printVerifyReport(report) {
  console.log('\n═══ Dataset Verification Report ═══');
  console.log(`Verified at: ${report.verified_at}`);
  console.log(`Total tiles: ${report.total}\n`);

  console.log('Stats:');
  console.log(`  raw_ok/missing:        ${report.stats.raw_ok}/${report.stats.raw_missing}`);
  console.log(`  ai_ok/missing:         ${report.stats.ai_ok}/${report.stats.ai_missing}`);
  console.log(`  sidecar_ok/missing:    ${report.stats.sidecar_ok}/${report.stats.sidecar_missing}`);
  console.log(`  dim_mismatch:          ${report.stats.dim_mismatch}`);
  console.log(`  mapped_to_grid:        ${report.stats.mapped_to_grid}`);

  if (report.issues.length > 0) {
    console.log(`\n ${report.issues.length} tile wrong:`);
    for (const issue of report.issues.slice(0, 20)) {
      console.log(`  (${issue.qx},${issue.qy}): ${issue.issues.join(', ')}`);
    }
    if (report.issues.length > 20) {
      console.log(`  ... and ${report.issues.length - 20} more`);
    }
  } else {
    console.log('\n All tiles OK');
  }
}