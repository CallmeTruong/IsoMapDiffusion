import fs from 'fs';
import path from 'path';
import { DATASET_PATHS } from './constants.mjs';

function formatName(qx, qy) {
  return {
    png:  `tile_${qx}_${qy}.png`,
    json: `tile_${qx}_${qy}.json`,
  };
}

export async function exportSelectedTiles(selected, destDir = DATASET_PATHS.rawTiles) {
  fs.mkdirSync(destDir, { recursive: true });

  let copiedPng = 0;
  let copiedMeta = 0;
  const errors = [];

  for (const t of selected) {
    const qx = t.qx;
    const qy = t.qy;
    const renderFile = t.render_file ?? t.renderFile ?? null;
    const metaFile   = t.meta_file   ?? t.metaFile   ?? null;

    const { png, json } = formatName(qx, qy);

    // Copy PNG
    if (renderFile) {
      const srcPng = path.join(DATASET_PATHS.sourceRenders, renderFile);
      const dstPng = path.join(destDir, png);
      try {
        fs.copyFileSync(srcPng, dstPng);
        copiedPng++;
      } catch (e) {
        errors.push(`PNG copy failed for (${qx},${qy}): ${e.message}`);
      }
    } else {
      errors.push(`No render_file for (${qx},${qy})`);
    }

    // Copy metadata
    if (metaFile) {
      const srcMeta = path.join(DATASET_PATHS.sourceMeta, metaFile);
      const dstMeta = path.join(destDir, json);
      try {
        const metaContent = JSON.parse(fs.readFileSync(srcMeta, 'utf8'));
        if (t.score !== undefined) {
          metaContent.selection = {
            score:        Number((t.score ?? 0).toFixed(4)),
            zone_key:     t.zone_key ?? t.zoneKey ?? null,
            color_bucket: t.color_bucket ?? t.colorBucket ?? null,
            exported_at:  new Date().toISOString(),
          };
        }
        fs.writeFileSync(dstMeta, JSON.stringify(metaContent, null, 2));
        copiedMeta++;
      } catch (e) {
        errors.push(`Meta copy failed for (${qx},${qy}): ${e.message}`);
      }
    } else {
      errors.push(`No meta_file for (${qx},${qy})`);
    }
  }

  return { copiedPng, copiedMeta, errors };
}