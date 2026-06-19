/**
 * report.mjs — Export selection report (JSON + CSV)
 */

import fs from 'fs';
import path from 'path';
import { DATASET_PATHS } from './constants.mjs';
import { summarizeTiles } from './analyze.mjs';
import { summarizeSelection } from './select.mjs';

/**
 * Export report JSON.
 */
export async function exportReport(selected, allTiles, options = {}) {
  fs.mkdirSync(DATASET_PATHS.root, { recursive: true });

  const report = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    pool_summary: summarizeTiles(allTiles),
    selection_summary: summarizeSelection(selected, allTiles),
    config: {
      target: options.target ?? null,
      weights: options.weights ?? null,
      diversity: options.diversity ?? null,
    },
    tiles: selected.map(t => ({
      qx: t.qx,
      qy: t.qy,
      score: Number(t.score.toFixed(4)),
      zone_key: t.zoneKey ?? null,
      color_bucket: t.colorBucket ?? null,
      render_file: t.renderFile,
      meta_file: t.metaFile,
      variance: t.variance,
      edge_density: t.edgeDensity,
      mean_rgb: { r: t.meanR, g: t.meanG, b: t.meanB },
      size_kb: t.sizeKb,
      lat: t.lat,
      lng: t.lng,
      hash: t.hash,
    })),
  };

  fs.writeFileSync(DATASET_PATHS.reportJson, JSON.stringify(report, null, 2));
  fs.writeFileSync(DATASET_PATHS.reportCsv, toCsv(report.tiles));

  return { json: DATASET_PATHS.reportJson, csv: DATASET_PATHS.reportCsv };
}

function toCsv(tiles) {
  const header = [
    'qx', 'qy', 'score', 'zone_key', 'color_bucket',
    'variance', 'edge_density', 'mean_r', 'mean_g', 'mean_b',
    'size_kb', 'lat', 'lng', 'hash', 'render_file',
  ];
  const rows = tiles.map(t => [
    t.qx, t.qy,
    t.score.toFixed(4),
    t.zone_key ?? '',
    t.color_bucket ?? '',
    t.variance.toFixed(1),
    t.edge_density.toFixed(4),
    Math.round(t.mean_rgb.r),
    Math.round(t.mean_rgb.g),
    Math.round(t.mean_rgb.b),
    t.size_kb.toFixed(1),
    t.lat ?? '',
    t.lng ?? '',
    t.hash ?? '',
    t.render_file,
  ].join(','));

  return [header.join(','), ...rows].join('\n');
}