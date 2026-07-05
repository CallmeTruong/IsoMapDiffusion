import fs from 'fs';
import path from 'path';

import { TILE, TILE_SIZE_M, QUADRANT_M, PATHS } from '../config.mjs';
import { findTile, listTiles, tileBounds } from '../tile/tile_io.mjs';
import { tileIndexToLatLng } from '../tile/coords.mjs';
import { degToRad } from '../utils/geo.mjs';

/**
 * Build rotated polygon
 */
function buildRotatedPolygon(qx, qy, seedLat, seedLng, cfg) {
  const headingRad = degToRad(cfg.azimuth);
  const cosH = Math.cos(headingRad);
  const sinH = Math.sin(headingRad);
  const cellToTileStep = 1 / cfg.cameraMoveStep;

  const corners = [
    [-0.5, -0.5], [+0.5, -0.5], [+0.5, +0.5], [-0.5, +0.5], [-0.5, -0.5],
  ];
  const ring = corners.map(([dx, dy]) => {
    const rx =  dx * cosH + dy * sinH;
    const ry = -dx * sinH + dy * cosH;
    const corner = tileIndexToLatLng(qx + rx * cellToTileStep, qy + ry * cellToTileStep, seedLat, seedLng, cfg);
    return [corner.lng, corner.lat];
  });
  return [ring];
}

/**
 * Build quadrants.geojson FeatureCollection.
 *
 * @param {Object} opts
 * @param {Map} opts.quadrantStatus - Map<"qx,qy", "LAND"|"INFRA"|"SKIP">
 * @param {string} opts.outputDir
 * @param {number} opts.seedLat
 * @param {number} opts.seedLng
 * @param {Object} opts.cfg - renderCfg
 * @param {number} opts.blankSizeKb
 * @returns {string} path to saved geojson
 */
export function buildQuadrantsGeojson({ quadrantStatus, outputDir, seedLat, seedLng, cfg, blankSizeKb }) {
  const features = [];
  for (const [key, status] of quadrantStatus) {
    const [qx, qy] = key.split(',').map(Number);
    const found = findTile(outputDir, qx, qy);
    const exists = found && found.filepath && fs.statSync(found.filepath).size >= blankSizeKb * 1024;
    features.push({
      type: 'Feature',
      properties: {
        qx, qy, status, rendered: !!exists,
        filename: found?.filename || null,
      },
      geometry: {
        type: 'Polygon',
        coordinates: buildRotatedPolygon(qx, qy, seedLat, seedLng, cfg),
      },
    });
  }

  const geojson = {
    type: 'FeatureCollection',
    properties: {
      seed_lat: seedLat,
      seed_lng: seedLng,
      quadrant_size_m: QUADRANT_M,
      tile_step: TILE.cameraMoveStep,
      camera_move_step: TILE.cameraMoveStep,
      heading: cfg.azimuth,
      pitch: cfg.elevation,
      altitude: cfg.altitude,
      frustum_w: cfg.frustumW,
      total_quadrants: features.length,
      config_version: 2,
    },
    features,
  };
  const outPath = path.join(outputDir, path.basename(PATHS.quadrantsGeojson));
  fs.writeFileSync(outPath, JSON.stringify(geojson, null, 2));
  return outPath;
}

/**
 * Build generation_config.json
 */
export function buildGenerationConfig({ outputDir, seedLat, seedLng, cfg, manifestFile, projectRoot, sessionCount = 0 }) {
  const allTiles = listTiles(outputDir);
  const bounds = tileBounds(allTiles);
  const rendered = allTiles.length;

  const config = {
    seed: { lat: seedLat, lng: seedLng },
    camera_azimuth_degrees: cfg.azimuth,
    camera_elevation_degrees: cfg.elevation,
    camera_altitude_m: cfg.altitude,
    width_px: TILE.sizePx,
    height_px: TILE.sizePx,
    view_height_meters: TILE_SIZE_M,
    camera_move_step: TILE.cameraMoveStep,
    tile_size_m: TILE_SIZE_M,
    tile_size_px: TILE.sizePx,
    manifest: manifestFile ? path.relative(projectRoot, manifestFile) : null,
    coverage: {
      min_qx: bounds.minQx, max_qx: bounds.maxQx,
      min_qy: bounds.minQy, max_qy: bounds.maxQy,
      total_tiles_expected: bounds.expectedTiles,
      tiles_rendered: rendered,
      tiles_total: allTiles.length,
    },
    config_version: 2,
    generated_at: new Date().toISOString(),
  };
  const outPath = path.join(outputDir, PATHS.generationConfig);
  fs.writeFileSync(outPath, JSON.stringify(config, null, 2));
  return outPath;
}


export function exportRenderOutputs(args) {
  const geojsonPath = buildQuadrantsGeojson(args);
  const configPath = buildGenerationConfig(args);
  return { geojsonPath, configPath };
}
