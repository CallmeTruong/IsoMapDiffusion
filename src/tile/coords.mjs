import { GEO } from '../config.mjs';
import { mPerDegLng, degToRad, M_PER_DEG_LAT } from '../utils/geo.mjs';

export { M_PER_DEG_LAT };

export function qxyToLatLngAxial(qx, qy, seedLat, seedLng, renderCfg) {
  let stepM;
  if (typeof renderCfg.quadrantM === 'number' && renderCfg.quadrantM > 0) {
    stepM = renderCfg.quadrantM;
  } else {
    const mPerPx = renderCfg.frustumW / renderCfg.sizePx;
    const tileStep = renderCfg.tileStep ?? renderCfg.cameraMoveStep ?? 0.5;
    stepM = renderCfg.sizePx * tileStep * mPerPx;
  }

  const deltaEastM  = qx * stepM;
  const deltaNorthM = qy * stepM;

  const mPerLng = mPerDegLng(seedLat);
  const lat = seedLat + deltaNorthM / M_PER_DEG_LAT;
  const lng = seedLng + deltaEastM / mPerLng;

  return { lat, lng, east_m: deltaEastM, north_m: deltaNorthM };
}

export const qxyToLatLng = qxyToLatLngAxial;


export function quadrantRCToLatLng(col, row, seedLat, seedLng, renderCfg) {
  const {
    sizePx, frustumW, azimuth, elevation, cameraMoveStep, quadrantM,
  } = renderCfg;

  // Validate
  if (typeof azimuth !== 'number') {
    throw new Error('quadrantRCToLatLng: renderCfg.azimuth is required');
  }
  if (typeof elevation !== 'number') {
    throw new Error('quadrantRCToLatLng: renderCfg.elevation is required');
  }
  if (typeof cameraMoveStep !== 'number' || cameraMoveStep <= 0) {
    throw new Error('quadrantRCToLatLng: renderCfg.cameraMoveStep is required (>0)');
  }

  // 1. Pixel shift in image space.
  const stepPx = sizePx * cameraMoveStep;
  const shift_x_px = col * stepPx;
  const shift_y_px = -row * stepPx;

  // 2. Pixel → meters in screen space
  const metersPerPixel = frustumW / sizePx;
  const shift_right_meters = shift_x_px * metersPerPixel;
  const shift_up_meters = shift_y_px * metersPerPixel;

  // 3. Convert screen → rotated world
  const elev_rad = degToRad(elevation);
  const sin_elev = Math.sin(elev_rad);
  if (Math.abs(sin_elev) < 1e-6) {
    throw new Error(`Elevation ${elevation}° close to 0/180`);
  }
  const delta_rot_x = shift_right_meters;
  const delta_rot_y = -shift_up_meters / sin_elev;

  // 4. Rotate by azimuth
  const azimuth_rad = degToRad(azimuth);
  const cos_a = Math.cos(azimuth_rad);
  const sin_a = Math.sin(azimuth_rad);

  const delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a;
  const delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a;

  // 5. East/North meters → lat/lng
  const mPerLng = mPerDegLng(seedLat);
  const lat = seedLat + delta_north_meters / M_PER_DEG_LAT;
  const lng = seedLng + delta_east_meters / mPerLng;

  return {
    lat, lng,
    east_m: delta_east_meters,
    north_m: delta_north_meters,
    x_rot_m: delta_rot_x,
    y_rot_m: delta_rot_y,
  };
}

/**
 *Convert tile (qx, qy) in TILE INDEX space
 * → lat/lng camera center.
 * @param {number} qx — tile x index
 * @param {number} qy — tile y index
 * @param {number} seedLat
 * @param {number} seedLng
 * @param {Object} renderCfg
 * @returns {{lat, lng, east_m, north_m}}
 */
export function tileIndexToLatLng(qx, qy, seedLat, seedLng, renderCfg) {
  const { sizePx, frustumW, azimuth, elevation, cameraMoveStep } = renderCfg;

  // Validate
  if (typeof azimuth !== 'number') {
    throw new Error('tileIndexToLatLng: renderCfg.azimuth is required');
  }
  if (typeof elevation !== 'number') {
    throw new Error('tileIndexToLatLng: renderCfg.elevation is required');
  }
  if (typeof cameraMoveStep !== 'number' || cameraMoveStep <= 0) {
    throw new Error('tileIndexToLatLng: renderCfg.cameraMoveStep is required (>0)');
  }

  // Tile center index = (qx + 0.5, qy + 0.5)
  const stepPx = sizePx * cameraMoveStep;
  const shift_x_px = (qx + 0.5) * stepPx;
  const shift_y_px = -(qy + 0.5) * stepPx;

  // 2. Pixel → meters in screen space
  const metersPerPixel = frustumW / sizePx;
  const shift_right_meters = shift_x_px * metersPerPixel;
  const shift_up_meters = shift_y_px * metersPerPixel;

  // 3. Convert screen → rotated world (theo elevation)
  const elev_rad = degToRad(elevation);
  const sin_elev = Math.sin(elev_rad);
  if (Math.abs(sin_elev) < 1e-6) {
    throw new Error(`Elevation ${elevation}° close to 0/180 (sin=0)`);
  }
  const delta_rot_x = shift_right_meters;
  const delta_rot_y = -shift_up_meters / sin_elev;

  // 4. Rotate by azimuth
  const azimuth_rad = degToRad(azimuth);
  const cos_a = Math.cos(azimuth_rad);
  const sin_a = Math.sin(azimuth_rad);

  const delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a;
  const delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a;

  // 5. East/North meters → lat/lng
  const mPerLng = mPerDegLng(seedLat);
  const lat = seedLat + delta_north_meters / M_PER_DEG_LAT;
  const lng = seedLng + delta_east_meters / mPerLng;

  return {
    lat, lng,
    east_m: delta_east_meters,
    north_m: delta_north_meters,
  };
}

/**
 * [LEGACY] Convert tile (qx, qy) in quadrant space → lat/lng camera center.
 *
 * @param {number} qx — tile x index in quadrant coords
 * @param {number} qy — tile y index in quadrant coords
 * @param {number} seedLat
 * @param {number} seedLng
 * @param {Object} renderCfg
 * @returns {{lat, lng, east_m, north_m}}
 */
export function tileQxQyToLatLng(qx, qy, seedLat, seedLng, renderCfg) {
  const tileStep = renderCfg.tileStep ?? renderCfg.cameraMoveStep ?? 0.5;
  const quadsPerTile = Math.round(1 / tileStep);
  return quadrantRCToLatLng(qx, qy, seedLat, seedLng, renderCfg);
}

/**
 * Convert lat/lng to quadrant coordinates.
 */
export function latLngToQxy(lat, lng, seedLat, seedLng, quadrantStepM) {
  const deltaNorthM = (lat - seedLat) * M_PER_DEG_LAT;
  const deltaEastM  = (lng - seedLng) * mPerDegLng(seedLat);

  const qy = Math.round(deltaNorthM / quadrantStepM);
  const qx = Math.round(deltaEastM  / quadrantStepM);

  return { qx, qy };
}
