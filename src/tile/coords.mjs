import { GEO } from '../config.mjs';

export const M_PER_DEG_LAT = GEO.mPerDegLat;

export function qxyToLatLngAxial(qx, qy, seedLat, seedLng, renderCfg) {
  let stepM;
  if (typeof renderCfg.quadrantM === 'number' && renderCfg.quadrantM > 0) {
    stepM = renderCfg.quadrantM;
  } else {
    const mPerPx = renderCfg.frustumW / renderCfg.sizePx;
    const tileStep = renderCfg.tileStep ?? 0.5;
    stepM = renderCfg.sizePx * tileStep * mPerPx;
  }

  const deltaEastM  = qx * stepM;
  const deltaNorthM = qy * stepM;

  const mPerDegLng = M_PER_DEG_LAT * Math.cos(seedLat * Math.PI / 180);
  const lat = seedLat + deltaNorthM / M_PER_DEG_LAT;
  const lng = seedLng + deltaEastM / mPerDegLng;

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
  const elev_rad = elevation * Math.PI / 180;
  const sin_elev = Math.sin(elev_rad);
  if (Math.abs(sin_elev) < 1e-6) {
    throw new Error(`Elevation ${elevation}° close to 0/180`);
  }
  const delta_rot_x = shift_right_meters;
  const delta_rot_y = -shift_up_meters / sin_elev;

  // 4. Rotate by azimuth
  const azimuth_rad = azimuth * Math.PI / 180;
  const cos_a = Math.cos(azimuth_rad);
  const sin_a = Math.sin(azimuth_rad);

  const delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a;
  const delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a;

  // 5. East/North meters → lat/lng
  const mPerDegLng = M_PER_DEG_LAT * Math.cos(seedLat * Math.PI / 180);
  const lat = seedLat + delta_north_meters / M_PER_DEG_LAT;
  const lng = seedLng + delta_east_meters / mPerDegLng;

  return {
    lat, lng,
    east_m: delta_east_meters,
    north_m: delta_north_meters,
    x_rot_m: delta_rot_x,
    y_rot_m: delta_rot_y,
  };
}

// Convert tile index (qx, qy) → lat/lng camera center
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

  // 3. Convert screen → rotated world (according to elevation)
  const elev_rad = elevation * Math.PI / 180;
  const sin_elev = Math.sin(elev_rad);
  if (Math.abs(sin_elev) < 1e-6) {
    throw new Error(`Elevation ${elevation}° close to 0/180 (sin=0)`);
  }
  const delta_rot_x = shift_right_meters;
  const delta_rot_y = -shift_up_meters / sin_elev;

  // 4. Rotate by azimuth
  const azimuth_rad = azimuth * Math.PI / 180;
  const cos_a = Math.cos(azimuth_rad);
  const sin_a = Math.sin(azimuth_rad);

  const delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a;
  const delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a;

  // 5. East/North meters → lat/lng
  const mPerDegLng = M_PER_DEG_LAT * Math.cos(seedLat * Math.PI / 180);
  const lat = seedLat + delta_north_meters / M_PER_DEG_LAT;
  const lng = seedLng + delta_east_meters / mPerDegLng;

  return {
    lat, lng,
    east_m: delta_east_meters,
    north_m: delta_north_meters,
  };
}

// Convert lat/lng to quadrant coordinates
export function latLngToQxy(lat, lng, seedLat, seedLng, quadrantStepM) {
  const deltaNorthM = (lat - seedLat) * M_PER_DEG_LAT;
  const deltaEastM  = (lng - seedLng) * M_PER_DEG_LAT * Math.cos(seedLat * Math.PI / 180);

  const qy = Math.round(deltaNorthM / quadrantStepM);
  const qx = Math.round(deltaEastM  / quadrantStepM);

  return { qx, qy };
}
