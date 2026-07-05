/**
 * utils/geo.mjs — Geographic & angular math helpers
 *
 * Centralized so call-sites never see raw Math.PI / 180, 111111, 2π, etc.
 * All values come from src/config.mjs (GEO.*), with env-var / JSON-overlay
 * overrides applied via the standard config load pipeline.
 *
 * Pure module: no I/O, no side effects.
 */

import { GEO } from '../config.mjs';

export const DEG_TO_RAD       = GEO.degToRad;
export const RAD_TO_DEG       = GEO.radToDeg;
export const TWO_PI           = GEO.twoPi;
export const M_PER_DEG_LAT    = GEO.mPerDegLat;
export const EARTH_CIRC_M     = GEO.earthCircumferenceM;
export const MERCATOR_LAT_MAX = GEO.mercatorLatClipDeg;
export const SLIPPY_MIN_ZOOM  = GEO.minZoom;
export const SLIPPY_MAX_ZOOM  = GEO.maxZoom;

export const degToRad = (d) => d * DEG_TO_RAD;
export const radToDeg = (r) => r * RAD_TO_DEG;

export function mPerDegLng(latDeg) {
  return M_PER_DEG_LAT * Math.cos(latDeg * DEG_TO_RAD);
}

export function clampMercatorLat(lat) {
  return Math.max(-MERCATOR_LAT_MAX, Math.min(MERCATOR_LAT_MAX, lat));
}

export function slippyScale(z) {
  return Math.pow(2, z);
}

export function clampSlippyZ(z) {
  return Math.max(SLIPPY_MIN_ZOOM, Math.min(SLIPPY_MAX_ZOOM, z));
}