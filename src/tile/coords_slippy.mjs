/**
 * coords_slippy.mjs — Slippy map tile XYZ coordinates (lat/lng → x/y at zoom z)
 *
 * All math constants come from src/utils/geo.mjs → src/config.mjs.
 */

import {
  DEG_TO_RAD, TWO_PI, EARTH_CIRC_M, MERCATOR_LAT_MAX,
  SLIPPY_MIN_ZOOM, SLIPPY_MAX_ZOOM,
  clampMercatorLat, slippyScale, clampSlippyZ,
} from '../utils/geo.mjs';

/**
 * Convert longitude to slippy tile X coordinate at zoom z.
 *
 * @param {number} lng - longitude in degrees [-180, 180]
 * @param {number} z - zoom level [minZoom, maxZoom]
 * @returns {number} tile X coordinate [0, 2^z - 1]
 */
export function lngToTileX(lng, z) {
  const scale = slippyScale(z);
  const x = ((lng + 180) / 360) * scale;
  return Math.max(0, Math.min(scale - 1, Math.floor(x)));
}

/**
 * Convert latitude to slippy tile Y coordinate at zoom z.
 * @param {number} lat - latitude in degrees [-MERCATOR_LAT_MAX, MERCATOR_LAT_MAX]
 * @param {number} z - zoom level [minZoom, maxZoom]
 * @returns {number} tile Y coordinate [0, 2^z - 1]
 */
export function latToTileY(lat, z) {
  const clippedLat = clampMercatorLat(lat);
  const rad = clippedLat * DEG_TO_RAD;
  const scale = slippyScale(z);
  const y = (1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / TWO_PI) / 2 * scale;
  return Math.max(0, Math.min(scale - 1, Math.floor(y)));
}

/**
 * Convert (lat, lng) to slippy XYZ tuple.
 *
 * @param {number} lat - latitude in degrees
 * @param {number} lng - longitude in degrees
 * @param {number} z - zoom level
 * @returns {{x:number, y:number, z:number}}
 */
export function tileIndexToSlippyXYZ(lat, lng, z) {
  return {
    x: lngToTileX(lng, z),
    y: latToTileY(lat, z),
    z,
  };
}

/**
 * Compute the "footprint" (meters wide) of a slippy tile at zoom z, at a given latitude.
 * Used to pick the right zoom level for the tile footprint we want.
 *
 * @param {number} z - zoom level
 * @param {number} lat - latitude in degrees (for east-west scale)
 * @returns {number} approximate tile width in meters
 */
export function tileFootprintMeters(z, lat = 0) {
  const totalTiles = slippyScale(z);
  const cosLat = Math.cos(lat * DEG_TO_RAD);
  // Mercator: width at latitude = earthCircumference * cos(lat) / totalTiles
  return (EARTH_CIRC_M * cosLat) / totalTiles;
}

/**
 * Compute the zoom level whose tile footprint is closest to `targetMeters`.
 *
 * @param {number} targetMeters - desired tile size in meters
 * @param {number} lat - latitude in degrees (for east-west scale)
 * @returns {number} zoom level [minZoom, maxZoom]
 */
export function zoomForFootprint(targetMeters, lat = 0) {
  const cosLat = Math.cos(lat * DEG_TO_RAD);
  const idealTiles = (EARTH_CIRC_M * cosLat) / targetMeters;
  const idealZ = Math.log2(idealTiles);
  const z = Math.round(idealZ);
  return clampSlippyZ(z);
}