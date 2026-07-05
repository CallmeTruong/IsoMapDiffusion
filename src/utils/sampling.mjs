import * as turf from '@turf/turf';
import { GRID } from '../config.mjs';

export function hasLand(cell, waterSearch, waterFeatures, minLandM2) {
  const threshold = minLandM2 ?? GRID.minWaterM2;
  const cellBBox = turf.bbox(cell);
  const result = waterSearch(cellBBox);
  const candidates = result.features;

  // Fast path: no water nearby → cell is land
  if (candidates.length === 0) {
    return true;
  }

  let remainingCell = cell;
  let subtractedAtLeastOne = false;

  // Multi-pass
  for (const candidate of candidates) {
    const waterIdx = candidate.properties._idx;
    const waterPoly = waterFeatures[waterIdx];
    if (!waterPoly) continue;

    try {
      
      if (!turf.booleanIntersects(remainingCell, waterPoly)) continue;

      const diff = turf.difference(
        turf.featureCollection([remainingCell, waterPoly])
      );

      if (diff === null) {
        return false;
      }

      if (diff.type === 'FeatureCollection') {
        remainingCell = mergeFeatureCollection(diff);
      } else {
        remainingCell = diff;
      }
      subtractedAtLeastOne = true;

      if (!remainingCell) {
        return false;
      }
    } catch (e) {
      continue;
    }
  }

  if (!subtractedAtLeastOne) {
    return true;
  }

  const finalArea = turf.area(remainingCell);
  return finalArea >= threshold;
}

/**
 * Merge a FeatureCollection of polygons into a single polygon.
 * Returns null if collection is empty or merge fails.
 */
function mergeFeatureCollection(fc) {
  if (!fc.features || fc.features.length === 0) return null;
  if (fc.features.length === 1) return fc.features[0];

  let merged = fc.features[0];
  for (let i = 1; i < fc.features.length; i++) {
    try {
      const result = turf.union(
        turf.featureCollection([merged, fc.features[i]])
      );
      if (result) merged = result;
    } catch (e) {
      continue;
    }
  }
  return merged;
}
