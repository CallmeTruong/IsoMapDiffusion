import RBush from 'rbush';
import * as turf from '@turf/turf';

/**
 * Build a spatial index (rbush) from GeoJSON features.
 * Returns a wrapper object that mimics the search() response format
 * with .features array.
 *
 * @param {Array} features - Array of GeoJSON features
 * @returns {{ search: Function, features: Array }} - Searchable index and original features
 */
export function buildSpatialIndex(features) {
  const tree = new RBush();

  const bboxItems = features.map((feature, idx) => {
    const bbox = turf.bbox(feature);
    return {
      minX: bbox[0],
      minY: bbox[1],
      maxX: bbox[2],
      maxY: bbox[3],
      properties: { _idx: idx, ...feature.properties }
    };
  });

  tree.load(bboxItems);

  const search = (bbox) => {
    const results = tree.search({
      minX: Array.isArray(bbox) ? bbox[0] : bbox.minX,
      minY: Array.isArray(bbox) ? bbox[1] : bbox.minY,
      maxX: Array.isArray(bbox) ? bbox[2] : bbox.maxX,
      maxY: Array.isArray(bbox) ? bbox[3] : bbox.maxY,
    });
    return { features: results };
  };

  return { search, features };
}