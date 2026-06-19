import * as turf from '@turf/turf';
import fs from 'fs';
import path from 'path';
import { buildSpatialIndex } from '../utils/geo.mjs';
import { hasLand } from '../utils/sampling.mjs';
import { GRID } from '../config.mjs';

/**
 * Generate cell ID from grid coordinates
 */
export function cellIdFromGrid(col, row, centroid) {
  const clng = Math.round(centroid[0] * 100000) / 100000;
  const clat = Math.round(centroid[1] * 100000) / 100000;
  return `cell_${col}_${row}_${clng}_${clat}`;
}

/**
 * Load district boundaries from directory
 */
export function loadDistricts(dir) {
  if (!fs.existsSync(dir)) {
    throw new Error('dir not found: ' + dir);
  }

  const files = fs.readdirSync(dir)
    .filter(f => f.endsWith('.geojson') || f.endsWith('.json'))
    .sort();

  if (files.length === 0) {
    throw new Error('cant find any .geojson in ' + dir);
  }

  const features = [];
  for (const file of files) {
    const data = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));

    let raw = [];
    if (data.type === 'FeatureCollection') raw = data.features;
    else if (data.type === 'Feature')      raw = [data];
    else if (data.type === 'Polygon' || data.type === 'MultiPolygon') raw = [turf.feature(data)];

    const polygons = raw.filter(f =>
      f?.geometry &&
      (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon')
    );

    if (polygons.length === 0) {
      console.warn('Skip ( no Polygon ): ' + file);
      continue;
    }

    features.push(...polygons);
    console.log('  Loaded: ' + file + ' (' + polygons.length + ' polygon)');
  }

  if (features.length === 0) {
    throw new Error('No Polygon');
  }

  return features;
}

/**
 * Merge all districts into 1 polygon for fast bounds check
 */
export function buildUnion(features) {
  const rewound = features.map(f => turf.rewind(f, { reverse: true, mutate: false }));
  let union = rewound[0];
  for (let i = 1; i < rewound.length; i++) {
    try {
      union = turf.union(turf.featureCollection([union, rewound[i]])) ?? union;
    } catch { /* bo qua geometry loi */ }
  }
  return union;
}

/**
 * Load polygon file with filtering by area
 */
export function loadPolygonFile(filePath, minArea = 0) {
  if (!fs.existsSync(filePath)) return [];
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return turf.rewind(raw, { reverse: true, mutate: false })
    .features
    .filter(f =>
      f.geometry &&
      (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon') &&
      turf.area(f) > minArea
    );
}

/**
 * Load any geometry file
 */
export function loadAnyGeomFile(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return raw.features.filter(f => f.geometry);
}

/**
 * Classify a cell using spatial index
 */
export function classifyCell(cell, cellBBox, cityBBox, citySearch, cityFeatures,
                           infraSearch, infraFeatures, waterSearch, waterFeatures, minWaterM2) {
  // Fast path 1: outside city bbox
  if (cellBBox[2] < cityBBox[0] || cellBBox[0] > cityBBox[2] ||
      cellBBox[3] < cityBBox[1] || cellBBox[1] > cityBBox[3]) {
    return 'SKIP';
  }

  // Fast path 2: intersects city boundary
  const cityHits = citySearch(cellBBox);
  const inCity = cityHits.features.some(c => {
    try { return turf.booleanIntersects(cell, cityFeatures[c.properties._idx]); }
    catch { return false; }
  });
  if (!inCity) return 'SKIP';

  // Fast path 3: touching infra
  const infraHits = infraSearch(cellBBox);
  for (const c of infraHits.features) {
    try {
      if (turf.booleanIntersects(cell, infraFeatures[c.properties._idx])) {
        return 'INFRA';
      }
    } catch { /* skip */ }
  }

  // Fast path 4: no water nearby
  const waterHits = waterSearch(cellBBox);
  if (waterHits.features.length === 0) return 'LAND';

  // Fast path 5: water fully contains cell
  for (const c of waterHits.features) {
    try {
      if (turf.booleanContains(waterFeatures[c.properties._idx], cell)) {
        return 'SKIP';
      }
    } catch { /* MultiPolygon edge case */ }
  }

  // Exact path
  return hasLand(cell, waterSearch, waterFeatures, minWaterM2) ? 'LAND' : 'SKIP';
}

/**
 * Main grid generation function
 */
export async function generateGrid(options = {}) {
  const {
    districtsDir = './districts',
    waterFile = './water.geojson',
    infraFile = './infra.geojson',
    cellSizeKm = GRID.cellSizeKm,
    minWaterM2 = GRID.minWaterM2,
    outputDir = './output',
  } = options;

  console.log('\n=== HCMC Grid Generator (Optimized) ===\n');

  // 1. Load districts
  console.log('1. Load districts...');
  const districtFeatures = loadDistricts(districtsDir);
  const districtUnion = buildUnion(districtFeatures);
  const cityBBox = turf.bbox(districtUnion);
  console.log('   BBox: [' + cityBBox.map(v => v.toFixed(4)).join(', ') + ']\n');

  // 2. Load water
  console.log('2. Load water (' + waterFile + ')...');
  const waterFeaturesRaw = loadPolygonFile(waterFile, minWaterM2);
  console.log('   ' + waterFeaturesRaw.length + ' vung nuoc\n');

  // 3. Load infra
  console.log('3. Load infra (' + infraFile + ')...');
  const infraFeaturesRaw = loadAnyGeomFile(infraFile);
  console.log('   ' + infraFeaturesRaw.length + ' cong trinh\n');

  // 4. Build spatial indexes
  console.log('4. Build spatial indexes (rbush)...');
  const { search: citySearch }  = buildSpatialIndex(districtFeatures);
  const { search: infraSearch } = buildSpatialIndex(infraFeaturesRaw);
  const { search: waterSearch } = buildSpatialIndex(waterFeaturesRaw);
  console.log('   Done.\n');

  // 5. Generate grid
  console.log('5. Generate grid ' + cellSizeKm + 'km...');
  const rawGrid = turf.squareGrid(cityBBox, cellSizeKm, { units: 'kilometers' });
  console.log('   ' + rawGrid.features.length + ' raw cells\n');

  // 6. Classify cells
  console.log('6. Classify cells (with spatial index)...');

  let land = 0, infra = 0, skip = 0;
  let processed = 0;
  const total = rawGrid.features.length;
  const progressInterval = Math.ceil(total / 50);

  const features = rawGrid.features
    .map((cell) => {
      const cellBBox = turf.bbox(cell);
      const [lng, lat] = turf.centroid(cell).geometry.coordinates;

      const cosLat = Math.cos(cityBBox[3] * Math.PI / 180);
      const cellSizeDegLng = (cellSizeKm * 1000) / (111111 * cosLat);
      const cellSizeDegLat = (cellSizeKm * 1000) / 111111;
      const originX = cityBBox[0];
      const originY = cityBBox[3];

      const col = Math.floor((lng - originX) / cellSizeDegLng);
      const row = Math.floor((originY - lat) / cellSizeDegLat);
      
      const status = classifyCell(
        cell, cellBBox,
        cityBBox,
        citySearch, districtFeatures,
        infraSearch, infraFeaturesRaw,
        waterSearch, waterFeaturesRaw,
        minWaterM2
      );

      if (status === 'SKIP')  { skip++;  return null; }
      if (status === 'INFRA') { infra++; }
      if (status === 'LAND')  { land++;  }

      processed++;
      if (processed % progressInterval === 0) {
        const pct = Math.round(processed / total * 100);
        process.stdout.write(`\r   Progress: ${pct}% (${processed}/${total})`);
      }

      return {
        ...cell,
        properties: {
          cell_id: cellIdFromGrid(col, row, [lng, lat]),
          col,
          row,
          centroid_lng: lng,
          centroid_lat: lat,
          status,
          area_m2: Math.round(turf.area(cell)),
          bbox_minX: cellBBox[0],
          bbox_minY: cellBBox[1],
          bbox_maxX: cellBBox[2],
          bbox_maxY: cellBBox[3],
        }
      };
    })
    .filter(Boolean);

  console.log('\n\n Result:');
  console.log('  LAND  : ' + land);
  console.log('  INFRA : ' + infra);
  console.log('  SKIP  : ' + skip);
  console.log('  TOTAL : ' + features.length);

  // Check for duplicate cell_ids
  const ids = features.map(f => f.properties.cell_id);
  const dupSet = new Set();
  const dups = ids.filter(id => dupSet.has(id) ? true : (dupSet.add(id), false)).length;

  if (dups > 0) console.warn('\nWarn: ' + dups + ' cell_id is duplicate!');
  console.log('  Unique IDs: ' + (features.length - dups));

  // Output
  fs.mkdirSync(outputDir, { recursive: true });

  const geojsonPath = path.join(outputDir, 'final_grid.geojson');

  // Snapshot seed point FIRST (stable across runs unless input changes)
  // seed = (cityBBox min + max) / 2 → independent of which cells are classified
  const seed = {
    seed_lat: cityBBox[1] < cityBBox[3]
      ? (cityBBox[1] + cityBBox[3]) / 2
      : (cityBBox[3] + cityBBox[1]) / 2,
    seed_lng: (cityBBox[0] + cityBBox[2]) / 2,
    city_bbox: cityBBox,
    cell_size_km: cellSizeKm,
    tile_size_m: cellSizeKm * 1000 * 2,
    quadrant_m: cellSizeKm * 1000,
    created_at: new Date().toISOString(),
    config: {
      minWaterM2,
      heading: undefined, // filled by render pipeline
      pitch: undefined,
    },
  };
  const seedPath = path.join(outputDir, 'seed.json');

  // Manifest: timestamped (immutable history) + stable (latest) symlink/copy
  const ts = seed.created_at.replace(/[:.]/g, '-');
  const manifestPath = path.join(outputDir, `render_manifest_${ts}.json`);
  const stableManifestPath = path.join(outputDir, 'render_manifest.json');

  const featureCollection = turf.featureCollection(features);
  fs.writeFileSync(geojsonPath, JSON.stringify(featureCollection, null, 2));

  const manifest = features.map(f => ({
    cell_id: f.properties.cell_id,
    col: f.properties.col,
    row: f.properties.row,
    centroid_lng: f.properties.centroid_lng,
    centroid_lat: f.properties.centroid_lat,
    status: f.properties.status,
  }));

  // Write timestamped manifest (full history) + stable manifest (latest for render_tiles to read)
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  fs.writeFileSync(stableManifestPath, JSON.stringify(manifest));
  fs.writeFileSync(seedPath, JSON.stringify(seed, null, 2));

  console.log('\nDone.');
  console.log('  Grid: ' + geojsonPath);
  console.log('  Seed: ' + seedPath);
  console.log('  Manifest (latest): ' + stableManifestPath + ' (' + features.length + ' entries)');
  console.log('  Manifest (archive): ' + manifestPath);

  return { features, manifest, geojsonPath, manifestPath: stableManifestPath, seedPath, seed };
}
