import path from 'path';
import { fileURLToPath } from 'url';
import { generateGrid } from './grid/index.mjs';
import { GRID } from './config.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');

// Parse CLI arguments
const ARGS = Object.fromEntries(
  process.argv.slice(2)
    .filter(a => a.startsWith('--'))
    .map(a => { const [k, v] = a.slice(2).split('='); return [k, v ?? true]; })
);

const options = {
  districtsDir: ARGS.districts
    ? path.resolve(PROJECT_ROOT, ARGS.districts)
    : path.resolve(PROJECT_ROOT, PATHS.districts),

  waterFile: ARGS.water
    ? path.resolve(PROJECT_ROOT, ARGS.water)
    : path.resolve(PROJECT_ROOT, PATHS.water),

  infraFile: ARGS.infra
    ? path.resolve(PROJECT_ROOT, ARGS.infra)
    : path.resolve(PROJECT_ROOT, PATHS.infra),
  cellSizeKm: Number(ARGS.cell ?? GRID.cellSizeKm),
  minWaterM2: Number(ARGS.minwater ?? GRID.minWaterM2),
  outputDir: ARGS.out
    ? path.resolve(PROJECT_ROOT, ARGS.out)
    : path.join(PROJECT_ROOT, 'output'),
};

console.log('Config:');
console.log('  districts:', options.districtsDir);
console.log('  water    :', options.waterFile);
console.log('  infra    :', options.infraFile);
console.log('  output   :', options.outputDir);
console.log('  cell     :', options.cellSizeKm, 'km');

// Run
generateGrid(options);
