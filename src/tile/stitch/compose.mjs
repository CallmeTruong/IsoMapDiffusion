import sharp from 'sharp';
import { STITCH } from '../../config.mjs';
import { computeLayout } from './layout.mjs';

/** Default background color (RGBA) */
const DEFAULT_BG = STITCH.background;

/** Default gap color for stitchPair (RGBA) */
const DEFAULT_PAIR_GAP = STITCH.pairGapColor;

/**
 *sharp instance
 */
export async function createBaseCanvas(width, height, bg = DEFAULT_BG) {
  const channels = 4;
  const raw = Buffer.alloc(width * height * channels);
  for (let i = 0; i < raw.length; i += channels) {
    raw[i + 0] = bg.r;
    raw[i + 1] = bg.g;
    raw[i + 2] = bg.b;
    raw[i + 3] = Math.round(bg.alpha * 255);
  }
  return sharp(raw, { raw: { width, height, channels } });
}



export async function stitchTiles({ tiles, gridSize, tileSize, stride, background, gridCols, gridRows, flipX = false }) {
  if (!Array.isArray(tiles)) throw new Error('stitchTiles: tiles must be an array');
  const cols = gridCols ?? gridSize;
  const rows = gridRows ?? gridSize;
  const expected = cols * rows;
  if (tiles.length !== expected) {
    throw new Error(`stitchTiles: tiles.length=${tiles.length} ≠ cols*rows=${expected}`);
  }

  for (const t of tiles) {
    if (typeof t.r !== 'number' || typeof t.c !== 'number') {
      throw new Error(`stitchTiles: tile missing (r, c)`);
    }
    if (t.r < 0 || t.r >= rows || t.c < 0 || t.c >= cols) {
      throw new Error(`stitchTiles: tile (r=${t.r}, c=${t.c}) out of range`);
    }
  }

  const layout = computeLayout({ gridSize, tileSize, stride, gridCols: cols, gridRows: rows });
  const { mapW, mapH } = layout;
  const channels = 4;

  // Canvas
  const baseSharp = await createBaseCanvas(mapW, mapH, background);
  let canvasBuf = await baseSharp.raw().toBuffer();

  // tiles row-major: (0,0), (0,1), ... (0,cols-1), (1,0), ...
  const sortedTiles = [...tiles].sort((a, b) => (a.r - b.r) || (a.c - b.c));

  for (const t of sortedTiles) {
    const pos = layout.positions.find(p => p.r === t.r && p.c === t.c);
    if (!pos) throw new Error(`stitchTiles: no position for (r=${t.r}, c=${t.c})`);

    // Load tile image + optional flip
    let tileImg = sharp(t.png);
    if (flipX) tileImg = tileImg.flop();
    const tileMeta = await tileImg.metadata();
    if (tileMeta.width !== tileSize || tileMeta.height !== tileSize) {
      tileImg = tileImg.resize({ width: tileSize, height: tileSize });
    }
    const { data: tileRaw } = await tileImg.ensureAlpha().raw()
      .toBuffer({ resolveWithObject: true });

    // Blend tile on canvas pixel-by-pixel
    const overlapX = Math.max(0, tileSize - (mapW - pos.x));
    const overlapY = Math.max(0, tileSize - (mapH - pos.y));

    for (let ty = 0; ty < tileSize; ty++) {
      const by = pos.y + ty;
      if (by < 0 || by >= mapH) continue;

      for (let tx = 0; tx < tileSize; tx++) {
        const bx = pos.x + tx;
        if (bx < 0 || bx >= mapW) continue;

        const tIdx = (ty * tileSize + tx) * channels;
        const cIdx = (by * mapW + bx) * channels;

        //weight for overlap zone
        let w = 1.0;
        if (overlapX > 0) {
          const t = tx / overlapX;
          w = Math.min(w, Math.max(0, t));
        }
        if (overlapY > 0) {
          const t = ty / overlapY;
          w = Math.min(w, Math.max(0, t));
        }

        if (w >= 1) {
          // Direct overwrite
          canvasBuf[cIdx]     = tileRaw[tIdx];
          canvasBuf[cIdx + 1] = tileRaw[tIdx + 1];
          canvasBuf[cIdx + 2] = tileRaw[tIdx + 2];
          canvasBuf[cIdx + 3] = tileRaw[tIdx + 3];
        } else if (w > 0) {
          // Weighted blend: base*(1-w) + tile*w
          const invW = 1 - w;
          canvasBuf[cIdx]     = Math.round(canvasBuf[cIdx]     * invW + tileRaw[tIdx]     * w);
          canvasBuf[cIdx + 1] = Math.round(canvasBuf[cIdx + 1] * invW + tileRaw[tIdx + 1] * w);
          canvasBuf[cIdx + 2] = Math.round(canvasBuf[cIdx + 2] * invW + tileRaw[tIdx + 2] * w);
          canvasBuf[cIdx + 3] = Math.round(canvasBuf[cIdx + 3] * invW + tileRaw[tIdx + 3] * w);
        }
      }
    }
  }

  return sharp(canvasBuf, { raw: { width: mapW, height: mapH, channels } });
}


export async function stitchPair({ pngA, pngB, offset, tileSize = 1024, gapColor, seamColor }) {
  if (typeof offset !== 'number' || offset < 0) {
    throw new Error(`stitchPair: offset must be ≥ 0, got ${offset}`);
  }

  const bg = gapColor ?? DEFAULT_PAIR_GAP;
  const stitchW = tileSize + offset;

  const base = await createBaseCanvas(stitchW, tileSize, bg);
  const composite = [
    { input: pngA, top: 0, left: 0 },
    { input: pngB, top: 0, left: offset },
  ];

  let result = base.composite(composite);

  if (seamColor && offset > 0 && offset < tileSize) {
    const svg = `<svg width="${stitchW}" height="${tileSize}" xmlns="http://www.w3.org/2000/svg">
      <line x1="${offset}" y1="0" x2="${offset}" y2="${tileSize}" stroke="${rgbaStr(seamColor)}" stroke-width="1" />
    </svg>`;
    const overlayBuf = await sharp(Buffer.from(svg)).png().toBuffer();
    result = result.composite([{ input: overlayBuf, top: 0, left: 0 }]);
  }

  return result;
}

function rgbaStr(c) {
  return `rgba(${c.r},${c.g},${c.b},${c.alpha})`;
}
