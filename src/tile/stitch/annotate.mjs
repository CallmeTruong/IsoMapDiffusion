import sharp from 'sharp';
import { STITCH } from '../../config.mjs';

const DEFAULT_SEAM = STITCH.seamColor;
const DEFAULT_SEAM_THICKNESS = STITCH.seamThicknessPx;

/**
 * Convert RGBA color object → CSS rgba() string.
 */
export function rgbaStr(c) {
  return `rgba(${c.r},${c.g},${c.b},${c.alpha})`;
}

/**
 * Build SVG for 1 set seam lines base on layout.
 */
export function buildSeamSvg(layout, opts = {}) {
  const color = opts.color ?? DEFAULT_SEAM;
  const thickness = opts.thickness ?? DEFAULT_SEAM_THICKNESS;
  const { mapW, mapH, tileSize, gridSize, gridRows } = layout;
  const rows = gridRows ?? gridSize;

  const lines = [];
  for (const pos of layout.positions) {
    if (pos.c < gridSize - 1) {
      const x = pos.x + tileSize;
      lines.push(
        `<line x1="${x}" y1="${pos.y}" x2="${x}" y2="${pos.y + tileSize}" stroke="${rgbaStr(color)}" stroke-width="${thickness}" />`
      );
    }
    if (pos.r < rows - 1) {
      const y = pos.y + tileSize;
      lines.push(
        `<line x1="${pos.x}" y1="${y}" x2="${pos.x + tileSize}" y2="${y}" stroke="${rgbaStr(color)}" stroke-width="${thickness}" />`
      );
    }
  }

  return `<svg width="${mapW}" height="${mapH}" xmlns="http://www.w3.org/2000/svg">${lines.join('')}</svg>`;
}

export async function annotateSeams(base, layout, opts = {}) {
  const svg = buildSeamSvg(layout, opts);
  const overlayBuf = await sharp(Buffer.from(svg)).png().toBuffer();
  return base.composite([{ input: overlayBuf, top: 0, left: 0 }]);
}
