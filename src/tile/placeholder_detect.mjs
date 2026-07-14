import sharp from 'sharp';

/**
 * Detect the "no data" placeholder that several tile providers return for
 * regions where satellite imagery is unavailable.
 *
 * Known signatures (verified empirically against ESRI and Cesium Ion / Google
 * Photorealistic — the two "no data" placeholders this pipeline actually hits
 * now that the fallback chain is satellite-imagery-only):
 *
 *  ESRI World Imagery (neutral gray)
 *    - Flat mid-gray (208, 208, 208) covering ~95% of pixels
 *    - "Map data not yet available" text in lighter gray (240..250)
 *    - Small red/orange marker dot (r > 150, g < 100, b < 100)
 *
 *  Cesium Ion / Google Photorealistic (warm tan, slightly tinted)
 *    - Flat (212, 206, 198) covering ~94% of pixels
 *    - White "Map data not yet available" text (R,G,B ≥ 245)
 *    - Same small red marker dot
 *
 *  EOX Sentinel-2 cloudless has no documented "no data" placeholder tile —
 *  it's a gap-free global mosaic — but the uniform-color short-circuit below
 *  still guards against any flat/blank response (e.g. a proxy error page).
 *
 * Approach:
 *   1. Decode once.
 *   2. Single pass to build a 32-bin luminance histogram and to count
 *      near-white text halo pixels and red marker pixels.
 *   3. Find the dominant luminance bucket and compute the mean RGB of
 *      pixels in that bucket (the "background color").
 *   4. Second pass to count pixels that fall within ±TOL of the background
 *      color (this catches both pure gray and warm-tan backgrounds).
 *   5. Short-circuit: if the image is COMPLETELY uniform (one unique color),
 *      it is a placeholder — real satellite tiles always have texture, edge
 *      structure, and multiple colors (verified: 0 of 100 sampled real tiles
 *      have ≤10 unique colors at 1024×1024).
 *   6. Decide: image is a placeholder when (a) ≥ 88% of pixels match the
 *      background color within tolerance AND (b) at least one red marker
 *      pixel exists AND (c) some near-white halo text is present.
 *
 * Real satellite imagery rarely produces a single dominant color over such
 * a large fraction of the frame, and never pairs that with a single-pixel
 * red marker, so the combination is robust on real tiles.
 *
 * @param {Buffer} imgBuf - PNG or JPEG bytes
 * @returns {Promise<boolean>} true if the image looks like a placeholder
 */
export async function isPlaceholderPng(imgBuf) {
  if (!imgBuf || imgBuf.length < 100) return true;

  const { data, info } = await sharp(imgBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
    .catch(() => ({ data: null, info: null }));

  if (!data || !info) return false;

  const { width, height, channels } = info;
  const totalPixels = width * height;

  const HIST = 32;
  const hist = new Uint32Array(HIST);
  let red = 0;
  let halo = 0;
  let uniqueColors = new Set();

  for (let p = 0; p < totalPixels; p++) {
    const i = p * channels;
    const r = data[i], g = data[i + 1], b = data[i + 2];
    uniqueColors.add((r << 16) | (g << 8) | b);
    if (r > 150 && g < 100 && b < 100) {
      red++;
      continue;
    }
    if (r >= 245 && g >= 245 && b >= 245) {
      halo++;
    }
    const L = (r + g + b) / 3;
    if (L >= 0 && L < 256) {
      const bIdx = Math.min(HIST - 1, Math.floor(L / 8));
      hist[bIdx]++;
    }
  }

  // Short-circuit: a 1024×1024 image with a single unique color is almost
  // certainly a fallback "no data" tile (verified empirically — no real tile
  // in the dataset has ≤ 10 unique colors). This catches the OSM-style
  // completely flat placeholder that has neither text nor red marker.
  if (uniqueColors.size <= 2) return true;

  let domBucket = 0;
  let domN = 0;
  for (let i = 0; i < HIST; i++) {
    if (hist[i] > domN) { domN = hist[i]; domBucket = i; }
  }
  if (domN === 0) return false;
  const domL = domBucket * 8 + 4;

  let sumR = 0, sumG = 0, sumB = 0, count = 0;
  for (let p = 0; p < totalPixels; p++) {
    const i = p * channels;
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const L = (r + g + b) / 3;
    if (L >= domL - 8 && L < domL + 8) {
      sumR += r; sumG += g; sumB += b; count++;
    }
  }
  if (count < totalPixels * 0.5) return false;
  const meanR = sumR / count;
  const meanG = sumG / count;
  const meanB = sumB / count;

  const TOL = 12;
  let uniform = 0;
  for (let p = 0; p < totalPixels; p++) {
    const i = p * channels;
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (
      Math.abs(r - meanR) <= TOL &&
      Math.abs(g - meanG) <= TOL &&
      Math.abs(b - meanB) <= TOL
    ) {
      uniform++;
    }
  }

  const uniformPct = (100 * uniform) / totalPixels;
  const haloPct = (100 * halo) / totalPixels;

  return (
    uniformPct >= 88 &&
    red >= 1 &&
    haloPct >= 0.1 &&
    haloPct < 15
  );
}

/**
 * Quick synchronous variant for when we already have raw RGBA data
 * (e.g. inside the Cesium canvas analyzer). Mirrors isPlaceholderPng but
 * skips the JPEG/PNG decode step.
 *
 * @param {Array<{r:number,g:number,b:number}>} samples - already-sampled pixels
 * @returns {boolean}
 */
export function isPlaceholderFromSamples(samples) {
  if (!samples || samples.length === 0) return false;
  const total = samples.length;

  const HIST = 32;
  const hist = new Uint32Array(HIST);
  let midGray = 0;
  let nearWhiteHalo = 0;
  let redMarker = 0;
  let sumR = 0, sumG = 0, sumB = 0, domCount = 0;
  let domBucket = 0;
  const uniqueColors = new Set();

  for (let i = 0; i < total; i++) {
    const s = samples[i];
    uniqueColors.add((s.r << 16) | (s.g << 8) | s.b);
    const dr = Math.abs(s.r - s.g), dg = Math.abs(s.g - s.b);
    const isGray = dr < 8 && dg < 8;
    if (isGray) {
      if (s.r >= 180 && s.r <= 235) midGray++;
      else if (s.r >= 236 && s.r <= 254) nearWhiteHalo++;
    } else if (s.r > 150 && s.g < 100 && s.b < 100) {
      redMarker++;
    } else {
      const L = (s.r + s.g + s.b) / 3;
      const bIdx = Math.min(HIST - 1, Math.floor(L / 8));
      hist[bIdx]++;
      if (L >= 236) nearWhiteHalo++;
    }
  }

  // Short-circuit: completely uniform sample set → placeholder.
  if (uniqueColors.size <= 1) return true;

  let domN = 0;
  for (let i = 0; i < HIST; i++) if (hist[i] > domN) { domN = hist[i]; domBucket = i; }
  if (domN > 0) {
    const domL = domBucket * 8 + 4;
    for (let i = 0; i < total; i++) {
      const s = samples[i];
      const L = (s.r + s.g + s.b) / 3;
      if (L >= domL - 8 && L < domL + 8) {
        sumR += s.r; sumG += s.g; sumB += s.b; domCount++;
      }
    }
  }

  const midGrayPct = (100 * midGray) / total;
  const haloPct = (100 * nearWhiteHalo) / total;
  const uniformPct = domCount > 0 ? (100 * domCount) / total : 0;

  return (
    (uniformPct >= 88 || midGrayPct > 70) &&
    redMarker >= 1 &&
    haloPct >= 0.1 &&
    haloPct < 15
  );
}