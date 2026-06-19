const CHANNELS = 4;  // RGBA

/**
 * MSE (Mean Squared Error)
 */
export function mseRaw(bufA, bufB, w, h) {
  if (bufA.length !== bufB.length) {
    throw new Error(`mseRaw: buffer size mismatch (${bufA.length} vs ${bufB.length})`);
  }
  const n = w * h;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const o = i * CHANNELS;
    const dr = bufA[o + 0] - bufB[o + 0];
    const dg = bufA[o + 1] - bufB[o + 1];
    const db = bufA[o + 2] - bufB[o + 2];
    sum += dr * dr + dg * dg + db * db;
  }
  return sum / (n * 3);
}

export function mirrorHorizontal(src, w, h) {
  const out = Buffer.allocUnsafe(src.length);
  const rowBytes = w * CHANNELS;
  for (let y = 0; y < h; y++) {
    const rowStart = y * rowBytes;
    for (let x = 0; x < w; x++) {
      const srcOff = rowStart + x * CHANNELS;
      const dstOff = rowStart + (w - 1 - x) * CHANNELS;
      out[dstOff + 0] = src[srcOff + 0];
      out[dstOff + 1] = src[srcOff + 1];
      out[dstOff + 2] = src[srcOff + 2];
      out[dstOff + 3] = src[srcOff + 3];
    }
  }
  return out;
}
