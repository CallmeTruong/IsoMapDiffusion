/**
 * dispatch.mjs — Array chunking utility for worker pool distribution.
 */

export function chunkArray(arr, n) {
  if (n <= 0) return [arr];
  if (n >= arr.length) return arr.map(item => [item]);
  const chunks = Array.from({ length: n }, () => []);
  arr.forEach((item, i) => chunks[i % n].push(item));
  return chunks;
}
