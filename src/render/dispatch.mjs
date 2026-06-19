import { Worker } from 'worker_threads';

export function chunkArray(arr, n) {
  if (n <= 0) return [arr];
  if (n >= arr.length) return arr.map(item => [item]);
  const chunks = Array.from({ length: n }, () => []);
  arr.forEach((item, i) => chunks[i % n].push(item));
  return chunks;
}

export function dispatchWorkers({ chunks, workerDataTemplate, workerUrl, onMessage }) {
  return chunks.map((chunk, i) => new Promise((resolve, reject) => {
    const worker = new Worker(workerUrl, {
      workerData: { ...workerDataTemplate, tiles: chunk, workerId: i },
    });

    worker.on('message', m => {
      if (onMessage) onMessage(i, m);
      if (m.type === 'done') resolve(m.stats);
      if (m.type === 'error') reject(new Error(m.error));
    });
    worker.on('error', reject);
    worker.on('exit', code => {
      if (code !== 0) reject(new Error(`Worker ${i} exit code ${code}`));
    });
  }));
}
