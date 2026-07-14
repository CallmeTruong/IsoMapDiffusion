import { isMainThread, parentPort, workerData } from 'worker_threads';
import { runWorker } from './worker.mjs';

if (!isMainThread) {
  runWorker(workerData).catch(e => {
    parentPort?.postMessage({
      type: 'error',
      workerId: workerData.workerId,
      error: e.message,
      stack: e.stack,
      code: e.code,
      syscall: e.syscall,
      path: e.path,
    });
    process.exit(1);
  });
}
