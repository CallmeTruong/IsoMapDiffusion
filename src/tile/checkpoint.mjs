import fs from 'fs';
import path from 'path';
import { CHECKPOINT } from '../config.mjs';

const SCHEMA_VERSION  = CHECKPOINT.schemaVersion;
const FLUSH_INTERVAL  = CHECKPOINT.flushIntervalMs;


export function loadCheckpoint(checkpointPath) {
  if (!checkpointPath || !fs.existsSync(checkpointPath)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(checkpointPath, 'utf8'));
    if (raw.version !== SCHEMA_VERSION) {
      console.warn(`[checkpoint] schema version mismatch: file=${raw.version}, code=${SCHEMA_VERSION} — skip`);
      return null;
    }
    return raw;
  } catch (e) {
    console.warn(`[checkpoint] failed to read: ${e.message}`);
    return null;
  }
}


export function saveCheckpoint(checkpointPath, doneTiles) {
  if (!checkpointPath) return;
  try {
    fs.mkdirSync(path.dirname(checkpointPath), { recursive: true });
    const data = {
      version: SCHEMA_VERSION,
      doneTiles,
      updatedAt: new Date().toISOString(),
    };
    // Atomic write
    const tmp = checkpointPath + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(data));
    fs.renameSync(tmp, checkpointPath);
  } catch (e) {
    console.warn(`[checkpoint] failed to save: ${e.message}`);
  }
}

export class CheckpointTracker {
  /**
   * @param {string} checkpointPath
   * @param {number} [flushIntervalMs] - Flush interval (ms, default từ CHECKPOINT.flushIntervalMs)
   */
  constructor(checkpointPath, flushIntervalMs = FLUSH_INTERVAL) {
    this.path = checkpointPath;
    this.flushIntervalMs = flushIntervalMs;
    this.doneTiles = new Map(); // key = "qx,qy" → {qx, qy}
    this.dirty = false;
    this.timer = null;

    // Load existing
    const existing = loadCheckpoint(checkpointPath);
    if (existing) {
      for (const t of existing.doneTiles) {
        this.doneTiles.set(`${t.qx},${t.qy}`, t);
      }
    }
  }

  /** Mark a tile as done. */
  markDone(qx, qy) {
    const key = `${qx},${qy}`;
    if (!this.doneTiles.has(key)) {
      this.doneTiles.set(key, { qx, qy });
      this.dirty = true;
      this._scheduleFlush();
    }
  }

  _scheduleFlush() {
    if (this.timer) return;
    this.timer = setTimeout(() => {
      this.flush();
      this.timer = null;
    }, this.flushIntervalMs);
  }

  /** Force flush to disk. */
  flush() {
    if (!this.dirty) return;
    saveCheckpoint(this.path, Array.from(this.doneTiles.values()));
    this.dirty = false;
  }

  /** Cleanup. */
  close() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.flush();
  }

  /** Number of done tiles. */
  get size() {
    return this.doneTiles.size;
  }
}
