/**
 * db.mjs — SQLite database helpers
 */

let db = null;
let sqlite3 = null;

/**
 * Initialize SQLite database connection
 * @param {string} dbPath - Path to SQLite database file
 * @returns {Promise<Object|null>} - Database instance or null if unavailable
 */
export async function initDB(dbPath) {
  if (db) return db;

  try {
    const BetterSQLite3 = (await import('better-sqlite3')).default;
    db = new BetterSQLite3(dbPath);
    sqlite3 = 'better';
  } catch {
    try {
      const SQLite = (await import('sqlite3')).default;
      sqlite3 = 'sqlite3';
      db = await new Promise((res, rej) => {
        const d = new SQLite.Database(dbPath, err => err ? rej(err) : res(d));
      });
    } catch {
      console.warn('SQLite not available, using JSON fallback');
      return null;
    }
  }

  if (sqlite3 === 'better') {
    db.exec(`
      CREATE TABLE IF NOT EXISTS quadrants (
        qx INTEGER NOT NULL,
        qy INTEGER NOT NULL,
        status TEXT DEFAULT 'pending',
        render BLOB,
        generation BLOB,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_attempt_at TEXT,
        worker_id INTEGER,
        attempt_count INTEGER DEFAULT 0,
        last_variance REAL,
        last_error TEXT,
        render_ms INTEGER,
        PRIMARY KEY (qx, qy)
      );
      CREATE INDEX IF NOT EXISTS idx_status ON quadrants(status);
      CREATE INDEX IF NOT EXISTS idx_attempt ON quadrants(attempt_count);
    `);
  } else {
    await runDB(`CREATE TABLE IF NOT EXISTS quadrants (
      qx INTEGER NOT NULL,
      qy INTEGER NOT NULL,
      status TEXT DEFAULT 'pending',
      render BLOB,
      generation BLOB,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      last_attempt_at TEXT,
      worker_id INTEGER,
      attempt_count INTEGER DEFAULT 0,
      last_variance REAL,
      last_error TEXT,
      render_ms INTEGER,
      PRIMARY KEY (qx, qy)
    )`);
    await runDB(`CREATE INDEX IF NOT EXISTS idx_status ON quadrants(status)`);
    await runDB(`CREATE INDEX IF NOT EXISTS idx_attempt ON quadrants(attempt_count)`);
  }

  return db;
}

/**
 * Execute a SQL statement
 * @param {string} sql - SQL statement
 * @param {Array} params - Query parameters
 * @returns {Promise} - Query result
 */
export async function runDB(sql, params = []) {
  if (!db) return null;
  if (sqlite3 === 'better') {
    return db.prepare(sql).run(...params);
  } else {
    return new Promise((res, rej) => {
      db.run(sql, params, function(err) { err ? rej(err) : res(this); });
    });
  }
}

/**
 * Insert a quadrant record (ignore if exists)
 * @param {number} qx - Quadrant X coordinate
 * @param {number} qy - Quadrant Y coordinate
 * @param {string} status - Quadrant status
 */
export async function insertOrIgnoreQuadrant(qx, qy, status = 'pending') {
  if (!db) return;
  if (sqlite3 === 'better') {
    db.prepare(`INSERT OR IGNORE INTO quadrants (qx, qy, status) VALUES (?, ?, ?)`)
      .run(qx, qy, status);
  } else {
    await runDB(`INSERT OR IGNORE INTO quadrants (qx, qy, status) VALUES (?, ?, ?)`, [qx, qy, status]);
  }
}

/**
 * Update quadrant render attempt metadata.
 * Called sau khi render một quadrant (thành công hoặc thất bại).
 *
 * @param {number} qx
 * @param {number} qy
 * @param {Object} info
 * @param {number} [info.workerId]
 * @param {number} [info.variance]
 * @param {string} [info.error]
 * @param {number} [info.renderMs]
 * @param {boolean} [info.success]
 */
export async function updateQuadrantAttempt(qx, qy, info = {}) {
  if (!db) return;
  const now = new Date().toISOString();
  if (sqlite3 === 'better') {
    db.prepare(`
      UPDATE quadrants SET
        last_attempt_at = ?,
        worker_id = COALESCE(?, worker_id),
        attempt_count = attempt_count + 1,
        last_variance = COALESCE(?, last_variance),
        last_error = COALESCE(?, last_error),
        render_ms = COALESCE(?, render_ms)
      WHERE qx = ? AND qy = ?
    `).run(
      now,
      info.workerId ?? null,
      info.variance ?? null,
      info.error ?? null,
      info.renderMs ?? null,
      qx, qy
    );
  } else {
    await runDB(`
      UPDATE quadrants SET
        last_attempt_at = ?,
        worker_id = COALESCE(?, worker_id),
        attempt_count = attempt_count + 1,
        last_variance = COALESCE(?, last_variance),
        last_error = COALESCE(?, last_error),
        render_ms = COALESCE(?, render_ms)
      WHERE qx = ? AND qy = ?
    `, [
      now,
      info.workerId ?? null,
      info.variance ?? null,
      info.error ?? null,
      info.renderMs ?? null,
      qx, qy,
    ]);
  }
}

/**
 * Get database instance
 * @returns {Object|null}
 */
export function getDB() {
  return db;
}
