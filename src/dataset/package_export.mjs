import fs from 'fs';
import path from 'path';
import { DATASET_PATHS } from './constants.mjs';

const SIG_LOCAL   = 0x04034b50;
const SIG_CENTRAL = 0x02014b50;
const SIG_END     = 0x06054b50;

export async function createZip(srcDir, outPath) {
  if (!fs.existsSync(srcDir)) {
    throw new Error(`Source directory note exist: ${srcDir}`);
  }

  const files = fs.readdirSync(srcDir);
  const fileEntries = [];

  // Build local file headers + data
  const buffers = [];
  let offset = 0;

  for (const name of files) {
    const fullPath = path.join(srcDir, name);
    const stat = fs.statSync(fullPath);
    if (!stat.isFile()) continue;

    const data = fs.readFileSync(fullPath);
    const crc = crc32(data);
    const nameBuf = Buffer.from(name, 'utf8');

    // Local file header
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(SIG_LOCAL, 0);
    localHeader.writeUInt16LE(20, 4);                // version needed
    localHeader.writeUInt16LE(0, 6);                 // flags
    localHeader.writeUInt16LE(0, 8);                 // method = STORE
    localHeader.writeUInt16LE(0, 10);                // mod time
    localHeader.writeUInt16LE(0x21, 12);             // mod date (1980-01-01)
    localHeader.writeUInt32LE(crc, 14);
    localHeader.writeUInt32LE(data.length, 18);      // compressed size
    localHeader.writeUInt32LE(data.length, 22);      // uncompressed size
    localHeader.writeUInt16LE(nameBuf.length, 26);
    localHeader.writeUInt16LE(0, 28);                // extra field length

    buffers.push(localHeader, nameBuf, data);

    fileEntries.push({
      name: nameBuf,
      crc,
      size: data.length,
      offset,
    });

    offset += localHeader.length + nameBuf.length + data.length;
  }

  // Central directory
  const centralBuffers = [];
  let centralSize = 0;

  for (const e of fileEntries) {
    const central = Buffer.alloc(46);
    central.writeUInt32LE(SIG_CENTRAL, 0);
    central.writeUInt16LE(20, 4);                   // version made by
    central.writeUInt16LE(20, 6);                   // version needed
    central.writeUInt16LE(0, 8);                    // flags
    central.writeUInt16LE(0, 10);                   // method
    central.writeUInt16LE(0, 12);                   // mod time
    central.writeUInt16LE(0x21, 14);                // mod date
    central.writeUInt32LE(e.crc, 16);
    central.writeUInt32LE(e.size, 20);              // compressed
    central.writeUInt32LE(e.size, 24);              // uncompressed
    central.writeUInt16LE(e.name.length, 28);
    central.writeUInt16LE(0, 30);                   // extra
    central.writeUInt16LE(0, 32);                   // comment
    central.writeUInt16LE(0, 34);                   // disk
    central.writeUInt16LE(0, 36);                   // internal attrs
    central.writeUInt32LE(0, 38);                   // external attrs
    central.writeUInt32LE(e.offset, 42);
    centralBuffers.push(central, e.name);
    centralSize += central.length + e.name.length;
  }

  // End of central directory
  const endRecord = Buffer.alloc(22);
  endRecord.writeUInt32LE(SIG_END, 0);
  endRecord.writeUInt16LE(0, 4);                    // disk
  endRecord.writeUInt16LE(0, 6);                    // disk start
  endRecord.writeUInt16LE(fileEntries.length, 8);
  endRecord.writeUInt16LE(fileEntries.length, 10);
  endRecord.writeUInt32LE(centralSize, 12);
  endRecord.writeUInt32LE(offset, 16);              // central dir offset
  endRecord.writeUInt16LE(0, 20);                   // comment length

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const out = fs.createWriteStream(outPath);
  for (const buf of buffers) out.write(buf);
  for (const buf of centralBuffers) out.write(buf);
  out.write(endRecord);
  await new Promise(resolve => out.end(resolve));

  return outPath;
}

/**
 * CRC32 implementation (zip-compatible).
 */
function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}