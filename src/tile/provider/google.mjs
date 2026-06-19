import { TileProvider } from './base.mjs';
import { GOOGLE } from '../../config.mjs';

export class GoogleProvider extends TileProvider {
  constructor() {
    super('google', 'GOOGLE_KEY', 'Google Photorealistic 3D Tiles');
  }

  async validateKey(key) {
    try {
      const url = `${GOOGLE.rootJsonUrl}?key=${encodeURIComponent(key)}`;
      const res = await fetch(url, {
        method: 'HEAD',
        signal: AbortSignal.timeout(GOOGLE.requestTimeoutMs),
      });
      return res.ok || res.status === 404;
    } catch {
      return false;
    }
  }

  getCesiumToken(key) {
    return { apiKey: key };
  }

  buildTilesetJs({ apiKey }) {
    return `
    tileset = await Cesium.createGooglePhotorealistic3DTileset({
      key: '${apiKey}',
      showCreditsOnScreen: false,
    });`;
  }
}
