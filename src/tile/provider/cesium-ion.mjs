import { TileProvider } from './base.mjs';
import { PROVIDERS } from '../../config.mjs';

export class CesiumIonProvider extends TileProvider {
  constructor() {
    super('cesium-ion', 'CESIUM_ION_TOKEN', 'Cesium Ion (Google Photorealistic)');
  }

  async validateKey(token) {
    if (!token) return false;
    const { googleAssetId, requestTimeoutMs } = PROVIDERS.cesiumIon;
    try {
      const url = `https://api.cesium.com/v1/assets/${googleAssetId}/endpoint?access_token=${encodeURIComponent(token)}`;
      const res = await fetch(url, {
        method: 'GET',
        signal: AbortSignal.timeout(requestTimeoutMs),
      });
      
      if (res.status === 200) return true;
      if (res.status === 401 || res.status === 403) return false;
      
      if (res.status === 404 || res.status >= 500) {
        const me = await fetch(`https://api.cesium.com/v1/me?access_token=${encodeURIComponent(token)}`, {
          method: 'GET',
          signal: AbortSignal.timeout(requestTimeoutMs),
        });
        return me.ok;
      }
      return res.ok;
    } catch {
      return false;
    }
  }

  getCesiumToken(token) {
    return {
      accessToken: token,
      assetId: PROVIDERS.cesiumIon.googleAssetId,
    };
  }

  buildTilesetJs({ accessToken, assetId }) {
    return `
    Cesium.Ion.defaultAccessToken = '${accessToken}';
    try {
      tileset = await Cesium.createGooglePhotorealistic3DTileset({
        accessToken: '${accessToken}',
        showCreditsOnScreen: false,
      });
    } catch (e) {
      console.warn('[provider] createGooglePhotorealistic3DTileset failed, fallback to fromIonAssetId:', e.message);
      tileset = await Cesium.Cesium3DTileset.fromIonAssetId(${assetId}, {
        accessToken: '${accessToken}',
      });
      tileset.showCreditsOnScreen = false;
    }`;
  }
}
