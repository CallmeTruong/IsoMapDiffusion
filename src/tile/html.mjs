import { CESIUM, TILE, RENDER } from '../config.mjs';

export const CESIUM_VERSION = CESIUM.version;


 /** Escape a string for safe interpolation into a JS string literal */

function escapeJsString(s) {
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\$/g, '\\$');
}

/**
 * Generate Cesium HTML template for tile rendering
 *
 * @param {string|Object} apiKeyOrToken - Backward compat: Google API key string.
 *                                         Or object { tilesetJs: string, ionReset?: boolean }.
 * @param {Object} cfg - Render configuration
 * @returns {string} - Complete HTML string
 */
export function makeCesiumHTML(apiKeyOrToken, cfg) {
    let tilesetJs;
    let resetIonToken = false;
    if (typeof apiKeyOrToken == 'string') {
        const safeKey = escapeJsString(apiKeyOrToken);
        tilesetJs = `
        tileset = await Cesium.createGooglePhotorealistic3DTileset({
            key: '${safeKey}',
            showCreditsOnScreen: false,
        });`;
    }   else {
        tilesetJs = apiKeyOrToken.tilesetJs;
        resetIonToken = apiKeyOrToken.ionReset === true;
    }
    return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cesium.com/downloads/cesiumjs/releases/${CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/${CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  #c { width:${cfg.sizePx}px; height:${cfg.sizePx}px; }
  .cesium-viewer-toolbar, .cesium-viewer-animationContainer,
  .cesium-viewer-timelineContainer, .cesium-viewer-bottom,
  .cesium-credit-logoContainer, .cesium-credit-textContainer,
  .cesium-viewer-fullscreenContainer { display:none !important; }
</style>
</head><body><div id="c"></div>
<script>
${resetIonToken ? 'Cesium.Ion.defaultAccessToken = undefined;' : ''}

const viewer = new Cesium.Viewer('c', {
  contextOptions: { webgl: { preserveDrawingBuffer: true } },
  imageryProvider: false, baseLayerPicker: false, geocoder: false,
  homeButton: false, sceneModePicker: false, navigationHelpButton: false,
  animation: false, timeline: false, fullscreenButton: false,
  infoBox: false, selectionIndicator: false,
  requestRenderMode: false,
});

viewer.targetFrameRate = ${RENDER.targetFrameRate};
viewer.scene.globe.show         = false;
viewer.scene.skyAtmosphere.show = false;
viewer.scene.skyBox.show        = false;
viewer.scene.sun.show           = false;
viewer.scene.moon.show          = false;
viewer.scene.backgroundColor    = Cesium.Color.WHITE;
viewer.scene.light = new Cesium.DirectionalLight({
  direction: new Cesium.Cartesian3(
    ${RENDER.lightDirection.x},
    ${RENDER.lightDirection.y},
    ${RENDER.lightDirection.z}
  ),
});
viewer.clock.shouldAnimate = false;

let tileset   = null;
let sessionOk = false;

async function initSession() {
  if (tileset) { viewer.scene.primitives.remove(tileset); tileset = null; }
  sessionOk = false;
  try {
    ${tilesetJs}
    if (tileset) {
      tileset.maximumScreenSpaceError = ${cfg.sse};
      viewer.scene.primitives.add(tileset);
      sessionOk = true;
    } else {
      console.error('[session] tileset is null after init');
    }
  } catch (e) {
    console.error('[session] Failed:', e.message);
  }
}

initSession();

window.isSessionOk   = () => sessionOk;
window.reinitSession = async () => { await initSession(); return sessionOk; };

let _cachedFrustum = null;

function setCamera(lng, lat) {
  // ─── ENU-based camera positioning ────────────────────────────────────────

  const target = Cesium.Cartesian3.fromDegrees(lng, lat, 0);

  // ECEF
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(target);

  const az  = Cesium.Math.toRadians(${cfg.azimuth});   // azimuth from North
  const el  = Cesium.Math.toRadians(${cfg.elevation}); // elevation

  const dirENU = new Cesium.Cartesian3(
    Math.sin(az) * Math.cos(el),   // East component
    Math.cos(az) * Math.cos(el),   // North component
    -Math.sin(el)                   // Up component
  );

  // Convert dirENU to ECEF
  const dirECEF = new Cesium.Cartesian3();
  Cesium.Matrix4.multiplyByPointAsVector(enu, dirENU, dirECEF);
  Cesium.Cartesian3.normalize(dirECEF, dirECEF);

  // camera position = target + dir * range
  const camPos = Cesium.Cartesian3.add(
    target,
    Cesium.Cartesian3.multiplyByScalar(dirECEF, ${cfg.altitude}, new Cesium.Cartesian3()),
    new Cesium.Cartesian3()
  );

  // Up vector
  const upECEF = new Cesium.Cartesian3(enu[4], enu[5], enu[6]);

  const lookDir = Cesium.Cartesian3.negate(dirECEF, new Cesium.Cartesian3());

  viewer.camera.setView({
    destination: camPos,
    orientation: { direction: lookDir, up: upECEF },
  });

  // ─── Orthographic frustum ────────────────────────────────────────────────
  if (!_cachedFrustum) {
    _cachedFrustum = new Cesium.OrthographicFrustum();
    _cachedFrustum.aspectRatio = 1.0;
  }
  _cachedFrustum.width = ${cfg.frustumW};
  viewer.camera.frustum = _cachedFrustum;
}

window.renderTile = async function(lng, lat, extraWait = 0) {
  setCamera(lng, lat);

  // ─── Phase 1: wait
  await new Promise(resolve => {
    let done = false;
    let stableLoadedFrames = 0;
    const REQUIRED_STABLE_FRAMES = ${RENDER.requiredStableFrames};

    const timer = setTimeout(() => {
      if (!done) { done = true; unsub(); resolve(); }
    }, ${cfg.tileWaitMs} + extraWait);

    const unsub = viewer.scene.postRender.addEventListener(() => {
      if (done) return;
      if (tileset && tileset.tilesLoaded) {
        stableLoadedFrames++;
        if (stableLoadedFrames >= REQUIRED_STABLE_FRAMES) {
          done = true;
          clearTimeout(timer);
          unsub();
          resolve();
        }
      } else {
        stableLoadedFrames = 0;
      }
    });
  });

  // ─── Phase 2: wait for tiles render + materials load ─────
  await new Promise(resolve => {
    let count = 0;
    const NEEDED = ${RENDER.postRenderExtraFrames};
    const unsub = viewer.scene.postRender.addEventListener(() => {
      if (++count >= NEEDED) { unsub(); resolve(); }
    });
  });

  // ─── Phase 3: adaptive settle — wait for pixel stability ─────────────────
  const maxPolls = Math.ceil(${cfg.settleMaxMs} / ${cfg.settlePollMs});
  let stableCount = 0;

  for (let i = 0; i < maxPolls; i++) {
    await new Promise(r => setTimeout(r, ${cfg.settlePollMs}));

    await new Promise(r => {
      viewer.scene.requestRender();
      const unsub = viewer.scene.postRender.addEventListener(() => {
        unsub();
        r();
      });
    });

    const { variance } = window.analyzeCanvas();
    if (variance > ${cfg.varianceThr}) {
      stableCount = 0;
    } else if (++stableCount >= ${cfg.stableHits}) {
      break;
    }
  }

  // ─── Phase 4: render final frame + capture ──────────────────────────────
  await new Promise(r => {
    viewer.scene.requestRender();
    const unsub = viewer.scene.postRender.addEventListener(() => {
      unsub();
      setTimeout(r, ${RENDER.postRenderBufferMs});
    });
  });

  return viewer.scene.canvas.toDataURL('image/png');
};

window.analyzeCanvas = function() {
  const canvas = viewer.scene.canvas;
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) return { isBlank: false, variance: 999, edgeDensity: 1 };

  const w = canvas.width, h = canvas.height;
  const pixel = new Uint8Array(4);
  const samples = [];
  // Sample grid ${RENDER.sampleGridSize}x${RENDER.sampleGridSize} = ${RENDER.sampleGridSize * RENDER.sampleGridSize} samples
  const GRID = ${RENDER.sampleGridSize};
  const stepX = Math.floor(w / GRID);
  const stepY = Math.floor(h / GRID);

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      gl.readPixels(
        col * stepX + Math.floor(stepX / 2),
        h - 1 - (row * stepY + Math.floor(stepY / 2)),
        1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel
      );
      samples.push({ r: pixel[0], g: pixel[1], b: pixel[2] });
    }
  }

  const meanR    = samples.reduce((s, p) => s + p.r, 0) / samples.length;
  const meanG    = samples.reduce((s, p) => s + p.g, 0) / samples.length;
  const meanB    = samples.reduce((s, p) => s + p.b, 0) / samples.length;
  const variance = samples.reduce((s, p) =>
    s + Math.pow(p.r - meanR, 2) + Math.pow(p.g - meanG, 2), 0) / samples.length;

  // ─── Edge density
  let edgeCount = 0;
  let totalPairs = 0;
  const EDGE_THR = ${RENDER.edgeGradThr};  // gradient threshold (0-255 per channel, sum 3 channels = 90)
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID - 1; col++) {
      const i = row * GRID + col;
      const j = row * GRID + col + 1;
      const grad = Math.abs(samples[i].r - samples[j].r)
                  + Math.abs(samples[i].g - samples[j].g)
                  + Math.abs(samples[i].b - samples[j].b);
      if (grad > EDGE_THR) edgeCount++;
      totalPairs++;
    }
  }
  for (let row = 0; row < GRID - 1; row++) {
    for (let col = 0; col < GRID; col++) {
      const i = row * GRID + col;
      const j = (row + 1) * GRID + col;
      const grad = Math.abs(samples[i].r - samples[j].r)
                  + Math.abs(samples[i].g - samples[j].g)
                  + Math.abs(samples[i].b - samples[j].b);
      if (grad > EDGE_THR) edgeCount++;
      totalPairs++;
    }
  }
  const edgeDensity = totalPairs > 0 ? edgeCount / totalPairs : 0;

  // ─── Blank/blur detection (separate from stable detection) ──────────────
  const blankVarThr        = ${cfg.blankVarianceThr};
  const blankEdgeThr       = ${cfg.blankEdgeThr ?? 0.15};
  const blankMeanRThr      = ${TILE.blankMeanRThr};
  const cornerWhiteThr     = 250;
  const topStripMeanRThr   = 240;
  const topStripVarThr     = 1500;
  const isLowVar           = variance < blankVarThr;
  const isWhiteFlat        = meanR > blankMeanRThr;
  const isLowEdge          = edgeDensity < blankEdgeThr;

  // V2 extras
  const corners = [
    samples[0], samples[GRID - 1],
    samples[(GRID - 1) * GRID], samples[GRID * GRID - 1],
  ];
  const allCornersWhite = corners.every(
    s => s.r > cornerWhiteThr && s.g > cornerWhiteThr && s.b > cornerWhiteThr
  );
  let topStripSum = 0;
  for (let c = 0; c < GRID; c++) topStripSum += samples[c].r;
  const topStripMeanR = topStripSum / GRID;
  const topStripLowVar = variance < topStripVarThr;

  const isBlank = isLowVar || isWhiteFlat || isLowEdge
               || allCornersWhite
               || (topStripMeanR > topStripMeanRThr && topStripLowVar);

  // ─── Google "Map data not yet available" placeholder detection ──────────
  // When Google's Photorealistic 3D Tiles have no coverage for a location,
  // it bakes an actual placeholder TEXTURE (English text + a small red
  // marker dot) directly into the 3D scene — not a JS error, not a DOM
  // overlay. Because it contains real text/edges, it has enough variance
  // and edge density to sail past every check above and get saved as if
  // it were valid terrain.
  //
  // Signature: this placeholder always uses one exact flat beige color as
  // its background — measured as rgb(212, 206, 198) — and the text/dot are
  // small and centered, so the four corners are *always* pure background.
  // Real photorealistic terrain essentially never lands all 4 corners in
  // this narrow, specific color range simultaneously.
  const PLACEHOLDER_BG = { r: 212, g: 206, b: 198 };
  const PLACEHOLDER_TOL = 12;
  const isPlaceholderColor = s =>
    Math.abs(s.r - PLACEHOLDER_BG.r) <= PLACEHOLDER_TOL &&
    Math.abs(s.g - PLACEHOLDER_BG.g) <= PLACEHOLDER_TOL &&
    Math.abs(s.b - PLACEHOLDER_BG.b) <= PLACEHOLDER_TOL;
  const isGooglePlaceholder = corners.every(isPlaceholderColor);

  return {
    isBlank: isBlank || isGooglePlaceholder,
    isGooglePlaceholder,
    variance: Math.round(variance),
    meanR: Math.round(meanR),
    meanG: Math.round(meanG),
    meanB: Math.round(meanB),
    edgeDensity: +edgeDensity.toFixed(4)
  };
};
</script></body></html>`;

}
