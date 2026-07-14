import fs from 'fs';
import path from 'path';

const rootDir = process.cwd();
const rendersDir = path.join(rootDir, 'output', 'renders'); 

async function main() {
  try {
    console.log("Reading generation_config.json...");
    
    const configPath = path.join(rendersDir, 'generation_config.json');
    const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    
    const tileSize = configData.width_px;
    const cameraMoveStep = configData.camera_move_step;
    const stride = Math.round(tileSize * cameraMoveStep);

    console.log(`Config: Tile ${tileSize}px, Stride: ${stride}px`);

    const files = fs.readdirSync(rendersDir);
    const regex = /^tile_([+-]?\d+)_([+-]?\d+)_([a-f0-9]+)\.png$/;
    
    const myTiles = [];
    let minQx = Infinity, maxQx = -Infinity;
    let minQy = Infinity, maxQy = -Infinity;

    for (const file of files) {
      const match = file.match(regex);
      if (match) {
        const qx = parseInt(match[1], 10);
        const qy = parseInt(match[2], 10);

        if (qx < minQx) minQx = qx;
        if (qx > maxQx) maxQx = qx;
        if (qy < minQy) minQy = qy;
        if (qy > maxQy) maxQy = qy;

        myTiles.push({
          qx, qy,
          path: path.join(rendersDir, file)
        });
      }
    }

    if (myTiles.length === 0) {
      console.log("Error matching tile.");
      return;
    }
    
    
    const canvasWidth = (maxQx - minQx) * stride + tileSize;
    const canvasHeight = (maxQy - minQy) * stride + tileSize;

    console.log("Start plan tiles");

    const planTiles = myTiles.map(t => ({
      path: t.path,
      x: (t.qx - minQx) * stride,
      y: (t.qy - minQy) * stride
    }));

    const plan = {
      canvasWidth,
      canvasHeight,
      tiles: planTiles
    };

    const outputJsonPath = path.join(rootDir, 'output', 'map_plan.json');
    fs.writeFileSync(outputJsonPath, JSON.stringify(plan, null, 2));

    console.log("-----------------------------------------");
    console.log("Done!");
    console.log(`Size: ${canvasWidth} x ${canvasHeight} pixel`);

  } catch (err) {
    console.error("Error during execute:", err);
  }
}

// Run
main();