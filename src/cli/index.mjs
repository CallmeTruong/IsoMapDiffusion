/**
 * cli/index.mjs — Re-export facade
 */

export { parseCliArgs, showHelp } from './args.mjs';
export { getSeed } from './seed.mjs';
export { cmdRender, cmdTest, runRender } from './render.mjs';
export { cmdStitch } from './stitch.mjs';
export { cmdList, cmdInfo } from './list_info.mjs';
export { runXaxisDetect } from './xaxis.mjs';
export { scanRenderDirs, dirHasGrid, findBestContiguousGrid } from './discover.mjs';
