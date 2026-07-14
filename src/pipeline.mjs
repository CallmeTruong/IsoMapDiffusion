import { parseCliArgs, showHelp, cmdRender, cmdTest, cmdStitch, cmdList, cmdInfo } from './cli/index.mjs';
import { projectRoot, loadEnv } from './render/index.mjs';

const projectRootDir = projectRoot(import.meta.url);
loadEnv(projectRootDir);

async function main() {
  const { positional, flags } = parseCliArgs(process.argv);
  const cmd = positional[0] ?? 'help';
  const ctx = { positional: positional.slice(1), flags, projectRootDir };

  try {
    switch (cmd) {
      case 'render': await cmdRender(ctx); break;
      case 'test':   await cmdTest(ctx); break;
      case 'stitch': await cmdStitch(ctx); break;
      case 'list':   await cmdList(ctx); break;
      case 'info':   cmdInfo(ctx); break;
      case 'help':
      case '--help':
      case '-h':
      default: showHelp();
    }
  } catch (e) {
    console.error(`\n[error] ${e.message}`);
    if (process.env.DEBUG) console.error(e.stack);
    process.exit(1);
  }
}

main();
