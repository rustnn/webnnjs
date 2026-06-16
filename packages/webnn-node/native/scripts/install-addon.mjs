#!/usr/bin/env node
/**
 * Stage the cargo release artifact as index.node for Node to require().
 *
 * This must run after `cargo build --release`, not in build.rs: build.rs executes
 * before the linker produces the cdylib.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dest = path.join(packageRoot, 'index.node');

const artifactName =
  process.platform === 'win32'
    ? 'webnn_node_native.dll'
    : process.platform === 'darwin'
      ? 'libwebnn_node_native.dylib'
      : 'libwebnn_node_native.so';

const artifact = path.join(packageRoot, 'target', 'release', artifactName);

if (!fs.existsSync(artifact)) {
  console.error(`Native artifact not found: ${artifact}`);
  console.error('Run: cargo build --release');
  process.exit(1);
}

try {
  if (fs.existsSync(dest)) {
    fs.rmSync(dest, { force: true });
  }
  fs.copyFileSync(artifact, dest);
  console.log(`Installed ${artifactName} -> index.node`);
} catch (error) {
  if (error && typeof error === 'object' && 'code' in error) {
    const code = error.code;
    if (code === 'EBUSY' || code === 'EPERM') {
      const pending = path.join(packageRoot, 'index.staged.node');
      fs.copyFileSync(artifact, pending);
      console.error(
        `Could not replace index.node (${code}): file is in use. ` +
          `Staged build at ${path.basename(pending)}. ` +
          'Close running Node processes, then run: npm run install-addon'
      );
      process.exit(1);
    }
  }
  throw error;
}
