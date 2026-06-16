#!/usr/bin/env node

import { copyFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const wptDir = process.env.WPT_DIR ?? path.join(root, '.cache', 'wpt');
const src = path.join(wptDir, 'interfaces', 'webnn.idl');
const dest = path.join(root, 'packages', 'webnn-node', 'idl', 'webnn.idl');

if (!existsSync(src)) {
  console.error(`WPT IDL not found at ${src}. Run: npm run test:wpt:fetch`);
  process.exit(2);
}

copyFileSync(src, dest);
console.log(`Synced ${src} -> ${dest}`);
