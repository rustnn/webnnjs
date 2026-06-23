#!/usr/bin/env node

import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';

const root = process.cwd();
const cacheDir = path.join(root, '.cache');
const wptDir = process.env.WPT_DIR ?? path.join(cacheDir, 'wpt');
const repo = 'https://github.com/web-platform-tests/wpt.git';

// webnnjs only needs IDL + WebNN tests/resources (not the full ~160k-file WPT tree).
const SPARSE_CONE_PATHS = ['interfaces', 'webnn'];

function run(cmd, args, cwd = root) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd, stdio: 'inherit' });
    p.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} failed with code ${code}`));
    });
  });
}

async function ensureSparseCheckout() {
  await run('git', ['sparse-checkout', 'init', '--cone'], wptDir);
  await run('git', ['sparse-checkout', 'set', ...SPARSE_CONE_PATHS], wptDir);
}

await mkdir(cacheDir, { recursive: true });

const hasGitRepo = existsSync(path.join(wptDir, '.git'));

if (!hasGitRepo) {
  console.log(`Cloning WPT (sparse: ${SPARSE_CONE_PATHS.join(', ')}) into ${wptDir}...`);
  await run('git', [
    'clone',
    '--depth',
    '1',
    '--filter=blob:none',
    '--sparse',
    '--single-branch',
    '--branch',
    'master',
    repo,
    wptDir
  ]);
  await ensureSparseCheckout();
} else {
  console.log(`Updating WPT in ${wptDir}...`);
  await run(
    'git',
    ['fetch', '--depth', '1', '--filter=blob:none', 'origin', 'master'],
    wptDir
  );
  await ensureSparseCheckout();
  await run('git', ['reset', '--hard', 'origin/master'], wptDir);
}

console.log('WPT ready.');
