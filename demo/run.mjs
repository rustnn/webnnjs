#!/usr/bin/env node

import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const demoDir = path.dirname(fileURLToPath(import.meta.url));
const envFile = path.join(demoDir, '.env');
const target = process.argv[2];

if (!target) {
  console.error('Usage: node run.mjs <script> [args...]');
  process.exit(1);
}

const hasOrtDylibPath = Boolean(process.env.ORT_DYLIB_PATH?.trim());
const hasEnvFile = existsSync(envFile);

if (!hasOrtDylibPath && !hasEnvFile) {
  console.error('ONNX Runtime is not configured.');
  console.error('Set ORT_DYLIB_PATH in your environment or create demo/.env, for example:');
  console.error('  ORT_DYLIB_PATH=C:\\path\\to\\onnxruntime.dll');
  process.exit(1);
}

const nodeArgs = [];
if (hasEnvFile) {
  nodeArgs.push(`--env-file=${envFile}`);
}
nodeArgs.push(path.resolve(demoDir, target), ...process.argv.slice(3));

const child = spawn(process.execPath, nodeArgs, {
  cwd: demoDir,
  stdio: 'inherit',
  env: process.env
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
