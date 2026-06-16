import * as fs from 'node:fs';
import * as path from 'node:path';

import { findFileRecursive } from './fs-utils.js';

export function ensureOrtDylibPath(): void {
  const configured = process.env.ORT_DYLIB_PATH?.trim();
  if (configured) {
    if (fs.existsSync(configured) && fs.statSync(configured).isFile()) {
      return;
    }
    const dllInConfiguredDir = path.join(configured, 'onnxruntime.dll');
    if (fs.existsSync(dllInConfiguredDir)) {
      process.env.ORT_DYLIB_PATH = dllInConfiguredDir;
      return;
    }
  }

  const libraryNames =
    process.platform === 'darwin'
      ? ['libonnxruntime.dylib']
      : process.platform === 'linux'
        ? ['libonnxruntime.so']
        : process.platform === 'win32'
          ? ['onnxruntime.dll']
          : ['libonnxruntime.dylib', 'libonnxruntime.so', 'onnxruntime.dll'];

  const ortLibDirs = (process.env.ORT_LIB_DIR ?? '')
    .split(path.delimiter)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

  const home = process.env.HOME ?? process.env.USERPROFILE ?? '';
  const commonRoots = [
    path.resolve(process.cwd(), '../../'),
    path.resolve(process.cwd(), '../'),
    path.resolve(process.cwd(), '../../onnxruntime'),
    path.resolve(process.cwd(), '../onnxruntime'),
    path.resolve(process.cwd(), 'onnxruntime'),
    path.resolve(process.cwd(), 'onnxruntime/lib'),
    path.resolve(process.cwd(), 'target/onnxruntime'),
    path.resolve(process.cwd(), 'target/onnxruntime/lib'),
    path.resolve(process.cwd(), '../target/onnxruntime'),
    path.resolve(process.cwd(), '../target/onnxruntime/lib'),
    path.resolve(process.cwd(), '../../target/onnxruntime'),
    path.resolve(process.cwd(), '../../target/onnxruntime/lib'),
    path.resolve(process.cwd(), 'node_modules/onnxruntime-node'),
    path.resolve(process.cwd(), 'node_modules/onnxruntime-node/bin'),
    path.resolve(process.cwd(), '../node_modules/onnxruntime-node'),
    path.resolve(process.cwd(), '../node_modules/onnxruntime-node/bin'),
    '/usr/local/lib',
    '/usr/lib',
    '/opt/homebrew/lib',
    '/opt/onnxruntime/lib',
    '/usr/local/onnxruntime/lib',
    home ? path.join(home, '.local', 'lib') : '',
  ]
    .filter((entry) => entry.length > 0)
    .filter((entry, index, entries) => entries.indexOf(entry) === index);

  const roots = [...ortLibDirs, ...commonRoots];

  for (const root of roots) {
    for (const libraryName of libraryNames) {
      const direct = path.join(root, libraryName);
      if (fs.existsSync(direct)) {
        process.env.ORT_DYLIB_PATH = direct;
        return;
      }
      const discovered = findFileRecursive(root, libraryName, 8);
      if (discovered) {
        process.env.ORT_DYLIB_PATH = discovered;
        return;
      }
    }
  }
}
