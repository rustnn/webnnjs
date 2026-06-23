import * as fs from 'node:fs';
import * as path from 'node:path';

export function findFileRecursive(
  rootDir: string,
  targetFileName: string,
  maxDepth: number
): string | undefined {
  if (maxDepth < 0 || !fs.existsSync(rootDir)) {
    return undefined;
  }

  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isFile() && entry.name === targetFileName) {
      return fullPath;
    }
    if (entry.isDirectory()) {
      const nested = findFileRecursive(fullPath, targetFileName, maxDepth - 1);
      if (nested) {
        return nested;
      }
    }
  }

  return undefined;
}
