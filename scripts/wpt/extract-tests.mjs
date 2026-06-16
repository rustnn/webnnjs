import { loadWptConformanceFile } from './load-wpt-file.mjs';

/** @deprecated Use loadWptConformanceFile */
export function extractTestsFromSource(sourceText, sourceName = 'wpt-test.js', options = {}) {
  return loadWptConformanceFile(sourceText, sourceName, options);
}

export { loadWptConformanceFile };
