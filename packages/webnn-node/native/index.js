const fs = require('node:fs');
const path = require('node:path');

const dir = __dirname;
const candidates = fs
  .readdirSync(dir)
  .filter((name) => name.endsWith('.node'))
  .map((name) => path.join(dir, name))
  .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);

if (candidates.length === 0) {
  throw new Error(
    'Native addon not found. Run: npm run build -w @webnnjs/webnn-node-native'
  );
}

module.exports = require(candidates[0]);
