const fs = require('node:fs');
const path = require('node:path');

const direct = path.join(__dirname, 'index.node');
if (fs.existsSync(direct)) {
  module.exports = require(direct);
} else {
  const candidates = fs
    .readdirSync(__dirname)
    .filter((name) => name.endsWith('.node'));

  if (candidates.length === 0) {
    throw new Error(
      'Native addon not found. Run: npm run build -w @webnnjs/webnn-node-native'
    );
  }

  module.exports = require(path.join(__dirname, candidates[0]));
}
