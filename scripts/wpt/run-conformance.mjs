#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ml } from '@webnnjs/webnn-node';

import { executeGraphResources } from './execute-graph.mjs';
import { extractTestsFromSource } from './extract-tests.mjs';
import { ensureOrtDylibPath } from './ort-env.mjs';
import { assertOutputClose, normalizeOpName } from './tolerance.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../..');

const SUPPORTED_DTYPES = new Set([
  'float32',
  'float16',
  'int8',
  'uint8',
  'int32',
  'uint32',
  'int64',
  'uint64'
]);

const UNIMPLEMENTED_OPS = new Set();

function parseArgs(argv) {
  const opts = {
    wptDir: process.env.WPT_DIR ?? path.join(repoRoot, '.cache', 'wpt'),
    op: null,
    file: null,
    limitTests: Number.POSITIVE_INFINITY,
    limitFiles: Number.POSITIVE_INFINITY,
    stopOnFail: false,
    reportJson: null,
    exitZero: false,
    failureSummaryMax: 20
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--wpt-dir') opts.wptDir = argv[++i];
    else if (arg === '--op') opts.op = argv[++i];
    else if (arg === '--file') opts.file = argv[++i];
    else if (arg === '--limit-tests') opts.limitTests = Number(argv[++i]);
    else if (arg === '--limit-files') opts.limitFiles = Number(argv[++i]);
    else if (arg === '--stop-on-fail') opts.stopOnFail = true;
    else if (arg === '--report-json') opts.reportJson = argv[++i];
    else if (arg === '--exit-zero') opts.exitZero = true;
    else if (arg === '--all-failures') opts.failureSummaryMax = Number.POSITIVE_INFINITY;
    else if (arg === '--failure-summary-limit') {
      const n = Number(argv[++i]);
      opts.failureSummaryMax =
        !Number.isFinite(n) || n <= 0 ? Number.POSITIVE_INFINITY : Math.floor(n);
    } else if (arg === '--help') {
      console.log(
        'Usage: node scripts/wpt/run-conformance.mjs [options]\n' +
          '  Runs WebNN conformance subtests from .cache/wpt/webnn/conformance_tests only.\n' +
          '  [--wpt-dir PATH] [--op NAME] [--file FILE] [--limit-tests N] [--limit-files N]\n' +
          '  [--stop-on-fail] [--report-json PATH] [--exit-zero]\n' +
          '  [--all-failures | --failure-summary-limit N]  (default: first 20 failures)\n' +
          '\n' +
          'Examples:\n' +
          '  npm run test:wpt:run -- --op add --limit-tests 5\n' +
          '  npm run test:wpt:run -- --file add.https.any.js'
      );
      process.exit(0);
    }
  }

  return opts;
}

async function listConformanceFiles(wptDir) {
  const { readdir } = await import('node:fs/promises');
  const base = path.join(wptDir, 'webnn', 'conformance_tests');
  const names = await readdir(base);
  return names
    .filter((n) => n.endsWith('.https.any.js'))
    .sort()
    .map((n) => path.join(base, n));
}

function collectUnimplementedOps(test) {
  const ops = Array.isArray(test?.graph?.operators) ? test.graph.operators : [];
  const missing = new Set();
  for (const op of ops) {
    const normalized = normalizeOpName(op?.name ?? '');
    if (UNIMPLEMENTED_OPS.has(normalized)) {
      missing.add(normalized);
    }
  }
  return [...missing];
}

function shouldSkipTest(test) {
  const inputs = Object.values(test.graph?.inputs ?? {});
  const outputs = Object.values(test.graph?.expectedOutputs ?? {});
  const tensors = [...inputs, ...outputs];

  for (const t of tensors) {
    const dt = t?.descriptor?.dataType;
    if (!SUPPORTED_DTYPES.has(dt)) {
      return `unsupported dataType: ${dt}`;
    }
  }

  const missingOps = collectUnimplementedOps(test);
  if (missingOps.length > 0) {
    return `unimplemented op(s): ${missingOps.join(', ')}`;
  }

  return null;
}

async function runSingleTest(context, test, testName) {
  const graph = test.graph;
  const skipReason = shouldSkipTest(test);
  if (skipReason) {
    return { status: 'skip', reason: skipReason };
  }

  const outputs = await executeGraphResources(context, graph);
  const lastOp = normalizeOpName(graph?.operators?.[graph.operators.length - 1]?.name ?? 'unknown');
  const graphOperatorNames = (graph.operators ?? []).map((o) => normalizeOpName(o?.name ?? ''));

  for (const [name, expected] of Object.entries(graph.expectedOutputs ?? {})) {
    const actual = outputs[name];
    if (!actual) {
      throw new Error(`missing output: ${name}`);
    }
    assertOutputClose({
      operatorName: lastOp,
      graphOperatorNames,
      outputName: name,
      expected,
      actual
    });
  }

  return { status: 'pass' };
}

async function main() {
  const opts = parseArgs(process.argv);

  ensureOrtDylibPath();
  if (process.env.ORT_DYLIB_PATH) {
    console.log(`Using ORT_DYLIB_PATH=${process.env.ORT_DYLIB_PATH}`);
  }

  if (!existsSync(path.join(opts.wptDir, 'webnn', 'conformance_tests'))) {
    console.error(`WPT WebNN tests not found at ${opts.wptDir}. Run: npm run test:wpt:fetch`);
    process.exit(2);
  }

  let files = await listConformanceFiles(opts.wptDir);
  if (opts.file) {
    files = files.filter((f) => path.basename(f).endsWith(opts.file) || path.basename(f) === opts.file);
  }
  if (opts.op) {
    files = files.filter(
      (f) =>
        path.basename(f).startsWith(`${opts.op}.`) || path.basename(f).startsWith(`${opts.op}_`)
    );
  }
  files = files.slice(0, opts.limitFiles);

  if (files.length === 0) {
    console.error('No matching WebNN conformance files.');
    process.exit(2);
  }

  const context = await ml.createContext({ accelerated: true });
  console.log(`Context accelerated: ${context.accelerated}`);

  let passed = 0;
  let failed = 0;
  let skipped = 0;
  const failures = [];
  const report = {
    meta: {
      startedAt: new Date().toISOString(),
      endedAt: null,
      options: opts,
      runner: 'webnnjs'
    },
    summary: { passed: 0, failed: 0, skipped: 0 },
    files: [],
    failures
  };

  try {
    for (const file of files) {
      const fileName = path.basename(file);
      const fileReport = {
        fileName,
        selectedTests: 0,
        summary: { passed: 0, failed: 0, skipped: 0 },
        cases: [],
        fileError: null
      };
      report.files.push(fileReport);

      let tests = [];
      try {
        const source = await readFile(file, 'utf8');
        tests = extractTestsFromSource(source, fileName).slice(0, opts.limitTests);
      } catch (err) {
        const skipReason = err.message.includes('No <name>Tests array');
        if (skipReason) {
          skipped += 1;
          fileReport.summary.skipped += 1;
          console.log(`\n[FILE] ${fileName} (non-graph test)`);
          console.log(`  - SKIP ${err.message}`);
          continue;
        }
        failures.push(`${fileName} :: FILE_PARSE :: ${err.message}`);
        failed += 1;
        fileReport.summary.failed += 1;
        fileReport.fileError = err.message;
        console.log(`\n[FILE] ${fileName} (parse error)`);
        console.log(`  - FAIL file parse: ${err.message}`);
        continue;
      }

      fileReport.selectedTests = tests.length;
      console.log(`\n[FILE] ${fileName} (${tests.length} tests)`);

      for (let testIndex = 0; testIndex < tests.length; testIndex += 1) {
        const test = tests[testIndex];
        const testName = test?.name ?? `[unnamed-${testIndex}]`;
        const started = Date.now();

        if (!test || typeof test !== 'object' || !test.graph) {
          skipped += 1;
          fileReport.summary.skipped += 1;
          console.log(`  - SKIP ${testName}: invalid extracted test case`);
          continue;
        }

        try {
          const res = await runSingleTest(context, test, testName);
          const durationMs = Date.now() - started;
          if (res.status === 'skip') {
            skipped += 1;
            fileReport.summary.skipped += 1;
            fileReport.cases.push({ testName, status: 'skip', reason: res.reason, durationMs });
            console.log(`  - SKIP ${testName}: ${res.reason}`);
            continue;
          }

          passed += 1;
          fileReport.summary.passed += 1;
          fileReport.cases.push({ testName, status: 'pass', durationMs });
          console.log(`  - PASS ${testName} (${durationMs}ms)`);
        } catch (err) {
          const durationMs = Date.now() - started;
          failed += 1;
          fileReport.summary.failed += 1;
          const message = err?.message ?? String(err);
          failures.push(`${fileName} :: ${testName} :: ${message}`);
          fileReport.cases.push({ testName, status: 'fail', error: message, durationMs });
          console.log(`  - FAIL ${testName}: ${message}`);
          if (opts.stopOnFail) {
            break;
          }
        }
      }

      if (opts.stopOnFail && failed > 0) {
        break;
      }
    }
  } finally {
    context.destroy();
  }

  report.summary = { passed, failed, skipped };
  report.meta.endedAt = new Date().toISOString();

  console.log('\n--- summary ---');
  console.log(`passed: ${passed}, failed: ${failed}, skipped: ${skipped}`);

  if (failures.length > 0) {
    const limit = opts.failureSummaryMax;
    const shown = failures.slice(0, limit);
    console.log(`\nFailures (${shown.length}${failures.length > shown.length ? ` of ${failures.length}` : ''}):`);
    for (const line of shown) {
      console.log(`  ${line}`);
    }
  }

  if (opts.reportJson) {
    await mkdir(path.dirname(path.resolve(opts.reportJson)), { recursive: true });
    await writeFile(path.resolve(opts.reportJson), JSON.stringify(report, null, 2), 'utf8');
    console.log(`\nWrote report: ${opts.reportJson}`);
  }

  if (!opts.exitZero && failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
