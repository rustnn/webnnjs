import {
  MLTensor,
  installWebNNPolyfill,
  ml,
  type LoadModelTensorMeta,
} from '@webnnjs/webnn-node';
import * as fs from 'node:fs';
import * as path from 'node:path';

const REPO_ID = 'tarekziade/SmolLM-135M-webnn';
const PROMPT_TOKEN_IDS = [7454];

function findFileRecursive(
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

function ensureOrtDylibPath(): void {
  if (process.env.ORT_DYLIB_PATH && fs.existsSync(process.env.ORT_DYLIB_PATH)) {
    return;
  }

  const directCandidates = [
    '/Users/tarek/Dev/rustnn/target/onnxruntime/onnxruntime-osx-arm64-1.23.2/lib/libonnxruntime.dylib',
    '/Users/tarek/Dev/rustnn/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.dylib',
  ];

  for (const candidate of directCandidates) {
    if (fs.existsSync(candidate)) {
      process.env.ORT_DYLIB_PATH = candidate;
      return;
    }
  }

  const searchRoots = [
    '/Users/tarek/Dev/rustnn/target/onnxruntime',
    path.resolve(process.cwd(), '../../rustnn/target/onnxruntime'),
    path.resolve(process.cwd(), '../../rustnn/.venv/lib'),
  ];

  for (const root of searchRoots) {
    const found = findFileRecursive(root, 'libonnxruntime.dylib', 8);
    if (found) {
      process.env.ORT_DYLIB_PATH = found;
      return;
    }
  }
}

function checkedElementCount(shape: number[]): number {
  if (shape.length === 0) {
    return 1;
  }
  return shape.reduce((acc, dim) => acc * dim, 1);
}

function chooseInputShape(
  name: string,
  modelMeta: LoadModelTensorMeta | undefined,
  promptLength: number
): number[] {
  const modelShape = modelMeta?.shape ?? [1];

  if (name === 'input_ids') {
    return [1, 1];
  }

  if (name === 'attention_mask') {
    return [1, 1];
  }

  if (name === 'position_ids') {
    return [1, 1];
  }

  if (name.startsWith('past_key_values_') && modelShape.length === 4) {
    return [1, Math.max(1, modelShape[1] ?? 1), 0, Math.max(1, modelShape[3] ?? 1)];
  }

  return modelShape.map((dim) => (dim < 0 ? 1 : dim));
}

function makeInputData(
  name: string,
  shape: number[],
  promptTokenIds: number[]
): Float32Array {
  const total = checkedElementCount(shape);
  const values = new Float32Array(total);

  if (name === 'input_ids') {
    for (let i = 0; i < Math.min(total, promptTokenIds.length); i += 1) {
      values[i] = promptTokenIds[i];
    }
    return values;
  }

  if (name === 'attention_mask') {
    values.fill(1);
    return values;
  }

  if (name === 'position_ids') {
    for (let i = 0; i < total; i += 1) {
      values[i] = i;
    }
    return values;
  }

  return values;
}

function topK(values: Float32Array, k: number): Array<{ index: number; value: number }> {
  const scored = Array.from(values, (value, index) => ({ value, index }));
  scored.sort((a, b) => b.value - a.value);
  return scored.slice(0, k);
}

async function main(): Promise<void> {
  installWebNNPolyfill();
  ensureOrtDylibPath();
  if (process.env.ORT_DYLIB_PATH) {
    console.log(`Using ORT_DYLIB_PATH=${process.env.ORT_DYLIB_PATH}`);
  } else {
    console.log('ORT_DYLIB_PATH not found automatically; relying on system loader search path');
  }

  const { context, graph, meta, snapshotPath } = await ml.loadModelFromHub(REPO_ID, {
    deviceType: 'gpu',
    accelerated: true,
    powerPreference: 'high-performance',
  });

  console.log(`Downloaded snapshot: ${snapshotPath}`);
  console.log(`Resolved WebNN graph: ${meta.graphPath}`);
  if (meta.warnings.length > 0) {
    console.log('Warnings:');
    for (const warning of meta.warnings) {
      console.log(`  - ${warning}`);
    }
  }

  const inputTensors: Record<string, MLTensor> = {};
  const outputTensors: Record<string, MLTensor> = {};

  for (const inputName of meta.inputNames) {
    const inputMeta = meta.inputs[inputName];
    const shape = chooseInputShape(inputName, inputMeta, PROMPT_TOKEN_IDS.length);

    const tensor = await context.createTensor({ dataType: 'float32', shape });
    const values = makeInputData(inputName, shape, PROMPT_TOKEN_IDS);
    context.writeTensor(tensor, values);
    inputTensors[inputName] = tensor;

    const modelType = inputMeta?.dataType ?? 'unknown';
    console.log(
      `Prepared input '${inputName}' shape=${JSON.stringify(shape)} modelType=${modelType}`
    );
  }

  const outputName =
    meta.outputNames.find((name) => name.toLowerCase().includes('logits')) ??
    meta.outputNames[0];

  if (!outputName) {
    throw new Error('Model has no outputs');
  }

  outputTensors[outputName] = await context.createTensor({
    dataType: 'float32',
    shape: [1],
  });

  context.dispatch(graph, inputTensors, outputTensors);
  const logits = await context.readTensor(outputTensors[outputName]);

  const top5 = topK(logits, 5);
  console.log(`Top-5 logits from output '${outputName}':`);
  for (const item of top5) {
    console.log(`  index=${item.index} value=${item.value.toFixed(6)}`);
  }

  graph.destroy();
  for (const tensor of Object.values(inputTensors)) {
    tensor.destroy();
  }
  for (const tensor of Object.values(outputTensors)) {
    tensor.destroy();
  }
  context.destroy();
}

main().catch((error) => {
  console.error('Demo failed:', error);
  process.exitCode = 1;
});
