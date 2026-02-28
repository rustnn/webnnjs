import {
  MLTensor,
  installWebNNPolyfill,
  ml,
  type LoadModelTensorMeta,
} from '@webnnjs/webnn-node';
import * as fs from 'node:fs';
import * as path from 'node:path';

const REPO_ID = 'tarekziade/SmolLM-135M-webnn';
const PROMPT_TEXT = 'Once upon a time';
const MAX_NEW_TOKENS = 24;

type KvTensorInfo = {
  name: string;
  layer: number;
  kind: 'key' | 'value';
};

type KvLayout = {
  numHeads: number;
  headDim: number;
  layerCount: number;
  pastInputs: KvTensorInfo[];
  presentOutputs: KvTensorInfo[];
};

type StepState = {
  position: number;
  pastLength: number;
  kvCache: Map<string, Float32Array<ArrayBufferLike>>;
};

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

function findTokenizerPath(snapshotPath: string): string {
  const direct = path.join(snapshotPath, 'tokenizer.json');
  if (fs.existsSync(direct)) {
    return direct;
  }

  const discovered = findFileRecursive(snapshotPath, 'tokenizer.json', 4);
  if (!discovered) {
    throw new Error(`Unable to locate tokenizer.json in snapshot: ${snapshotPath}`);
  }
  return discovered;
}

function parseKvName(
  name: string,
  prefix: 'past_key_values_' | 'present_'
): KvTensorInfo | undefined {
  if (!name.startsWith(prefix)) {
    return undefined;
  }

  const suffix = name.slice(prefix.length);
  const match = suffix.match(/^(\d+)_(key|value)$/);
  if (!match) {
    return undefined;
  }

  return {
    name,
    layer: Number(match[1]),
    kind: match[2] as 'key' | 'value',
  };
}

function detectKvLayout(meta: {
  inputNames: string[];
  outputNames: string[];
  inputs: Record<string, LoadModelTensorMeta>;
}): KvLayout {
  const pastInputs: KvTensorInfo[] = [];
  const presentOutputs: KvTensorInfo[] = [];

  for (const name of meta.inputNames) {
    const parsed = parseKvName(name, 'past_key_values_');
    if (parsed) {
      pastInputs.push(parsed);
    }
  }

  for (const name of meta.outputNames) {
    const parsed = parseKvName(name, 'present_');
    if (parsed) {
      presentOutputs.push(parsed);
    }
  }

  if (pastInputs.length === 0 || presentOutputs.length === 0) {
    throw new Error('Model metadata does not expose expected KV cache inputs/outputs');
  }

  const firstPastMeta = meta.inputs[pastInputs[0].name];
  if (!firstPastMeta || firstPastMeta.shape.length < 4) {
    throw new Error('Unable to infer KV layout from past_key_values metadata');
  }

  const numHeads = Math.max(1, firstPastMeta.shape[1] ?? 1);
  const headDim = Math.max(1, firstPastMeta.shape[3] ?? 1);
  const maxLayer = Math.max(...pastInputs.map((entry) => entry.layer));

  return {
    numHeads,
    headDim,
    layerCount: maxLayer + 1,
    pastInputs,
    presentOutputs,
  };
}

async function createAndFillTensor(
  context: Awaited<ReturnType<typeof ml.createContext>>,
  shape: number[],
  values: Float32Array<ArrayBufferLike>
): Promise<MLTensor> {
  const tensor = await context.createTensor({ dataType: 'float32', shape });
  context.writeTensor(tensor, values);
  return tensor;
}

function argmax(values: Float32Array<ArrayBufferLike>): number {
  if (values.length === 0) {
    throw new Error('Cannot argmax empty logits tensor');
  }

  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

async function runGenerationStep(args: {
  context: Awaited<ReturnType<typeof ml.createContext>>;
  graph: { dispatch: (inputs: Record<string, MLTensor>, outputs: Record<string, MLTensor>) => void };
  modelMeta: {
    inputNames: string[];
    outputNames: string[];
  };
  kvLayout: KvLayout;
  state: StepState;
  tokenId: number;
}): Promise<Float32Array<ArrayBufferLike>> {
  const { context, graph, modelMeta, kvLayout, state, tokenId } = args;

  const inputTensors: Record<string, MLTensor> = {};
  const outputTensors: Record<string, MLTensor> = {};

  try {
    for (const inputName of modelMeta.inputNames) {
      if (inputName === 'input_ids') {
        inputTensors[inputName] = await createAndFillTensor(
          context,
          [1, 1],
          new Float32Array([tokenId])
        );
      } else if (inputName === 'position_ids') {
        inputTensors[inputName] = await createAndFillTensor(
          context,
          [1, 1],
          new Float32Array([state.position])
        );
      } else if (inputName === 'attention_mask') {
        const length = state.pastLength + 1;
        inputTensors[inputName] = await createAndFillTensor(
          context,
          [1, length],
          new Float32Array(length).fill(1)
        );
      } else {
        const kv = parseKvName(inputName, 'past_key_values_');
        if (!kv) {
          throw new Error(`Unsupported model input for demo generation: ${inputName}`);
        }

        const shape = [1, kvLayout.numHeads, state.pastLength, kvLayout.headDim];
        const expected = checkedElementCount(shape);
        const cached = state.kvCache.get(inputName) ?? new Float32Array(expected);

        if (cached.length !== expected) {
          throw new Error(
            `KV cache size mismatch for ${inputName}: expected ${expected}, got ${cached.length}`
          );
        }

        inputTensors[inputName] = await createAndFillTensor(context, shape, cached);
      }
    }

    const logitsName =
      modelMeta.outputNames.find((name) => name.toLowerCase().includes('logits')) ??
      modelMeta.outputNames[0];
    if (!logitsName) {
      throw new Error('No output tensor found for logits');
    }

    outputTensors[logitsName] = await context.createTensor({
      dataType: 'float32',
      shape: [1],
    });

    for (const kvOut of kvLayout.presentOutputs) {
      outputTensors[kvOut.name] = await context.createTensor({
        dataType: 'float32',
        shape: [1],
      });
    }

    graph.dispatch(inputTensors, outputTensors);

    const logits = await context.readTensor(outputTensors[logitsName]);

    let nextPastLength = state.pastLength;
    for (const kvOut of kvLayout.presentOutputs) {
      const values = await context.readTensor(outputTensors[kvOut.name]);
      const perToken = kvLayout.numHeads * kvLayout.headDim;
      const seqLen = perToken === 0 ? 0 : Math.floor(values.length / perToken);
      const cacheName = `past_key_values_${kvOut.layer}_${kvOut.kind}`;
      state.kvCache.set(cacheName, values);
      nextPastLength = Math.max(nextPastLength, seqLen);
    }

    state.pastLength = nextPastLength;
    state.position += 1;

    return logits;
  } finally {
    for (const tensor of Object.values(inputTensors)) {
      tensor.destroy();
    }
    for (const tensor of Object.values(outputTensors)) {
      tensor.destroy();
    }
  }
}

type LoadedTokenizer = {
  encode(
    text: string,
    addSpecialTokens?: boolean
  ): { ids?: number[]; getIds?: () => number[] };
  decode(ids: number[], skipSpecialTokens?: boolean): string;
};

async function loadTokenizerFromFile(tokenizerPath: string): Promise<LoadedTokenizer> {
  const tokenizerJson = JSON.parse(fs.readFileSync(tokenizerPath, 'utf-8'));
  const tokenizerConfigPath = path.join(path.dirname(tokenizerPath), 'tokenizer_config.json');
  const tokenizerConfig = fs.existsSync(tokenizerConfigPath)
    ? JSON.parse(fs.readFileSync(tokenizerConfigPath, 'utf-8'))
    : {};

  const module = (await import('@huggingface/tokenizers')) as {
    Tokenizer: new (tokenizer: unknown, config?: unknown) => LoadedTokenizer;
  };

  return new module.Tokenizer(tokenizerJson, tokenizerConfig);
}

function encodedIds(encoded: { ids?: number[]; getIds?: () => number[] }): number[] {
  if (Array.isArray(encoded.ids)) {
    return encoded.ids;
  }
  if (typeof encoded.getIds === 'function') {
    return encoded.getIds();
  }
  throw new Error('Tokenizer encode() result did not expose token IDs');
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

  try {
    console.log(`Downloaded snapshot: ${snapshotPath}`);
    console.log(`Resolved WebNN graph: ${meta.graphPath}`);
    if (meta.warnings.length > 0) {
      console.log('Warnings:');
      for (const warning of meta.warnings) {
        console.log(`  - ${warning}`);
      }
    }

    const tokenizerPath = findTokenizerPath(snapshotPath);
    const tokenizer = await loadTokenizerFromFile(tokenizerPath);
    const encodedPrompt = tokenizer.encode(PROMPT_TEXT, true);
    const promptIds = encodedIds(encodedPrompt);

    if (promptIds.length === 0) {
      throw new Error(`Prompt encoded to zero tokens: "${PROMPT_TEXT}"`);
    }

    const kvLayout = detectKvLayout({
      inputNames: meta.inputNames,
      outputNames: meta.outputNames,
      inputs: meta.inputs,
    });

    console.log(`Prompt: ${PROMPT_TEXT}`);
    console.log(`Prompt token count: ${promptIds.length}`);

    const state: StepState = {
      position: 0,
      pastLength: 0,
      kvCache: new Map(),
    };

    let logits: Float32Array<ArrayBufferLike> = new Float32Array();

    for (const tokenId of promptIds) {
      logits = await runGenerationStep({
        context,
        graph: {
          dispatch: (inputs, outputs) => context.dispatch(graph, inputs, outputs),
        },
        modelMeta: {
          inputNames: meta.inputNames,
          outputNames: meta.outputNames,
        },
        kvLayout,
        state,
        tokenId,
      });
    }

    const generatedIds: number[] = [];
    for (let i = 0; i < MAX_NEW_TOKENS; i += 1) {
      const nextId = argmax(logits);
      generatedIds.push(nextId);

      logits = await runGenerationStep({
        context,
        graph: {
          dispatch: (inputs, outputs) => context.dispatch(graph, inputs, outputs),
        },
        modelMeta: {
          inputNames: meta.inputNames,
          outputNames: meta.outputNames,
        },
        kvLayout,
        state,
        tokenId: nextId,
      });
    }

    const fullIds = [...promptIds, ...generatedIds];
    const generatedText = tokenizer.decode(fullIds, true);

    console.log(`Generated token IDs: ${generatedIds.join(', ')}`);
    console.log('\nGenerated text:');
    console.log(generatedText);
  } finally {
    graph.destroy();
    context.destroy();
  }
}

main().catch((error) => {
  console.error('Demo failed:', error);
  process.exitCode = 1;
});
