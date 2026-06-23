import {
  MLContext,
  MLGraph,
  MLTensor,
  installWebNNPolyfill,
  ml,
  type LoadedModelMeta,
  type MLOperandDataType,
} from '@webnnjs/webnn-node';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { findFileRecursive } from './fs-utils.js';
import { ensureOrtDylibPath } from './ort-env.js';

const REPO_ID = 'tarekziade/SmolLM-135M-webnn';
const PROMPT_TEXT = process.env.DEMO_PROMPT ?? 'Once upon a time';
const MAX_NEW_TOKENS = Number(process.env.DEMO_MAX_NEW_TOKENS ?? '24');

type KvLayout = {
  numHeads: number;
  headDim: number;
  layerCount: number;
  maxCacheLen: number;
  logitsName: string;
  vocabSize: number;
};

type StepState = {
  position: number;
  cache: Map<string, Float32Array>;
};

type StepTensors = {
  inputIds: MLTensor;
  positionIds: MLTensor;
  attentionMask: MLTensor;
  pastK: MLTensor[];
  pastV: MLTensor[];
  presentK: MLTensor[];
  presentV: MLTensor[];
  logits: MLTensor;
};

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

function detectLayout(meta: LoadedModelMeta): KvLayout {
  let numLayers = 0;
  let numHeads: number | undefined;
  let maxCacheLen: number | undefined;
  let headDim: number | undefined;
  let logitsName: string | undefined;
  let vocabSize: number | undefined;

  for (const [name, desc] of Object.entries(meta.inputs)) {
    if (name.startsWith('past_key_values_')) {
      const layer = Number(name.split('_')[3]);
      if (Number.isFinite(layer)) {
        numLayers = Math.max(numLayers, layer + 1);
      }
      if (desc.shape.length === 4) {
        numHeads = desc.shape[1];
        maxCacheLen = desc.shape[2];
        headDim = desc.shape[3];
      }
    }
  }

  for (const [name, desc] of Object.entries(meta.outputs)) {
    if (name.toLowerCase().includes('logits')) {
      logitsName = name;
      if (desc.shape.length > 0) {
        vocabSize = desc.shape[desc.shape.length - 1];
      }
    }
  }

  if (
    numLayers === 0 ||
    numHeads === undefined ||
    maxCacheLen === undefined ||
    headDim === undefined ||
    !logitsName ||
    !vocabSize
  ) {
    throw new Error('Unable to infer SmolLM layout from loaded model metadata');
  }

  return {
    numHeads,
    headDim,
    layerCount: numLayers,
    maxCacheLen,
    logitsName,
    vocabSize,
  };
}

async function makeTensor(
  context: MLContext,
  dataType: MLOperandDataType,
  shape: number[],
  readable: boolean,
  writable: boolean
): Promise<MLTensor> {
  return context.createTensor({
    dataType,
    shape,
    readable,
    writable,
  });
}

async function initStepTensors(
  context: MLContext,
  layout: KvLayout
): Promise<StepTensors> {
  const h = layout.numHeads;
  const d = layout.headDim;
  const maxSeq = layout.maxCacheLen;
  const maxPast = Math.max(0, layout.maxCacheLen - 1);

  const inputIds = await makeTensor(context, 'int64', [1, 1], false, true);
  const positionIds = await makeTensor(context, 'int64', [1, 1], false, true);
  const attentionMask = await makeTensor(context, 'int64', [1, 1], false, true);
  context.rustnnSetTensorCapacity(attentionMask, [1, maxSeq]);

  const pastK: MLTensor[] = [];
  const pastV: MLTensor[] = [];
  const presentK: MLTensor[] = [];
  const presentV: MLTensor[] = [];

  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    const pk = await makeTensor(context, 'float32', [1, h, 0, d], false, true);
    context.rustnnSetTensorCapacity(pk, [1, h, maxPast, d]);
    pastK.push(pk);

    const pv = await makeTensor(context, 'float32', [1, h, 0, d], false, true);
    context.rustnnSetTensorCapacity(pv, [1, h, maxPast, d]);
    pastV.push(pv);

    const prk = await makeTensor(context, 'float32', [1, h, 1, d], true, false);
    context.rustnnSetTensorCapacity(prk, [1, h, maxSeq, d]);
    presentK.push(prk);

    const prv = await makeTensor(context, 'float32', [1, h, 1, d], true, false);
    context.rustnnSetTensorCapacity(prv, [1, h, maxSeq, d]);
    presentV.push(prv);
  }

  const logitsShape = [1, 1, layout.vocabSize];
  const logits = await makeTensor(context, 'float32', logitsShape, true, false);

  return {
    inputIds,
    positionIds,
    attentionMask,
    pastK,
    pastV,
    presentK,
    presentV,
    logits,
  };
}

function initState(layout: KvLayout): StepState {
  const cache = new Map<string, Float32Array>();
  const elems = layout.numHeads * layout.maxCacheLen * layout.headDim;
  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    cache.set(`past_key_values_${layer}_key`, new Float32Array(elems));
    cache.set(`past_key_values_${layer}_value`, new Float32Array(elems));
  }
  return { position: 0, cache };
}

function compactKv(
  state: StepState,
  layout: KvLayout,
  layer: number,
  kind: 'key' | 'value',
  pastLen: number
): Float32Array {
  if (pastLen === 0) {
    return new Float32Array();
  }
  const cache = state.cache.get(`past_key_values_${layer}_${kind}`);
  if (!cache) {
    throw new Error(`Missing KV cache for layer ${layer} ${kind}`);
  }
  const out = new Float32Array(layout.numHeads * pastLen * layout.headDim);
  for (let h = 0; h < layout.numHeads; h += 1) {
    for (let t = 0; t < pastLen; t += 1) {
      const src = (h * layout.maxCacheLen + t) * layout.headDim;
      const dst = (h * pastLen + t) * layout.headDim;
      out.set(cache.subarray(src, src + layout.headDim), dst);
    }
  }
  return out;
}

function storePresent(
  state: StepState,
  layout: KvLayout,
  layer: number,
  kind: 'key' | 'value',
  present: Float32Array,
  seqLen: number
): void {
  const cache = state.cache.get(`past_key_values_${layer}_${kind}`);
  if (!cache) {
    throw new Error(`Missing KV cache for layer ${layer} ${kind}`);
  }
  for (let h = 0; h < layout.numHeads; h += 1) {
    const dst = (h * layout.maxCacheLen + state.position) * layout.headDim;
    const src = (h * seqLen + (seqLen - 1)) * layout.headDim;
    cache.set(present.subarray(src, src + layout.headDim), dst);
  }
}

function argmax(values: Float32Array): number {
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

async function runStep(args: {
  context: MLContext;
  graph: MLGraph;
  layout: KvLayout;
  tensors: StepTensors;
  state: StepState;
  tokenId: number;
}): Promise<number> {
  const { context, graph, layout, tensors, state, tokenId } = args;
  const pastLen = state.position;
  const seqLen = pastLen + 1;
  const h = layout.numHeads;
  const d = layout.headDim;

  context.rustnnResizeTensor(tensors.attentionMask, [1, seqLen]);
  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    context.rustnnResizeTensor(tensors.pastK[layer], [1, h, pastLen, d]);
    context.rustnnResizeTensor(tensors.pastV[layer], [1, h, pastLen, d]);
    context.rustnnResizeTensor(tensors.presentK[layer], [1, h, seqLen, d]);
    context.rustnnResizeTensor(tensors.presentV[layer], [1, h, seqLen, d]);
  }

  context.writeTensor(tensors.inputIds, new BigInt64Array([BigInt(tokenId)]));
  context.writeTensor(tensors.positionIds, new BigInt64Array([BigInt(pastLen)]));
  context.writeTensor(tensors.attentionMask, new BigInt64Array(Array(seqLen).fill(1n)));

  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    const kData = compactKv(state, layout, layer, 'key', pastLen);
    if (kData.length > 0) {
      context.writeTensor(tensors.pastK[layer], kData);
    }
    const vData = compactKv(state, layout, layer, 'value', pastLen);
    if (vData.length > 0) {
      context.writeTensor(tensors.pastV[layer], vData);
    }
  }

  const inputs: Record<string, MLTensor> = {
    input_ids: tensors.inputIds,
    position_ids: tensors.positionIds,
    attention_mask: tensors.attentionMask,
  };
  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    inputs[`past_key_values_${layer}_key`] = tensors.pastK[layer];
    inputs[`past_key_values_${layer}_value`] = tensors.pastV[layer];
  }

  const outputs: Record<string, MLTensor> = {
    [layout.logitsName]: tensors.logits,
  };
  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    outputs[`present_${layer}_key`] = tensors.presentK[layer];
    outputs[`present_${layer}_value`] = tensors.presentV[layer];
  }

  context.dispatch(graph, inputs, outputs);

  const logitsView = (await context.readTensorTyped(tensors.logits)) as Float32Array;
  const nextToken = argmax(logitsView);

  const kvElems = layout.numHeads * seqLen * layout.headDim;
  for (let layer = 0; layer < layout.layerCount; layer += 1) {
    const presentK = (await context.readTensorTyped(tensors.presentK[layer])) as Float32Array;
    const presentV = (await context.readTensorTyped(tensors.presentV[layer])) as Float32Array;
    if (presentK.length >= kvElems) {
      storePresent(state, layout, layer, 'key', presentK, seqLen);
    }
    if (presentV.length >= kvElems) {
      storePresent(state, layout, layer, 'value', presentV, seqLen);
    }
  }

  state.position += 1;
  return nextToken;
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
  if (!Number.isFinite(MAX_NEW_TOKENS) || MAX_NEW_TOKENS <= 0) {
    throw new Error(
      `DEMO_MAX_NEW_TOKENS must be a positive number, got: ${String(process.env.DEMO_MAX_NEW_TOKENS)}`
    );
  }

  installWebNNPolyfill();
  ensureOrtDylibPath();

  console.log(`Using ORT_DYLIB_PATH=${process.env.ORT_DYLIB_PATH}`);

  const { context, graph, meta, snapshotPath } = await ml.loadModelFromHub(REPO_ID, {
    accelerated: true,
    powerPreference: 'high-performance',
  });

  try {
    console.log(`Downloaded snapshot: ${snapshotPath}`);
    console.log(`Resolved WebNN graph: ${meta.graphPath}`);
    console.log(`Context accelerated: ${context.accelerated}`);

    const tokenizerPath = findTokenizerPath(snapshotPath);
    const tokenizer = await loadTokenizerFromFile(tokenizerPath);
    const promptIds = encodedIds(tokenizer.encode(PROMPT_TEXT, true));

    if (promptIds.length === 0) {
      throw new Error(`Prompt encoded to zero tokens: "${PROMPT_TEXT}"`);
    }

    const layout = detectLayout(meta);
    if (promptIds.length >= layout.maxCacheLen) {
      throw new Error(
        `Prompt too long: ${promptIds.length} tokens (must be < ${layout.maxCacheLen})`
      );
    }

    const tensors = await initStepTensors(context, layout);
    const state = initState(layout);

    console.log(`Prompt: ${PROMPT_TEXT}`);
    console.log(`Prompt token count: ${promptIds.length}`);

    let lastToken = 0;
    for (const tokenId of promptIds) {
      lastToken = await runStep({
        context,
        graph,
        layout,
        tensors,
        state,
        tokenId,
      });
    }

    const generatedIds: number[] = [];
    for (let i = 0; i < MAX_NEW_TOKENS; i += 1) {
      generatedIds.push(lastToken);
      lastToken = await runStep({
        context,
        graph,
        layout,
        tensors,
        state,
        tokenId: lastToken,
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
