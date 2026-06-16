import { packConstantBuffer, shouldInlineConstant } from './pack-constants.mjs';

const POOL2D_LIKE_OPS = new Set([
  'averagePool2d',
  'maxPool2d',
  'l2Pool2d',
  'globalAveragePool',
  'globalMaxPool'
]);

const MULTI_OUTPUT_OPS = new Set(['split', 'gru', 'lstm', 'lstmCell']);

const OPTION_OPERAND_KEYS = {
  batchNormalization: ['scale', 'bias'],
  conv2d: ['bias'],
  convTranspose2d: ['bias'],
  gemm: ['c'],
  gru: ['bias', 'recurrentBias', 'initialHiddenState'],
  gruCell: ['bias', 'recurrentBias'],
  instanceNormalization: ['scale', 'bias'],
  layerNormalization: ['scale', 'bias'],
  lstm: ['bias', 'recurrentBias', 'peepholeWeight', 'initialHiddenState', 'initialCellState'],
  lstmCell: ['bias', 'recurrentBias', 'peepholeWeight']
};

function isOperandOption(opName, optKey) {
  const keys = OPTION_OPERAND_KEYS[opName];
  return keys && keys.includes(optKey);
}

function normalizeOptionKey(opName, key) {
  if (opName === 'cast' && key === 'type') return 'to';
  if (POOL2D_LIKE_OPS.has(opName) && key === 'roundingType') {
    return 'outputShapeRounding';
  }
  return key;
}

function normalizeValue(v) {
  if (typeof v === 'number' && !Number.isFinite(v)) {
    if (Number.isNaN(v)) return 'NaN';
    return v > 0 ? 'Infinity' : '-Infinity';
  }
  if (typeof v === 'bigint') return v.toString();
  if (Array.isArray(v)) return v.map(normalizeValue);
  if (v && typeof v === 'object') {
    const out = {};
    for (const [k, val] of Object.entries(v)) out[k] = normalizeValue(val);
    return out;
  }
  return v;
}

function isOperandRef(value, operandNames) {
  return typeof value === 'string' && operandNames.has(value);
}

function resolveOptionsObject(rawOptions, operandMap, operandNames, opName) {
  const out = {};
  for (const [key, value] of Object.entries(rawOptions ?? {})) {
    const optKey = normalizeOptionKey(opName, key);
    if (isOperandRef(value, operandNames)) {
      out[optKey] = operandMap.get(value);
      continue;
    }
    if (Array.isArray(value) && value.every((x) => isOperandRef(x, operandNames))) {
      out[optKey] = value.map((x) => operandMap.get(x));
      continue;
    }
    out[optKey] = normalizeValue(value);
  }
  return out;
}

function buildMethodArgs(opName, wptArguments, operandMap, operandNames) {
  const callArgs = [];
  let options = {};

  for (const arg of wptArguments ?? []) {
    for (const [key, value] of Object.entries(arg)) {
      if (key === 'options') {
        Object.assign(options, resolveOptionsObject(value, operandMap, operandNames, opName));
        continue;
      }

      if (isOperandRef(value, operandNames)) {
        if (isOperandOption(opName, key)) {
          options[normalizeOptionKey(opName, key)] = operandMap.get(value);
        } else {
          callArgs.push(operandMap.get(value));
        }
        continue;
      }

      if (Array.isArray(value) && value.every((x) => isOperandRef(x, operandNames))) {
        callArgs.push(value.map((x) => operandMap.get(x)));
        continue;
      }

      if (opName === 'cast' && key === 'type') {
        callArgs.push(value);
        continue;
      }

      if (isOperandOption(opName, key)) {
        throw new Error(`operand option '${key}' must reference a graph operand`);
      }

      callArgs.push(normalizeValue(value));
    }
  }

  if (Object.keys(options).length > 0) {
    callArgs.push(options);
  }

  return callArgs;
}

function typedArrayToPlainData(typed, dataType) {
  if (dataType === 'int4') {
    return Array.from(new Int8Array(typed.buffer, typed.byteOffset, typed.byteLength), (v) => v);
  }
  if (dataType === 'uint4') {
    return Array.from(typed);
  }
  if (dataType === 'uint64') {
    return Array.from(typed, (v) => (typeof v === 'bigint' ? v : BigInt(v)).toString());
  }
  if (dataType === 'int64') {
    return Array.from(typed, (v) => v.toString());
  }
  if (dataType === 'float16') {
    return Array.from(typed, (v) => Number(v));
  }
  return Array.from(typed);
}

function typedArrayCtor(dataType) {
  switch (dataType) {
    case 'float32':
      return Float32Array;
    case 'float16':
      return Float16Array;
    case 'int8':
      return Int8Array;
    case 'uint8':
      return Uint8Array;
    case 'uint4':
    case 'int4':
      return Uint8Array;
    case 'int32':
      return Int32Array;
    case 'uint32':
      return Uint32Array;
    case 'int64':
      return BigInt64Array;
    case 'uint64':
      return BigUint64Array;
    default:
      return Float32Array;
  }
}

function packRuntimeInputBuffer(input) {
  const data = Array.isArray(input.data) ? input.data : [input.data ?? 0];
  const fake = { descriptor: input.descriptor, data };
  return packConstantBuffer(fake);
}

/**
 * Build and execute a WPT graphResources object via webnnjs MLGraphBuilder.
 * @param {import('@webnnjs/webnn-node').MLContext} context
 * @param {object} graphResources
 * @returns {Promise<Record<string, { descriptor: object, data: unknown[] }>>}
 */
export async function executeGraphResources(context, graphResources) {
  const { MLGraphBuilder } = await import('@webnnjs/webnn-node');

  const builder = new MLGraphBuilder(context);
  const operandMap = new Map();
  const operandNames = new Set(Object.keys(graphResources.inputs ?? {}));
  const runtimeInputs = {};

  for (const [name, input] of Object.entries(graphResources.inputs ?? {})) {
    const descriptor = {
      dataType: input.descriptor.dataType,
      shape: [...input.descriptor.shape]
    };

    const useRuntimeInput = input.constant === true && !shouldInlineConstant(input);

    if (input.constant === true && !useRuntimeInput) {
      const buffer = packConstantBuffer(input);
      const operand = builder.constant(descriptor, buffer);
      operandMap.set(name, operand);
      continue;
    }

    const operand = builder.input(name, descriptor);
    operandMap.set(name, operand);
    runtimeInputs[name] = {
      descriptor,
      buffer: packRuntimeInputBuffer(input)
    };
  }

  for (const op of graphResources.operators ?? []) {
    const opName = op.name;
    const method = builder[opName];
    if (typeof method !== 'function') {
      throw new Error(`MLGraphBuilder.${opName} is not available`);
    }

    const callArgs = buildMethodArgs(opName, op.arguments, operandMap, operandNames);
    const result = method.apply(builder, callArgs);
    const outputNames = Array.isArray(op.outputs) ? op.outputs : [op.outputs];

    if (MULTI_OUTPUT_OPS.has(opName)) {
      if (!Array.isArray(result) || result.length !== outputNames.length) {
        throw new Error(
          `${opName}: expected ${outputNames.length} outputs, got ${Array.isArray(result) ? result.length : 1}`
        );
      }
      outputNames.forEach((outName, index) => {
        operandMap.set(outName, result[index]);
        operandNames.add(outName);
      });
      continue;
    }

    operandMap.set(outputNames[0], result);
    operandNames.add(outputNames[0]);
  }

  const buildOutputs = {};
  for (const name of Object.keys(graphResources.expectedOutputs ?? {})) {
    const operand = operandMap.get(name);
    if (!operand) {
      throw new Error(`missing operand for expected output '${name}'`);
    }
    buildOutputs[name] = operand;
  }

  const graph = await builder.build(buildOutputs);

  const inputTensors = {};
  const outputTensors = {};
  const tensorsToDestroy = [];

  try {
    for (const [name, payload] of Object.entries(runtimeInputs)) {
      const tensor = await context.createTensor({
        ...payload.descriptor,
        readable: false,
        writable: true
      });
      context.writeTensor(tensor, payload.buffer);
      inputTensors[name] = tensor;
      tensorsToDestroy.push(tensor);
    }

    for (const [name, expected] of Object.entries(graphResources.expectedOutputs ?? {})) {
      const tensor = await context.createTensor({
        dataType: expected.descriptor.dataType,
        shape: [...expected.descriptor.shape],
        readable: true,
        writable: true
      });
      outputTensors[name] = tensor;
      tensorsToDestroy.push(tensor);
    }

    context.dispatch(graph, inputTensors, outputTensors);

    const outputs = {};
    for (const [name, expected] of Object.entries(graphResources.expectedOutputs ?? {})) {
      const tensor = outputTensors[name];
      let data;
      try {
        const typed = await context.readTensorTyped(tensor);
        data = typedArrayToPlainData(typed, expected.descriptor.dataType);
      } catch (err) {
        const buffer = Buffer.from(await context.readTensor(tensor));
        const Ctor = typedArrayCtor(expected.descriptor.dataType);
        const typed = new Ctor(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        data = typedArrayToPlainData(typed, expected.descriptor.dataType);
      }
      outputs[name] = {
        descriptor: {
          dataType: expected.descriptor.dataType,
          shape: [...expected.descriptor.shape]
        },
        data
      };
    }

    return outputs;
  } finally {
    graph.destroy();
    for (const tensor of tensorsToDestroy) {
      tensor.destroy();
    }
  }
}
