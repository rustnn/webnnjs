import { snapshotDownload } from '@huggingface/hub';
import * as native from '@webnnjs/webnn-node-native';

export type MLDeviceType = 'cpu' | 'gpu';
export type MLPowerPreference = 'default' | 'low-power' | 'high-performance';

export interface MLContextOptions {
  deviceType?: MLDeviceType;
  powerPreference?: MLPowerPreference;
  accelerated?: boolean;
}

export interface MLTensorDescriptor {
  dataType: 'float32';
  shape: number[];
}

interface BuilderOperand {
  id: number;
  descriptor: MLTensorDescriptor;
  kind: 'input' | 'constant' | 'intermediate';
  name?: string;
  constantData?: Float32Array;
}

interface BuilderOperation {
  type: 'add' | 'mul' | 'identity';
  inputOperands: number[];
  outputOperand: number;
}

interface NativeLoadModelMeta {
  graphPath: string;
  inputNames: string[];
  outputNames: string[];
  inputs: Record<string, LoadModelTensorMeta>;
  outputs: Record<string, LoadModelTensorMeta>;
  warnings: string[];
}

export interface LoadModelTensorMeta {
  dataType: string;
  shape: number[];
}

export interface LoadedModelMeta {
  graphPath: string;
  inputNames: string[];
  outputNames: string[];
  inputs: Record<string, LoadModelTensorMeta>;
  outputs: Record<string, LoadModelTensorMeta>;
  warnings: string[];
}

export interface LoadModelFromHubResult {
  context: MLContext;
  graph: MLGraph;
  meta: LoadedModelMeta;
  snapshotPath: string;
}

function assertFloat32Descriptor(
  descriptor: MLTensorDescriptor,
  api: string
): asserts descriptor is MLTensorDescriptor {
  if (!descriptor || typeof descriptor !== 'object') {
    throw new TypeError(`${api}: descriptor must be an object`);
  }

  if (descriptor.dataType !== 'float32') {
    throw new TypeError(`${api}: only dataType='float32' is supported in MVP`);
  }

  if (!Array.isArray(descriptor.shape)) {
    throw new TypeError(`${api}: descriptor.shape must be an array`);
  }

  for (const dim of descriptor.shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new TypeError(`${api}: shape dimensions must be non-negative integers`);
    }
  }
}

function checkedElementCount(shape: number[]): number {
  if (shape.length === 0) {
    return 1;
  }

  let total = 1;
  for (const dim of shape) {
    total *= dim;
  }
  return total;
}

function normalizeFloat32Data(
  data: ArrayBufferView | ArrayBuffer | ArrayLike<number>,
  api: string
): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }

  if (ArrayBuffer.isView(data)) {
    return Float32Array.from(data as unknown as ArrayLike<number>);
  }

  if (data instanceof ArrayBuffer) {
    if (data.byteLength % 4 !== 0) {
      throw new TypeError(`${api}: ArrayBuffer byteLength must be divisible by 4`);
    }
    return new Float32Array(data);
  }

  if (typeof data === 'object' && data !== null && 'length' in data) {
    return Float32Array.from(data as ArrayLike<number>);
  }

  throw new TypeError(
    `${api}: data must be Float32Array, ArrayBufferView, ArrayBuffer, or number[]`
  );
}

function bufferFromFloat32(view: Float32Array): Buffer {
  return Buffer.from(view.buffer, view.byteOffset, view.byteLength);
}

function float32FromBuffer(buffer: Buffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error('native readTensor returned invalid byteLength for float32 buffer');
  }

  const view = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    Math.floor(buffer.byteLength / 4)
  );
  return new Float32Array(view);
}

const contextFinalizer = new FinalizationRegistry<number>((contextHandle) => {
  try {
    native.destroyContext(contextHandle);
  } catch {
    // Best-effort cleanup during GC.
  }
});

const graphFinalizer = new FinalizationRegistry<{
  contextHandle: number;
  graphHandle: number;
}>(({ contextHandle, graphHandle }) => {
  try {
    native.destroyGraph(contextHandle, graphHandle);
  } catch {
    // Best-effort cleanup during GC.
  }
});

const tensorFinalizer = new FinalizationRegistry<{
  contextHandle: number;
  tensorHandle: number;
}>(({ contextHandle, tensorHandle }) => {
  try {
    native.destroyTensor(contextHandle, tensorHandle);
  } catch {
    // Best-effort cleanup during GC.
  }
});

export class MLTensor {
  private _destroyed = false;

  constructor(
    private readonly _context: MLContext,
    private readonly _nativeHandle: number,
    private _descriptor: MLTensorDescriptor
  ) {
    tensorFinalizer.register(
      this,
      {
        contextHandle: _context.nativeHandle,
        tensorHandle: _nativeHandle,
      },
      this
    );
  }

  get descriptor(): MLTensorDescriptor {
    return {
      dataType: this._descriptor.dataType,
      shape: [...this._descriptor.shape],
    };
  }

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Tensor has been destroyed');
    }
    return this._nativeHandle;
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    tensorFinalizer.unregister(this);
    native.destroyTensor(this._context.nativeHandle, this._nativeHandle);
  }
}

export class MLGraph {
  private _destroyed = false;

  constructor(
    readonly context: MLContext,
    private readonly _nativeHandle: number,
    readonly meta?: LoadedModelMeta
  ) {
    graphFinalizer.register(
      this,
      {
        contextHandle: context.nativeHandle,
        graphHandle: _nativeHandle,
      },
      this
    );
  }

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Graph has been destroyed');
    }
    return this._nativeHandle;
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    graphFinalizer.unregister(this);
    native.destroyGraph(this.context.nativeHandle, this._nativeHandle);
  }
}

export class MLContext {
  private _destroyed = false;
  private _dispatchFence: Promise<void> = Promise.resolve();

  constructor(private readonly _nativeHandle: number) {
    contextFinalizer.register(this, _nativeHandle, this);
  }

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Context has been destroyed');
    }
    return this._nativeHandle;
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    contextFinalizer.unregister(this);
    native.destroyContext(this._nativeHandle);
  }

  async createTensor(descriptor: MLTensorDescriptor): Promise<MLTensor> {
    assertFloat32Descriptor(descriptor, 'createTensor');

    const handle = native.createTensor(
      this.nativeHandle,
      JSON.stringify(descriptor)
    );

    return new MLTensor(this, handle, {
      dataType: 'float32',
      shape: [...descriptor.shape],
    });
  }

  writeTensor(
    tensor: MLTensor,
    data: ArrayBufferView | ArrayBuffer | ArrayLike<number>
  ): void {
    if (!(tensor instanceof MLTensor)) {
      throw new TypeError('writeTensor: tensor must be an MLTensor');
    }

    const values = normalizeFloat32Data(data, 'writeTensor');
    const expected = checkedElementCount(tensor.descriptor.shape);
    if (values.length !== expected) {
      throw new TypeError(
        `writeTensor: expected ${expected} elements for shape ${JSON.stringify(
          tensor.descriptor.shape
        )}, got ${values.length}`
      );
    }

    native.writeTensor(
      this.nativeHandle,
      tensor.nativeHandle,
      bufferFromFloat32(values)
    );
  }

  async readTensor(tensor: MLTensor): Promise<Float32Array> {
    if (!(tensor instanceof MLTensor)) {
      throw new TypeError('readTensor: tensor must be an MLTensor');
    }

    await this._dispatchFence;
    const bytes = native.readTensor(this.nativeHandle, tensor.nativeHandle);
    return float32FromBuffer(bytes);
  }

  dispatch(
    graph: MLGraph,
    inputs: Record<string, MLTensor>,
    outputs: Record<string, MLTensor>
  ): void {
    if (!(graph instanceof MLGraph)) {
      throw new TypeError('dispatch: graph must be an MLGraph');
    }

    const inputHandles: Record<string, number> = {};
    const outputHandles: Record<string, number> = {};

    for (const [name, tensor] of Object.entries(inputs)) {
      if (!(tensor instanceof MLTensor)) {
        throw new TypeError(`dispatch: input '${name}' must be an MLTensor`);
      }
      inputHandles[name] = tensor.nativeHandle;
    }

    for (const [name, tensor] of Object.entries(outputs)) {
      if (!(tensor instanceof MLTensor)) {
        throw new TypeError(`dispatch: output '${name}' must be an MLTensor`);
      }
      outputHandles[name] = tensor.nativeHandle;
    }

    const runDispatch = (): Promise<void> =>
      native.dispatch(
        this.nativeHandle,
        graph.nativeHandle,
        JSON.stringify(inputHandles),
        JSON.stringify(outputHandles)
      );

    this._dispatchFence = this._dispatchFence.then(runDispatch, runDispatch);
  }

  async loadModel(
    pathOrDir: string,
    options: MLContextOptions = {}
  ): Promise<{ graph: MLGraph; meta: LoadedModelMeta }> {
    if (typeof pathOrDir !== 'string' || pathOrDir.length === 0) {
      throw new TypeError('loadModel: pathOrDir must be a non-empty string');
    }

    const result = await native.loadWebnnModel(
      this.nativeHandle,
      pathOrDir,
      JSON.stringify(options)
    );

    const meta = JSON.parse(result.metaJson) as NativeLoadModelMeta;
    const normalizedMeta: LoadedModelMeta = {
      graphPath: meta.graphPath,
      inputNames: [...meta.inputNames],
      outputNames: [...meta.outputNames],
      inputs: { ...meta.inputs },
      outputs: { ...meta.outputs },
      warnings: [...meta.warnings],
    };

    return {
      graph: new MLGraph(this, result.graphHandle, normalizedMeta),
      meta: normalizedMeta,
    };
  }

  async _waitForDispatches(): Promise<void> {
    await this._dispatchFence;
  }
}

export class MLGraphBuilder {
  private _nextOperandId = 0;
  private readonly _operands = new Map<number, BuilderOperand>();
  private readonly _operations: BuilderOperation[] = [];

  constructor(private readonly _context: MLContext) {}

  input(name: string, descriptor: MLTensorDescriptor): BuilderOperand {
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('input: name must be a non-empty string');
    }
    assertFloat32Descriptor(descriptor, 'input');

    const operand: BuilderOperand = {
      id: this._nextOperandId++,
      descriptor: {
        dataType: descriptor.dataType,
        shape: [...descriptor.shape],
      },
      kind: 'input',
      name,
    };

    this._operands.set(operand.id, operand);
    return operand;
  }

  constant(
    descriptor: MLTensorDescriptor,
    data: ArrayBufferView | ArrayBuffer | ArrayLike<number>
  ): BuilderOperand {
    assertFloat32Descriptor(descriptor, 'constant');
    const values = normalizeFloat32Data(data, 'constant');

    const expected = checkedElementCount(descriptor.shape);
    if (values.length !== expected) {
      throw new TypeError(
        `constant: expected ${expected} elements for shape ${JSON.stringify(
          descriptor.shape
        )}, got ${values.length}`
      );
    }

    const operand: BuilderOperand = {
      id: this._nextOperandId++,
      descriptor: {
        dataType: descriptor.dataType,
        shape: [...descriptor.shape],
      },
      kind: 'constant',
      constantData: values,
    };

    this._operands.set(operand.id, operand);
    return operand;
  }

  add(a: BuilderOperand, b: BuilderOperand): BuilderOperand {
    return this._binaryOp('add', a, b);
  }

  mul(a: BuilderOperand, b: BuilderOperand): BuilderOperand {
    return this._binaryOp('mul', a, b);
  }

  private _binaryOp(
    type: 'add' | 'mul',
    a: BuilderOperand,
    b: BuilderOperand
  ): BuilderOperand {
    this._assertOperand(a, '_binaryOp(a)');
    this._assertOperand(b, '_binaryOp(b)');

    if (a.descriptor.dataType !== 'float32' || b.descriptor.dataType !== 'float32') {
      throw new TypeError(`${type}: only float32 operands are supported`);
    }

    if (a.descriptor.shape.length !== b.descriptor.shape.length) {
      throw new TypeError(
        `${type}: rank mismatch (${a.descriptor.shape.length} vs ${b.descriptor.shape.length})`
      );
    }

    for (let i = 0; i < a.descriptor.shape.length; i += 1) {
      if (a.descriptor.shape[i] !== b.descriptor.shape[i]) {
        throw new TypeError(
          `${type}: shape mismatch (${JSON.stringify(
            a.descriptor.shape
          )} vs ${JSON.stringify(b.descriptor.shape)})`
        );
      }
    }

    const output: BuilderOperand = {
      id: this._nextOperandId++,
      descriptor: {
        dataType: 'float32',
        shape: [...a.descriptor.shape],
      },
      kind: 'intermediate',
    };

    this._operands.set(output.id, output);
    this._operations.push({
      type,
      inputOperands: [a.id, b.id],
      outputOperand: output.id,
    });

    return output;
  }

  private _assertOperand(operand: BuilderOperand, label: string): void {
    if (!operand || typeof operand !== 'object' || typeof operand.id !== 'number') {
      throw new TypeError(`${label}: operand is invalid`);
    }

    if (!this._operands.has(operand.id)) {
      throw new TypeError(`${label}: operand does not belong to this builder`);
    }
  }

  async build(outputs: Record<string, BuilderOperand>): Promise<MLGraph> {
    if (!outputs || typeof outputs !== 'object') {
      throw new TypeError('build: outputs must be an object');
    }

    const outputEntries = Object.entries(outputs);
    if (outputEntries.length === 0) {
      throw new TypeError('build: outputs must have at least one entry');
    }

    if (this._operations.length === 0) {
      throw new TypeError('build: graph has no operations');
    }

    type IrOperand = {
      kind: 'input' | 'constant' | 'output';
      descriptor: {
        data_type: 'float32';
        shape: number[];
        pending_permutation: number[];
      };
      name?: string;
    };

    const irOperands: IrOperand[] = [];
    const inputOperands: number[] = [];
    const outputOperands: number[] = [];
    const irOperations: Array<{
      type: 'add' | 'mul' | 'identity';
      input_operands: number[];
      output_operand: number;
      attributes: Record<string, never>;
    }> = [];
    const constantOperandIdsToHandles: Record<string, { data: string }> = {};

    const sortedOperandIds = [...this._operands.keys()].sort((a, b) => a - b);
    for (const operandId of sortedOperandIds) {
      const operand = this._operands.get(operandId);
      if (!operand) {
        continue;
      }

      irOperands[operand.id] = {
        kind: operand.kind === 'input' ? 'input' : operand.kind === 'constant' ? 'constant' : 'output',
        descriptor: {
          data_type: 'float32',
          shape: [...operand.descriptor.shape],
          pending_permutation: [],
        },
      };

      if (operand.kind === 'input') {
        irOperands[operand.id].name = operand.name;
        inputOperands.push(operand.id);
      }

      if (operand.kind === 'constant') {
        const data = operand.constantData;
        if (!data) {
          throw new Error(`build: missing constant data for operand ${operand.id}`);
        }

        constantOperandIdsToHandles[String(operand.id)] = {
          data: Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString(
            'base64'
          ),
        };
      }
    }

    for (const op of this._operations) {
      irOperations.push({
        type: op.type,
        input_operands: [...op.inputOperands],
        output_operand: op.outputOperand,
        attributes: {},
      });
    }

    const claimedOutputOperandNames = new Map<number, string>();

    for (const [name, operand] of outputEntries) {
      if (typeof name !== 'string' || name.length === 0) {
        throw new TypeError('build: output names must be non-empty strings');
      }
      this._assertOperand(operand, `build output '${name}'`);

      const existingName = claimedOutputOperandNames.get(operand.id);
      if (!existingName) {
        claimedOutputOperandNames.set(operand.id, name);
        irOperands[operand.id].name = name;
        outputOperands.push(operand.id);
      } else {
        const duplicateOutputId = this._nextOperandId++;
        irOperands[duplicateOutputId] = {
          kind: 'output',
          name,
          descriptor: {
            data_type: 'float32',
            shape: [...operand.descriptor.shape],
            pending_permutation: [],
          },
        };

        irOperations.push({
          type: 'identity',
          input_operands: [operand.id],
          output_operand: duplicateOutputId,
          attributes: {},
        });
        outputOperands.push(duplicateOutputId);
      }
    }

    const ir = {
      operands: irOperands,
      input_operands: inputOperands,
      output_operands: outputOperands,
      operations: irOperations,
      constant_operand_ids_to_handles: constantOperandIdsToHandles,
      id_to_constant_tensor_operand_map: {},
      quantized: false,
    };

    const graphHandle = await native.compileGraph(
      this._context.nativeHandle,
      JSON.stringify(ir)
    );

    return new MLGraph(this._context, graphHandle);
  }
}

class MLNamespace {
  async createContext(options: MLContextOptions = {}): Promise<MLContext> {
    if (options && typeof options !== 'object') {
      throw new TypeError('createContext: options must be an object');
    }

    const contextHandle = native.createContext(JSON.stringify(options ?? {}));
    return new MLContext(contextHandle);
  }

  async loadModelFromHub(
    repoId: string,
    options: MLContextOptions & {
      revision?: string;
      allowPatterns?: string[];
      ignorePatterns?: string[];
    } = {}
  ): Promise<LoadModelFromHubResult> {
    if (typeof repoId !== 'string' || repoId.length === 0) {
      throw new TypeError('loadModelFromHub: repoId must be a non-empty string');
    }

    const context = await this.createContext(options);

    const downloadArgs: {
      repo: string;
      revision?: string;
      allowPatterns?: string[];
      ignorePatterns?: string[];
    } = {
      repo: repoId,
    };

    if (options.revision) {
      downloadArgs.revision = options.revision;
    }
    if (options.allowPatterns) {
      downloadArgs.allowPatterns = options.allowPatterns;
    }
    if (options.ignorePatterns) {
      downloadArgs.ignorePatterns = options.ignorePatterns;
    }

    const snapshotPath = await snapshotDownload(downloadArgs as any);
    const { graph, meta } = await context.loadModel(snapshotPath, options);

    return {
      context,
      graph,
      meta,
      snapshotPath,
    };
  }
}

export const ml = new MLNamespace();

export function installWebNNPolyfill(): void {
  const globalObject = globalThis as Record<string, unknown>;
  if (!globalObject.navigator || typeof globalObject.navigator !== 'object') {
    globalObject.navigator = {};
  }

  const navigatorObject = globalObject.navigator as Record<string, unknown>;
  if (!navigatorObject.ml) {
    navigatorObject.ml = ml;
  }
}
