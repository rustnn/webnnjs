import { snapshotDownload } from '@huggingface/hub';
import * as native from '@webnnjs/webnn-node-native';
import {
  installGeneratedBuilderMethods,
  type GeneratedMLGraphBuilderMethods,
} from './generated/builder-methods.js';

export * from './generated/builder-options.js';

export type MLPowerPreference = 'default' | 'high-performance' | 'low-power';

export interface MLContextOptions {
  powerPreference?: MLPowerPreference;
  accelerated?: boolean;
}

export type MLOperandDataType =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'uint32'
  | 'int64'
  | 'uint64'
  | 'int8'
  | 'uint8';

export interface MLOperandDescriptor {
  dataType: MLOperandDataType;
  shape: number[];
}

export interface MLTensorDescriptor extends MLOperandDescriptor {
  readable?: boolean;
  writable?: boolean;
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
}

export interface LoadModelFromHubResult {
  context: MLContext;
  graph: MLGraph;
  meta: LoadedModelMeta;
  snapshotPath: string;
}

function assertDescriptor(
  descriptor: MLTensorDescriptor | MLOperandDescriptor,
  api: string
): void {
  if (!descriptor || typeof descriptor !== 'object') {
    throw new TypeError(`${api}: descriptor must be an object`);
  }
  if (!descriptor.dataType || typeof descriptor.dataType !== 'string') {
    throw new TypeError(`${api}: descriptor.dataType is required`);
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

function descriptorToJson(descriptor: MLTensorDescriptor | MLOperandDescriptor): string {
  return JSON.stringify({
    dataType: descriptor.dataType,
    shape: descriptor.shape,
    readable: 'readable' in descriptor ? descriptor.readable : undefined,
    writable: 'writable' in descriptor ? descriptor.writable : undefined,
  });
}

function bufferFromArrayBufferView(data: ArrayBufferView): Buffer {
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
}

function typedArrayFromBuffer(
  buffer: Buffer,
  dataType: MLOperandDataType
): ArrayBufferView {
  if (buffer.byteLength === 0) {
    switch (dataType) {
      case 'float32':
        return new Float32Array();
      case 'int64':
        return new BigInt64Array();
      case 'int32':
        return new Int32Array();
      case 'uint32':
        return new Uint32Array();
      case 'float16':
        throw new TypeError('float16 readTensor is not supported yet');
      default:
        return new Uint8Array();
    }
  }

  switch (dataType) {
    case 'float32':
      return new Float32Array(
        buffer.buffer,
        buffer.byteOffset,
        Math.floor(buffer.byteLength / 4)
      );
    case 'int64':
      return new BigInt64Array(
        buffer.buffer,
        buffer.byteOffset,
        Math.floor(buffer.byteLength / 8)
      );
    case 'int32':
      return new Int32Array(
        buffer.buffer,
        buffer.byteOffset,
        Math.floor(buffer.byteLength / 4)
      );
    case 'uint32':
      return new Uint32Array(
        buffer.buffer,
        buffer.byteOffset,
        Math.floor(buffer.byteLength / 4)
      );
    case 'uint64':
      return new BigInt64Array(
        buffer.buffer,
        buffer.byteOffset,
        Math.floor(buffer.byteLength / 8)
      );
    case 'int8':
      return new Int8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    case 'uint8':
      return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    case 'float16':
      throw new TypeError('float16 readTensor is not supported yet');
    default:
      throw new TypeError(`unsupported readTensor dataType: ${dataType}`);
  }
}

export class MLOperand {
  constructor(
    readonly builder: MLGraphBuilder,
    readonly handle: number,
    readonly dataType: MLOperandDataType,
    readonly shape: number[]
  ) {}
}

export class MLTensor {
  private _destroyed = false;

  constructor(
    private readonly _context: MLContext,
    private readonly _handle: number,
    readonly dataType: MLOperandDataType,
    readonly shape: number[],
    readonly readable: boolean,
    readonly writable: boolean
  ) {}

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Tensor has been destroyed');
    }
    return this._handle;
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    native.destroyTensor(this._context.nativeHandle, this._handle);
  }
}

export class MLGraph {
  private _destroyed = false;

  constructor(
    readonly context: MLContext,
    private readonly _handle: number,
    readonly meta?: LoadedModelMeta
  ) {}

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Graph has been destroyed');
    }
    return this._handle;
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    native.destroyGraph(this.context.nativeHandle, this._handle);
  }
}

export class MLContext {
  private _destroyed = false;

  constructor(private readonly _handle: number) {}

  get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Context has been destroyed');
    }
    return this._handle;
  }

  get accelerated(): boolean {
    return native.contextAccelerated(this.nativeHandle);
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    native.destroyContext(this._handle);
  }

  async createTensor(descriptor: MLTensorDescriptor): Promise<MLTensor> {
    assertDescriptor(descriptor, 'createTensor');
    const handle = native.createTensor(
      this.nativeHandle,
      descriptorToJson(descriptor)
    );
    return new MLTensor(
      this,
      handle,
      descriptor.dataType,
      [...descriptor.shape],
      descriptor.readable ?? false,
      descriptor.writable ?? false
    );
  }

  writeTensor(tensor: MLTensor, data: ArrayBufferView | ArrayBuffer): void {
    if (!(tensor instanceof MLTensor)) {
      throw new TypeError('writeTensor: tensor must be an MLTensor');
    }
    const buffer =
      data instanceof ArrayBuffer
        ? Buffer.from(data)
        : bufferFromArrayBufferView(data);
    native.writeTensor(this.nativeHandle, tensor.nativeHandle, buffer);
  }

  async readTensor(tensor: MLTensor): Promise<ArrayBuffer> {
    if (!(tensor instanceof MLTensor)) {
      throw new TypeError('readTensor: tensor must be an MLTensor');
    }
    const bytes = native.readTensor(this.nativeHandle, tensor.nativeHandle);
    const copy = Buffer.allocUnsafe(bytes.byteLength);
    bytes.copy(copy);
    return copy.buffer.slice(copy.byteOffset, copy.byteOffset + copy.byteLength);
  }

  async readTensorTyped(tensor: MLTensor): Promise<ArrayBufferView> {
    const buffer = Buffer.from(await this.readTensor(tensor));
    return typedArrayFromBuffer(buffer, tensor.dataType);
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

    native.dispatch(
      this.nativeHandle,
      graph.nativeHandle,
      JSON.stringify(inputHandles),
      JSON.stringify(outputHandles)
    );
  }

  /** rustnn extension: resize tensor logical shape before dispatch. */
  rustnnResizeTensor(tensor: MLTensor, shape: number[]): void {
    native.rustnnResizeTensor(
      this.nativeHandle,
      tensor.nativeHandle,
      JSON.stringify(shape)
    );
  }

  /** rustnn extension: pre-size tensor storage for dynamic shapes. */
  rustnnSetTensorCapacity(tensor: MLTensor, maxShape: number[]): void {
    native.rustnnSetTensorCapacity(
      this.nativeHandle,
      tensor.nativeHandle,
      JSON.stringify(maxShape)
    );
  }

  async loadModel(pathOrDir: string): Promise<{ graph: MLGraph; meta: LoadedModelMeta }> {
    if (typeof pathOrDir !== 'string' || pathOrDir.length === 0) {
      throw new TypeError('loadModel: pathOrDir must be a non-empty string');
    }

    const result = await native.loadWebnnModel(this.nativeHandle, pathOrDir);
    const meta = JSON.parse(result.metaJson) as LoadedModelMeta;
    return {
      graph: new MLGraph(this, result.graphHandle, meta),
      meta,
    };
  }
}

export interface MLGraphBuilder extends GeneratedMLGraphBuilderMethods {}

export class MLGraphBuilder {
  private _destroyed = false;
  private readonly _handle: number;

  constructor(readonly context: MLContext) {
    if (!(context instanceof MLContext)) {
      throw new TypeError('MLGraphBuilder constructor requires an MLContext');
    }
    this._handle = native.createGraphBuilder(context.nativeHandle);
  }

  private get nativeHandle(): number {
    if (this._destroyed) {
      throw new Error('Graph builder has been destroyed');
    }
    return this._handle;
  }

  input(name: string, descriptor: MLOperandDescriptor): MLOperand {
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('input: name must be a non-empty string');
    }
    assertDescriptor(descriptor, 'input');
    const handle = native.builderInput(
      this.nativeHandle,
      name,
      descriptorToJson(descriptor)
    );
    return new MLOperand(this, handle, descriptor.dataType, [...descriptor.shape]);
  }

  constant(descriptor: MLOperandDescriptor, buffer: ArrayBufferView): MLOperand;
  constant(dataType: MLOperandDataType, value: number): MLOperand;
  constant(tensor: MLTensor): MLOperand;
  constant(
    descriptorOrType: MLOperandDescriptor | MLOperandDataType | MLTensor,
    bufferOrValue?: ArrayBufferView | number
  ): MLOperand {
    if (typeof descriptorOrType === 'string') {
      throw new TypeError(
        'constant(dataType, value): scalar constants are not implemented in rustnn yet'
      );
    }

    if (descriptorOrType instanceof MLTensor) {
      throw new TypeError(
        'constant(tensor): tensor constants are not implemented in rustnn yet'
      );
    }

    if (!bufferOrValue || ArrayBuffer.isView(bufferOrValue) === false) {
      throw new TypeError(
        'constant(descriptor, buffer): buffer must be an ArrayBufferView'
      );
    }

    assertDescriptor(descriptorOrType, 'constant');
    const resultJson = native.builderConstantBuffer(
      this.nativeHandle,
      descriptorToJson(descriptorOrType),
      bufferFromArrayBufferView(bufferOrValue)
    );
    return this._operandFromInvokeJson(resultJson);
  }

  _invokeOp(multi: false, wire: Record<string, unknown>): MLOperand;
  _invokeOp(multi: true, wire: Record<string, unknown>): MLOperand[];
  _invokeOp(
    multi: boolean,
    wire: Record<string, unknown>
  ): MLOperand | MLOperand[] {
    const resultJson = native.builderInvoke(
      this.nativeHandle,
      JSON.stringify(wire)
    );
    const result = JSON.parse(resultJson) as {
      operands: Array<{
        handle: number;
        dataType: MLOperandDataType;
        shape: number[];
      }>;
    };
    const mapped = result.operands.map(
      (o) => new MLOperand(this, o.handle, o.dataType, o.shape)
    );
    return multi ? mapped : mapped[0];
  }

  private _operandFromInvokeJson(resultJson: string): MLOperand {
    const result = JSON.parse(resultJson) as {
      operands: Array<{
        handle: number;
        dataType: MLOperandDataType;
        shape: number[];
      }>;
    };
    const o = result.operands[0];
    return new MLOperand(this, o.handle, o.dataType, o.shape);
  }

  async build(outputs: Record<string, MLOperand>): Promise<MLGraph> {
    if (!outputs || typeof outputs !== 'object') {
      throw new TypeError('build: outputs must be an object');
    }

    const outputHandles: Record<string, number> = {};
    for (const [name, operand] of Object.entries(outputs)) {
      this._assertOperand(operand, `build output '${name}'`);
      outputHandles[name] = operand.handle;
    }

    const graphHandle = await native.builderBuild(
      this.context.nativeHandle,
      this.nativeHandle,
      JSON.stringify(outputHandles)
    );
    this._destroyed = true;
    return new MLGraph(this.context, graphHandle);
  }

  destroy(): void {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    native.destroyGraphBuilder(this._handle);
  }

  private _assertOperand(operand: MLOperand, label: string): void {
    if (!(operand instanceof MLOperand) || operand.builder !== this) {
      throw new TypeError(`${label}: operand does not belong to this builder`);
    }
  }
}

installGeneratedBuilderMethods(MLGraphBuilder.prototype);

class MLNamespace {
  async createContext(options: MLContextOptions = {}): Promise<MLContext> {
    if (options && typeof options !== 'object') {
      throw new TypeError('createContext: options must be an object');
    }

    const handle = native.createContext(JSON.stringify(options ?? {}));
    return new MLContext(handle);
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
    const { graph, meta } = await context.loadModel(snapshotPath);

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
