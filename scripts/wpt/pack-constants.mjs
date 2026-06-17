function shapeElementCount(shape) {
  if (!shape || shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}

function bytesPerElement(dataType) {
  switch (dataType) {
    case 'float32':
    case 'int32':
    case 'uint32':
      return 4;
    case 'float16':
      return 2;
    case 'int64':
    case 'uint64':
      return 8;
    default:
      return 1;
  }
}

/** rustnn packed int4/uint4 storage: even indices in high nibble, odd in low. */
export function packedStorageByteLength(elements) {
  const n = Math.max(1, elements);
  return Math.max(1, (4 * n + 7) >> 3);
}

function storageByteCount(dataType, elementCount) {
  if (dataType === 'int4' || dataType === 'uint4') {
    return packedStorageByteLength(elementCount);
  }
  return Math.max(1, elementCount) * bytesPerElement(dataType);
}

function constantRawLength(raw) {
  if (Array.isArray(raw)) return raw.length;
  return raw == null ? 0 : 1;
}

const LARGE_SCALAR_INLINE_BYTES_THRESHOLD = 8 * 1024 * 1024;

/**
 * Match rustnnpt build-graph-json: large scalar-fill constants are runtime inputs,
 * not inlined graph constants.
 */
export function shouldInlineConstant(input) {
  if (input.constant !== true) return false;
  const dt = input?.descriptor?.dataType ?? 'float32';
  const shape = input?.descriptor?.shape ?? [];
  const rawLen = constantRawLength(input.data);
  const scalarFillLike = rawLen <= 1;
  const estBytes = storageByteCount(dt, shapeElementCount(shape));
  return !(scalarFillLike && estBytes >= LARGE_SCALAR_INLINE_BYTES_THRESHOLD);
}

function parseNumericLoose(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'bigint') return Number(v);
  if (typeof v === 'string') {
    const t = v.trim();
    if (t === 'NaN') return NaN;
    if (t === 'Infinity' || t === '+Infinity') return Infinity;
    if (t === '-Infinity') return -Infinity;
    const noN = t.endsWith('n') ? t.slice(0, -1) : t;
    return Number(noN);
  }
  return Number(v);
}

function int4Nibble(value) {
  const v = Math.trunc(parseNumericLoose(value));
  return Math.max(-8, Math.min(7, v)) & 0x0f;
}

function uint4Nibble(value) {
  const v = Math.trunc(parseNumericLoose(value));
  return Math.max(0, Math.min(15, v)) & 0x0f;
}

/** @param {ArrayLike<number>} values */
export function packInt4(values) {
  const n = values.length;
  const out = new Uint8Array(packedStorageByteLength(n));
  for (let i = 0; i < n; i++) {
    const nibble = int4Nibble(values[i]);
    if (i % 2 === 0) {
      out[i >> 1] = nibble << 4;
    } else {
      out[i >> 1] |= nibble;
    }
  }
  return out;
}

/** @param {ArrayLike<number>} values */
export function packUint4(values) {
  const n = values.length;
  const out = new Uint8Array(packedStorageByteLength(n));
  for (let i = 0; i < n; i++) {
    const nibble = uint4Nibble(values[i]);
    if (i % 2 === 0) {
      out[i >> 1] = nibble << 4;
    } else {
      out[i >> 1] |= nibble;
    }
  }
  return out;
}

/** @param {Uint8Array} data */
export function unpackInt4(data, elementCount) {
  const out = new Array(elementCount);
  for (let i = 0; i < elementCount; i++) {
    const byte = data[i >> 1] ?? 0;
    const nibble = i % 2 === 0 ? (byte >> 4) & 0x0f : byte & 0x0f;
    out[i] = nibble >= 8 ? nibble - 16 : nibble;
  }
  return out;
}

/** @param {Uint8Array} data */
export function unpackUint4(data, elementCount) {
  const out = new Array(elementCount);
  for (let i = 0; i < elementCount; i++) {
    const byte = data[i >> 1] ?? 0;
    out[i] = i % 2 === 0 ? (byte >> 4) & 0x0f : byte & 0x0f;
  }
  return out;
}

/**
 * Pack WPT tensor values to a typed array for MLGraphBuilder.constant() / writeTensor().
 * int4/uint4 use rustnn nibble packing (two logical values per byte).
 * @param {{ descriptor: { dataType: string, shape: number[] }, data?: unknown }} input
 * @returns {ArrayBufferView}
 */
export function packConstantBuffer(input) {
  const dt = input.descriptor.dataType;
  const shape = input.descriptor.shape ?? [];
  const n = Math.max(1, shapeElementCount(shape));
  let raw = input.data;
  if (raw == null || (Array.isArray(raw) && raw.length === 0)) {
    raw = n === 1 ? [0] : new Array(n).fill(0);
  }

  const arr = Array.isArray(raw) ? raw : [raw];
  const getNorm = (i) => {
    const v = arr.length === 1 ? arr[0] : arr[i];
    if (typeof v === 'bigint') return v;
    if (typeof v === 'string' && (v.endsWith('n') || v === 'NaN' || v.includes('Infinity'))) {
      return parseNumericLoose(v);
    }
    return v;
  };

  switch (dt) {
    case 'float32': {
      const ta = new Float32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return ta;
    }
    case 'float16': {
      const ta = new Float16Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return ta;
    }
    case 'int8': {
      const ta = new Int8Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) | 0;
      return ta;
    }
    case 'uint8': {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = parseNumericLoose(getNorm(i)) & 0xff;
      return out;
    }
    case 'int4': {
      const values = new Array(n);
      for (let i = 0; i < n; i++) values[i] = parseNumericLoose(getNorm(i));
      return packInt4(values);
    }
    case 'uint4': {
      const values = new Array(n);
      for (let i = 0; i < n; i++) values[i] = parseNumericLoose(getNorm(i));
      return packUint4(values);
    }
    case 'int32': {
      const ta = new Int32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) | 0;
      return ta;
    }
    case 'uint32': {
      const ta = new Uint32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i)) >>> 0;
      return ta;
    }
    case 'int64': {
      const ta = new BigInt64Array(n);
      for (let i = 0; i < n; i++) {
        const v = getNorm(i);
        ta[i] = typeof v === 'bigint' ? v : BigInt(Math.trunc(parseNumericLoose(v)));
      }
      return ta;
    }
    case 'uint64': {
      const ta = new BigUint64Array(n);
      for (let i = 0; i < n; i++) {
        const v = getNorm(i);
        ta[i] =
          typeof v === 'bigint'
            ? v
            : BigInt.asUintN(64, BigInt(Math.trunc(parseNumericLoose(v))));
      }
      return ta;
    }
    default: {
      const ta = new Float32Array(n);
      for (let i = 0; i < n; i++) ta[i] = parseNumericLoose(getNorm(i));
      return ta;
    }
  }
}
