// IEEE float bits reinterpreted as integers for ordered ULP distance (see Hamming / ordering trick).
const F32_SIGN_MASK = 0x8000_0000;
/** Exponent + fraction: all bits except the sign bit. */
const F32_NOT_SIGN_MASK = 0x7fff_ffff;
const F16_SIGN_MASK = 0x8000;
/** Exponent + fraction: all bits except the sign bit. */
const F16_NOT_SIGN_MASK = 0x7fff;

/** `f32` and `u32` share one 4-byte buffer; do not reuse concurrently across overlapping calls. */
/** @param {number} v @param {{ f32: Float32Array, u32: Uint32Array }} scratch */
function float32Bits(v, scratch) {
  scratch.f32[0] = v;
  return scratch.u32[0];
}

/** @param {number} a @param {number} b @param {{ f32: Float32Array, u32: Uint32Array }} scratch */
function ulpDistanceF32(a, b, scratch) {
  if (Object.is(a, b)) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  const aBits = float32Bits(a, scratch);
  const bBits = float32Bits(b, scratch);
  const toOrdered = (bits) =>
    bits & F32_SIGN_MASK ? F32_SIGN_MASK - (bits & F32_NOT_SIGN_MASK) : bits + F32_SIGN_MASK;
  return Math.abs(toOrdered(aBits) - toOrdered(bBits));
}

/** `f16` and `u16` share one 2-byte buffer; do not reuse concurrently across overlapping calls. */
/** @param {number} v @param {{ f16: Float16Array, u16: Uint16Array }} scratch */
function float16Bits(v, scratch) {
  scratch.f16[0] = v;
  return scratch.u16[0];
}

/** ULP distance in IEEE binary16; required for float16 outputs (f32 ULP inflates ~1 f16 step to thousands). */
/** @param {number} a @param {number} b @param {{ f16: Float16Array, u16: Uint16Array }} scratch */
function ulpDistanceF16(a, b, scratch) {
  if (Object.is(a, b)) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return a === b ? 0 : Number.POSITIVE_INFINITY;
  }
  const aBits = float16Bits(a, scratch);
  const bBits = float16Bits(b, scratch);
  const toOrdered = (bits) =>
    bits & F16_SIGN_MASK ? F16_SIGN_MASK - (bits & F16_NOT_SIGN_MASK) : bits + F16_SIGN_MASK;
  return Math.abs(toOrdered(aBits) - toOrdered(bBits));
}

const OP_ULP = {
  add: 1,
  sub: 1,
  mul: 1,
  div: 2,
  relu: 0,
  sigmoid: 34,
  tanh: 16,
  softmax: 256,
  matmul: 512,
  conv2d: 16384,
  conv_transpose2d: 16384,
  exp: 4,
  log: 4,
  sqrt: 2,
  reduce_sum: 8,
  reduce_mean: 16,
  reduce_max: 0,
  reduce_min: 0,
  reduce_product: 32,
  reduce_l1: 8,
  reduce_l2: 16,
  reduce_log_sum: 16,
  reduce_log_sum_exp: 32,
  reduce_sum_square: 16,
  instance_normalization: 12,
  layer_normalization: 16
};

const OP_ABS_TOL = {
  cos: { float32: 2 ** -10, float16: 2 ** -7 },
  sin: { float32: 2 ** -11, float16: 2 ** -7 },
  conv2d: { float32: 5e-4, float16: 1e-2 },
  conv_transpose2d: { float32: 5e-4, float16: 1e-2 }
};

function normalizeOpName(opName) {
  return opName.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
}

/**
 * @param {string} primaryOp normalized op name (typically last node)
 * @param {string[] | undefined} allOps normalized names of all operators in order
 * @param {string} dataType
 * @returns {{ ulpTol: number, absTol: number }}
 */
function mergedFloatTolerance(primaryOp, allOps, dataType) {
  let ulpTol = OP_ULP[primaryOp];
  if (ulpTol === undefined) ulpTol = 4;

  let absTol =
    OP_ABS_TOL[primaryOp]?.[dataType] ?? (dataType.startsWith('float') ? 1e-4 : 0);

  if (Array.isArray(allOps)) {
    for (const op of allOps) {
      if (!op) continue;
      const u = OP_ULP[op];
      if (u !== undefined && u > ulpTol) ulpTol = u;
      const at = OP_ABS_TOL[op]?.[dataType];
      if (at !== undefined && at > absTol) absTol = at;
    }
  }

  return { ulpTol, absTol };
}

function shapeElementCount(shape) {
  if (!shape || shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * WPT often stores a single scalar for uniform tensors (e.g. large scalar-fill cases).
 * Compare every actual element against that scalar when lengths differ but shapes match.
 */
function isScalarFillExpected(expectedData, shape) {
  return expectedData.length === 1 && shapeElementCount(shape) > 1;
}

const INTEGER_DTYPES = new Set([
  'int4',
  'uint4',
  'int8',
  'uint8',
  'int32',
  'uint32'
]);

/**
 * Resolve per-graph tolerance the same way WPT does in assertResultsEquals:
 * toleranceFunc(graphResources) → { metricType: 'ULP' | 'ATOL', value }.
 * For integer outputs, WPT "ULP" means absolute difference.
 * @param {object | null | undefined} getTolerance
 * @param {object | null | undefined} graph
 * @returns {{ metricType: 'ULP' | 'ATOL', value: number } | null}
 */
function resolveWptTolerance(getTolerance, graph) {
  if (typeof getTolerance !== 'function' || !graph) {
    return null;
  }
  try {
    const info = getTolerance(graph, {});
    if (!info || typeof info.value !== 'number') {
      return null;
    }
    if (info.metricType !== 'ULP' && info.metricType !== 'ATOL') {
      return null;
    }
    return { metricType: info.metricType, value: info.value };
  } catch {
    return null;
  }
}

export function assertOutputClose({
  operatorName,
  graphOperatorNames,
  graph,
  getTolerance,
  outputName,
  expected,
  actual
}) {
  const dataType = expected.descriptor.dataType;
  const expectedData = Array.isArray(expected.data) ? expected.data : [expected.data];
  const actualData = actual.data ?? [];
  const expectedElementCount = shapeElementCount(expected.descriptor.shape);

  if (expectedData.length !== actualData.length) {
    if (!isScalarFillExpected(expectedData, expected.descriptor.shape)
        || actualData.length !== expectedElementCount) {
      throw new Error(
        `length mismatch for ${outputName}: expected ${expectedData.length}, got ${actualData.length}`
      );
    }
  }

  const expectedShape = JSON.stringify(expected.descriptor.shape);
  const actualShape = JSON.stringify(actual.descriptor.shape);
  if (expectedShape !== actualShape) {
    throw new Error(`shape mismatch for ${outputName}: expected ${expectedShape}, got ${actualShape}`);
  }

  const scalarFill = isScalarFillExpected(expectedData, expected.descriptor.shape);
  const compareLength = scalarFill ? actualData.length : expectedData.length;

  if (INTEGER_DTYPES.has(dataType)) {
    const wptTolerance = resolveWptTolerance(getTolerance, graph);
    const intTol =
      wptTolerance?.metricType === 'ULP'
        ? wptTolerance.value
        : wptTolerance?.metricType === 'ATOL'
          ? wptTolerance.value
          : 0;
    const scalar = scalarFill ? Number(expectedData[0]) : null;
    for (let i = 0; i < compareLength; i += 1) {
      const a = Number(actualData[i]);
      const e = scalar ?? Number(expectedData[i]);
      if (Math.abs(a - e) > intTol) {
        throw new Error(
          `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}` +
            (intTol > 0 ? ` (tolerance ±${intTol})` : '')
        );
      }
    }
    return;
  }

  if (dataType === 'int64' || dataType === 'uint64') {
    const scalar = scalarFill ? BigInt(expectedData[0]) : null;
    for (let i = 0; i < compareLength; i += 1) {
      const a = BigInt(actualData[i]);
      const e = scalar ?? BigInt(expectedData[i]);
      if (a !== e) {
        throw new Error(`value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}`);
      }
    }
    return;
  }

  const wptTolerance = resolveWptTolerance(getTolerance, graph);
  let ulpTol;
  let absTol;
  let wptUlpOnly = false;
  if (wptTolerance?.metricType === 'ULP') {
    ulpTol = wptTolerance.value;
    absTol = 0;
    wptUlpOnly = true;
  } else if (wptTolerance?.metricType === 'ATOL') {
    ulpTol = Number.POSITIVE_INFINITY;
    absTol = wptTolerance.value;
  } else {
    ({ ulpTol, absTol } = mergedFloatTolerance(operatorName, graphOperatorNames, dataType));
    if (dataType === 'float16' && ulpTol === 0) ulpTol = 4;
  }

  let f32BitScratch;
  let f16BitScratch;
  if (dataType === 'float16') {
    const f16 = new Float16Array(1);
    f16BitScratch = { f16, u16: new Uint16Array(f16.buffer) };
  } else {
    const f32 = new Float32Array(1);
    f32BitScratch = { f32, u32: new Uint32Array(f32.buffer) };
  }

  const scalar = scalarFill ? Number(expectedData[0]) : null;

  for (let i = 0; i < compareLength; i += 1) {
    const a = Number(actualData[i]);
    const e = scalar ?? Number(expectedData[i]);

    if (Number.isNaN(e) && Number.isNaN(a)) continue;
    if (!Number.isFinite(e) || !Number.isFinite(a)) {
      if (e !== a) {
        throw new Error(`value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}`);
      }
      continue;
    }

    const absDiff = Math.abs(a - e);
    let ulp;
    if (dataType === 'float16') {
      ulp = ulpDistanceF16(a, e, f16BitScratch);
    } else {
      ulp = ulpDistanceF32(a, e, f32BitScratch);
    }

    if (wptUlpOnly) {
      if (ulp > ulpTol) {
        throw new Error(
          `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}, absDiff=${absDiff}, ulp=${ulp}, ulpTol=${ulpTol}`
        );
      }
    } else if (wptTolerance?.metricType === 'ATOL') {
      if (absDiff > absTol) {
        throw new Error(
          `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}, absDiff=${absDiff}, absTol=${absTol}`
        );
      }
    } else if (absDiff > absTol && ulp > ulpTol) {
      throw new Error(
        `value mismatch for ${outputName}[${i}]: expected ${e}, got ${a}, absDiff=${absDiff}, ulp=${ulp}, ulpTol=${ulpTol}`
      );
    }
  }
}

export { normalizeOpName };
