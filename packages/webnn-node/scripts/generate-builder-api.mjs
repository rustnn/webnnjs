#!/usr/bin/env node
/**
 * Generates MLGraphBuilder dispatch (Rust) and methods (TypeScript) from webnn.idl.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(pkgRoot, '../..');

function resolveIdlPath() {
  const wptRoot = process.env.WPT_DIR ?? path.join(repoRoot, '.cache', 'wpt');
  const cachedIdl = path.join(wptRoot, 'interfaces', 'webnn.idl');
  const vendoredIdl = path.join(pkgRoot, 'idl', 'webnn.idl');

  if (fs.existsSync(cachedIdl)) {
    return cachedIdl;
  }
  return vendoredIdl;
}

const idlPath = resolveIdlPath();
const rustOut = path.join(pkgRoot, 'native', 'src', 'generated', 'builder_ops.rs');
const tsOut = path.join(pkgRoot, 'src', 'generated', 'builder-methods.ts');
const tsTypesOut = path.join(pkgRoot, 'src', 'generated', 'builder-options.ts');

function camelToSnake(name) {
  return name
    .replace(/([A-Z])/g, '_$1')
    .toLowerCase()
    .replace(/^_/, '');
}

function rustMethodName(idlName) {
  if (RUST_METHOD[idlName]) return RUST_METHOD[idlName];
  if (idlName === 'where') return 'where_';
  return camelToSnake(idlName);
}

function rustWithOptions(idlName) {
  if (RUST_WITH_OPTIONS[idlName]) return RUST_WITH_OPTIONS[idlName];
  return `${rustMethodName(idlName)}_with_options`;
}

const RUST_METHOD = {
  argMin: 'arg_min',
  argMax: 'arg_max',
  averagePool2d: 'average_pool2d',
  batchNormalization: 'batch_normalization',
  conv2d: 'conv2d',
  convTranspose2d: 'conv_transpose2d',
  cumulativeSum: 'cumulative_sum',
  dequantizeLinear: 'dequantize_linear',
  gatherElements: 'gather_elements',
  gatherND: 'gather_nd',
  gemm: 'gemm',
  greaterOrEqual: 'greater_or_equal',
  hardSigmoid: 'hard_sigmoid',
  hardSwish: 'hard_swish',
  instanceNormalization: 'instance_normalization',
  isInfinite: 'is_infinite',
  isNaN: 'is_nan',
  l2Pool2d: 'l2_pool2d',
  layerNormalization: 'layer_normalization',
  leakyRelu: 'leaky_relu',
  lesserOrEqual: 'lesser_or_equal',
  logicalAnd: 'logical_and',
  logicalNot: 'logical_not',
  logicalOr: 'logical_or',
  logicalXor: 'logical_xor',
  maxPool2d: 'max_pool2d',
  notEqual: 'not_equal',
  quantizeLinear: 'quantize_linear',
  reduceL1: 'reduce_l1',
  reduceL2: 'reduce_l2',
  reduceLogSum: 'reduce_log_sum',
  reduceLogSumExp: 'reduce_log_sum_exp',
  reduceMax: 'reduce_max',
  reduceMean: 'reduce_mean',
  reduceMin: 'reduce_min',
  reduceProduct: 'reduce_product',
  reduceSum: 'reduce_sum',
  reduceSumSquare: 'reduce_sum_square',
  resample2d: 'resample2d',
  roundEven: 'round_even',
  scatterND: 'scatter_nd',
  where: 'where_',
};

const RUST_WITH_OPTIONS = {
  concat: 'concat_with_options',
  conv2d: 'conv2_with_options',
  gather: 'gather_with_options',
  gatherElements: 'gather_elements_with_options',
  slice: 'slice_with_options',
  split: 'split_with_options',
  where: 'where_with_options',
};

function stripIdlComments(idl) {
  return idl.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
}

function parseBuilderMethods(idl) {
  const cleaned = stripIdlComments(idl);
  const blocks = [...cleaned.matchAll(/partial interface MLGraphBuilder\s*\{([\s\S]*?)\};/g)];
  const methods = [];

  for (const block of blocks) {
    const body = block[1];
    const sigRe =
      /(sequence<MLOperand>|MLOperand)\s+(\w+)\s*\(([\s\S]*?)\)\s*;/g;
    let match;
    while ((match = sigRe.exec(body)) !== null) {
      const returnKind = match[1] === 'sequence<MLOperand>' ? 'sequence' : 'single';
      const name = match[2];
      const paramsRaw = match[3]
        .replace(/\[[^\]]*\]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
      methods.push({ name, returnKind, paramsRaw });
    }
  }

  return methods;
}

function parseParamPart(part) {
  part = part.trim();
  if (!part) return null;

  if (part.startsWith('optional ')) {
    const m = part.match(/^optional\s+(\w+)\s+options\s*=/);
    return { kind: 'options', optionsType: m?.[1] ?? 'MLOperatorOptions' };
  }

  if (part.includes(' or sequence<')) {
    const name = part.match(/(\w+)\s*$/)?.[1];
    if (!name) return null;
    return { name, kind: 'splits', typeHint: 'unsigned long | sequence' };
  }

  if (part.startsWith('sequence<MLOperand>')) {
    const name = part.match(/sequence<MLOperand>\s+(\w+)/)?.[1];
    if (!name) return null;
    return { name, kind: 'operand_sequence', typeHint: 'sequence<MLOperand>' };
  }

  if (part.startsWith('MLOperand ')) {
    return {
      name: part.slice('MLOperand '.length).trim(),
      kind: 'operand',
      typeHint: 'MLOperand',
    };
  }

  if (part.startsWith('MLOperandDataType ')) {
    return {
      name: part.slice('MLOperandDataType '.length).trim(),
      kind: 'data_type',
      typeHint: 'MLOperandDataType',
    };
  }

  if (part.startsWith('unsigned long ')) {
    return {
      name: part.slice('unsigned long '.length).trim(),
      kind: 'u32',
      typeHint: 'unsigned long',
    };
  }

  if (part.startsWith('sequence<')) {
    const name = part.match(/sequence<[^>]+>\s+(\w+)/)?.[1];
    if (!name) return null;
    return { name, kind: 'u32_sequence', typeHint: 'sequence' };
  }

  return null;
}

function parseParams(paramsRaw) {
  const parts = [];
  let depth = 0;
  let current = '';
  for (const ch of paramsRaw) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ',' && depth === 0) {
      parts.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim()) parts.push(current.trim());

  const params = [];
  let optionsType = 'MLOperatorOptions';

  for (const part of parts) {
    const parsed = parseParamPart(part);
    if (!parsed) continue;
    if (parsed.kind === 'options') {
      optionsType = parsed.optionsType;
      continue;
    }
    params.push(parsed);
  }

  return { params, optionsType };
}

const NOT_IMPLEMENTED = new Set();

const OPTION_OPERAND_FIELDS = {
  batchNormalization: ['scale', 'bias'],
  conv2d: ['bias'],
  convTranspose2d: ['bias'],
  gemm: ['c'],
  instanceNormalization: ['scale', 'bias'],
  layerNormalization: ['scale', 'bias'],
  gru: ['bias', 'recurrentBias', 'initialHiddenState'],
  gruCell: ['bias', 'recurrentBias'],
  lstm: ['bias', 'recurrentBias', 'peepholeWeight', 'initialHiddenState', 'initialCellState'],
  lstmCell: ['bias', 'recurrentBias', 'peepholeWeight'],
};

function classify(method) {
  const { name, returnKind, paramsRaw } = method;
  const { params, optionsType } = parseParams(paramsRaw);
  const operandParams = params.filter((p) => p.kind === 'operand');
  const base = {
    idlName: name,
    rustName: rustMethodName(name),
    returnKind,
    params,
    optionsType,
    operandFields: OPTION_OPERAND_FIELDS[name] ?? [],
  };

  if (NOT_IMPLEMENTED.has(name)) {
    return { ...base, category: 'not_implemented' };
  }

  if (name === 'scatterElements') {
    return { ...base, category: 'ternary' };
  }
  if (name === 'gruCell') {
    return { ...base, category: 'recurrent_cell' };
  }
  if (name === 'lstmCell') {
    return { ...base, category: 'recurrent_cell_multi' };
  }

  if (returnKind === 'sequence') {
    if (name === 'split') return { ...base, category: 'split' };
    if (name === 'gru' || name === 'lstm') return { ...base, category: 'recurrent' };
    return { ...base, category: 'not_implemented' };
  }

  if (name === 'concat') return { ...base, category: 'concat' };
  if (name === 'slice') return { ...base, category: 'slice' };
  if (name === 'pad') return { ...base, category: 'pad' };
  if (name === 'cast') return { ...base, category: 'cast' };
  if (name === 'expand' || name === 'reshape') return { ...base, category: 'new_shape' };
  if (name === 'tile') return { ...base, category: 'tile' };

  const axisOps = new Set(['argMin', 'argMax', 'softmax', 'cumulativeSum']);
  if (axisOps.has(name)) return { ...base, category: 'unary_axis' };

  if (operandParams.length === 1 && params.length === 1) {
    return { ...base, category: 'unary' };
  }
  if (operandParams.length === 2 && params.length === 2) {
    return { ...base, category: 'binary' };
  }
  if (operandParams.length === 3) {
    return { ...base, category: 'ternary' };
  }

  return { ...base, category: 'not_implemented' };
}

function rustOptionsType(optionsType) {
  return optionsType;
}

function emitRustArm(op) {
  const opts = rustOptionsType(op.optionsType);
  const patchFields =
    op.operandFields.length > 0
      ? `resolve_option_operands(builder, wire.options.take(), &[${op.operandFields.map((f) => `"${f}"`).join(', ')}])?`
      : 'wire.options.take()';

  switch (op.category) {
    case 'not_implemented':
      return `        "${op.idlName}" => Err(op_err("${op.idlName}", "not implemented in rustnn yet")),`;
    case 'unary':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.${rustWithOptions(op.idlName)}(input, opts)
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'unary_axis':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let axis = wire.axis.ok_or_else(|| missing_field("${op.idlName}", "axis"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.${rustWithOptions(op.idlName)}(input, axis, opts)
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'binary':
      return `        "${op.idlName}" => {
            let a = operand_at(builder, &wire, 0)?;
            let b = operand_at(builder, &wire, 1)?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.${rustWithOptions(op.idlName)}(a, b, opts)
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'ternary':
      return `        "${op.idlName}" => {
            let a = operand_at(builder, &wire, 0)?;
            let b = operand_at(builder, &wire, 1)?;
            let c = operand_at(builder, &wire, 2)?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.${rustWithOptions(op.idlName)}(a, b, ${
              op.idlName === 'quantizeLinear' || op.idlName === 'dequantizeLinear'
                ? 'Some(c)'
                : 'c'
            }, opts)
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'cast':
      return `        "cast" => {
            let input = operand_at(builder, &wire, 0)?;
            let data_type = parse_data_type(
                wire.data_type.as_deref().ok_or_else(|| missing_field("cast", "dataType"))?,
            )?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.cast_with_options(input, data_type, opts)
                .map_err(|e| op_err("cast", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'new_shape':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let new_shape = parse_mldimensions(
                wire.new_shape.as_ref().ok_or_else(|| missing_field("${op.idlName}", "newShape"))?,
            )?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.${rustWithOptions(op.idlName)}(input, new_shape, opts)
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'tile':
      return `        "tile" => {
            let input = operand_at(builder, &wire, 0)?;
            let repetitions = wire
                .repetitions
                .clone()
                .ok_or_else(|| missing_field("tile", "repetitions"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.tile_with_options(input, repetitions, opts)
                .map_err(|e| op_err("tile", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'pad':
      return `        "pad" => {
            let input = operand_at(builder, &wire, 0)?;
            let beginning = wire
                .beginning_padding
                .clone()
                .ok_or_else(|| missing_field("pad", "beginningPadding"))?;
            let ending = wire
                .ending_padding
                .clone()
                .ok_or_else(|| missing_field("pad", "endingPadding"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.pad_with_options(input, beginning, ending, opts)
                .map_err(|e| op_err("pad", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'concat':
      return `        "concat" => {
            let inputs = operands_all(builder, &wire)?;
            let axis = wire.axis.ok_or_else(|| missing_field("concat", "axis"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.concat_with_options(&inputs, axis, opts)
                .map_err(|e| op_err("concat", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'slice':
      return `        "slice" => {
            let input = operand_at(builder, &wire, 0)?;
            let starts = wire.starts.clone().ok_or_else(|| missing_field("slice", "starts"))?;
            let sizes = parse_mldimensions(
                wire.sizes.as_ref().ok_or_else(|| missing_field("slice", "sizes"))?,
            )?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder.builder.slice_with_options(input, &starts, &sizes, opts)
                .map_err(|e| op_err("slice", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'split':
      return `        "split" => {
            let input = operand_at(builder, &wire, 0)?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let outs = if let Some(n) = wire.split_equal_parts {
                split_equal_parts(builder, input, n, opts)?
            } else {
                let splits = wire
                    .splits
                    .clone()
                    .ok_or_else(|| missing_field("split", "splits"))?;
                builder
                    .builder
                    .split_with_options(input, &splits, opts)
                    .map_err(|e| op_err("split", e))?
            };
            Ok(multi_result(builder, outs)?)
        }`;
    case 'recurrent':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let weight = operand_at(builder, &wire, 1)?;
            let recurrent_weight = operand_at(builder, &wire, 2)?;
            let steps = wire.steps.ok_or_else(|| missing_field("${op.idlName}", "steps"))?;
            let hidden_size = wire
                .hidden_size
                .ok_or_else(|| missing_field("${op.idlName}", "hiddenSize"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let outs = builder
                .builder
                .${rustMethodName(op.idlName)}_with_options(
                    input,
                    weight,
                    recurrent_weight,
                    steps,
                    hidden_size,
                    opts,
                )
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(multi_result(builder, outs)?)
        }`;
    case 'recurrent_cell':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let weight = operand_at(builder, &wire, 1)?;
            let recurrent_weight = operand_at(builder, &wire, 2)?;
            let hidden_state = operand_at(builder, &wire, 3)?;
            let hidden_size = wire
                .hidden_size
                .ok_or_else(|| missing_field("${op.idlName}", "hiddenSize"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let out = builder
                .builder
                .${rustMethodName(op.idlName)}_with_options(
                    input,
                    weight,
                    recurrent_weight,
                    hidden_state,
                    hidden_size,
                    opts,
                )
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(single_result(builder, out)?)
        }`;
    case 'recurrent_cell_multi':
      return `        "${op.idlName}" => {
            let input = operand_at(builder, &wire, 0)?;
            let weight = operand_at(builder, &wire, 1)?;
            let recurrent_weight = operand_at(builder, &wire, 2)?;
            let hidden_state = operand_at(builder, &wire, 3)?;
            let cell_state = operand_at(builder, &wire, 4)?;
            let hidden_size = wire
                .hidden_size
                .ok_or_else(|| missing_field("${op.idlName}", "hiddenSize"))?;
            let opts: ${opts} = parse_options(${patchFields})?;
            let outs = builder
                .builder
                .${rustMethodName(op.idlName)}_with_options(
                    input,
                    weight,
                    recurrent_weight,
                    hidden_state,
                    cell_state,
                    hidden_size,
                    opts,
                )
                .map_err(|e| op_err("${op.idlName}", e))?;
            Ok(multi_result(builder, outs)?)
        }`;
    default:
      return `        "${op.idlName}" => Err(op_err("${op.idlName}", "not implemented in rustnn yet")),`;
  }
}

function tsParamType(param) {
  switch (param.kind) {
    case 'operand':
      return `${param.name}: MLOperand`;
    case 'operand_sequence':
      return `${param.name}: MLOperand[]`;
    case 'data_type':
      return `${param.name}: MLOperandDataType`;
    case 'u32':
      return `${param.name}: number`;
    case 'u32_sequence':
      return `${param.name}: number[]`;
    case 'splits':
      return `${param.name}: number | number[]`;
    default:
      return `${param.name}: unknown`;
  }
}

function wireFieldName(paramName) {
  if (paramName === 'newShape') return 'newShape';
  if (paramName === 'beginningPadding') return 'beginningPadding';
  if (paramName === 'endingPadding') return 'endingPadding';
  return paramName;
}

function emitTsMethod(op) {
  const valueParams = op.params;
  const paramDecl = valueParams.map(tsParamType).join(', ');
  const optionsDecl =
    op.optionsType === 'MLOperatorOptions'
      ? 'options: MLOperatorOptions = {}'
      : `options: ${op.optionsType} = {}`;
  const fullParams = paramDecl ? `${paramDecl}, ${optionsDecl}` : optionsDecl;

  const wireLines = [`op: '${op.idlName}'`];

  const operandParams = valueParams.filter((p) => p.kind === 'operand');
  const sequenceParam = valueParams.find((p) => p.kind === 'operand_sequence');

  if (sequenceParam) {
    wireLines.push(`operands: ${sequenceParam.name}.map((o) => o.handle)`);
  } else if (operandParams.length > 0) {
    wireLines.push(
      `operands: [${operandParams.map((p) => `${p.name}.handle`).join(', ')}]`
    );
  }

  for (const p of valueParams) {
    if (p.kind === 'u32') {
      wireLines.push(`${wireFieldName(p.name)}: ${p.name}`);
    }
    if (p.kind === 'data_type') {
      wireLines.push(`dataType: ${p.name}`);
    }
    if (p.kind === 'u32_sequence') {
      wireLines.push(`${wireFieldName(p.name)}: [...${p.name}]`);
    }
    if (p.kind === 'splits') {
      wireLines.push(
        `...(typeof ${p.name} === 'number' ? { splitEqualParts: ${p.name} } : { splits: [...${p.name}] })`
      );
    }
  }

  wireLines.push('options: serializeBuilderOptions(options)');

  const returnType = op.returnKind === 'sequence' ? 'MLOperand[]' : 'MLOperand';
  const body =
    op.category === 'not_implemented'
      ? `throw new Error('MLGraphBuilder.${op.idlName} is not implemented in rustnn yet');`
      : `return this._invokeOp(${returnType === 'MLOperand[]' ? 'true' : 'false'}, {
      ${wireLines.join(',\n      ')},
    });`;

  return `  ${op.idlName}(this: BuilderMethodsHost, ${fullParams}): ${returnType} {
    ${body}
  }`;
}

function emitTsInterfaceMethod(op) {
  const valueParams = op.params;
  const paramDecl = valueParams.map(tsParamType).join(', ');
  const optionsDecl =
    op.optionsType === 'MLOperatorOptions'
      ? 'options?: MLOperatorOptions'
      : `options?: ${op.optionsType}`;
  const fullParams = paramDecl ? `${paramDecl}, ${optionsDecl}` : optionsDecl;
  const returnType = op.returnKind === 'sequence' ? 'MLOperand[]' : 'MLOperand';
  return `  ${op.idlName}(${fullParams}): ${returnType};`;
}

function collectOptionTypes(ops) {
  const types = new Set(['MLOperatorOptions']);
  for (const op of ops) types.add(op.optionsType);
  return [...types].sort();
}

function main() {
  if (!fs.existsSync(idlPath)) {
    console.error(`webnn.idl not found at ${idlPath}`);
    console.error('Run: npm run test:wpt:fetch   (or commit packages/webnn-node/idl/webnn.idl)');
    process.exit(1);
  }

  const idl = fs.readFileSync(idlPath, 'utf8');
  const rawMethods = parseBuilderMethods(idl);
  const skip = new Set(['input', 'constant', 'build']);
  const ops = rawMethods.filter((m) => !skip.has(m.name)).map(classify);

  const optionTypes = collectOptionTypes(ops);
  const rustOptionUses = optionTypes
    .filter((t) => t !== 'MLOperatorOptions')
    .map((t) => `    ${t},`)
    .join('\n');

  const rustArms = ops.map(emitRustArm).join('\n');

  const rust = `// Auto-generated by scripts/generate-builder-api.mjs — do not edit.

use napi::{Error, Result, Status};
use rustnn::operator_options::{
    MLOperatorOptions,
${rustOptionUses}
};

use crate::builder_dispatch::{
    missing_field, multi_result, op_err, operand_at, operands_all,
    parse_data_type, parse_mldimensions, parse_options, resolve_option_operands, single_result,
    split_equal_parts, BuilderInvokeWire, InvokeResult,
};
use crate::BuilderEntry;

pub fn dispatch_builder_op(
    builder: &mut BuilderEntry,
    mut wire: BuilderInvokeWire,
) -> Result<InvokeResult> {
    match wire.op.as_str() {
${rustArms}
        other => Err(Error::new(
            Status::InvalidArg,
            format!("unknown MLGraphBuilder operation '{other}'"),
        )),
    }
}
`;

  const tsTypes = `// Auto-generated by scripts/generate-builder-api.mjs — do not edit.

import type { MLOperandDataType } from '../index.js';

export interface MLOperatorOptions {
  label?: string;
}

${optionTypes
  .filter((t) => t !== 'MLOperatorOptions')
  .map((t) => `export interface ${t} extends MLOperatorOptions {}`)
  .join('\n\n')}
`;

  const tsMethods = `// Auto-generated by scripts/generate-builder-api.mjs — do not edit.

import type { MLOperand, MLOperandDataType } from '../index.js';
import type { ${optionTypes.join(', ')} } from './builder-options.js';

export type BuilderMethodsHost = {
  _invokeOp(multi: false, wire: Record<string, unknown>): MLOperand;
  _invokeOp(multi: true, wire: Record<string, unknown>): MLOperand[];
};

export function serializeBuilderOptions(options: object): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(options ?? {})) {
    if (value === undefined) continue;
    if (value && typeof value === 'object' && 'handle' in (value as object)) {
      out[key] = (value as MLOperand).handle;
    } else {
      out[key] = value;
    }
  }
  return out;
}

export function installGeneratedBuilderMethods<T extends BuilderMethodsHost>(proto: T): void {
  Object.assign(proto, generatedBuilderMethods);
}

export interface GeneratedMLGraphBuilderMethods {
${ops.map(emitTsInterfaceMethod).join('\n')}
}

const generatedBuilderMethods: Record<
  string,
  (this: BuilderMethodsHost, ...args: never[]) => MLOperand | MLOperand[]
> = {
${ops.map(emitTsMethod).join(',\n\n')}
};
`;

  fs.mkdirSync(path.dirname(rustOut), { recursive: true });
  fs.mkdirSync(path.dirname(tsOut), { recursive: true });
  fs.writeFileSync(rustOut, rust);
  fs.writeFileSync(tsTypesOut, tsTypes);
  fs.writeFileSync(tsOut, tsMethods);

  console.log(`Generated ${ops.length} builder ops ->`);
  console.log(`  IDL: ${path.relative(pkgRoot, idlPath)}`);
  console.log(`  ${path.relative(pkgRoot, rustOut)}`);
  console.log(`  ${path.relative(pkgRoot, rustOut)}`);
  console.log(`  ${path.relative(pkgRoot, tsOut)}`);
  console.log(`  ${path.relative(pkgRoot, tsTypesOut)}`);
}

main();
