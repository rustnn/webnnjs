use napi::{Error, Result, Status};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLDimension, MLSplitOptions};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    nerr, operand_descriptor, register_operand_inferred, BuilderEntry, TensorDescriptorWire,
};

pub(crate) use crate::parse_data_type;
pub use crate::generated::builder_ops::dispatch_builder_op;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BuilderInvokeWire {
    pub op: String,
    #[serde(default)]
    pub operands: Vec<u32>,
    pub axis: Option<u32>,
    pub data_type: Option<String>,
    pub new_shape: Option<Vec<Value>>,
    pub repetitions: Option<Vec<u32>>,
    pub beginning_padding: Option<Vec<u32>>,
    pub ending_padding: Option<Vec<u32>>,
    pub starts: Option<Vec<u32>>,
    pub sizes: Option<Vec<Value>>,
    pub splits: Option<Vec<u32>>,
    pub split_equal_parts: Option<u32>,
    pub options: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvokeOperandWire {
    pub handle: u32,
    pub data_type: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvokeResult {
    pub operands: Vec<InvokeOperandWire>,
}

pub fn missing_field(op: &str, field: &str) -> Error {
    nerr(
        Status::InvalidArg,
        format!("MLGraphBuilder.{op}: missing required field '{field}'"),
    )
}

pub fn not_implemented_op(op: &str) -> Error {
    nerr(
        Status::GenericFailure,
        format!("MLGraphBuilder.{op} is not implemented in rustnn yet"),
    )
}

pub fn op_err(op: &str, err: impl std::fmt::Display) -> Error {
    nerr(
        Status::GenericFailure,
        format!("MLGraphBuilder.{op} failed: {err}"),
    )
}

pub fn parse_options<T: DeserializeOwned + Default>(value: Option<Value>) -> Result<T> {
    match value {
        Some(raw) => serde_json::from_value(raw).map_err(|e| {
            nerr(
                Status::InvalidArg,
                format!("invalid operator options JSON: {e}"),
            )
        }),
        None => Ok(T::default()),
    }
}

pub fn parse_mldimensions(raw: &[Value]) -> Result<Vec<MLDimension>> {
    raw.iter()
        .map(|value| match value {
            Value::Number(n) => n
                .as_u64()
                .map(|v| MLDimension::Static(v as u32))
                .ok_or_else(|| nerr(Status::InvalidArg, "invalid static MLDimension")),
            Value::Object(obj) => serde_json::from_value(Value::Object(obj.clone())).map_err(|e| {
                nerr(
                    Status::InvalidArg,
                    format!("invalid dynamic MLDimension: {e}"),
                )
            }),
            _ => Err(nerr(
                Status::InvalidArg,
                "MLDimension must be a number or dynamic-dimension object",
            )),
        })
        .collect()
}

pub fn resolve_option_operands(
    builder: &BuilderEntry,
    options: Option<Value>,
    fields: &[&str],
) -> Result<Option<Value>> {
    let Some(Value::Object(mut map)) = options else {
        return Ok(options);
    };

    for field in fields {
        if let Some(Value::Number(handle)) = map.get(*field) {
            let handle = handle.as_u64().ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("options.{field} must be an operand handle"),
                )
            })? as u32;
            let (_, meta) = builder.operands.get(&handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown operand handle {handle} in options.{field}"),
                )
            })?;
            map.insert(
                field.to_string(),
                Value::Number((meta.rustnn_id as u64).into()),
            );
        }
    }

    Ok(Some(Value::Object(map)))
}

pub fn operand_at(
    builder: &BuilderEntry,
    wire: &BuilderInvokeWire,
    index: usize,
) -> Result<MLOperand> {
    let handle = wire.operands.get(index).copied().ok_or_else(|| {
        nerr(
            Status::InvalidArg,
            format!(
                "MLGraphBuilder.{}: expected operand at index {index}",
                wire.op
            ),
        )
    })?;
    builder
        .operands
        .get(&handle)
        .map(|(operand, _)| *operand)
        .ok_or_else(|| nerr(Status::InvalidArg, format!("unknown operand handle {handle}")))
}

pub fn operands_all(builder: &BuilderEntry, wire: &BuilderInvokeWire) -> Result<Vec<MLOperand>> {
    wire.operands
        .iter()
        .map(|handle| {
            builder
                .operands
                .get(handle)
                .map(|(operand, _)| *operand)
                .ok_or_else(|| nerr(Status::InvalidArg, format!("unknown operand handle {handle}")))
        })
        .collect()
}

fn infer_operand_wire(
    builder: &mut BuilderEntry,
    operand: MLOperand,
) -> Result<InvokeOperandWire> {
    let shape = builder
        .builder
        .rustnn_operand_shape(operand)
        .map_err(|e| op_err("operand", e))?;
    let data_type = builder
        .builder
        .rustnn_operand_data_type(operand)
        .map_err(|e| op_err("operand", e))?;
    let data_type_str = ml_operand_dtype_to_string(data_type);
    let handle = register_operand_inferred(builder, operand, data_type_str.clone(), shape.clone());
    Ok(InvokeOperandWire {
        handle,
        data_type: data_type_str,
        shape,
    })
}

pub fn single_result(builder: &mut BuilderEntry, operand: MLOperand) -> Result<InvokeResult> {
    Ok(InvokeResult {
        operands: vec![infer_operand_wire(builder, operand)?],
    })
}

pub fn multi_result(builder: &mut BuilderEntry, operands: Vec<MLOperand>) -> Result<InvokeResult> {
    operands
        .into_iter()
        .map(|operand| infer_operand_wire(builder, operand))
        .collect::<Result<Vec<_>>>()
        .map(|operands| InvokeResult { operands })
}

pub fn split_equal_parts(
    builder: &mut BuilderEntry,
    input: MLOperand,
    parts: u32,
    options: MLSplitOptions,
) -> Result<Vec<MLOperand>> {
    if parts == 0 {
        return Err(nerr(
            Status::InvalidArg,
            "MLGraphBuilder.split: splitEqualParts must be greater than 0",
        ));
    }

    let shape = builder
        .builder
        .rustnn_operand_shape(input)
        .map_err(|e| op_err("split", e))?;
    let axis = options.axis as usize;
    if axis >= shape.len() {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "MLGraphBuilder.split: axis {axis} out of bounds for rank {}",
                shape.len()
            ),
        ));
    }

    let dim = shape[axis];
    if dim % parts as u64 != 0 {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "MLGraphBuilder.split: axis size {dim} is not evenly divisible by {parts}"
            ),
        ));
    }

    let chunk = (dim / parts as u64) as u32;
    let splits = vec![chunk; parts as usize];
    builder
        .builder
        .split_with_options(input, &splits, options)
        .map_err(|e| op_err("split", e))
}

pub fn constant_from_buffer(
    builder: &mut BuilderEntry,
    descriptor: &TensorDescriptorWire,
    data: &[u8],
) -> Result<InvokeResult> {
    let operand_desc = operand_descriptor(descriptor)?;
    let output = match descriptor.data_type.as_str() {
        "float32" => {
            let values = crate::bytes_to_pod::<f32>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "float16" => {
            return Err(nerr(
                Status::GenericFailure,
                "float16 constant is not supported yet",
            ))
        }
        "int32" => {
            let values = crate::bytes_to_pod::<i32>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "uint32" => {
            let values = crate::bytes_to_pod::<u32>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "int64" => {
            let values = crate::bytes_to_pod::<i64>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "uint64" => {
            let values = crate::bytes_to_pod::<u64>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "int8" => {
            let values = crate::bytes_to_pod::<i8>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        "uint8" => {
            let values = crate::bytes_to_pod::<u8>(data)?;
            builder
                .builder
                .constant_from_slice(&operand_desc, &values)
                .map_err(|e| op_err("constant", e))?
        }
        other => {
            return Err(nerr(
                Status::InvalidArg,
                format!("unsupported constant dataType '{other}'"),
            ))
        }
    };

    single_result(builder, output)
}

pub fn ml_operand_dtype_to_string(dtype: MLOperandDataType) -> String {
    match dtype {
        MLOperandDataType::Float32 => "float32",
        MLOperandDataType::Float16 => "float16",
        MLOperandDataType::Int32 => "int32",
        MLOperandDataType::Uint32 => "uint32",
        MLOperandDataType::Int64 => "int64",
        MLOperandDataType::Uint64 => "uint64",
        MLOperandDataType::Int8 => "int8",
        MLOperandDataType::Uint8 => "uint8",
    }
    .to_string()
}
