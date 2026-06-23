use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use napi::bindgen_prelude::Buffer;
use napi::{Error, Result, Status};
use napi_derive::napi;
use once_cell::sync::Lazy;
use rustnn::graph::{get_static_or_max_size, DataType, GraphInfo};
use rustnn::loader::load_graph_from_path;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLOperand, MLOperandDescriptor,
    MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::validator::{ContextProperties, GraphValidator, ValidationArtifacts};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

mod builder_dispatch;
mod generated;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ContextOptionsWire {
    power_preference: Option<String>,
    accelerated: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TensorDescriptorWire {
    data_type: String,
    shape: Vec<u64>,
    readable: Option<bool>,
    writable: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadTensorMeta {
    data_type: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadMeta {
    graph_path: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    inputs: HashMap<String, LoadTensorMeta>,
    outputs: HashMap<String, LoadTensorMeta>,
}

#[napi(object)]
pub struct NativeModelLoadResult {
    pub graph_handle: u32,
    pub meta_json: String,
}

struct GraphEntry {
    graph: MLGraph<'static>,
}

pub(crate) struct BuilderEntry {
    builder: MLGraphBuilder<'static, 'static>,
    operands: HashMap<u32, (MLOperand, OperandWireMeta)>,
    next_operand_handle: u32,
    next_rustnn_operand_id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct OperandWireMeta {
    rustnn_id: u32,
}

struct ContextSession {
    context: MLContext<'static>,
    graphs: HashMap<u32, GraphEntry>,
    tensors: HashMap<u32, MLTensor>,
    builders: HashMap<u32, BuilderEntry>,
    next_graph_handle: u32,
    next_tensor_handle: u32,
    next_builder_handle: u32,
}

struct Store {
    next_context_handle: u32,
    contexts: HashMap<u32, ContextSession>,
}

impl Store {
    fn new() -> Self {
        Self {
            next_context_handle: 1,
            contexts: HashMap::new(),
        }
    }
}

// MLContext/MLGraphBuilder use trait objects that are not auto-Send. Node calls into
// this addon are serialized through the mutex; ORT dispatch stays on the calling thread.
unsafe impl Send for Store {}
unsafe impl Sync for Store {}

static STATE: Lazy<Mutex<Store>> = Lazy::new(|| Mutex::new(Store::new()));

fn nerr(status: Status, message: impl Into<String>) -> Error {
    Error::new(status, message.into())
}

fn lock_store() -> Result<std::sync::MutexGuard<'static, Store>> {
    STATE
        .lock()
        .map_err(|_| nerr(Status::GenericFailure, "native state lock poisoned"))
}

fn parse_json<T: DeserializeOwned>(raw: &str, label: &str) -> Result<T> {
    serde_json::from_str(raw)
        .map_err(|e| nerr(Status::InvalidArg, format!("invalid JSON for {label}: {e}")))
}

fn parse_power_preference(raw: Option<&str>) -> MLPowerPreference {
    match raw.unwrap_or("default") {
        "high-performance" => MLPowerPreference::HighPerformance,
        "low-power" => MLPowerPreference::LowPower,
        _ => MLPowerPreference::Default,
    }
}

pub(crate) fn parse_data_type(raw: &str) -> Result<MLOperandDataType> {
    match raw {
        "float32" => Ok(MLOperandDataType::Float32),
        "float16" => Ok(MLOperandDataType::Float16),
        "int32" => Ok(MLOperandDataType::Int32),
        "uint32" => Ok(MLOperandDataType::Uint32),
        "int64" => Ok(MLOperandDataType::Int64),
        "uint64" => Ok(MLOperandDataType::Uint64),
        "int8" => Ok(MLOperandDataType::Int8),
        "uint8" => Ok(MLOperandDataType::Uint8),
        "int4" => Ok(MLOperandDataType::Int4),
        "uint4" => Ok(MLOperandDataType::Uint4),
        other => Err(nerr(
            Status::InvalidArg,
            format!("unsupported dataType '{other}'"),
        )),
    }
}

fn rustnn_dtype_to_string(dtype: DataType) -> String {
    match dtype {
        DataType::Int4 => "int4",
        DataType::Uint4 => "uint4",
        DataType::Float16 => "float16",
        DataType::Float32 => "float32",
        DataType::Int32 => "int32",
        DataType::Uint32 => "uint32",
        DataType::Int8 => "int8",
        DataType::Uint8 => "uint8",
        DataType::Int64 => "int64",
        DataType::Uint64 => "uint64",
    }
    .to_string()
}

fn descriptor_shape(desc: &rustnn::graph::OperandDescriptor) -> Vec<usize> {
    desc.shape
        .iter()
        .map(|d| get_static_or_max_size(d) as usize)
        .collect()
}

fn wire_descriptor(raw: &TensorDescriptorWire) -> Result<MLTensorDescriptor> {
    let data_type = parse_data_type(&raw.data_type)?;
    let mut descriptor = MLTensorDescriptor::new(data_type, raw.shape.clone());
    if raw.readable.unwrap_or(false) {
        descriptor.set_readable(true);
    }
    if raw.writable.unwrap_or(false) {
        descriptor.set_writable(true);
    }
    Ok(descriptor)
}

pub(crate) fn operand_descriptor(raw: &TensorDescriptorWire) -> Result<MLOperandDescriptor> {
    Ok(MLOperandDescriptor::new(
        parse_data_type(&raw.data_type)?,
        raw.shape.clone(),
    ))
}

fn io_names_from_graph_info(graph_info: &GraphInfo) -> (Vec<String>, Vec<String>) {
    let input_names = graph_info
        .input_operands
        .iter()
        .filter_map(|&id| {
            graph_info
                .operands
                .get(id as usize)
                .and_then(|op| op.name.clone())
        })
        .collect();
    let output_names = graph_info
        .output_operands
        .iter()
        .filter_map(|&id| {
            graph_info
                .operands
                .get(id as usize)
                .and_then(|op| op.name.clone())
        })
        .collect();
    (input_names, output_names)
}

fn meta_from_artifacts(artifacts: &ValidationArtifacts, graph_path: &Path) -> LoadMeta {
    let mut inputs = HashMap::new();
    for (name, desc) in &artifacts.input_names_to_descriptors {
        inputs.insert(
            name.clone(),
            LoadTensorMeta {
                data_type: rustnn_dtype_to_string(desc.data_type),
                shape: descriptor_shape(desc),
            },
        );
    }

    let mut outputs = HashMap::new();
    for (name, desc) in &artifacts.output_names_to_descriptors {
        outputs.insert(
            name.clone(),
            LoadTensorMeta {
                data_type: rustnn_dtype_to_string(desc.data_type),
                shape: descriptor_shape(desc),
            },
        );
    }

    let (input_names, output_names) = if !artifacts.input_names_to_descriptors.is_empty() {
        let mut input_names: Vec<String> = artifacts
            .input_names_to_descriptors
            .keys()
            .cloned()
            .collect();
        input_names.sort();
        let mut output_names: Vec<String> = artifacts
            .output_names_to_descriptors
            .keys()
            .cloned()
            .collect();
        output_names.sort();
        (input_names, output_names)
    } else {
        (Vec::new(), Vec::new())
    };

    LoadMeta {
        graph_path: graph_path.display().to_string(),
        input_names,
        output_names,
        inputs,
        outputs,
    }
}

fn candidate_graph_names() -> &'static [&'static str] {
    &[
        "graph.webnn",
        "model.webnn",
        "smollm.webnn",
        "webnn.webnn",
        "graph.json",
        "model.json",
        "webnn.json",
    ]
}

fn looks_like_graph_json(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
        return false;
    };

    let lower = name.to_ascii_lowercase();
    if lower.contains("manifest")
        || lower.contains("tokenizer")
        || lower == "config.json"
        || lower == "generation_config.json"
    {
        return false;
    }

    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };

    contents.contains("\"operations\"") && contents.contains("\"operands\"")
}

fn find_webnn_graph_path(path_or_dir: &Path) -> Result<PathBuf> {
    if path_or_dir.is_file() {
        return Ok(path_or_dir.to_path_buf());
    }

    if !path_or_dir.is_dir() {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "path does not exist or is not a file/directory: {}",
                path_or_dir.display()
            ),
        ));
    }

    for name in candidate_graph_names() {
        let candidate = path_or_dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let mut webnn_files = Vec::new();
    let mut json_files = Vec::new();

    for entry in WalkDir::new(path_or_dir)
        .max_depth(6)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| v.to_ascii_lowercase());

        match ext.as_deref() {
            Some("webnn") => webnn_files.push(path.to_path_buf()),
            Some("json") => {
                if looks_like_graph_json(path) {
                    json_files.push(path.to_path_buf());
                }
            }
            _ => {}
        }
    }

    webnn_files.sort();
    json_files.sort();

    if let Some(path) = webnn_files.first() {
        return Ok(path.clone());
    }

    if let Some(path) = json_files.first() {
        return Ok(path.clone());
    }

    Err(nerr(
        Status::GenericFailure,
        format!(
            "unable to locate WebNN graph file under {}",
            path_or_dir.display()
        ),
    ))
}

fn with_context<F, T>(context_handle: u32, f: F) -> Result<T>
where
    F: FnOnce(&mut ContextSession) -> Result<T>,
{
    let mut store = lock_store()?;
    let session = store.contexts.get_mut(&context_handle).ok_or_else(|| {
        nerr(
            Status::InvalidArg,
            format!("unknown context handle: {context_handle}"),
        )
    })?;
    f(session)
}

fn with_builder<F, T>(builder_handle: u32, f: F) -> Result<T>
where
    F: FnOnce(&mut BuilderEntry) -> Result<T>,
{
    let mut store = lock_store()?;
    for session in store.contexts.values_mut() {
        if let Some(builder) = session.builders.get_mut(&builder_handle) {
            return f(builder);
        }
    }
    Err(nerr(
        Status::InvalidArg,
        format!("unknown builder handle: {builder_handle}"),
    ))
}

pub(crate) fn register_operand(
    builder: &mut BuilderEntry,
    operand: MLOperand,
    meta: OperandWireMeta,
) -> u32 {
    let handle = builder.next_operand_handle;
    builder.next_operand_handle = builder.next_operand_handle.saturating_add(1);
    builder.operands.insert(handle, (operand, meta));
    handle
}

pub(crate) fn register_operand_inferred(
    builder: &mut BuilderEntry,
    operand: MLOperand,
) -> u32 {
    let rustnn_id = builder.next_rustnn_operand_id;
    builder.next_rustnn_operand_id = builder.next_rustnn_operand_id.saturating_add(1);
    register_operand(
        builder,
        operand,
        OperandWireMeta { rustnn_id },
    )
}

fn write_tensor_bytes(context: &mut MLContext, tensor: &MLTensor, data: &[u8]) -> Result<()> {
    match tensor.data_type() {
        MLOperandDataType::Float32 => {
            let values = bytes_to_pod::<f32>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Float16 => {
            let values = bytes_to_pod::<u16>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Int32 => {
            let values = bytes_to_pod::<i32>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Uint32 => {
            let values = bytes_to_pod::<u32>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Int64 => {
            let values = bytes_to_pod::<i64>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Uint64 => {
            let values = bytes_to_pod::<u64>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Int8 => {
            let values = bytes_to_pod::<i8>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Uint8 => {
            let values = bytes_to_pod::<u8>(data)?;
            context
                .write_tensor(tensor, &values)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
        MLOperandDataType::Int4 | MLOperandDataType::Uint4 => {
            let required = tensor.rustnn_required_bytes();
            if data.len() != required {
                return Err(nerr(
                    Status::InvalidArg,
                    format!(
                        "writeTensor buffer length {} does not match required {} for {}",
                        data.len(),
                        required,
                        tensor.data_type().as_str()
                    ),
                ));
            }
            context
                .write_tensor(tensor, data)
                .map_err(|e| nerr(Status::GenericFailure, format!("writeTensor failed: {e}")))
        }
    }
}

fn read_tensor_bytes(context: &mut MLContext, tensor: &MLTensor) -> Result<Vec<u8>> {
    let logical = tensor.rustnn_required_bytes();
    match tensor.data_type() {
        MLOperandDataType::Float32 => {
            let mut values = vec![0f32; logical / 4];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(bytemuck::cast_slice(&values).to_vec());
        }
        MLOperandDataType::Int64 => {
            let mut values = vec![0i64; logical / 8];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(bytemuck::cast_slice(&values).to_vec());
        }
        MLOperandDataType::Int32 => {
            let mut values = vec![0i32; logical / 4];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(bytemuck::cast_slice(&values).to_vec());
        }
        MLOperandDataType::Uint32 => {
            let mut values = vec![0u32; logical / 4];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(bytemuck::cast_slice(&values).to_vec());
        }
        MLOperandDataType::Uint64 => {
            let mut values = vec![0u64; logical / 8];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(bytemuck::cast_slice(&values).to_vec());
        }
        MLOperandDataType::Int8 => {
            let mut values = vec![0i8; logical];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            Ok(bytemuck::cast_slice(&values).to_vec())
        }
        MLOperandDataType::Uint8 => {
            let mut values = vec![0u8; logical];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            return Ok(values);
        }
        MLOperandDataType::Int4 | MLOperandDataType::Uint4 => {
            let mut values = vec![0u8; logical];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            Ok(values)
        }
        MLOperandDataType::Float16 => {
            let mut values = vec![0u16; logical / 2];
            context
                .read_tensor(tensor, &mut values)
                .map_err(|e| nerr(Status::GenericFailure, format!("readTensor failed: {e}")))?;
            Ok(bytemuck::cast_slice(&values).to_vec())
        }
    }
}

pub(crate) fn bytes_to_pod<T: bytemuck::Pod>(input: &[u8]) -> Result<Vec<T>> {
    if input.len() % std::mem::size_of::<T>() != 0 {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "buffer length {} is not aligned to {}-byte elements",
                input.len(),
                std::mem::size_of::<T>()
            ),
        ));
    }
    Ok(bytemuck::cast_slice(input).to_vec())
}

#[napi(js_name = "createContext")]
pub fn create_context(options_json: String) -> Result<u32> {
    let options: ContextOptionsWire = parse_json(&options_json, "createContext options")?;
    let accelerated = options.accelerated.unwrap_or(true);
    let rustnn_options =
        MLContextOptions::new(parse_power_preference(options.power_preference.as_deref()), accelerated);

    let context = MLContext::create(&rustnn_options).map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("MLContext::create failed: {e}"),
        )
    })?;

    let mut store = lock_store()?;
    let handle = store.next_context_handle;
    store.next_context_handle = store.next_context_handle.saturating_add(1);
    store.contexts.insert(
        handle,
        ContextSession {
            context,
            graphs: HashMap::new(),
            tensors: HashMap::new(),
            builders: HashMap::new(),
            next_graph_handle: 1,
            next_tensor_handle: 1,
            next_builder_handle: 1,
        },
    );
    Ok(handle)
}

#[napi(js_name = "destroyContext")]
pub fn destroy_context(context_handle: u32) -> Result<()> {
    let mut store = lock_store()?;
    store.contexts.remove(&context_handle);
    Ok(())
}

#[napi(js_name = "contextAccelerated")]
pub fn context_accelerated(context_handle: u32) -> Result<bool> {
    with_context(context_handle, |session| Ok(session.context.accelerated()))
}

#[napi(js_name = "createGraphBuilder")]
pub fn create_graph_builder(context_handle: u32) -> Result<u32> {
    with_context(context_handle, |session| {
        let builder = MLGraphBuilder::new(&mut session.context).map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!("MLGraphBuilder::new failed: {e}"),
            )
        })?;
        let handle = session.next_builder_handle;
        session.next_builder_handle = session.next_builder_handle.saturating_add(1);
        session.builders.insert(
            handle,
            BuilderEntry {
                builder,
                operands: HashMap::new(),
                next_operand_handle: 1,
                next_rustnn_operand_id: 0,
            },
        );
        Ok(handle)
    })
}

#[napi(js_name = "destroyGraphBuilder")]
pub fn destroy_graph_builder(builder_handle: u32) -> Result<()> {
    let mut store = lock_store()?;
    for session in store.contexts.values_mut() {
        if session.builders.remove(&builder_handle).is_some() {
            return Ok(());
        }
    }
    Ok(())
}

#[napi(js_name = "builderInput")]
pub fn builder_input(
    builder_handle: u32,
    name: String,
    descriptor_json: String,
) -> Result<u32> {
    let descriptor: TensorDescriptorWire =
        parse_json(&descriptor_json, "builderInput descriptor")?;
    let operand_desc = operand_descriptor(&descriptor)?;

    with_builder(builder_handle, |builder| {
        let operand = builder
            .builder
            .input(&name, &operand_desc)
            .map_err(|e| nerr(Status::GenericFailure, format!("builder.input failed: {e}")))?;
        let rustnn_id = builder.next_rustnn_operand_id;
        builder.next_rustnn_operand_id = builder.next_rustnn_operand_id.saturating_add(1);
        Ok(register_operand(
            builder,
            operand,
            OperandWireMeta { rustnn_id },
        ))
    })
}

#[napi(js_name = "builderInvoke")]
pub fn builder_invoke(builder_handle: u32, invoke_json: String) -> Result<String> {
    let wire: builder_dispatch::BuilderInvokeWire =
        parse_json(&invoke_json, "builderInvoke payload")?;

    with_builder(builder_handle, |builder| {
        let result = builder_dispatch::dispatch_builder_op(builder, wire)?;
        serde_json::to_string(&result).map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!("builderInvoke result serialization failed: {e}"),
            )
        })
    })
}

#[napi(js_name = "builderConstantBuffer")]
pub fn builder_constant_buffer(
    builder_handle: u32,
    descriptor_json: String,
    data: Buffer,
) -> Result<String> {
    let descriptor: TensorDescriptorWire =
        parse_json(&descriptor_json, "builderConstantBuffer descriptor")?;

    with_builder(builder_handle, |builder| {
        let result =
            builder_dispatch::constant_from_buffer(builder, &descriptor, data.as_ref())?;
        serde_json::to_string(&result).map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!("builderConstantBuffer result serialization failed: {e}"),
            )
        })
    })
}

#[napi(js_name = "builderBuild")]
pub async fn builder_build(
    context_handle: u32,
    builder_handle: u32,
    outputs_json: String,
) -> Result<u32> {
    let outputs_map: HashMap<String, u32> = parse_json(&outputs_json, "builderBuild outputs")?;

    tokio::task::spawn_blocking(move || {
        with_context(context_handle, |session| {
            let builder_entry = session.builders.remove(&builder_handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown builder handle: {builder_handle}"),
                )
            })?;

            let mut rust_outputs = HashMap::new();
            for (name, operand_handle) in &outputs_map {
                let (operand, _) = builder_entry.operands.get(operand_handle).ok_or_else(|| {
                    nerr(
                        Status::InvalidArg,
                        format!("unknown output operand handle {operand_handle}"),
                    )
                })?;
                rust_outputs.insert(name.as_str(), *operand);
            }

            let mut builder = builder_entry.builder;
            let graph = builder.build(&rust_outputs).map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("MLGraphBuilder::build failed: {e}"),
                )
            })?;

            let graph_handle = session.next_graph_handle;
            session.next_graph_handle = session.next_graph_handle.saturating_add(1);
            session.graphs.insert(graph_handle, GraphEntry { graph });
            Ok(graph_handle)
        })
    })
    .await
    .map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("builderBuild worker join failure: {e}"),
        )
    })?
}

#[napi(js_name = "createTensor")]
pub fn create_tensor(context_handle: u32, descriptor_json: String) -> Result<u32> {
    let descriptor = wire_descriptor(&parse_json(&descriptor_json, "createTensor descriptor")?)?;

    with_context(context_handle, |session| {
        let tensor = session
            .context
            .create_tensor(&descriptor)
            .map_err(|e| nerr(Status::GenericFailure, format!("createTensor failed: {e}")))?;
        let handle = session.next_tensor_handle;
        session.next_tensor_handle = session.next_tensor_handle.saturating_add(1);
        session.tensors.insert(handle, tensor);
        Ok(handle)
    })
}

#[napi(js_name = "destroyTensor")]
pub fn destroy_tensor(context_handle: u32, tensor_handle: u32) -> Result<()> {
    with_context(context_handle, |session| {
        session.tensors.remove(&tensor_handle);
        Ok(())
    })
}

#[napi(js_name = "writeTensor")]
pub fn write_tensor(context_handle: u32, tensor_handle: u32, data: Buffer) -> Result<()> {
    with_context(context_handle, |session| {
        let tensor = session.tensors.get(&tensor_handle).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("unknown tensor handle: {tensor_handle}"),
            )
        })?;
        write_tensor_bytes(&mut session.context, tensor, data.as_ref())
    })
}

#[napi(js_name = "readTensor")]
pub fn read_tensor(context_handle: u32, tensor_handle: u32) -> Result<Buffer> {
    with_context(context_handle, |session| {
        let tensor = session.tensors.get(&tensor_handle).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("unknown tensor handle: {tensor_handle}"),
            )
        })?;
        let bytes = read_tensor_bytes(&mut session.context, tensor)?;
        Ok(Buffer::from(bytes))
    })
}

#[napi(js_name = "rustnnResizeTensor")]
pub fn rustnn_resize_tensor(
    context_handle: u32,
    tensor_handle: u32,
    shape_json: String,
) -> Result<()> {
    let shape: Vec<u64> = parse_json(&shape_json, "rustnnResizeTensor shape")?;
    with_context(context_handle, |session| {
        let tensor = session
            .tensors
            .get_mut(&tensor_handle)
            .ok_or_else(|| nerr(Status::InvalidArg, format!("unknown tensor handle: {tensor_handle}")))?;
        session
            .context
            .rustnn_resize_tensor(tensor, &shape)
            .map_err(|e| nerr(Status::GenericFailure, format!("rustnn_resize_tensor failed: {e}")))
    })
}

#[napi(js_name = "rustnnSetTensorCapacity")]
pub fn rustnn_set_tensor_capacity(
    context_handle: u32,
    tensor_handle: u32,
    shape_json: String,
) -> Result<()> {
    let shape: Vec<u64> = parse_json(&shape_json, "rustnnSetTensorCapacity shape")?;
    with_context(context_handle, |session| {
        let tensor = session
            .tensors
            .get_mut(&tensor_handle)
            .ok_or_else(|| nerr(Status::InvalidArg, format!("unknown tensor handle: {tensor_handle}")))?;
        session
            .context
            .rustnn_set_tensor_capacity(tensor, &shape)
            .map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("rustnn_set_tensor_capacity failed: {e}"),
                )
            })
    })
}

#[napi(js_name = "dispatch")]
pub fn dispatch(
    context_handle: u32,
    graph_handle: u32,
    inputs_json: String,
    outputs_json: String,
) -> Result<()> {
    let inputs_map: HashMap<String, u32> = parse_json(&inputs_json, "dispatch inputs")?;
    let outputs_map: HashMap<String, u32> = parse_json(&outputs_json, "dispatch outputs")?;

    with_context(context_handle, |session| {
        let graph_entry = session.graphs.get_mut(&graph_handle).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("unknown graph handle: {graph_handle}"),
            )
        })?;

        let mut inputs = HashMap::new();
        for (name, tensor_handle) in &inputs_map {
            let tensor = session.tensors.get(tensor_handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown input tensor handle {tensor_handle} for '{name}'"),
                )
            })?;
            inputs.insert(name.as_str(), tensor);
        }

        let mut outputs = HashMap::new();
        for (name, tensor_handle) in &outputs_map {
            let tensor = session.tensors.get(tensor_handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown output tensor handle {tensor_handle} for '{name}'"),
                )
            })?;
            outputs.insert(name.as_str(), tensor);
        }

        session
            .context
            .dispatch(&mut graph_entry.graph, &inputs, &outputs)
            .map_err(|e| nerr(Status::GenericFailure, format!("dispatch failed: {e}")))
    })
}

#[napi(js_name = "destroyGraph")]
pub fn destroy_graph(context_handle: u32, graph_handle: u32) -> Result<()> {
    with_context(context_handle, |session| {
        session.graphs.remove(&graph_handle);
        Ok(())
    })
}

#[napi(js_name = "loadWebnnModel")]
pub async fn load_webnn_model(context_handle: u32, path_or_dir: String) -> Result<NativeModelLoadResult> {
    tokio::task::spawn_blocking(move || {
        let resolved_path = find_webnn_graph_path(Path::new(&path_or_dir))?;
        let graph_info = load_graph_from_path(&resolved_path).map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!(
                    "failed to load WebNN graph from {}: {e}",
                    resolved_path.display()
                ),
            )
        })?;

        let context_props = ContextProperties {
            tensor_byte_length_limit: 1_000_000_000_000usize,
            ..Default::default()
        };
        let artifacts = GraphValidator::new(&graph_info, context_props)
            .validate()
            .map_err(|e| nerr(Status::GenericFailure, format!("graph validation failed: {e}")))?;
        let meta = meta_from_artifacts(&artifacts, &resolved_path);
        let (input_names, output_names) = io_names_from_graph_info(&graph_info);
        let mut meta = meta;
        if !input_names.is_empty() {
            meta.input_names = input_names;
        }
        if !output_names.is_empty() {
            meta.output_names = output_names;
        }

        with_context(context_handle, |session| {
            let mut builder = MLGraphBuilder::new(&mut session.context).map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("MLGraphBuilder::new failed: {e}"),
                )
            })?;
            let graph = builder.build_graph_info(graph_info).map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("build_graph_info failed: {e}"),
                )
            })?;

            let graph_handle = session.next_graph_handle;
            session.next_graph_handle = session.next_graph_handle.saturating_add(1);
            session.graphs.insert(graph_handle, GraphEntry { graph });

            let meta_json = serde_json::to_string(&meta).map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("failed to serialize loadWebnnModel metadata: {e}"),
                )
            })?;

            Ok(NativeModelLoadResult {
                graph_handle,
                meta_json,
            })
        })
    })
    .await
    .map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("loadWebnnModel worker join failure: {e}"),
        )
    })?
}
