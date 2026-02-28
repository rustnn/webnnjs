use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::Buffer;
use napi::{Error, Result, Status};
use napi_derive::napi;
use once_cell::sync::Lazy;
use prost::Message;
use rustnn::graph::{get_static_or_max_size, DataType, OperandDescriptor};
use rustnn::protos::onnx::ModelProto;
use rustnn::{
    load_graph_from_path, run_onnx_with_inputs, ContextProperties, ConverterRegistry, GraphInfo,
    GraphValidator, OnnxInput, TensorData,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ContextOptions {
    device_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TensorDescriptorWire {
    data_type: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TensorState {
    context_handle: u32,
    descriptor: TensorDescriptorWire,
    data: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ContextState {
    _options: ContextOptions,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct GraphState {
    context_handle: u32,
    onnx_bytes: Arc<Vec<u8>>,
    input_order: Vec<String>,
    output_order: Vec<String>,
    input_types: HashMap<String, String>,
    output_types: HashMap<String, String>,
    input_shapes: HashMap<String, Vec<usize>>,
    output_shapes: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Default)]
struct NativeState {
    next_context_handle: u32,
    next_graph_handle: u32,
    next_tensor_handle: u32,
    contexts: HashMap<u32, ContextState>,
    graphs: HashMap<u32, GraphState>,
    tensors: HashMap<u32, TensorState>,
}

impl NativeState {
    fn new() -> Self {
        Self {
            next_context_handle: 1,
            next_graph_handle: 1,
            next_tensor_handle: 1,
            ..Self::default()
        }
    }
}

static STATE: Lazy<Mutex<NativeState>> = Lazy::new(|| Mutex::new(NativeState::new()));

#[derive(Debug, Clone)]
struct CompiledGraph {
    onnx_bytes: Arc<Vec<u8>>,
    input_order: Vec<String>,
    output_order: Vec<String>,
    input_types: HashMap<String, String>,
    output_types: HashMap<String, String>,
    input_shapes: HashMap<String, Vec<usize>>,
    output_shapes: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadMeta {
    graph_path: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    inputs: HashMap<String, LoadTensorMeta>,
    outputs: HashMap<String, LoadTensorMeta>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadTensorMeta {
    data_type: String,
    shape: Vec<usize>,
}

#[napi(object)]
pub struct NativeModelLoadResult {
    pub graph_handle: u32,
    pub meta_json: String,
}

fn nerr(status: Status, message: impl Into<String>) -> Error {
    Error::new(status, message.into())
}

fn lock_state() -> Result<std::sync::MutexGuard<'static, NativeState>> {
    STATE
        .lock()
        .map_err(|_| nerr(Status::GenericFailure, "native state lock poisoned"))
}

fn parse_json<T: DeserializeOwned>(raw: &str, label: &str) -> Result<T> {
    serde_json::from_str(raw)
        .map_err(|e| nerr(Status::InvalidArg, format!("invalid JSON for {label}: {e}")))
}

fn checked_element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut total = 1usize;
    for dim in shape {
        total = total.checked_mul(*dim).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("shape element count overflow for shape {shape:?}"),
            )
        })?;
    }
    Ok(total)
}

fn bytes_to_f32(input: &[u8]) -> Result<Vec<f32>> {
    if input.len() % 4 != 0 {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "buffer length must be divisible by 4 for float32, got {}",
                input.len()
            ),
        ));
    }

    let mut values = Vec::with_capacity(input.len() / 4);
    for chunk in input.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn f32_to_bytes(input: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len() * 4);
    for value in input {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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

fn descriptor_shape(desc: &OperandDescriptor) -> Vec<usize> {
    desc.shape
        .iter()
        .map(|d| get_static_or_max_size(d) as usize)
        .collect()
}

fn descriptor_maps(
    descriptors: &HashMap<String, OperandDescriptor>,
) -> (HashMap<String, String>, HashMap<String, Vec<usize>>) {
    let mut types = HashMap::with_capacity(descriptors.len());
    let mut shapes = HashMap::with_capacity(descriptors.len());

    for (name, desc) in descriptors {
        types.insert(name.clone(), rustnn_dtype_to_string(desc.data_type));
        shapes.insert(name.clone(), descriptor_shape(desc));
    }

    (types, shapes)
}

fn extract_onnx_io_order(model_bytes: &[u8]) -> Result<(Vec<String>, Vec<String>)> {
    let model = ModelProto::decode(model_bytes).map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("failed to decode ONNX model bytes: {e}"),
        )
    })?;

    let graph = model
        .graph
        .ok_or_else(|| nerr(Status::GenericFailure, "ONNX model has no graph"))?;

    let input_order = graph.input.into_iter().map(|value| value.name).collect();
    let output_order = graph.output.into_iter().map(|value| value.name).collect();
    Ok((input_order, output_order))
}

fn compile_graph_info(graph: GraphInfo) -> Result<CompiledGraph> {
    let mut context_props = ContextProperties::default();
    context_props.tensor_byte_length_limit = 1_000_000_000_000usize;
    let artifacts = GraphValidator::new(&graph, context_props)
        .validate()
        .map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!("graph validation failed: {e}"),
            )
        })?;

    let converted = ConverterRegistry::with_defaults()
        .convert("onnx", &graph)
        .map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!("graph conversion to ONNX failed: {e}"),
            )
        })?;

    let (input_order, output_order) = extract_onnx_io_order(&converted.data)?;
    let (input_types, input_shapes) = descriptor_maps(&artifacts.input_names_to_descriptors);
    let (output_types, output_shapes) = descriptor_maps(&artifacts.output_names_to_descriptors);

    Ok(CompiledGraph {
        onnx_bytes: Arc::new(converted.data),
        input_order,
        output_order,
        input_types,
        output_types,
        input_shapes,
        output_shapes,
    })
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

fn to_onnx_input(name: &str, dtype: &str, shape: &[usize], data: &[f32]) -> Result<OnnxInput> {
    let tensor_data = match dtype {
        "float32" => TensorData::Float32(data.to_vec()),
        "int64" => TensorData::Int64(data.iter().map(|v| *v as i64).collect()),
        "int32" => TensorData::Int32(data.iter().map(|v| *v as i32).collect()),
        "uint32" => TensorData::Uint32(data.iter().map(|v| *v as u32).collect()),
        "uint64" => TensorData::Uint64(data.iter().map(|v| *v as u64).collect()),
        "int8" => TensorData::Int8(data.iter().map(|v| *v as i8).collect()),
        "uint8" => TensorData::Uint8(data.iter().map(|v| *v as u8).collect()),
        other => {
            return Err(nerr(
                Status::GenericFailure,
                format!("unsupported model input dtype in dispatch: {other}"),
            ));
        }
    };

    Ok(OnnxInput {
        name: name.to_string(),
        shape: shape.to_vec(),
        data: tensor_data,
    })
}

fn output_to_f32(output: &rustnn::OnnxOutputWithData) -> Vec<f32> {
    if let Some(int64_values) = &output.int64_data {
        return int64_values.iter().map(|v| *v as f32).collect();
    }

    if let Some(uint64_values) = &output.uint64_data {
        return uint64_values.iter().map(|v| *v as f32).collect();
    }

    output.data.iter().map(|v| *v as f32).collect()
}

#[napi(js_name = "createContext")]
pub fn create_context(options_json: String) -> Result<u32> {
    let options: ContextOptions = parse_json(&options_json, "createContext options")?;
    let mut warnings = Vec::new();

    let device_type = options
        .device_type
        .clone()
        .unwrap_or_else(|| "cpu".to_string());
    if device_type.eq_ignore_ascii_case("gpu") {
        let warning = "deviceType='gpu' requested; current rustnn ONNX wrapper initializes CPU EP, falling back to CPU".to_string();
        eprintln!("[webnn-node-native] warning: {warning}");
        warnings.push(warning);
    }

    let mut state = lock_state()?;
    let handle = state.next_context_handle;
    state.next_context_handle = state.next_context_handle.saturating_add(1);

    state.contexts.insert(
        handle,
        ContextState {
            _options: options,
            warnings,
        },
    );

    Ok(handle)
}

#[napi(js_name = "destroyContext")]
pub fn destroy_context(context_handle: u32) -> Result<()> {
    let mut state = lock_state()?;
    state.contexts.remove(&context_handle);

    let graph_ids: Vec<u32> = state
        .graphs
        .iter()
        .filter_map(|(handle, graph)| {
            if graph.context_handle == context_handle {
                Some(*handle)
            } else {
                None
            }
        })
        .collect();

    for graph_id in graph_ids {
        state.graphs.remove(&graph_id);
    }

    let tensor_ids: Vec<u32> = state
        .tensors
        .iter()
        .filter_map(|(handle, tensor)| {
            if tensor.context_handle == context_handle {
                Some(*handle)
            } else {
                None
            }
        })
        .collect();

    for tensor_id in tensor_ids {
        state.tensors.remove(&tensor_id);
    }

    Ok(())
}

#[napi(js_name = "createTensor")]
pub fn create_tensor(context_handle: u32, descriptor_json: String) -> Result<u32> {
    let descriptor: TensorDescriptorWire = parse_json(&descriptor_json, "createTensor descriptor")?;

    if descriptor.data_type != "float32" {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "only float32 tensors are supported in MVP, got {}",
                descriptor.data_type
            ),
        ));
    }

    let element_count = checked_element_count(&descriptor.shape)?;

    let mut state = lock_state()?;
    if !state.contexts.contains_key(&context_handle) {
        return Err(nerr(
            Status::InvalidArg,
            format!("unknown context handle: {context_handle}"),
        ));
    }

    let handle = state.next_tensor_handle;
    state.next_tensor_handle = state.next_tensor_handle.saturating_add(1);

    state.tensors.insert(
        handle,
        TensorState {
            context_handle,
            descriptor,
            data: vec![0.0_f32; element_count],
        },
    );

    Ok(handle)
}

#[napi(js_name = "destroyTensor")]
pub fn destroy_tensor(context_handle: u32, tensor_handle: u32) -> Result<()> {
    let mut state = lock_state()?;

    let Some(tensor) = state.tensors.get(&tensor_handle) else {
        return Ok(());
    };

    if tensor.context_handle != context_handle {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "tensor handle {} does not belong to context {}",
                tensor_handle, context_handle
            ),
        ));
    }

    state.tensors.remove(&tensor_handle);
    Ok(())
}

#[napi(js_name = "writeTensor")]
pub fn write_tensor(context_handle: u32, tensor_handle: u32, data: Buffer) -> Result<()> {
    let values = bytes_to_f32(data.as_ref())?;

    let mut state = lock_state()?;
    let tensor = state.tensors.get_mut(&tensor_handle).ok_or_else(|| {
        nerr(
            Status::InvalidArg,
            format!("unknown tensor handle: {tensor_handle}"),
        )
    })?;

    if tensor.context_handle != context_handle {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "tensor handle {} does not belong to context {}",
                tensor_handle, context_handle
            ),
        ));
    }

    let expected = checked_element_count(&tensor.descriptor.shape)?;
    if values.len() != expected {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "tensor data length mismatch for tensor {}: expected {}, got {}",
                tensor_handle,
                expected,
                values.len()
            ),
        ));
    }

    tensor.data = values;
    Ok(())
}

#[napi(js_name = "readTensor")]
pub fn read_tensor(context_handle: u32, tensor_handle: u32) -> Result<Buffer> {
    let state = lock_state()?;
    let tensor = state.tensors.get(&tensor_handle).ok_or_else(|| {
        nerr(
            Status::InvalidArg,
            format!("unknown tensor handle: {tensor_handle}"),
        )
    })?;

    if tensor.context_handle != context_handle {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "tensor handle {} does not belong to context {}",
                tensor_handle, context_handle
            ),
        ));
    }

    Ok(Buffer::from(f32_to_bytes(&tensor.data)))
}

#[napi(js_name = "compileGraph")]
pub async fn compile_graph(context_handle: u32, graph_json: String) -> Result<u32> {
    {
        let state = lock_state()?;
        if !state.contexts.contains_key(&context_handle) {
            return Err(nerr(
                Status::InvalidArg,
                format!("unknown context handle: {context_handle}"),
            ));
        }
    }

    let compiled = tokio::task::spawn_blocking(move || {
        let graph: GraphInfo = serde_json::from_str(&graph_json).map_err(|e| {
            nerr(
                Status::InvalidArg,
                format!("failed to parse graph JSON for compileGraph: {e}"),
            )
        })?;
        compile_graph_info(graph)
    })
    .await
    .map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("compileGraph worker join failure: {e}"),
        )
    })??;

    let mut state = lock_state()?;
    if !state.contexts.contains_key(&context_handle) {
        return Err(nerr(
            Status::InvalidArg,
            format!("context {context_handle} was destroyed during compileGraph"),
        ));
    }

    let graph_handle = state.next_graph_handle;
    state.next_graph_handle = state.next_graph_handle.saturating_add(1);

    state.graphs.insert(
        graph_handle,
        GraphState {
            context_handle,
            onnx_bytes: compiled.onnx_bytes,
            input_order: compiled.input_order,
            output_order: compiled.output_order,
            input_types: compiled.input_types,
            output_types: compiled.output_types,
            input_shapes: compiled.input_shapes,
            output_shapes: compiled.output_shapes,
        },
    );

    Ok(graph_handle)
}

#[napi(js_name = "loadWebnnModel")]
pub async fn load_webnn_model(
    context_handle: u32,
    path_or_dir: String,
    options_json: String,
) -> Result<NativeModelLoadResult> {
    let options: ContextOptions = parse_json(&options_json, "loadWebnnModel options")?;

    let context_warnings = {
        let state = lock_state()?;
        let Some(ctx) = state.contexts.get(&context_handle) else {
            return Err(nerr(
                Status::InvalidArg,
                format!("unknown context handle: {context_handle}"),
            ));
        };
        ctx.warnings.clone()
    };

    let input_path = PathBuf::from(path_or_dir);
    let compiled_result = tokio::task::spawn_blocking(move || {
        let resolved_path = find_webnn_graph_path(&input_path)?;
        let graph = load_graph_from_path(&resolved_path).map_err(|e| {
            nerr(
                Status::GenericFailure,
                format!(
                    "failed to load WebNN graph from {}: {e}",
                    resolved_path.display()
                ),
            )
        })?;

        let compiled = compile_graph_info(graph)?;

        let mut warnings = Vec::new();
        if options
            .device_type
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("gpu"))
            .unwrap_or(false)
        {
            warnings.push(
                "deviceType='gpu' requested; ONNX Runtime CPU EP was used by rustnn wrapper"
                    .to_string(),
            );
        }

        Ok::<(PathBuf, CompiledGraph, Vec<String>), Error>((resolved_path, compiled, warnings))
    })
    .await
    .map_err(|e| {
        nerr(
            Status::GenericFailure,
            format!("loadWebnnModel worker join failure: {e}"),
        )
    })??;

    let (resolved_path, compiled, mut warnings) = compiled_result;
    warnings.extend(context_warnings);

    let mut state = lock_state()?;
    if !state.contexts.contains_key(&context_handle) {
        return Err(nerr(
            Status::InvalidArg,
            format!("context {context_handle} was destroyed during loadWebnnModel"),
        ));
    }

    let graph_handle = state.next_graph_handle;
    state.next_graph_handle = state.next_graph_handle.saturating_add(1);

    let graph_state = GraphState {
        context_handle,
        onnx_bytes: compiled.onnx_bytes,
        input_order: compiled.input_order,
        output_order: compiled.output_order,
        input_types: compiled.input_types,
        output_types: compiled.output_types,
        input_shapes: compiled.input_shapes,
        output_shapes: compiled.output_shapes,
    };

    let inputs_meta: HashMap<String, LoadTensorMeta> = graph_state
        .input_order
        .iter()
        .filter_map(|name| {
            let data_type = graph_state.input_types.get(name)?;
            let shape = graph_state.input_shapes.get(name)?;
            Some((
                name.clone(),
                LoadTensorMeta {
                    data_type: data_type.clone(),
                    shape: shape.clone(),
                },
            ))
        })
        .collect();

    let outputs_meta: HashMap<String, LoadTensorMeta> = graph_state
        .output_order
        .iter()
        .filter_map(|name| {
            let data_type = graph_state.output_types.get(name)?;
            let shape = graph_state.output_shapes.get(name)?;
            Some((
                name.clone(),
                LoadTensorMeta {
                    data_type: data_type.clone(),
                    shape: shape.clone(),
                },
            ))
        })
        .collect();

    let meta = LoadMeta {
        graph_path: resolved_path.display().to_string(),
        input_names: graph_state.input_order.clone(),
        output_names: graph_state.output_order.clone(),
        inputs: inputs_meta,
        outputs: outputs_meta,
        warnings,
    };

    state.graphs.insert(graph_handle, graph_state);

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
}

struct DispatchPayload {
    onnx_bytes: Arc<Vec<u8>>,
    onnx_inputs: Vec<OnnxInput>,
    outputs_map: HashMap<String, u32>,
    context_handle: u32,
}

#[napi(js_name = "dispatch")]
pub async fn dispatch(
    context_handle: u32,
    graph_handle: u32,
    inputs_json: String,
    outputs_json: String,
) -> Result<()> {
    let inputs_map: HashMap<String, u32> = parse_json(&inputs_json, "dispatch inputs")?;
    let outputs_map: HashMap<String, u32> = parse_json(&outputs_json, "dispatch outputs")?;

    let payload = {
        let state = lock_state()?;
        let graph = state.graphs.get(&graph_handle).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("unknown graph handle: {graph_handle}"),
            )
        })?;

        if graph.context_handle != context_handle {
            return Err(nerr(
                Status::InvalidArg,
                format!(
                    "graph handle {} does not belong to context {}",
                    graph_handle, context_handle
                ),
            ));
        }

        let mut onnx_inputs = Vec::with_capacity(graph.input_order.len());
        for input_name in &graph.input_order {
            let tensor_handle = inputs_map.get(input_name).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("missing input tensor for model input '{input_name}'"),
                )
            })?;

            let tensor = state.tensors.get(tensor_handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown tensor handle for input '{input_name}': {tensor_handle}"),
                )
            })?;

            if tensor.context_handle != context_handle {
                return Err(nerr(
                    Status::InvalidArg,
                    format!(
                        "input tensor {} does not belong to context {}",
                        tensor_handle, context_handle
                    ),
                ));
            }

            let expected_dtype = graph
                .input_types
                .get(input_name)
                .map(String::as_str)
                .unwrap_or("float32");

            let shape = if let Some(expected_shape) = graph.input_shapes.get(input_name) {
                if !expected_shape.is_empty() && *expected_shape != tensor.descriptor.shape {
                    tensor.descriptor.shape.clone()
                } else {
                    expected_shape.clone()
                }
            } else {
                tensor.descriptor.shape.clone()
            };

            let expected_elements = checked_element_count(&shape)?;
            if tensor.data.len() != expected_elements {
                return Err(nerr(
                    Status::InvalidArg,
                    format!(
                        "input tensor '{}' has {} elements but shape {:?} requires {}",
                        input_name,
                        tensor.data.len(),
                        shape,
                        expected_elements
                    ),
                ));
            }

            onnx_inputs.push(to_onnx_input(
                input_name,
                expected_dtype,
                &shape,
                &tensor.data,
            )?);
        }

        for (output_name, tensor_handle) in &outputs_map {
            let tensor = state.tensors.get(tensor_handle).ok_or_else(|| {
                nerr(
                    Status::InvalidArg,
                    format!("unknown tensor handle for output '{output_name}': {tensor_handle}"),
                )
            })?;
            if tensor.context_handle != context_handle {
                return Err(nerr(
                    Status::InvalidArg,
                    format!(
                        "output tensor {} does not belong to context {}",
                        tensor_handle, context_handle
                    ),
                ));
            }
        }

        DispatchPayload {
            onnx_bytes: graph.onnx_bytes.clone(),
            onnx_inputs,
            outputs_map,
            context_handle,
        }
    };

    let DispatchPayload {
        onnx_bytes,
        onnx_inputs,
        outputs_map,
        context_handle: dispatch_context_handle,
    } = payload;

    let outputs =
        tokio::task::spawn_blocking(move || run_onnx_with_inputs(onnx_bytes.as_ref(), onnx_inputs))
            .await
            .map_err(|e| {
                nerr(
                    Status::GenericFailure,
                    format!("dispatch worker join failure: {e}"),
                )
            })?
            .map_err(|e| nerr(Status::GenericFailure, format!("dispatch failed: {e}")))?;

    let outputs_by_name: HashMap<String, rustnn::OnnxOutputWithData> = outputs
        .into_iter()
        .map(|output| (output.name.clone(), output))
        .collect();

    let mut state = lock_state()?;
    if !state.contexts.contains_key(&dispatch_context_handle) {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "context {} was destroyed during dispatch",
                dispatch_context_handle
            ),
        ));
    }

    for (output_name, tensor_handle) in outputs_map {
        let output = outputs_by_name.get(&output_name).ok_or_else(|| {
            nerr(
                Status::GenericFailure,
                format!("model output '{output_name}' was not produced by ONNX execution"),
            )
        })?;

        let output_values = output_to_f32(output);
        let tensor = state.tensors.get_mut(&tensor_handle).ok_or_else(|| {
            nerr(
                Status::InvalidArg,
                format!("output tensor handle no longer exists: {tensor_handle}"),
            )
        })?;

        tensor.data = output_values;
        tensor.descriptor.shape = output.shape.clone();
    }

    Ok(())
}

#[napi(js_name = "destroyGraph")]
pub fn destroy_graph(context_handle: u32, graph_handle: u32) -> Result<()> {
    let mut state = lock_state()?;

    let Some(graph) = state.graphs.get(&graph_handle) else {
        return Ok(());
    };

    if graph.context_handle != context_handle {
        return Err(nerr(
            Status::InvalidArg,
            format!(
                "graph handle {} does not belong to context {}",
                graph_handle, context_handle
            ),
        ));
    }

    state.graphs.remove(&graph_handle);
    Ok(())
}
