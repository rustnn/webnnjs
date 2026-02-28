# webnnjs

Node.js WebNN-flavor polyfill API backed by `rustnn` and ONNX Runtime via a Rust napi-rs addon.

This project is a Node polyfill (not a browser implementation). It keeps WebNN naming close to the W3C WebNN spec while adding Node-only model loading helpers.

## Layout

- `packages/webnn-node/` - TypeScript WebNN-flavor API
- `packages/webnn-node/native/` - Rust napi-rs addon (`rustnn` + ONNX Runtime)
- `demo/` - Demo app that downloads and runs a WebNN model from Hugging Face Hub

## Prerequisites

- Node.js `>= 20`
- Rust toolchain (`cargo`, `rustc`)
- A local `rustnn` checkout at:
  - `/Users/tarek/Dev/rustnn`

The native addon depends on `rustnn` with feature `onnx-runtime`.

### ONNX Runtime dynamic library

`rustnn` enables ONNX Runtime dynamic loading (`ort` with `load-dynamic`). You must provide ONNX Runtime shared libraries on your system.

Typical approach:

- Install ONNX Runtime shared library for your platform.
- Set environment variable expected by `ort` crate (for example `ORT_DYLIB_PATH`) to point to the ONNX Runtime dynamic library.
- If you have CUDA-enabled ORT, that can be used by ORT itself. This MVP currently falls back to CPU through `rustnn`'s ONNX executor path.

## Install and Build

```bash
npm install
npm run build
```

## Run Demo

```bash
npm run demo
```

Demo behavior:

1. Downloads/caches `tarekziade/SmolLM-135M-webnn` via `@huggingface/hub` `snapshotDownload`.
2. Loads WebNN graph files from the snapshot directory.
3. Uses `rustnn` to parse/validate/lower to ONNX.
4. Executes with ONNX Runtime backend.
5. Prints top-5 logits.

## API Surface (MVP)

- `installWebNNPolyfill()` attaches `navigator.ml` on `globalThis`
- `ml.createContext(options)`
- `new MLGraphBuilder(context)`
- `builder.input(name, descriptor)`
- `builder.constant(descriptor, data)`
- `builder.add(a, b)`
- `builder.mul(a, b)`
- `builder.build(outputs)`
- `context.createTensor(descriptor)`
- `context.writeTensor(tensor, data)`
- `context.dispatch(graph, inputs, outputs)`
- `context.readTensor(tensor)`
- `graph.destroy()`

Node-only helper:

- `ml.loadModelFromHub(repoId, options)`
