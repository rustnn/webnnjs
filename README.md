# webnnjs

Node.js WebNN-flavor polyfill API backed by `rustnn` and ONNX Runtime via a Rust napi-rs addon.

This is a Node polyfill (not a browser implementation). It keeps WebNN naming close to the W3C WebNN API, with Node-only helpers for model loading.

## Layout

- `packages/webnn-node/` - TypeScript WebNN-flavor API
- `packages/webnn-node/native/` - Rust napi-rs addon (`rustnn` + ONNX Runtime)
- `demo/` - Demo app that downloads and runs a WebNN model from Hugging Face Hub

## Prerequisites

- Node.js `>= 20`
- Rust toolchain (`cargo`, `rustc`)
- Build toolchain for native addons (C/C++ compiler, linker, Python if required by your platform)

## rustnn dependency

The native addon depends on upstream `rustnn` from GitHub (configured in `packages/webnn-node/native/Cargo.toml`):

- `https://github.com/rustnn/rustnn` (`branch = "main"`)

If you want to test with a local `rustnn` checkout, change that dependency to a local `path` in `Cargo.toml`.

## ONNX Runtime shared library

`rustnn` uses ONNX Runtime dynamic loading (`ort` with `load-dynamic`). You need ONNX Runtime shared libraries installed and discoverable.

Preferred setup:

1. Install ONNX Runtime shared library for your platform.
2. Set one of:
   - `ORT_DYLIB_PATH` to the exact ORT library file.
   - `ORT_LIB_DIR` to a directory containing ORT libs.

Examples:

```bash
# macOS
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib

# Linux
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so

# Windows (PowerShell)
$env:ORT_DYLIB_PATH="C:\path\to\onnxruntime.dll"
```

The demo also tries common install locations automatically; explicit env vars are still the most reliable.

## Install and Build

```bash
npm install
npm run build
```

Equivalent Makefile flow:

```bash
make install
make build
```

## Run Demo

```bash
npm run demo
```

Makefile flow (downloads ORT like rustnn then runs demo with env vars):

```bash
make demo
```

Optional demo overrides:

```bash
DEMO_PROMPT="The future of AI is" DEMO_MAX_NEW_TOKENS=32 npm run demo
```

With `make`:

```bash
DEMO_PROMPT="The future of AI is" DEMO_MAX_NEW_TOKENS=32 make demo
```

Demo behavior:

1. Downloads/caches `tarekziade/SmolLM-135M-webnn` via `@huggingface/hub` `snapshotDownload`.
2. Loads WebNN graph files from the snapshot directory.
3. Uses `rustnn` to parse/validate/lower to ONNX.
4. Executes with ONNX Runtime backend.
5. Performs autoregressive generation and prints generated text.

## Makefile Targets

This repo now includes a Makefile patterned after rustnn's ORT setup logic:

- `make onnxruntime-download` downloads the platform-specific ONNX Runtime package into `target/onnxruntime`.
- `make demo` injects `ORT_DYLIB_PATH` (and `LD_LIBRARY_PATH` on Linux) before running the demo.
- `make demo-only` runs the already-built demo with the same ORT env setup.
- `make clean` removes build artifacts and downloaded ORT files.

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
