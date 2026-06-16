# webnnjs

Node.js WebNN-flavor API backed by `rustnn` and ONNX Runtime via a Rust napi-rs addon.

This is a Node polyfill (not a browser implementation). The public API follows the [W3C WebNN IDL](https://www.w3.org/TR/webnn/), with Node-only helpers for Hub model loading.

## Layout

- `packages/webnn-node/` — TypeScript WebNN API (`MLContext`, `MLGraphBuilder`, …)
- `packages/webnn-node/native/` — Rust napi-rs addon
- `packages/webnn-node/idl/webnn.idl` — WPT IDL snapshot used to codegen builder ops
- `demo/` — Runnable examples

## Prerequisites

- Node.js `>= 20`
- Rust toolchain (`cargo`, `rustc`)
- Native build toolchain (C/C++ compiler, linker)
- ONNX Runtime shared library (see below)
- Local [`rustnn`](https://github.com/rustnn/rustnn) checkout at `../rustnn` relative to this repo (see `packages/webnn-node/native/Cargo.toml`)

## ONNX Runtime

`rustnn` loads ONNX Runtime dynamically. Point it at your ORT shared library:

| Platform | Example |
|----------|---------|
| Windows | `ORT_DYLIB_PATH=C:\path\to\onnxruntime.dll` |
| Linux | `ORT_DYLIB_PATH=/path/to/libonnxruntime.so` |
| macOS | `ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib` |

Alternatively, set `ORT_LIB_DIR` to a directory that contains the library.

Both demos read `demo/.env` via `node --env-file=.env`. Copy or edit that file for your machine:

```env
ORT_DYLIB_PATH=C:\git\rustnn-workspace\onnxruntime.dll
```

The demo code also searches common install locations if the variable is unset, but an explicit path is most reliable.

## Build

From the repo root:

```bash
npm install
npm run build
```

This runs, in order:

1. **Native addon** (`packages/webnn-node/native`) — `cargo build --release`, then `scripts/install-addon.mjs` copies the release cdylib to `index.node`
2. **webnn-node** — regenerates `MLGraphBuilder` methods from `webnn.idl`, compiles TypeScript
3. **demo** — compiles demo TypeScript

Build individual packages:

```bash
# Native only
npm run build -w @webnnjs/webnn-node-native

# TypeScript package (includes native)
npm --prefix packages/webnn-node run build

# Demo only (requires webnn-node built first)
npm --prefix demo run build
```

### Native addon staging

The copy to `index.node` happens in `scripts/install-addon.mjs` after `cargo build`, not in `build.rs` (`build.rs` only runs `napi_build::setup()` before linking).

If `index.node` is locked on Windows (a Node process still running), the install script stages `index.staged.node` instead. Close Node processes, then:

```bash
npm run install-addon -w @webnnjs/webnn-node-native
```

`index.js` loads the newest `*.node` file in the native package directory.

Optional: `npm run build:napi -w @webnnjs/webnn-node-native` uses `@napi-rs/cli` instead (also regenerates `index.d.ts` from `#[napi]` exports).

## Run demos

Demos require a successful build and a valid `ORT_DYLIB_PATH` in `demo/.env` (or auto-discovered ORT).

### 1. MLGraphBuilder `add` example

Builds a 2×2 float32 `add` graph in TypeScript, executes it, and prints element-wise sums.

```bash
npm run demo:builder
```

Skip the rebuild if already built:

```bash
npm --prefix demo run demo:builder
```

Expected output:

```
Input A: [ 1, 2, 3, 4 ]
Input B: [ 5, 6, 7, 8 ]
A + B =   [ 6, 8, 10, 12 ]
```

Source: [`demo/src/builder-add.ts`](demo/src/builder-add.ts)

### 2. SmolLM text generation (Hugging Face Hub)

Downloads `tarekziade/SmolLM-135M-webnn`, loads the WebNN graph via `MLContext`, and runs autoregressive generation.

```bash
npm run demo
```

Skip the rebuild:

```bash
npm run demo:run
```

Optional overrides:

```bash
DEMO_PROMPT="The future of AI is" DEMO_MAX_NEW_TOKENS=32 npm run demo:run
```

Source: [`demo/src/index.ts`](demo/src/index.ts)

## Makefile (optional)

On Unix, `make` can download ONNX Runtime and run the SmolLM demo with env vars set:

```bash
make install
make build
make demo              # download ORT + build + run SmolLM demo
make demo-only         # run SmolLM demo (already built)
make onnxruntime-download
make clean
```

There is no `make` target for `demo:builder` yet; use `npm run demo:builder`.

## API overview

Polyfill entry:

- `installWebNNPolyfill()` — attaches `navigator.ml`
- `ml.createContext(options)`
- `ml.loadModelFromHub(repoId, options)` — Node-only Hub download helper

WebNN IDL types:

- `MLContext` — `createTensor`, `writeTensor`, `readTensor`, `dispatch`
- `MLGraphBuilder` — full IDL operator set (codegen from `webnn.idl`), plus `input`, `constant`, `build`
- `MLGraph`, `MLTensor`, `MLOperand`

Rustnn extensions on `MLContext`:

- `rustnnResizeTensor`, `rustnnSetTensorCapacity` — dynamic shapes for dispatch

Some IDL ops are not implemented in rustnn yet (`gru`, `lstm`, `scatterElements`, etc.) and throw at runtime.

Regenerate builder bindings after IDL changes:

```bash
npm run generate -w @webnnjs/webnn-node
```
