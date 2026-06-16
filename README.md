# webnnjs

Node.js WebNN-flavor API backed by `rustnn` and ONNX Runtime via a Rust napi-rs addon.

This is a Node polyfill (not a browser implementation). The public API follows the [W3C WebNN IDL](https://www.w3.org/TR/webnn/), with Node-only helpers for Hub model loading.

## Layout

- `packages/webnn-node/` — TypeScript WebNN API (`MLContext`, `MLGraphBuilder`, …)
- `packages/webnn-node/native/` — Rust napi-rs addon
- `packages/webnn-node/idl/webnn.idl` — vendored IDL fallback for codegen (see WPT cache below)
- `demo/` — Runnable examples
- `.cache/wpt/` — local clone of [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt) (gitignored; created by `npm run test:wpt:fetch`)

## Prerequisites

- Node.js `>= 20`
- Rust toolchain (`cargo`, `rustc`)
- Native build toolchain (C/C++ compiler, linker)
- ONNX Runtime shared library (see below)
- Local [`rustnn`](https://github.com/rustnn/rustnn) checkout at `../rustnn` relative to this repo (see root `Cargo.toml` `[workspace.dependencies]`)

## Local `rustnn` and native addon

The Node addon is a **Cargo workspace member**. It always links the path dependency in the root `Cargo.toml`, not a crates.io release:

```toml
rustnn = { path = "../rustnn", features = ["onnx-runtime", "dynamic-inputs"] }
```

Release builds land in **`target/release/` at the repo root** (`webnnjs/target/release/`), not under `packages/webnn-node/native/target/`. After changing `rustnn` or the native crate, you must rebuild **and** re-stage `index.node`; otherwise Node may keep running an old binary.

### Recommended build (from repo root)

```bash
npm install
npm run build
```

That runs `cargo build --release`, then `scripts/install-addon.mjs`, which copies the workspace artifact into `packages/webnn-node/native/index.node` and prints the source path:

```text
Using native artifact: .../webnnjs/target/release/webnn_node_native.dll
Installed webnn_node_native.dll -> index.node
```

If you only changed Rust/`rustnn`:

```bash
cargo build --release -p webnn-node-native
npm run install-addon -w @webnnjs/webnn-node-native
```

### Verify the linked `rustnn`

**Dependency tree** (should show your local path, not crates.io):

```bash
cargo tree -p webnn-node-native -i rustnn
```

Example:

```text
rustnn v0.5.11 (C:\git\rustnn-workspace\rustnn)
└── webnn-node-native v0.1.0 (...\webnnjs\packages\webnn-node\native)
```

**Install script** — confirm it points at `webnnjs/target/release/`, not an old `native/target/` copy. If the path or timestamp looks wrong after a rebuild, Node is likely loading a stale `.node` file (see [Native addon staging](#native-addon-staging)).

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

1. **Native addon** — `cargo build --release` (workspace → `target/release/`), then `install-addon.mjs` copies the cdylib to `packages/webnn-node/native/index.node`
2. **webnn-node** — regenerates `MLGraphBuilder` methods from `webnn.idl`, compiles TypeScript
3. **demo** — compiles demo TypeScript

Build individual packages:

```bash
# Native only (build + stage index.node)
npm run build -w @webnnjs/webnn-node-native

# Or explicitly from the workspace root:
cargo build --release -p webnn-node-native
npm run install-addon -w @webnnjs/webnn-node-native

# TypeScript package (includes native)
npm --prefix packages/webnn-node run build

# Demo only (requires webnn-node built first)
npm --prefix demo run build
```

### Native addon staging

The copy to `index.node` happens in `packages/webnn-node/native/scripts/install-addon.mjs` **after** `cargo build`, not in `build.rs` (`build.rs` only runs `napi_build::setup()` before linking).

The install script prefers the **workspace** artifact (`webnnjs/target/release/`) over `packages/webnn-node/native/target/release/`. Always read its `Using native artifact:` line after a build.

`index.js` loads the **newest** `*.node` file in the native package directory (by modification time). That means:

- **`index.staged.node`** can win over `index.node` if it is newer — close Node, run `install-addon`, or delete stray `*.staged.node` files after copying.
- A forgotten old `index.node` causes confusing behaviour (e.g. quantize tests failing with stale shape inference). Re-run `npm run install-addon -w @webnnjs/webnn-node-native` after every native rebuild.

If `index.node` is locked on Windows (a Node process still running), the install script writes `index.staged.node` instead. Close Node processes, then:

```bash
npm run install-addon -w @webnnjs/webnn-node-native
```

Optional: `npm run build:napi -w @webnnjs/webnn-node-native` uses `@napi-rs/cli` instead (also regenerates `index.d.ts` from `#[napi]` exports).

## WPT cache

Same pattern as [rustnnpt](https://github.com/rustnn/rustnnpt): shallow-clone the WPT repo into `.cache/wpt` (never committed).

```bash
npm run test:wpt:fetch
```

This runs `scripts/fetch-wpt.mjs`, which:

1. Creates `.cache/` if needed
2. On first run: `git clone --depth 1` of `https://github.com/web-platform-tests/wpt.git` → `.cache/wpt`
3. On later runs: `git fetch` + `reset --hard origin/master`

Override the location with `WPT_DIR` (same as rustnnpt).

After fetch, useful paths include:

| Path | Contents |
|------|----------|
| `.cache/wpt/interfaces/webnn.idl` | WebNN IDL |
| `.cache/wpt/webnn/conformance_tests/` | WPT WebNN conformance tests |

**IDL for codegen:** `npm run generate` prefers `.cache/wpt/interfaces/webnn.idl` when the cache exists; otherwise it uses the vendored `packages/webnn-node/idl/webnn.idl`.

To refresh the committed vendor copy from the cache:

```bash
npm run test:wpt:fetch
npm run sync:idl
```

Requires `git` on PATH for fetch (no Python needed for the cache step).

### Run WebNN conformance subtests

The runner executes **only** tests under `.cache/wpt/webnn/conformance_tests/` (not the full WPT tree). Each subtest builds a graph with `MLGraphBuilder`, runs `dispatch`, and compares outputs with WPT tolerances.

```bash
npm run test:wpt:fetch
npm run test:wpt:run -- --op add --limit-tests 5
```

Common options (pass after `--`):

| Flag | Description |
|------|-------------|
| `--op NAME` | Only files like `add.https.any.js` |
| `--file FILE` | Single file (basename) |
| `--limit-tests N` | Cap subtests per file |
| `--limit-files N` | Cap files |
| `--stop-on-fail` | Stop at first failure |
| `--report-json PATH` | Write JSON report |

`int4` and `uint4` tests are skipped until supported. Float16 WPT tests require `Float16Array` (Node 24+ by default, or Node 22 with `--js-float16array`).

Set `ORT_DYLIB_PATH` (or `demo/.env`) before running; the harness searches common ORT locations if unset.

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

Some IDL ops may still throw at runtime if rustnn or the native bridge does not implement them yet.

Regenerate builder bindings after IDL changes:

```bash
npm run test:wpt:fetch          # optional: refresh WPT + use cached IDL
npm run generate -w @webnnjs/webnn-node
```
