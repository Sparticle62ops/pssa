# WGSL recurrent-stage findings

## Scope

This branch adds a portable `wgpu 0.19` implementation for the recurrent SSM
scan and episodic-memory stage. The CPU code and CUDA dispatch remain separate;
no checkpoint bytes or model configuration fields were changed.

The WGSL path now contains:

- a tiled 256-lane affine Blelloch scan, recursive tile-summary scan, and tile
  prefix application;
- SSM forward map preparation, scan materialization, reverse-time backward map
  scan, and token-local SSM derivatives;
- the full memory forward stage (query projections, open-ball projection,
  hyperbolic retrieval, gate, value projection, and injection);
- memory elementwise and retrieval backward kernels; and
- the small softplus elementwise dispatch needed by the GPU projection path.

`GpuDispatch` selects these kernels for `--backend wgpu` (and the `webgpu`
spelling remains accepted). The resident single-lane path passes the SSM output
buffer directly to the memory forward dispatch before reading the host tape.
Packed `SequenceBatch` lanes use the same SSM dispatcher per lane, then use the
packed device memory stage. SSM backward uses the reverse affine scan and keeps
its token-local derivatives on the device until the existing dense gradient
boundary. If a GPU stage fails after device selection, the recovery is explicit
and emits a `warning: GPU ... failed` line; an explicit `--backend wgpu` adapter
failure is an error, not a silent CPU selection.

## Local checks and measured parity

This sandbox has no WebGPU adapter (`/dev/dri` is absent and no Vulkan/Mesa
runtime was installed). The parity integration test therefore deliberately
skips at adapter initialization and prints:

```text
WebGPU parity skipped: no adapter (No compatible WebGPU compute adapter found.)
```

The actual run was:

```sh
CARGO_TARGET_DIR=/tmp/pssa-wgpu-target CARGO_BUILD_JOBS=1 \
  cargo test --test wgpu_parity -- --nocapture
```

Measured result: **3 passed, 0 failed, 3 adapter skips**. Consequently, no
numerical GPU relative-error number was measured in this environment; the
required `< 1e-3` forward/gradient number remains unverified here. The test
contains the small full-model forward/all-gradient comparison and a separate
latent-3584, vocabulary-2048, depth-1, loops-1, batch-2 SSM/memory/vocabulary
shape smoke. On a discrete adapter, both tests execute and the small test asserts every
model gradient and forward tape result at relative error `< 1e-3`. A software-
only adapter (reported as `Cpu`, including llvmpipe, lavapipe, or SwiftShader)
is now reported as a parity skip instead of failing device initialization. Set
`PSSA_WGPU_ALLOW_SOFTWARE=1` when deliberately exercising the parity kernels on
such an adapter.

The post-change Rust check also passed:

```sh
CARGO_TARGET_DIR=/tmp/pssa-wgpu-target CARGO_BUILD_JOBS=1 cargo check
```

It reported only the repository's existing dead-code warnings in
`src/inference.rs` and `src/tui/shadow.rs`.

## Blackwell/molab commands

Run these from the checkout on the RTX PRO 6000 Blackwell host. These commands
are verification instructions only; no training run was started in this
sandbox.

```sh
# Confirm the adapter and backend selection first.
cargo run --release -- gpu-probe

# Execute the small CPU-vs-WGSL forward/gradient parity and the training-shape smoke.
CARGO_BUILD_JOBS=1 cargo test --release --test wgpu_parity -- --nocapture

# A short, fresh, explicitly WGSL training smoke at the requested production shape.
# Use a new output path; do not overwrite an existing checkpoint.
CARGO_BUILD_JOBS=1 cargo run --release -- train \
  --data data/downloaded.txt \
  --out runs/wgpu-blackwell-smoke.pssa \
  --max-tokens 4096 --epochs 1 --no-tui \
  --latent 3584 --state 16 --key 32 --memory 512 --chunk 2 \
  --depth 1 --loops 1 --batch-size 2 --backend wgpu
```

Record the adapter/driver printed by startup, the parity relative errors, and
whether any `warning: GPU ... failed` fallback appeared. For a throughput
comparison, repeat the same command with identical data, chunk, batch, and
thread settings using `--backend cuda` and `--backend cpu`; do not compare a
cold first launch with a warmed GPU step.

## Unverified / limitations

- No WGSL shader was executed locally because there is no adapter. Runtime
  shader validation, Blackwell results, measured GPU parity, VRAM use, and
  speedup are unverified until the molab commands run.
- The portable stage API still copies the host activation tape at the
  boundaries needed by the existing backward and optimizer code. The
  SSM-to-memory forward buffer is resident; the entire model is not yet one
  device-resident graph.
- Stacked-depth/loop replay retains its existing host recurrent scheduling;
  the depth-1, loops-1 path is the device-native path covered by the requested
  production-shape smoke. CPU and CUDA paths were not intentionally rewritten.
- The required release build, release library tests, release clippy, and
  example compilation still need to be run serially before this branch is
  considered complete.
