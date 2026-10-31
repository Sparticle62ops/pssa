# Contributing to PSSA

Issues and pull requests are welcome. This is the short version; the full guide
with the code-layout map and the project roadmap lives in
[docs/contributors.md](docs/contributors.md).

## Build and test locally

For repeated local edits, use the optimized incremental `fast` profile:

```bash
cargo check --profile fast --tests
cargo build --profile fast
cargo test --profile fast
cargo run --profile fast -- help
# Optional NVIDIA CUDA backend:
cargo build --profile fast --features cuda
```

The profile enables incremental compilation, disables LTO, and keeps
optimization level 3. Its binaries and compiler cache live under `target/fast/`.
Use the normal release profile for reproducible performance measurements.

## Before you open a PR

Validate the branch with:

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets
cargo test --release
```

Keep changes focused and preserve the behavior the existing tests pin down. The
architecture depends on a scalar CPU reference path that batched and parallel
implementations are checked against; the gradient and checkpoint tests are part
of that contract.

## Where help is needed

- Kernel performance (CPU, CUDA, WebGPU).
- A modern recurrent baseline to compare against.
- Evaluation beyond next-token loss.
- Compute. The 105M run and the matched transformer baseline all bottleneck on
  GPU time; see [DONATE.md](DONATE.md).
