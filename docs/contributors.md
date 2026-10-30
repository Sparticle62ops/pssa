# Contributing

Issues and pull requests are welcome. This guide covers building and testing
locally, the code layout, what to validate before opening a PR, and where the
project needs help.

For user-facing setup see [setup.md](setup.md); for the command reference see
[cli.md](cli.md).

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

This profile enables incremental compilation, disables LTO, and uses 16 codegen
units while retaining optimization level 3. Its binaries and compiler cache live
under `target/fast/`. The first build populates the cache; subsequent edits can
reuse it. Debug builds (`cargo build`) already enable incremental compilation.
Keep using the normal release profile for reproducible performance measurements.

## Commit and pull request expectations

Validate any branch with `cargo test --release` before opening a PR. The
formatting, lint and test commands used in this repository are:

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets
cargo test --release
```

Clippy is clean of errors; a number of style warnings in the numeric kernels are
left in place deliberately, since rewriting indexed loops there would churn code
the gradient tests pin down.

Integration tests live in `tests/`: `allocations.rs`, `bpe_repair.rs`,
`checkpoint_repair.rs`, `core_repair.rs`, `linalg.rs`, and `runtime_repair.rs`,
with shared artifacts under `tests/fixtures/`. They cover tokenizer round trips,
checkpoint import/export across versions, linear-algebra kernels, allocation
behavior, and CLI runtime output.

Keep changes focused and preserve the behavior the existing tests pin down. The
architecture depends on a scalar CPU reference path that batched and parallel
implementations are checked against; gradient and checkpoint tests are part of
that contract.

## Code layout

| Path | Responsibility |
| --- | --- |
| `src/main.rs` | Binary entry point; forwards process arguments to the CLI. |
| `src/cli.rs` | Argument parsing, home screen, training, chat, generation, evaluation, status, download, and benchmark orchestration. |
| `src/ui.rs` | Terminal presentation: logo, panels, spinners, progress bars, ANSI-aware width handling. |
| `src/dataset.rs` | Tokenization, vocabulary construction, built-in corpora, local and remote loading, streaming WikiText cleaning. |
| `src/pssa.rs` | PSSA layer, forward pass, plastic learning, consolidation, and `.pssa` serialization. |
| `src/checkpoint.rs` | Checkpoint format versions, resume payloads, and import/export validation. |
| `src/inference.rs` | Autoregressive sampling and generation constraints. |
| `src/backend.rs` | GEMM dispatch, CPU reference kernels, and the WebGPU device probe. |
| `src/memory.rs` | Fixed-capacity hyperbolic memory bank and retrieval/update logic. |
| `src/adapter.rs` | Low-rank modular adapter projections and updates. |
| `src/defense.rs` | Refractory rate-limiter primitives for stable updates and overwrite defense. |
| `src/linalg.rs` | Small allocation-conscious vector, matrix, math, and deterministic RNG utilities. |
| `src/diagnostics.rs` | CLI banner formatting. |
| `tests/` | Integration tests and shared fixtures. |
| `scripts/` | Paired-evaluation harness and TUI helper scripts. |
| `experiments/` | Paired-evaluation manifests and the memory-capacity sweep. |
| `kaggle/` | Chained-training driver for long corpora on a hosted notebook. |
| `docs/` | Setup, architecture, benchmark, CLI, TUI and experiment documentation. |
| `assets/` | Math reference and evaluation prompts. |
| `data/downloaded.txt` | Checked-in corpus used as the default when present. |
| `data/model.pssa` | Checked-in serialized model artifact. |

## Project status and roadmap

- **The first big training run.** A larger prototype is training on a single
  NVIDIA RTX Pro 6000 (Blackwell) through the CUDA backend, as a resume chain:
  each link trains on the next 500,000 tokens of a mixed corpus and saves a
  checkpoint, so the run survives restarts. A sample from every checkpoint is
  being collected to show how the model improves over time.
- **Sleep and dreaming.** The opt-in sleep phase replays stored memories and
  optional generated rehearsal sequences between updates. It is described in
  [experiments.md](experiments.md#dreaming) and remains off by default.
- **Depth and loops.** Stacked PSSA blocks (`--depth`) and shared repeated passes
  (`--loops`), with notes in [STACKED-DEPTH.md](STACKED-DEPTH.md).
- **WebGPU training.** The recurrent scan and memory stage are being ported to
  WebGPU so AMD, Intel and Apple GPUs can train too. It still needs testing on a
  real hardware adapter.
- **Next up.** Tests of the model on a small symbolic world (does it learn the
  rules, not just the text), and a longer demo video of the dashboard.

## Where the project needs help

### Compute

The whole result in [benchmarks.md](benchmarks.md) was trained on a free hosted
notebook with a single entry-level GPU, in 200,000-token links, because a
session gets cut after a few hours. Every interesting question left, whether the
gap holds at 10x or 100x these parameters, whether the memory bank matters at
scale, how it does against a modern recurrent baseline, needs one thing: a GPU
with real VRAM and allocations measured in days instead of hours. Anything
meaningfully above the entry-level card this ran on changes what can be asked.

If you have compute to grant, or you work somewhere that does, that is the
single highest-leverage thing anyone can offer this project. The immediate
milestone is a 105M-parameter run trained from scratch, with a same-scale
transformer baseline, so the comparison finally gets asked at scale. Sponsorship
details are in the [README](../README.md#sponsorship).

### Code and evaluation

The parts most in need of hands are kernel performance, a modern recurrent
baseline to compare against, and evaluation beyond next-token loss.

### Experiment documentation

Deeper notes for the current research directions:

- [BATCHING.md](BATCHING.md) — independent-sequence batching.
- [STACKED-DEPTH.md](STACKED-DEPTH.md) — stacked-depth PSSA.
- [PAIRED-EVAL.md](PAIRED-EVAL.md) — paired multi-seed evaluation harness.
- [COMPARISON.md](COMPARISON.md) — replaying a chain against the transformer.
- [TRANSFORMER-BASELINE.md](TRANSFORMER-BASELINE.md) — the matched baseline.
- [GPU-CLIP-SPEED.md](GPU-CLIP-SPEED.md) — clipping and memory-cap validation.
- [distill_trainer.md](distill_trainer.md) — Qwen3-to-PSSA distillation trainer
  design.
- [TUI-EXTRAS.md](TUI-EXTRAS.md) — dashboard extras.

## License

See [../LICENSE](../LICENSE) for the project license.
