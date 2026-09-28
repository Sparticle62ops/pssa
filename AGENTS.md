# Working in this repo

Read this first. It exists so you do not have to re-read 8k lines of `src/`
to orient yourself. Trust it, spot-check what you touch, and keep it current.

## What this is

`oxide_ai_pssa` (crate v0.4.0, edition 2024): a from-scratch language model
built on a Plastic State-Space Architecture. No external ML framework. CPU
SIMD + rayon by default, optional WebGPU and native CUDA/cuBLAS backends.

## Map of `src/`

| file | what lives there |
| --- | --- |
| `pssa.rs` (1273) | the architecture itself: forward, backward, state updates |
| `cli.rs` (1247) | argument parsing and every subcommand entry point |
| `checkpoint.rs` (1027) | `.pssa` serialization, resume, repair |
| `gpu_batch.rs` | shared dense forward/backward dispatch stages |
| `sequence_batch.rs` | packed independent document lanes, per-lane recurrent carry and TBPTT |
| `backend.rs` (697) | backend selection and the CPU path |
| `dataset.rs` (602) | corpus loading, tokenizer, `--max-tokens` / `--skip-tokens` |
| `linalg.rs` (518) | matmul and friends |
| `tui.rs` (486), `ui.rs` (367) | ratatui progress display |
| `inference.rs` (302) | PSSA generation and shared sampling policy |
| `transformer.rs` | CPU decoder-only baseline, forward/backward, parameter counts |
| `transformer_checkpoint.rs` | separate `TRFM` v1 checkpoints, complete Adam/tokenizer resume |
| `transformer_training.rs`, `transformer_inference.rs` | baseline train/generate/evaluate runtime |
| `training.rs` | shared chunk plans, LR schedule horizons, token-stream audit fingerprints |
| `cuda.rs` (249) | cuBLAS SGEMM, device-resident weight cache |
| `memory.rs` (182), `adapter.rs` (93), `defense.rs` (68) | supporting pieces |

Backend order at runtime: CUDA, then WebGPU, then CPU.

## Build and test

```
cargo build --release              # CPU + WebGPU
cargo build --release --features cuda
cargo test
```

For the workspace toolchain, set `CC=/workspace/bin/zigcc`,
`CXX=/workspace/bin/zigcc`, and `AR=/workspace/bin/ar`; run tests with
`CARGO_TARGET_DIR=/workspace/oxide-target-test`.

PSSA `train --batch-size N` groups independent document lanes (default 1,
legacy single-lane path); `--accumulate` counts microbatches per update.
Batch workspaces are runtime-only, not checkpoint metadata. See
`tests/batch_training.rs` for CLI, planning, and schedule/resume coverage;
`tests/sequence_batch.rs` checks packed math against separate sequences.
`examples/sequence_batch_probe.rs` is the fixed-work CPU throughput baseline.

`cuda` is optional and dynamically loaded, so a CUDA build still runs on a
machine with no driver; the backend just reports itself unavailable. Tests
live in `tests/` (`core_repair`, `checkpoint_repair`, `bpe_repair`,
`runtime_repair`, `linalg`, `allocations`, `backward_blocked`). Two probes in
`examples/`: `perf_probe.rs`, `twin_check.rs` (CPU-twin verification of the GPU
path; run it after touching `gpu_batch.rs` or `cuda.rs`).

## Transformer baseline

`train-transformer` is a one-block width-256 / 4-head / FFN-448 decoder baseline.
At actual vocab 2048 it has 1,541,120 trainable parameters versus default PSSA's
1,544,704. Both print actual counts. It shares PSSA's document/window selector,
chunk plan, schedule, AdamW, and progress/loss format. Use `--tokenizer-from
<pssa-checkpoint>` on a fresh baseline run to import the exact tokenizer, then
resume each model's own checkpoint with the same window/accumulation options.
Baseline generation/evaluation use `generate-transformer` / `evaluate-transformer`.
The baseline is CPU-only and chunk-local; PSSA retains recurrent carry and memory.
See `docs/TRANSFORMER-BASELINE.md` for reproducible commands, counting conventions,
and comparison caveats. Tests live in `tests/transformer.rs`.

## Do not break the Kaggle contract

`kaggle/kaggle_continue.sh` drives long chained training runs on Kaggle and is
usually live. It depends on these exact invocations:

```
oxide_ai_pssa train <data> -o <out> --max-tokens N --skip-tokens N -e 1 --resume <ck>
oxide_ai_pssa generate -m <ck> -p <prompt>
oxide_ai_pssa help          # the script greps this output for the string --resume
```

Add new surface freely. Never rename, repurpose, or drop those flags, and never
change what `help` prints such that `--resume` disappears from it.

The chain writes `chain/ckNN.pssa`, one checkpoint per 200k-token window, each
resuming from the last. Checkpoint format changes are breaking: a chain in
flight must still load.

## Conventions

- Fix defects; do not refactor for taste alone.
- No panics on user input. Validation errors go through the CLI's error path
  with a message that says what to do instead.
- Keep the CPU and GPU paths numerically equivalent. If you change one, change
  the other or say plainly why it diverges.
- Commit in logical commits with real messages. Do not push to origin.

## Design intent and project history

`docs/` holds the architect's own handover material. Read it before judging the
math:

- `HANDOFF-2026-09-17.md`, `HANDOFF-2026-09-17b.md` — project state, verified
  training evidence, the frozen width/depth comparison protocol.
- `CONTINUATION-2026-09-17.md` — the repaired defects and the study readiness
  record from that session.

Config drift to be aware of: those documents describe width64 / depth1 / BPE
vocab 2048 / state 8 / memory-key 16 / capacity 32, and a separate technical
dossier describes vocab 10,000. The live chain trains latent 256, state 16, 512
memory slots, key width 32, vocab 2048. Trust the code and the checkpoint
header for SHAPES. Trust the handover docs for INTENT and for the success bar.

The success bar the architect set is unselected coherent output, not lower
perplexity, not compilation, not layer count.
