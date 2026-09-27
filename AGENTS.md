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
| `gpu_batch.rs` (904) | batched GPU dispatch stages |
| `backend.rs` (697) | backend selection and the CPU path |
| `dataset.rs` (602) | corpus loading, tokenizer, `--max-tokens` / `--skip-tokens` |
| `linalg.rs` (518) | matmul and friends |
| `tui.rs` (486), `ui.rs` (367) | ratatui progress display |
| `inference.rs` (302) | generation |
| `cuda.rs` (249) | cuBLAS SGEMM, device-resident weight cache |
| `memory.rs` (182), `adapter.rs` (93), `defense.rs` (68) | supporting pieces |

Backend order at runtime: CUDA, then WebGPU, then CPU.

## Build and test

```
cargo build --release              # CPU + WebGPU
cargo build --release --features cuda
cargo test
```

`cuda` is optional and dynamically loaded, so a CUDA build still runs on a
machine with no driver; the backend just reports itself unavailable. Tests
live in `tests/` (`core_repair`, `checkpoint_repair`, `bpe_repair`,
`runtime_repair`, `linalg`, `allocations`, `backward_blocked`). Two probes in
`examples/`: `perf_probe.rs`, `twin_check.rs` (CPU-twin verification of the GPU
path; run it after touching `gpu_batch.rs` or `cuda.rs`).

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
