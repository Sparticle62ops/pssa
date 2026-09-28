# Independent-sequence batching

PSSA `train` now accepts `--batch-size N`. It defaults to **1**, retaining the
historical single-lane forward/backward path. Larger values pack independent
documents into shared dense matrix operations while maintaining a separate
recurrent carry and truncated-backpropagation tape per lane.

## CLI and checkpoint behavior

```sh
# Existing invocations are unchanged; --batch-size 1 is implicit.
oxide_ai_pssa train corpus.txt -o serial.pssa --chunk 64 --accumulate 8 -e 1

# Up to eight independent documents per microbatch.
oxide_ai_pssa train corpus.txt -o batched.pssa --chunk 64 --batch-size 8 --accumulate 1 -e 1
```

- `--chunk` remains the maximum sequence length **per lane**, not batch size
  times sequence length. Consecutive chunks of one document never execute as
  independent concurrent sequences. A one-document corpus cannot fill several
  lanes.
- Finished lanes take the next document on the next microbatch and reset their
  carry. Continuing documents retain carry, including across optimizer updates;
  gradients stop at chunk boundaries. Epochs reset all carries.
- Short chunks and partial batches are packed without padding or duplicated
  targets. Gradient accumulation is weighted by actual target-token counts.
- `--accumulate` counts **microbatches per optimizer update**. For full batches,
  tokens/update is `chunk * batch_size * accumulate`. The example commands have
  the same nominal 512 tokens/update, but document lengths and refill can change
  the actual grouping. Updates/epoch is `ceil(actual_microbatches / accumulate)`;
  microbatch counts come from the lane plan, not simply
  `ceil(total_chunks / batch_size)`.
- Shared episodic memory stays unchanged through a microbatch's forward and
  backward. Then high-loss chunks insert their terminal key/value in lane order,
  using each chunk's own mean loss. At batch sizes above 1, this deliberately
  differs from interleaving a memory write after each individual document.
- Batch workspaces and lane carries are runtime-only. Checkpoint formats and
  stored chunk lengths are unchanged. Old checkpoints without a schedule
  horizon still load and retain legacy per-link scheduling; stored fixed
  horizons/warmup still resume at the global optimizer step. Choose batch size,
  accumulation, and total-update horizon consistently for a training chain.
- The legacy token-stream fingerprint is retained. Batched runs additionally
  print `sequence_plan_fnv1a64`, batch size, and microbatch count to identify the
  grouping.
- `--batch-size` is PSSA-only. The transformer train/generate/evaluate commands
  retain their existing interface. PSSA train/generate/evaluate/chat/help remain
  available, and `help` still contains `--resume`.

Batching changes update grouping and memory visibility, so throughput gains are
not evidence of identical learning trajectories or improved language quality.
The larger workspace also needs more RAM; allocation-limit errors recommend
reducing `--batch-size` or `--chunk`.

## Fixed-work throughput measurement

Measured on 2026-09-28 using `examples/sequence_batch_probe.rs`, extending the
serial baseline introduced in `9490cb4`. The measured source is `8f2895f`
(batching core `ade7dde`, CLI integration `3223b55`). This compares serial and
packed paths in the same release binary, not two different amounts of training
work.

| Measurement | Batch size | Microbatches/update | Median target tokens/second |
| --- | ---: | ---: | ---: |
| Before: separate sequences | 1 | 8 | **921.867** |
| After: packed sequences | 8 | 1 | **962.015** |

The pooled-median change is **+4.36% (1.044×)** on this host. This is a modest,
noisy improvement, not a demonstrated universal or statistically significant
speedup: individual samples overlap, and the second pair's packed median is
slightly below its serial median. All observations are retained below, in
execution order (tokens/second, rounded to three decimals):

| Run | Batch size | Five timed samples | Invocation median |
| --- | ---: | --- | ---: |
| 1 | 1 | 911.356, 728.538, 975.140, 862.892, 921.867 | 911.356 |
| 2 | 8 | 965.621, 1017.618, 1028.535, 1043.000, 962.015 | 1017.618 |
| 3 | 8 | 883.875, 919.851, 922.111, 878.340, 997.483 | 919.851 |
| 4 | 1 | 932.250, 923.766, 874.416, 930.795, 919.886 | 923.766 |
| 5 | 1 | 908.082, 945.463, 953.987, 867.309, 922.489 | 922.489 |
| 6 | 8 | 962.260, 954.191, 941.276, 1030.341, 907.129 | 954.191 |

### Work held constant

Each timed sample processes **eight independent length-64 sequences, 512 target
tokens, and one AdamW update**:

- `--batch-size 1`: eight serial forward/backward microbatches, each contributing
  `1/8` of the update's mean gradient; this is the pre-batching baseline path.
- `--batch-size 8`: one packed forward/backward microbatch contributing the
  entire update's mean gradient.

Model dimensions are latent **256**, state **16**, vocabulary **2048**, memory
key **32**, with **32 populated memory slots**. Both modes use seed 42, identical
synthetic inputs/targets, the same initialized model each round, and LR `1e-5`.
Nonzero MLP/adapter output weights exercise the backward branches. Carries reset
for each independent sequence. The memory bank is read during training but the
probe does not insert new entries.

Timing includes zeroing gradients, all forwards/backwards, carry resets, memory
retrieval, and AdamW. It excludes tokenization, file/checkpoint I/O, model and
workspace construction, sequence-descriptor construction, and post-update
finiteness checks. The packed API's input/workspace validation is timed.
This is a **CPU training-kernel baseline**, not an end-to-end corpus-training or
GPU benchmark.

### Environment and reproduction

- Linux x86-64 KVM guest with **2 exposed CPUs**, Intel family 6/model 79
  (AVX2/FMA available); CPU model name is not exposed by the guest.
- `RAYON_NUM_THREADS=2` for every measurement. The probe explicitly uses the CPU
  constructor rather than GPU auto-detection.
- Rust `1.98.1 (48a229cea 2026-09-01)`, Cargo `1.98.1`.
- Normal release profile: optimization level 3, fat LTO, one codegen unit;
  no custom `RUSTFLAGS`. No third-party dependencies added.
- Builds and tests completed before timing; benchmark invocations ran
  sequentially, without another build/test/benchmark from this session running.

```sh
export CC=/workspace/bin/zigcc
export CXX=/workspace/bin/zigcc
export AR=/workspace/bin/ar

cargo build --release
CARGO_TARGET_DIR=/workspace/oxide-target-test cargo test
CARGO_TARGET_DIR=/workspace/oxide-target-test cargo test --example sequence_batch_probe
cargo build --release --example sequence_batch_probe

# Three invocations per batch size, alternating pair order.
for batch in 1 8 8 1 1 8; do
  RAYON_NUM_THREADS=2 target/release/examples/sequence_batch_probe --batch-size "$batch"
done
```

Each invocation discards one warmup round and reports five timed samples plus
their median. The headline comparison pools the **15 timed samples per batch
size** and takes their median. Absolute throughput is host-specific; these are
short synthetic runs, not a universal speedup guarantee.

## Validation

`cargo build --release` passes. `cargo test` reports **104 passed, 0 failed,
4 ignored** (manual timing probes), and `cargo test --example
sequence_batch_probe` reports **2 passed, 0 failed**, with the toolchain/environment
above. Regression coverage includes:

- complete transition coverage, stable lane carry/refill, ragged/partial batches,
  and the imbalanced-plan optimizer-update count;
- packed outputs, gradients, and carry versus independent sequences;
- byte-identical checkpoints for implicit versus explicit batch size 1;
- exact fixed-schedule batched resume and horizonless legacy checkpoint loading;
- invalid batch sizes failing without writing checkpoints;
- generation, evaluation, and a chat turn from a batched PSSA checkpoint;
- the existing transformer training/resume/generation/evaluation CLI suite.
