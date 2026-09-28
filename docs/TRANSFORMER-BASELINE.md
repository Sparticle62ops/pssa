# Parameter-matched decoder-only baseline

`train-transformer` adds a from-scratch transformer without changing the existing
`train`, `generate`, or PSSA checkpoint formats. No dependencies were added.
`generate-transformer` and `evaluate-transformer` accept the corresponding
PSSA commands' prompt/model and data/model arguments.

## Architecture and size

One pre-norm decoder block, width 256, four causal attention heads of width 64,
FFN width 448 with SiLU, two affine RMSNorms (epsilon `1e-5`), residual connections,
fixed sinusoidal positions, and separate (untied) token embedding/output matrices.
The FFN width is chosen to match the live default PSSA, not a wider hypothetical
model. There are no learned positional embeddings, attention/MLP biases, dropout,
external frameworks, or untrained placeholder components. All trainable families
receive gradients and AdamW updates.

At **actual vocabulary size 2048**:

| Model | Embedding + output | Other trainable scalars | Total |
|---|---:|---:|---:|
| PSSA (latent 256, state 16, key 32, rank-16 adapter) | 1,048,576 | 496,128 | **1,544,704** |
| Transformer (width 256, heads 4, FFN 448, depth 1) | 1,048,576 | 492,544 | **1,541,120** |

Difference: 3,584 parameters, **0.232%** of PSSA. Counts include learned norm
scale/offset and the plastic adapter's trainable coefficients. They exclude
Adam moments, gradients, activation buffers, recurrent state, episodic memory,
and the adapter's non-independent consolidated coefficients. Fixed sinusoidal
positions add no parameters. A BPE vocabulary ceiling is not a guarantee that a
small corpus reaches that size; **both commands print actual counts**:

```text
model=pssa parameters=1544704 vocab=2048
model=transformer parameters=1541120 vocab=2048
```

## What is held constant

- The same `DatasetManager` loads the corpus, including directory ordering and
  newline handling.
- Use `--tokenizer-from <PSSA checkpoint>` on the transformer's **fresh** run to
  import the exact embedded BPE tokenizer or ordered word vocabulary. Only the
  tokenizer is imported: no PSSA weights, optimizer clock, memory, or learning
  schedule is copied. Independent tokenizer training is also supported with the
  same `--tokenizer bpe|word --vocab-size N` flags, but importing removes any
  ambiguity about the tokenizer used for an experiment.
- Both call `CLIHandler::documents`: encode each line independently, skip modulo
  the corpus's encoded token count, consume the global `--max-tokens` window,
  and wrap at EOF. There are no targets across line or wrap boundaries.
  Single-token fragments consume window budget but have no next-token target.
- Both use the same chunk plan and token-weighted gradient accumulation. A short
  chunk contributes `len / group_target_count` times its mean-loss gradient;
  partial final groups are not discarded.
- Both use the existing parameter containers and AdamW, with beta1 `0.9`, beta2
  `0.999`, epsilon `1e-8`, weight decay `0.01` (zero on norm scale/offset), and
  dense embedding updates. Both use a `1/sqrt(width)` output-logit scale.
- Both use the same `Schedule` and `learning_rate_for_update`. A fixed
  `--total-updates` horizon persists through checkpoints; conflicting horizons
  or a horizon ending before the new link finishes are errors. Omitting `--lr`
  on CLI resume restores the saved base LR; supplying it is an explicit override.
  As in PSSA, resumed links do **not** restart warmup. For exact split/uninterrupted
  equivalence, split after warmup or leave warmup at zero.
- Both print `token_stream_fnv1a64=... chunk=... accumulate=...` for the selected
  token IDs, document boundaries, and grouping. Matching fingerprints, token
  totals, and update totals audit the actual training exposure. FNV is an
  accidental-drift check, not a cryptographic identity guarantee.
- Both call the same progress reporter on every update (30-second cadence in
  redirected logs), and report token-weighted epoch loss in exactly this format:

  ```text
  epoch 1/1 loss=... tokens=... updates=...
  training_seconds=... optimizer_updates=...
  ```

## Reproducible first-window comparison

Build in this container with the existing Zig wrapper:

```sh
export PATH=/workspace/bin:$PATH
export CC=/workspace/bin/zigcc
export AR=/workspace/bin/ar
export CARGO_BUILD_JOBS=1
cargo build --release
cargo test
```

The following is an example **50,000-update fixed schedule**, not a claim that a
200k-token window contains 50,000 updates. Choose the horizon for the entire
intended experiment, then keep it fixed on both sides and across all links.
Document lengths affect the update count; do not infer it simply as tokens /
(chunk * accumulation).

Run sequentially against the same unchanged corpus file:

```sh
mkdir -p comparison

target/release/oxide_ai_pssa train data/downloaded.txt \
  -o comparison/pssa-01.pssa \
  --tokenizer bpe --vocab-size 2048 \
  --latent 256 --state 16 --key 32 --memory 512 \
  --chunk 64 --accumulate 8 --lr 0.001 --seed 42 \
  --total-updates 50000 --max-tokens 200000 --skip-tokens 0 -e 1 \
  > comparison/pssa-01.log 2>&1

target/release/oxide_ai_pssa train-transformer data/downloaded.txt \
  -o comparison/transformer-01.trfm \
  --tokenizer-from comparison/pssa-01.pssa \
  --chunk 64 --accumulate 8 --lr 0.001 --seed 42 \
  --total-updates 50000 --max-tokens 200000 --skip-tokens 0 -e 1 \
  > comparison/transformer-01.log 2>&1

grep -E '^(model=|token_stream_fnv1a64=|epoch |lr_schedule=)' comparison/*-01.log
```

## Next window / checkpoint continuation

Each model resumes its **own** checkpoint. Neither resumes an external dataset
cursor: `--skip-tokens` explicitly selects the next window, just as in the Kaggle
PSSA chain. The checkpoint owns its chunk capacity. Accumulation and epoch count
remain run options, as they do for PSSA. Repeat custom accumulation on resume.
Do not pass `--tokenizer-from` on a transformer resume: its own tokenizer is
already embedded.

```sh
target/release/oxide_ai_pssa train data/downloaded.txt \
  -o comparison/pssa-02.pssa --resume comparison/pssa-01.pssa \
  --accumulate 8 --total-updates 50000 \
  --max-tokens 200000 --skip-tokens 200000 -e 1 \
  > comparison/pssa-02.log 2>&1

target/release/oxide_ai_pssa train-transformer data/downloaded.txt \
  -o comparison/transformer-02.trfm --resume comparison/transformer-01.trfm \
  --accumulate 8 --total-updates 50000 \
  --max-tokens 200000 --skip-tokens 200000 -e 1 \
  > comparison/transformer-02.log 2>&1
```

Transformer checkpoints use separate `TRFM` v1 magic and the existing checked
binary I/O, checksum, and atomic-write primitives. They store model shape,
weights, gradients, Adam moments, step counter, RNG state, full tokenizer
identity, and optional horizon. They can resume between chunks, not midway
through backward. PSSA V5/V6/V7 remain unchanged and readable.

## Evaluation, generation, and limitations

```sh
target/release/oxide_ai_pssa evaluate data/heldout.txt -m comparison/pssa-02.pssa
target/release/oxide_ai_pssa evaluate-transformer data/heldout.txt -m comparison/transformer-02.trfm

target/release/oxide_ai_pssa generate -m comparison/pssa-02.pssa \
  -p "The purpose of a scientific experiment is" --temperature 0 --max-new-tokens 64
target/release/oxide_ai_pssa generate-transformer -m comparison/transformer-02.trfm \
  -p "The purpose of a scientific experiment is" --temperature 0 --max-new-tokens 64
```

Evaluation uses the embedded tokenizer, preserves document boundaries and prints
the same JSON fields. Generation reuses PSSA's sampler (seed 1337, temperature,
top-k/top-p, repetition penalty and `<unk>` exclusion) and decoding policy.

**Important distinctions:** the transformer is currently a CPU reference path,
not a CUDA/WebGPU throughput competitor. Its causal context is at most `--chunk`
tokens, reset for every training chunk; generation recomputes the most recent
context window with positions starting at zero. PSSA carries recurrent state
between chunks within each document and retains its episodic/plastic mechanisms.
Thus this is a same-parameter / same-token-exposure comparison, **not** equal
context, equal compute, or equal memory usage. Keep backend/hardware labels when
reporting speed. No long comparative training or language-quality claim is made
by the implementation tests. Review fixed, unselected generations as well as
held-out loss; lower training loss alone is not the project's success criterion.

Tests cover all small-model parameters by central differences, causal masking,
short chunks, training/inference parity, weighted accumulation, learning, exact
checkpoint and next-update round trips, malformed files/allocation limits, EOF
windowing, fixed-horizon resume, and a BPE CLI comparison with matching stream
fingerprints while the original PSSA generate/help contract remains exercised.
