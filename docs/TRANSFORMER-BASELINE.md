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
  As in PSSA, fixed-horizon checkpoints persist warmup and resume the complete
  schedule at the global optimizer step, including when a split falls inside
  warmup. Old checkpoints without warmup metadata retain legacy no-rewarmup
  behavior; old checkpoints without a horizon retain per-link horizons.
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

## Training curves and held-out corpus slices

Both trainers accept `--loss-csv PATH --loss-every N` (default cadence 10,000
**scored next-token targets**, not encoded input tokens). They append the same
header and schema:

```csv
tokens_seen,updates,loss,tokens_per_second
```

- `tokens_seen`: cumulative supervised next-token targets processed, including
  repeated epochs/windows. Line endings and one-token fragments have no target;
  this is normally less than the `--max-tokens` input budget.
- `updates`: global completed optimizer updates, including resumed links.
- `loss`: target-weighted mean pre-update cross entropy, in nats, since the
  previous row in this invocation (not an unweighted mean of chunk losses).
- `tokens_per_second`: targets in that interval divided by elapsed training
  seconds. Excludes loading/tokenizing, checkpoint saving and evaluation; includes
  training-loop reporting overhead. Label hardware/backend when comparing speed.

Rows are flushed at the first optimizer boundary crossing each global multiple
of N, plus a final partial interval. An update is never split and skipped
thresholds do not produce invented observations. Link ends may add extra rows.
On resume, reuse the CSV: its final update must match the checkpoint, and its
final token count is restored. A **new** CSV on resume requires `--tokens-seen N`
with the actual number of previously scored targets; do not substitute the raw
window budget. Malformed/stale CSVs fail rather than silently splice different
runs. A crashed link may leave CSV rows ahead of its saved checkpoint: archive
that tail and restore the CSV through the checkpoint's final row before retrying.
Old checkpoints need no new metadata and remain readable. Historical curves
cannot be recovered from checkpoints that were trained without logging.

```sh
oxide_ai_pssa train CORPUS -o ck01.pssa -e 1 --max-tokens 200000 \
  --loss-csv pssa.csv --loss-every 10000
oxide_ai_pssa train-transformer CORPUS -o ck01.trfm -e 1 --max-tokens 200000 \
  --tokenizer-from ck01.pssa --loss-csv transformer.csv --loss-every 10000
```

Both evaluation commands accept `--skip-tokens N --max-tokens N` to score a
slice using the checkpoint's embedded tokenizer. Unlike training windows,
evaluation slices **never wrap**: out-of-range slices, zero limits and slices
without transitions are errors. They preserve line boundaries, never score a
target outside the slice, and never update/save weights, optimizer state or the
memory bank. PSSA starts each document with zero recurrent carry but retains its
trained episodic bank; the transformer resets attention per chunk. Outputs retain
the existing JSON schema (`cross_entropy` is mean loss, `perplexity = exp(loss)`).

```sh
oxide_ai_pssa evaluate CORPUS -m chain/ck64.pssa \
  --skip-tokens 12800000 --max-tokens 200000
oxide_ai_pssa evaluate-transformer CORPUS -m comparison/ck64.trfm \
  --skip-tokens 12800000 --max-tokens 200000
```

A slice is only held out if the chain never trained on it. Verify corpus length
and training offsets: training wraps at EOF, so a wrapped chain may have no
unseen slice. Do not clean/change the corpus midway through a comparison.

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
