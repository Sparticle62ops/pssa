# Stacked PSSA: operational CPU pilot

Use the normal `train`, `generate`, and `evaluate` commands. The old research
`width_depth_study` runner is not part of this integration and must not be used
to establish corpus coverage: its group/chunk indexing can undertrain or fail.
A successful smoke test is not evidence of improved language quality.

## Architecture and budget

`train --depth N` accepts **1 through 32**, default **1**. A model owns one
embedding and one distinct, untied output head, with independently parameterized
continuous blocks between them. Each block has its own affine RMSNorm, selective
SSM, episodic bank, plastic adapter and MLP. At depth D:

```text
h0 = block0(embedding[token])
hi = h(i-1) + block_i(h(i-1)) / sqrt(D), for i = 1..D-1
logits = output_head(h(D-1)) / sqrt(width)
```

The first block has no outer identity residual. At depth four, the added branch
scales are 0.5. A memory write uses each block's own query and branch output,
not the upper layer's residual sum. Carries and memory banks are separate for
every layer; chunk boundaries detach recurrent adjoints (TBPTT).

For **actual** vocabulary 2048, state 16, key width 32, adapter rank 16:

| Width | Depth | Memory slots per layer | Trainable parameters |
| --- | --- | --- | --- |
| 256 | 1 | 512 | 1,544,704 |
| **166** | **4** | **512** | **1,548,448** |

The four-layer shape is +0.2424% in trainable parameters. The count is
`2*vocab*width + depth*(7*width^2 + (3*state + 2*key + 34)*width)`.
The CLI prints actual vocabulary and parameter counts: `--vocab-size 2048` is a
BPE ceiling, not a guarantee that a small corpus produces 2048 entries.

Memory slots, recurrent carry, gradients/moments, adapter consolidated copies
and tapes are not independent trainable parameters. Four layers have **2048
memory slots total**, four carries and four tapes. Equal parameter counts do
not imply equal RAM, compute, wall time or memory capacity. This is a fresh
architecture comparison, not a shape-changing resume of the width-256 chain.

## Backend and batching limits

- Depth one retains the existing CPU, CUDA and WebGPU dispatch paths and packed
  independent-document GEMMs. Omitting `--depth` keeps the original model.
- Depth greater than one is **CPU-only**. Training reports
  `backend=cpu (stacked depth N; GPU execution unsupported)` without attaching a
  GPU. The public whole-chunk batched entrypoints route a CPU stack through all
  layers; a stack attached to an actual GPU is explicitly rejected.
- `--batch-size N` works for stacks through **serial CPU replay**, not accelerated
  packed GEMMs. The CLI additionally reports `batch_backend=cpu-replay`. Each
  lane has its own carry for every layer; resets affect only that lane. A chunk
  is replayed from its incoming carry to reconstruct its tape for backward.
  Expect additional forward work; there is no stacked throughput claim.
- Loss and gradients use actual target-token weighting across unequal lane
  lengths and across `--accumulate` microbatches. There is no padding loss and
  no cross-lane recurrent adjoint. All lanes finish backward against the same
  banks before deterministic per-lane, per-layer memory writes occur.
- As with depth-one `SequenceBatch`, library callers must not mutate weights or
  banks between `forward` and `backward`. `state(lane)` contains all layer carries
  concatenated in layer order. For stacks, only packed per-token losses and
  terminal query/branch-output rows are published in the model tape; other
  activation/probability rows are replay scratch, not a packed diagnostics API.
- Lane workspaces are runtime-only, not checkpoint metadata. Resume occurs at
  the trainer's existing run/epoch boundaries, not midway through a microbatch.

## Matched-exposure fresh runs

Clean the corpus once, if desired, **before both fresh runs**. Do not switch
between cleaned and raw text midway through a chain. Freeze a sufficiently large
UTF-8 corpus, tokenizer policy, seeds, line boundaries, token slices, chunk size,
lane count, accumulation, epochs, learning rate and update schedule. Keep the
raw file immutable so deterministic tokenizer construction yields the same
ordered vocabulary and encoded stream. Confirm actual vocabulary is 2048 and
compare the printed stream fingerprints and target-token counts before claiming
the parameter match above.

Example single-window pilot, using existing regular CLI flags:

```sh
cargo build --release
BIN=target/release/oxide_ai_pssa
CORPUS=data/frozen-clean.txt

# Same corpus, 200k encoded-token window, one pass, chunk 64, accumulation 8.
"$BIN" train "$CORPUS" -o depth1.pssa \
  --latent 256 --depth 1 --state 16 --key 32 --memory 512 \
  --tokenizer bpe --vocab-size 2048 --seed 42 \
  --chunk 64 --batch-size 1 --accumulate 8 -e 1 \
  --skip-tokens 0 --max-tokens 200000 --lr 0.001 \
  --loss-csv depth1.csv --loss-every 10000

"$BIN" train "$CORPUS" -o depth4.pssa \
  --latent 166 --depth 4 --state 16 --key 32 --memory 512 \
  --tokenizer bpe --vocab-size 2048 --seed 42 \
  --chunk 64 --batch-size 1 --accumulate 8 -e 1 \
  --skip-tokens 0 --max-tokens 200000 --lr 0.001 \
  --loss-csv depth4.csv --loss-every 10000
```

Training windows retain the existing cyclic EOF behavior; the example is an
actual 200k-token prefix only if the encoded corpus is at least that long.
Target tokens exclude document/window boundaries. Compare **target-token
exposure**, not just update counts or nominal caps. For chained runs, choose one
fixed whole-run `--total-updates` horizon on each fresh model and keep the same
windows, chunk/lane/accumulation options on subsequent links. Do not alter
batching in the middle of an exact schedule comparison.

A continuation preserves the stored depth and tokenizer when `--depth` is
omitted:

```sh
"$BIN" train "$CORPUS" -o depth4-next.pssa --resume depth4.pssa \
  --skip-tokens 200000 --max-tokens 200000 -e 1 \
  --batch-size 1 --accumulate 8 --loss-csv depth4.csv --loss-every 10000
```

An explicitly different `--depth`, width, state, key, capacity or chunk length
is a validation error, not an expansion/conversion operation. Depth-one saves
remain V7; stacks save V8 with all per-layer model/optimizer/carry/bank state,
shared endpoints, tokenizer and schedule metadata. Existing V5/V6/V7 loads and
Kaggle `--resume`, `--skip-tokens`, `--max-tokens` invocations remain available.
Existing loss CSV append validation is unchanged; starting a **new** CSV on a
resume still requires explicit `--tokens-seen` target-token provenance.

## Held-out scoring and samples

Use a held-out file or a non-overlapping encoded slice, with the checkpoint's
embedded tokenizer. Evaluation is strict and **never wraps** at EOF:

```sh
"$BIN" evaluate data/heldout.txt -m depth1.pssa --skip-tokens 0 --max-tokens 50000
"$BIN" evaluate data/heldout.txt -m depth4.pssa --skip-tokens 0 --max-tokens 50000
"$BIN" generate -m depth1.pssa -p "The scientist observed" --temperature 0 --max-new-tokens 64
"$BIN" generate -m depth4.pssa -p "The scientist observed" --temperature 0 --max-new-tokens 64
```

Evaluation resets every layer's carry at document boundaries, carries state
across chunks inside each selected document, does not write banks, and restores
**all** incoming carries even when non-finite scores produce an error. Retain
all fixed-prompt generations, not only favorable samples. Lower loss, checkpoint
roundtrips, successful training and extra layers do not meet the project's
language-quality success bar: unselected coherent output must be demonstrated.

## Verification

`tests/stacked_training.rs` checks CLI bounds/default/resume semantics, explicit
CPU reporting, complete-stack batched dispatch, ragged independent stacked lanes
with accumulation and deferred all-layer writes, exact batched epoch-boundary
resume, and bounded evaluation's all-layer state preservation. Depth-one
regressions remain covered by `sequence_batch`, `batch_training`,
`backward_blocked`, the CPU-twin example and existing runtime tests. Architecture
and checkpoint suites separately cover residual gradients and V8 persistence.
