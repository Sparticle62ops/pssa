# Continual-learning benchmark findings

## What the original harness did

`src/feature_benchmark.rs` builds two independent eight-token synthetic documents:

- A: `0 1 2 0 1 2 0 1` -> `1 2 0 1 2 0 1 2`
- B: `3 4 5 3 4 5 3 4` -> `4 5 3 4 5 3 4 5`

With `epochs_per_task = 18`, each task was run for 18 repeated one-chunk updates, for 36 AdamW updates total. The original loop did a forward pass, zeroed gradients, backpropagated, optionally called `insert_training_memory(loss, inputs.len())`, applied AdamW, and optionally called `ema_consolidate_plasticity()`. It evaluated A initially, after A, and after B, and reset recurrent state before B. Evaluation saved/restored the persistent carry, so evaluation itself was not intended to become training.

The memory flag did reach production code. When enabled it called `PSSALayerV2::insert_training_memory`; that helper only writes when the chunk loss is greater than 3.5 and applies the normal refractory-protected bank insertion. It writes the terminal token's detached query/value, not every token. When disabled it skipped the call, leaving the bank empty in the relevant run.

The consolidation flag also reached production code. When enabled it called `PSSALayerV2::ema_consolidate_plasticity` after each update. That API transfers `alpha * up_proj.fast` into `consolidated_up` and scales the fast copy by `1-alpha`, preserving `fast + slow` (up to float roundoff). It is not a ridge solve despite the benchmark's feature name.

## Why the reported result was misleading

The checked-in scratch result was not produced by the current untracked harness: its JSON protocol says `d_vocab=8`, `mem_capacity=24`, and `weight_decay=0`, while the harness on disk used `d_vocab=64`, `mem_capacity=64`, and `weight_decay=0.01`. The old result was therefore also stale with respect to the source being audited.

Two independent issues explain its symptoms:

1. The old retention expression was:

   ```text
   1 - max(0, (A_after_B / A_before_B) - 1)
   ```

   followed by another lower clamp at zero. Thus every A-loss increase of at least 100% saturated to exactly `0.0`; it did not measure total forgetting. The raw losses and ratio were the real measurements.

2. In the old protocol's `weight_decay=0` configuration, consolidation's exact fast/slow transfer is functionally neutral: forward and backward use the sum, and Adam's future gradient update on the fast copy has the same effective result when no decay acts only on the fast copy. Bit-identical loss rows for consolidation on/off are therefore expected for that configuration, not evidence that the call was unreachable. The original output also showed that the bank had entries (`memory_slots_used=24`), so this was not a missed write interval.

## Changes made

Only the benchmark harness was changed; production PSSA training/inference, memory retrieval/write/refractory semantics, CLI training paths, and checkpoint formats were not changed.

- Added schema version 2 and an unclamped primary measurement. `loss_ratio_after_over_before`, A-loss delta, and an inverse-loss `retention_ratio = A_before_B / A_after_B` are emitted. A zero denominator is `null`, not fabricated with a floor. The old saturated value is retained as `legacy_clamped_retention_score` for comparison.
- Made the document boundary explicit for every repeated one-document epoch: recurrent carry resets before each task update, while the bank, parameters, slow copy, and optimizer state persist across the A/B boundary. This matches the production document selector rather than carrying the final token of one synthetic document into the next copy of it.
- Kept the production ordering visible in the harness: retrieval/backward first, optional gated memory write before AdamW, then optional consolidation after the update. Per-epoch audit rows record losses, bank slots, memory injection, whether each flag API was called, and consolidation's effective-weight change.
- Added adapter fast/slow norms and consolidation-call counts so flag plumbing is directly observable. Added regression tests for metric saturation, denominator handling, document-reset/update order, evaluation restoration, the production loss gate, all four controls, and the expected zero-weight-decay neutral behavior of exact consolidation.

## Re-run numbers

Command:

```text
CC=/workspace/bin/zigcc CXX=/workspace/bin/zigcc AR=/workspace/bin/ar CARGO_TARGET_DIR=/workspace/oxide-target-test cargo run --release -- benchmark --feature continual --out __agent__/feature_results
```

The final run with seed 4101, 18 epochs/task, and the current tiny configuration reports:

| variant | A before B -> A after B | loss ratio after/before | inverse-loss retention | memory slots A -> final | consolidation calls |
|---|---:|---:|---:|---:|---:|
| memory + consolidation | 0.005168186 -> 0.000010162 | 0.001966323 | 508.563371821 | 3 -> 7 | 36 |
| memory disabled | 0.007497854 -> 0.000981512 | 0.130905763 | 7.639083083 | 0 -> 0 | 36 |
| consolidation disabled | 0.005168491 -> 0.000010192 | 0.001971973 | 507.106276079 | 3 -> 7 | 0 |
| both disabled | 0.007498594 -> 0.000983481 | 0.131155373 | 7.624544687 | 0 -> 0 | 0 |

All four A-after-B values are distinct in the deterministic regression test. Memory-enabled runs actually write and retrieve memory (`memory_injection_l2` becomes nonzero); disabled runs do not. Consolidation-enabled runs have a nonzero slow-copy norm and 36 recorded calls; disabled runs have zero slow-copy norm and zero calls. The consolidation on/off numerical difference is small because the production transfer is intentionally effective-weight preserving; with the current nonzero AdamW decay it is nevertheless observable.

This particular synthetic probe does **not** show catastrophic forgetting after the harness repair: A's loss improves during B in all four variants, with memory giving the larger improvement. That is the honest result for this seed/task and is not evidence of general continual-learning ability; the record labels it as a single-seed training-document probe rather than held-out evidence.

## Verification

- `cargo build --release` — passed.
- `cargo test --release` — passed; all test binaries reported zero failures.
- `cargo run --release --example twin_check` — passed. The final line was `CPU TWIN CHECK PASSED`; maximum reported gradient difference was `5.960e-8` (below `1e-6`).
- The final benchmark rerun completed successfully and rewrote `__agent__/feature_results/continual_learning.json` and `summary.md`.

The repository commit is `e595c45` (`Repair continual learning feature benchmark`). The pre-existing modifications to `README.md`, `src/lib.rs`, and `src/memory.rs` were left unstaged and untouched.
