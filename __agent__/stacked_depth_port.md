# Stacked-depth port completion

## Test-helper fix

`tests/checkpoint_repair.rs` no longer builds an array of simultaneous mutable
references through `PSSALayerV2`'s `DerefMut` fields. The fixture now initializes
its twelve endpoint/block matrices one at a time through a narrow
`persistent_matrix_mut` match accessor. The malformed-tensor loop uses the same
accessor to hold only the one matrix it is corrupting. This preserves the
existing fourteen tensor cases and all assertions. The adapter-duplication
case also clones the existing adapter into a local before pushing it, avoiding
an immutable borrow overlapping the vector mutation.

No tests were deleted, ignored, weakened, or had their assertions changed.

## Build and test results

With the required compiler path exported:

```text
export PATH=/workspace/bin:/workspace/.local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
```

- `cargo build --release`: passed.
- `cargo test --release`: passed: 171 tests passed, 4 benchmark/timing tests
  ignored, and 0 failed. This includes all existing tests and the new
  stacked-depth suites.
- Targeted release verification:
  `cargo test --release --test depth_one_parity --test stacked_depth --test stacked_checkpoint`:
  12 tests passed, 0 failed.

The compiler emitted the existing linker warning about a deprecated linker
optimization setting; it did not affect the successful build or tests.

## Required verifications

1. **Depth-one parity and old checkpoints:**
   `depth_one_is_bit_exact_with_main_and_loads_unchanged_v7_checkpoints` passed.
   It compares the new depth-one initial checkpoint byte-for-byte with the
   unmodified-main fixture, loads that existing V7 checkpoint, compares the
   observed training/inference scalar bits, and compares the trained
   checkpoint bytes. Thus depth 1 remains numerically identical to the
   pre-port single-layer model and existing checkpoints still load.
2. **Parameter-matched depth four:**
   `parameter_matched_depth_four_config_has_the_required_count` passed with
   `baseline.parameter_count() == 1,544,704` for width 256/depth 1 and
   `stack.parameter_count() == 1,548,448` for width 166/depth 4.
3. **Depth-four checkpoint round trip:**
   `depth_four_roundtrips_all_state_and_next_optimizer_update` passed. The V8
   depth-four checkpoint reloads with all shared endpoint weights, all four
   blocks' parameters/optimizer state, carries, memory banks, tokenizer and
   schedule metadata equal; the serialized bytes round-trip identically,
   inference logits match, and the next backward/AdamW update matches and
   advances the step counter to 12.

## Remaining work

No functional work from the stacked-depth port remains undone. Depth greater
than one remains intentionally CPU-only as documented by the port; that is a
design constraint, not an unfinished test fix. I did not reformat unrelated
pre-existing files solely to make `cargo fmt --check` clean.
