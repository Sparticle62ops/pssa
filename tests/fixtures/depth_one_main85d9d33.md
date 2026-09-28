# Unmodified-main depth-one golden fixtures

Generated from commit `85d9d337875d58c709d5e32703d2f43825185f8e`, exported with
`git archive` to a separate directory. No stacked-depth code was used to produce
these files. The recipe is `tests/support/depth_one_reference.rs`, compiled as
an example against that original crate with its original release profile
(opt-level 3, fat LTO, one codegen unit) on x86_64 Linux.

- `*_initial.pssa`: `checkpoint::save_model(&reference::model(), ...)`.
- `*_observed.bin`: the little-endian `u32::to_le_bytes` of each f32 bit pattern
  returned by `reference::advance(&mut model)`.
- `*_trained.pssa`: save the same model after `advance`.

The recipe activates the normally zero-initialized MLP/adapter outputs, sets
nonzero adapter slow state and carry, populates two memory entries, attaches
vocabulary and schedule metadata, and performs two token-weighted updates with
unequal chunk lengths, consolidation, and token inference. Thus byte comparison
covers gradients and moments as well as weights, carry, memory, RNG and counters.

SHA-256:

```text
29b3266df8c26ace956bf945c5092b56efcb9caf251589587d999600944ef986  depth_one_main85d9d33_initial.pssa
34c5742255c81a7e30a0d7eda8e4bc40a668eef2f4ca16ab6a121dd749a25e49  depth_one_main85d9d33_observed.bin
bb25281eaf4ee967d29b8ca4c47a9816a0da9b4410493623152c1ed780e74c6e  depth_one_main85d9d33_trained.pssa
```

The test `depth_one_is_bit_exact_with_main_and_loads_unchanged_v7_checkpoints`
compares a fresh depth-one model with both old fixtures, then loads the old
initial/trained checkpoints and checks exact continuation. Existing V5/V6 and
BPE/V7 compatibility tests remain in the regression suite.
