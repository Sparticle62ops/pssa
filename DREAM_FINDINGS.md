# Offline dream replay findings

## What was built

PSSA now has an opt-in, runtime-only sleep phase. The default training path keeps
`--dream-every 0`, so it does not allocate a dream RNG or call replay and remains
on the historical update path. When enabled, a dream runs after every N completed
optimizer updates:

- `--dream-every N` enables the phase; `--dream-replay K` samples up to K occupied
  episodic entries (default 32).
- Memory mode replays each selected latent through the adapter that owns the
  memory entry, updates only the fast plastic adapter, and transfers its update
  into the consolidated adapter copy with the existing exact EMA split.
- `--dream-mode memory|generate|both` selects memory replay, generated replay, or
  both. Generate mode uses each selected memory value as a continuous seed,
  samples autoregressively at temperature 0.8 for `--dream-len` tokens (default
  64), and replays the resulting latent sequence before consolidation.
- Dream restores the live recurrent carry and never performs an Adam step or
  modifies main PSSA weights, optimizer moments, or the episodic bank.
- Each phase logs one line with mode, entries, generated token count,
  consolidation delta norm, and elapsed time. Dream controls are not checkpoint
  fields; they must be supplied again when resuming.
- Dream is host-only. GPU training emits an explicit warning; CUDA's
  device-resident optimizer weights are synchronized to the host before replay,
  and adapter state is refreshed back to CUDA after consolidation. The regular
  CUDA and wgpu training paths remain unchanged when dreaming is off.

The generated replay sampler excludes vocabulary ID 0 (`<unk>`), uses a fixed
caller-provided `SimpleRng`, and is therefore deterministic for a fixed model and
seed. The unit tests cover dream-off no-op behavior, empty memory, adapter-only
mutation, memory and generated replay, fixed-seed determinism, recurrent-carry
restoration, and mode parsing.

## Sequential-task probe

`examples/dream_probe.rs` is a tiny reproducible probe. It trains the same small
model on a two-token Task A, then a disjoint two-token Task B, with and without
memory dreaming, and prints Task-A loss before and after Task B plus the forgetting
delta. It can be run with:

```text
cargo run --release --example dream_probe
```

The probe was **not started in this session** because the task instructions
explicitly prohibited starting a training run. Consequently, no probe numbers
are claimed here; running the command above records the four values directly in
its two output lines. The example itself was release-checked successfully.

## Checks run

- `CARGO_BUILD_JOBS=1 csrun 'cargo test --release --lib dream'` — 8 passed.
- `CARGO_BUILD_JOBS=1 csrun 'cargo check --release --example dream_probe'` — passed.
- A CUDA all-target check remains part of the final verification pass.
