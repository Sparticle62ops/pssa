# Cleanup notes

- Normalized Rust formatting across the package and test/example targets; this is behavior-neutral and keeps future diffs focused on code changes.
- The RAM process wrapper treated telemetry switches as value-taking options and could forward or remove the wrong argument; it now recognizes every boolean training switch while stripping only `--ram-mib` and its value.
- Qwen3 1.7B converter fixed: borrow-after-move in the attention residual add and a usize-negate in RoPE frequency; memory-bank estimates now match HyperbolicEpisodicBankV2's real per-slot layout (norm_sq + confidence f32 + last_seen_step usize = 16 B/slot), tests updated to 16,744,448 / 119,504,896 bytes.
