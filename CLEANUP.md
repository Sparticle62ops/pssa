# Cleanup notes

- Normalized Rust formatting across the package and test/example targets; this is behavior-neutral and keeps future diffs focused on code changes.
- The RAM process wrapper treated telemetry switches as value-taking options and could forward or remove the wrong argument; it now recognizes every boolean training switch while stripping only `--ram-mib` and its value.
- Qwen3 1.7B converter fixed: borrow-after-move in the attention residual add and a usize-negate in RoPE frequency; memory-bank estimates now match HyperbolicEpisodicBankV2's real per-slot layout (norm_sq + confidence f32 + last_seen_step usize = 16 B/slot), tests updated to 16,744,448 / 119,504,896 bytes.
- Qwen3 1.7B emit step landed: `emit_pssa_checkpoint` writes a v6 checkpoint laid out for the requested target (compact or source-width). PSSA scan/memory/adapter weights keep their seeded init; the token table transfers through a fixed seeded projection into the target latent width (rows rescaled to unit RMS) and the unembedding mirrors it, preserving Qwen3's tied input/output relationship in projected space. Round-trip test added (load back, dims, tied equality, row RMS, byte-identical same-seed re-emit); `examples/qwen3_convert.rs` drives the full path.
- Added examples/qwen3_bench.rs: times prefill and greedy decode on a converted v6 checkpoint and probes fresh seeded models across chunk lengths 64/128/256 and memory capacities 512/2048 at depths 1 and 4, CPU and CUDA, for the 4B port.

- Switched emit_pssa_checkpoint to checkpoint::save_model: V6/V7 cannot serialize the stacked (depth > 1) Qwen3 target, so the emitted 1.7B checkpoint is V8 for stacks (V7 for depth one), which load_checkpoint reads transparently.
