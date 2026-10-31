# Qwen3 → PSSA distillation: the math, in full

This document records the exact math the Qwen3-to-PSSA conversion and
distillation pipeline uses, and how each piece is verified. It covers
`src/qwen3.rs` (teacher model + conversion), `src/teacher_frames.rs`
(frame protocol), `examples/teacher_dump.rs` (pre-computed teacher
frames) and `examples/distill_train.rs` (the trainer).

## What transfers and what does not

PSSA's scan, memory-bank and adapter weights have no Qwen3
counterpart, so they keep their seeded initialization. What transfers
is the token table: every Qwen3 embedding row is pushed through a
fixed seeded projection into the target latent width, then rescaled to
unit RMS per row. Because Qwen3-1.7B ties its input and output
embeddings, the projected student keeps the same tie: the unembedding
is a copy of the projected embedding table, so converted logits keep
the same geometry (unit test
`tied_logits_do_not_apply_an_extra_hidden_width_scale`).

### Logit-scale calibration

A freshly emitted stack starts logits far from softmax scale: the
first converted compact checkpoint scored about 60 CE/token against a
uniform floor of ln(vocab) ≈ 11.9. At emission time, 8 probe chunks of
random tokens are pushed through the seeded stack, the logit RMS is
measured, and the unembedding is rescaled by `1 / logit_rms` so the
initial logits sit near unit RMS. A cold start then begins near the
uniform floor instead of an exploding scale. The report printed by the
emitter carries `logit_rms_before` and `logit_scale`.

## Teacher forward, piece by piece

The teacher is a faithful scalar re-implementation of Qwen3 used both
as the distillation teacher and as a reference the device forward is
checked against (`tests/qwen3_gpu_parity.rs`).

- **RMSNorm** — `x * w / sqrt(mean(x²) + eps)`, `eps` from the config
  (1e-6), applied pre-attention and pre-MLP in the standard pre-norm
  residual layout.
- **QK-norm** — Qwen3's per-head RMSNorm over `head_dim` on queries
  and keys, applied *before* RoPE, with the same `eps` and no bias.
- **RoPE** — the Hugging Face split-half (rotate-half) layout: the
  first half of each head rotates against the second half, with
  frequencies `theta^(-2i/d)` and per-position angles. Unit test
  `rope_uses_qwen_split_half_layout` pins the layout.
- **Attention** — scaled dot-product with `1/sqrt(head_dim)`,
  grouped-query attention mapping query head `q` to key/value head
  `q / (n_heads / n_kv_heads)`, causal masking, softmax in the
  max-subtracted stable form.
- **MLP** — SwiGLU: `down(silu(gate(x)) * up(x))`, with the residual
  taken against the pre-norm input so each block computes
  `x + attn(x_norm) + mlp(norm(x + attn(x_norm)))`.
- **Head** — final RMSNorm, then logits as the dot product with the
  (tied) unembedding rows.

Config sanity: `qwen3_1_7b()` carries 16 query heads, 8 key/value
heads, `head_dim` 128, `rms_norm_eps` 1e-6 and `rope_theta` 1e6,
matching Qwen3-1.7B's published configuration; the 0.6B preset and
JSON loading (`from_json_file`) follow the same fields, with
validation rejecting GQA-incompatible widths, odd `head_dim`, missing
QK-norm and non-positive epsilon/theta.

## Teacher frames

`teacher_dump` tokenizes the corpus, runs the Qwen3 device forward per
chunk, and writes framed records the trainer can stream from stdin,
a file, or concatenated shards. Frame layout (little-endian):

```
u32 magic 0x4B4C4431 ("KLD1")
u32 n            # tokens in the chunk
u32 vocab        # d_vocab
u64 chunk_index
u32[n]  token ids
u32[n]  target ids
f32[n*vocab]  teacher probabilities
```

## The loss

Per token, with student logits `z`, teacher distribution `q` (the
teacher softmax at temperature `T`, taken from the frame or computed
in-process), hard target `t`, and `m = ce_mix`:

```
L = m * CE(z, t) + (1 - m) * KL(q || p_T),   p_T = softmax(z / T)
```

The per-logit gradient is

```
dL/dz_i = m * (p_i - 1{i = t}) + (1 - m) * (p_T,i - q_i) / T
```

Note the KL term carries a single `1/T` factor, not the classic
Hinton `T²` multiplier: the gradient is that of the plain
temperature-`T` KL divergence. This is deliberate and pinned by
`tests/distill_kl.rs`, which checks the mixed CE+KL gradient against
finite differences at depth 1 and depth 3, verifies the CE-only path
is unchanged when `ce_mix = 1`, and confirms the self-distill case
(student logits equal to the teacher's) yields zero loss and zero
gradient.

Defaults (`run_distill.sh` and the example): `--temp 2.0`,
`--ce-mix 0.5`, LR 1e-4, gradient clip 1.0, checkpoints every 2000
updates, output `pssa-distill-out.v6`.

## Verification summary

- `tests/qwen3.rs` — config presets, parameter counting, split-half
  RoPE layout, tied-logit geometry.
- `tests/qwen3_gpu_parity.rs` — the device forward path is
  cross-checked against the scalar reference above.
- `tests/distill_kl.rs` — finite-difference check of the mixed loss
  gradient, CE-only regression, self-distill zero-loss/zero-gradient.
