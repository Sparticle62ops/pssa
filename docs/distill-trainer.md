# Qwen3-1.7B -> PSSA distillation trainer (design, v2)

Student = qwen3-1-7b-pssa-compact.v6 (vocab 151936, latent 256, depth 28,
state 16, key 32, capacity 512, chunk 64). The token space was locked at
full 151,936 on 2026-10-27: the compact already has that vocab, so no
architecture change is needed for the distill student.

Teacher = Qwen3-1.7B safetensors, loaded through src/qwen3.rs
(Qwen3Model::load_safetensors + forward_logits), run on Modal L4.

## Objective
Temperature-scaled KL over the FULL teacher distribution, T = 2.0, mixed
with hard-token CE (ratio to calibrate in the smoke run):
    L = alpha * KL_T(p_teacher || p_student) + (1 - alpha) * CE

## Where it lands in the code (verified against current src)
- Forward loss: src/gpu_batch.rs stage_logits_loss (line ~1071) and the
  stacked-depth variant stacked_logits_loss (~1465). The stacked path is the
  one the depth-28 student actually takes.
- Backward: bwd_stage_logits (~1620) -> bwd_stage_logits_blocked (~1638)
  and bwd_stage_logits_scalar (~1705). CE gradient today is
      g[i] = (probs[i] - indicator[tgt]) * scale_loss * logit_scale
  KL swaps the indicator for the teacher distribution:
      g[i] = (probs[i] - p_teacher_T[i] * T * T_adj) * scale_loss * logit_scale
  with the exact T factor settled by the derivative of
  KL_T = sum_i p_T_i * (log p_T_i - log softmax(l_i / T)); implement once in
  the blocked path, mirror in scalar, verify both against a finite-difference
  probe on a tiny config (pattern: examples/gpu_train_check.rs).
- Tape: ChunkActivationTape (src/pssa.rs ~338) gains teacher_probs and a
  distill flag; empty/false by default so nothing else moves.

## Data
Qwen2 tokenizer via the tokenizers crate (tokenizer.json from the Qwen3-1.7B
HF snapshot), streaming corpus, TokenChunkIterator for chunk-64 batching.

## Run shape (per plan P2)
1. Smoke: ~10 min on L4, measure tok/s, verify KL falls, then quote
   the real budget before the full 500M-token run (~$110 on L4 vs burner
   Starter credits ~$30/mo).
2. Checkpoint every link, upload to volume pssa-artifacts.

## Status
- Trainer skeleton: examples/distill_train.rs (this pass).
- KL loss/backward surgery: next pass, after skeleton compiles.

## Pure-Rust pipeline (2026-10-27, python dropped)

- `examples/distill_train.rs` runs standalone: `--corpus` + `--teacher` load the
  Qwen3 teacher in-process, tokenize the corpus with the shared Qwen tokenizer
  (`tokenizers` crate, already a dependency), and run the teacher on the repo's
  own device path: `Qwen3Model::forward_logits_device` (new in src/qwen3.rs)
  does every large projection through the GPU GEMM helpers, with norms, RoPE,
  causal GQA attention and SwiGLU glue on the CPU where that is cheaper at
  chunk sizes. Teacher weights upload once through the GpuDispatch weight
  cache (pointer-keyed), not per chunk.
- The old frame-stream mode (`--teacher-input`) still exists for externally
  produced teacher targets; the python teacher_server.py was deleted.
- Cross-check: `tests/qwen3_gpu_parity.rs` compares forward_logits_device
  against the independent per-position CPU reference `forward_logits` on a
  small random model; with a GPU present it exercises the real dispatch.

## Wired pipeline (2026-10-27)

- `examples/distill_train.rs`: reads framed teacher targets from stdin/file, sets
  tape distill targets, trains with `forward_train_chunk_batched` +
  `backward_chunk_batched`, AdamW with grad clip, periodic v6 saves.
- `teacher_server.py`: tokenizes the corpus with the shared Qwen tokenizer, runs
  Qwen3 bf16 on the GPU, emits framed chunks (token ids, next-token targets,
  teacher probs at T). Frame: u32 magic 0x4B4C4431, u32 n, u32 vocab, u64 chunk
  index, u32 ids, u32 targets, f32 probs [token][vocab].
- `run_distill.sh`: teacher | trainer in one container. Env: CORPUS, STUDENT,
  OUT, LR (default 1e-4), CHUNKS (0 = whole corpus; small value for smoke).
- Student teacher_probs live only on the final block's tape (extra blocks
  unaffected); plain CE behavior preserved when distill_temp = 0.
