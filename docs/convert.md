# Universal checkpoint converter

`pssa convert` (and the `convert_any` example) turn a Hugging Face LLM
checkpoint into a PSSA v6 checkpoint. The converter transplants the token
table: every embedding row is pushed through a fixed seeded projection into
the target latent width, rescaled to unit RMS, the unembedding is tied to the
projected table, and the logit scale is calibrated on the emitted student.
The math and the reasoning live in [Qwen distillation math](qwen-distill.md).

## Usage

```
pssa convert <model-or-dir> <config.json> <out.v6> [--target compact|source-width] [--seed N]
```

`MODEL-OR-DIR` is a single `.safetensors` file or a model directory with
shards (they are scanned in sorted order for the embedding tensor).
`CONFIG.JSON` is the checkpoint's `config.json`. `--target compact` gives the
standard 256-latent student; `--source-width` keeps the source hidden width as
the latent width (large: check RAM before using it on big models).

## Supported families

Qwen3 (dense), Llama, Mistral, Qwen2, Mixtral (MoE), Gemma, Phi-3 and
GPT-2-style checkpoints are detected from `model_type` and converted. Newer
Qwen generations (3.5/3.6/3.8) use a hybrid linear-attention scan the
converter cannot read; they are rejected with a clear message instead of
being silently misread.

What transfers is the token table. Scan, memory-bank and adapter weights keep
their seeded initialization, exactly like the Qwen3-specific path
(`pssa convert` on a Qwen3 config produces the same shape of output as
`qwen3_convert`). The student still trains from the transplanted table; the
teacher calibration described in qwen-distill.md is a separate step.

## Notes

- Embedding rows are streamed one row at a time from disk, so large vocab
  tables do not have to be resident all at once.
- Source stacks deeper than 32 layers are clamped to the student's 32-layer
  shape with a visible note in the report.
- Mixture-of-experts checkpoints convert (the embedding is standard) and the
  report says so explicitly.
- Tests: `cargo test --test convert_detect` covers detection, the
  nested-`text_config` wrappers, the hybrid rejection, and an end-to-end
  conversion of a tiny synthetic safetensors file.
