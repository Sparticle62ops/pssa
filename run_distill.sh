#!/usr/bin/env bash
# Qwen3-1.7B -> PSSA compact student distillation. Pure Rust, one process:
# the in-process teacher runs the repo's own device forward on the same GPU
# as the student. No python, no pipe.
set -euo pipefail
cd "$(dirname "$0")"
CORPUS=${CORPUS:?set CORPUS to the corpus text file}
STUDENT=${STUDENT:?set STUDENT to the student checkpoint path}
TEACHER=${TEACHER:?set TEACHER to the Qwen3 checkpoint directory (safetensors + config.json + tokenizer.json)}
OUT=${OUT:-pssa-distill-out.v6}
LR=${LR:-1e-4}
STEPS=${STEPS:-0}
cargo build --release --example distill_train
./target/release/examples/distill_train \
  --student "$STUDENT" --corpus "$CORPUS" --teacher "$TEACHER" \
  --lr "$LR" --grad-clip 1.0 --save-every 2000 --out "$OUT" \
  --temp 2.0 --ce-mix 0.5 --steps "$STEPS"
