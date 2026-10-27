#!/usr/bin/env bash
# Qwen3-1.7B -> PSSA compact student distillation, one container:
# python teacher (GPU) pipes framed teacher targets into the Rust trainer.
set -euo pipefail
cd "$(dirname "$0")"
CORPUS=${CORPUS:?set CORPUS to the corpus text file}
STUDENT=${STUDENT:?set STUDENT to the student checkpoint path}
OUT=${OUT:-pssa-distill-out.v6}
LR=${LR:-1e-4}
CHUNKS=${CHUNKS:-0}
python3 teacher_server.py --corpus "$CORPUS" --temp 2.0 --chunk 64 --max-chunks "$CHUNKS" \
  | ./target/release/examples/distill_train --student "$STUDENT" --teacher-input - \
    --lr "$LR" --grad-clip 1.0 --save-every 2000 --out "$OUT" --temp 2.0 --ce-mix 0.5
