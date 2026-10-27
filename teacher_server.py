#!/usr/bin/env python3
"""Qwen3 teacher logit server for the PSSA distillation trainer.

Tokenizes the corpus with the shared Qwen tokenizer, runs the teacher over
fixed-size token chunks on the GPU, and writes the framed binary stream the
Rust trainer consumes on stdin (layout in examples/distill_train.rs):

    u32 magic 0x4B4C4431, u32 n, u32 vocab, u64 chunk_index,
    u32[n] token_ids, u32[n] target_ids, f32[n * vocab] teacher probs at T.

Teacher logits at position i predict the token after x[i], so a chunk's
targets are simply the next `chunk` tokens of the stream.
"""

import argparse
import struct
import sys

MAGIC = 0x4B4C4431


def corpus_blocks(paths, block_chars=1 << 24):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            while True:
                block = handle.read(block_chars)
                if not block:
                    break
                yield block


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="comma-separated text files")
    ap.add_argument("--teacher", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = (
        AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    if args.out == "-":
        stream = sys.stdout.buffer
    else:
        stream = open(args.out, "wb")

    paths = [p.strip() for p in args.corpus.split(",") if p.strip()]
    carry: list[int] = []
    pos = 0
    emitted = 0
    vocab = None

    def flush_chunk(carry, pos):
        nonlocal emitted
        n = args.chunk
        tokens = carry[pos : pos + n]
        targets = carry[pos + 1 : pos + 1 + n]
        x = torch.tensor([tokens], device=args.device, dtype=torch.long)
        with torch.inference_mode():
            logits = model(x).logits[0, :n].float()
        probs = torch.softmax(logits / args.temp, dim=-1).to(torch.float32)
        nonlocal vocab
        if vocab is None:
            vocab = int(probs.shape[1])
        stream.write(struct.pack("<IIQ", MAGIC, n, vocab, emitted))
        stream.write(struct.pack(f"<{n}I", *tokens))
        stream.write(struct.pack(f"<{n}I", *targets))
        stream.write(probs.cpu().numpy().tobytes())
        emitted += 1
        if emitted % 16 == 0:
            print(f"teacher_server: {emitted} chunks, {emitted * n} tokens", file=sys.stderr, flush=True)

    for block in corpus_blocks(paths):
        ids = tok(block, add_special_tokens=False)["input_ids"]
        carry.extend(ids)
        while len(carry) - pos >= args.chunk + 1:
            flush_chunk(carry, pos)
            pos += args.chunk
            if args.max_chunks and emitted >= args.max_chunks:
                stream.flush()
                print(f"teacher_server: stopped at {emitted} chunks (max)", file=sys.stderr)
                return
        if pos > (1 << 20):
            carry = carry[pos:]
            pos = 0
    stream.flush()
    print(f"teacher_server: done, {emitted} chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
