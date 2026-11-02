<p align="center">
  <img src="docs/img/pssa-mark.svg" alt="PSSA mark" width="220">
</p>

# PSSA: a plastic state-space architecture

[![Discord](https://img.shields.io/badge/discord-join%20PSlabs-5865F2?logo=discord&logoColor=white)](https://discord.gg/9sqfKeqWYF) [![Hugging Face](https://img.shields.io/badge/huggingface-sparticle62%2Fpssa-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/sparticle62/pssa) [![Blog](https://img.shields.io/badge/blog-read%20the%20notes-3BCB78)](https://sparticle62ops.github.io/pssa/blog.html)

[Technical deep dive](https://sparticle62ops.github.io/pssa/technical.html) · [Support PSSA](DONATE.md) · [The math, in full](MODELMATH.md) · [Qwen distillation math](docs/qwen-distill.md) · [Universal converter](docs/convert.md)

PSSA is a small language model that is not a transformer. It reads text one
token at a time through a recurrent state-space layer, keeps a bank of episodic
memories it can look things up in, and rewrites part of its own weights while it
runs. It is written in Rust from scratch, with no PyTorch, no TensorFlow, and no
ML framework of any kind underneath it.

At matched parameters and on the same corpus, it learns faster than a
transformer and generates text about twelve times quicker on the same CPU.

![The PSSA terminal dashboard during a live training run](docs/img/tui-demo.gif)

## Headline result

Two models, same corpus, same tokenizer, same optimizer schedule, same seed,
same number of parameters. One is PSSA, one is a standard transformer. Over
12.7M tokens of cleaned WikiText-103, PSSA finished at **3.98** training
cross-entropy against **4.43**: a gap of **0.45 nats**, perplexity 53.7 against
83.7.

![PSSA and the matched transformer on the same schedule](docs/img/pssa-vs-transformer-training.png)

The gap holds on text neither model has seen: on a 198,939-token held-out slice,
PSSA scores **3.997** cross-entropy against **4.429** (perplexity 54.4 against
83.8, next-token accuracy 24.1% against 18.0%). On matched CPU hardware PSSA also
trains at 1,716 tokens/second against 415 (4.1x), and generates 200 tokens in
226 ms against 2,735 ms (12x).

These are 1.5M-parameter models on 12.7M tokens: a research prototype, not a
competitor to anything you have heard of. The full curves, the matched-run
methodology, the caveats and the reproduction scripts are in
[docs/benchmarks.md](docs/benchmarks.md).

On the held-out slice and on matched CPU hardware:

| Held-out slice, 198,939 unseen tokens | PSSA | Transformer |
| --- | --- | --- |
| Cross-entropy | **3.997** | 4.429 |
| Perplexity | **54.4** | 83.8 |
| Next-token accuracy | **24.1%** | 18.0% |

| Same CPU, fixed work | PSSA | Transformer |
| --- | --- | --- |
| Training throughput | **1,716 tok/s** | 415 tok/s (4.13x) |
| Generate 200 tokens | **226 ms** | 2,735 ms (12x faster) |

## Models and weights

- [NSC64/pssa-100m](https://huggingface.co/NSC64/pssa-100m) — a 109M-parameter PyTorch port of this architecture, trained from scratch by NSC64 on 4B tokens of TinyStories and FineWeb-Edu. Ships bf16 safetensors weights, a 16k byte-level BPE tokenizer and a self-contained loader (plain scan on CPU, fused Triton kernel on CUDA).
- [sparticle62/pssa-checkpoints-105m](https://huggingface.co/sparticle62/pssa-checkpoints-105m) — training checkpoints from the 105M run in progress (raw .pssa resume files, not a packaged model).
- [NSC64/pssa-distill-105m](https://huggingface.co/NSC64/pssa-distill-105m) — a 107M PSSA student distilled from Qwen3-8B by NSC64, continuing the ck256.pssa base for 5,000 updates on about 2 GB of FineWeb-Edu, OpenWebMath and CodeParrot (0.5 CE + 0.5 KL at temperature 2.0, one RTX PRO 6000). Ships V7 .pssa checkpoints, GPL-3.0.

## What makes it different

- **A recurrent state-space core.** Learned continuous state matrices carry
  information forward in a fixed-size state, instead of attention over the full
  context window.
- **An episodic memory bank.** 512 slots with hyperbolic (Poincare-style)
  retrieval, written to and read from during the run.
- **Plastic weights.** Fast updates reinforce what works, novelty drives growth,
  and a refractory gate rate-limits overwrites.
- **Closed-form consolidation.** A ridge-regression step folds the fast plastic
  updates back into the base transition matrix.
- **No framework.** Hand-written linear algebra in Rust, with a CUDA path for
  training and a scalar CPU reference that every gradient is checked against
  (max gradient difference 2.98e-8).

The layer, the write path and the training pipeline are described in
[docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/Sparticle62ops/pssa.git
cd pssa
cargo install --path .                    # CPU build
cargo install --path . --features cuda    # NVIDIA GPU build (cuBLAS)
pssa
```

Running it with no arguments opens the dashboard. Everything the dashboard does
is also a plain CLI command, so it works the same over SSH, in a notebook
terminal, or in a script. Requirements, backends, datasets, Hugging Face caches
and long-corpus training chains are covered in [docs/setup.md](docs/setup.md).

## Documentation

Getting started and reference:

- [docs/setup.md](docs/setup.md) — install, build, backends, datasets, checkpoints.
- [docs/architecture.md](docs/architecture.md) — how the layer works.
- [docs/benchmarks.md](docs/benchmarks.md) — results and how they were measured.
- [docs/cli.md](docs/cli.md) — commands and options.
- [docs/tui.md](docs/tui.md) — the terminal dashboard.
- [docs/tui-setup.md](docs/tui-setup.md) — the TUI training wizard.
- [docs/experiments.md](docs/experiments.md) — dreaming, the loss guard, memory capacity, the world model.
- [docs/convert.md](docs/convert.md) — converting any Hugging Face checkpoint to PSSA.
- [docs/contributors.md](docs/contributors.md) — building, testing, contributing.
- [MODELMATH.md](MODELMATH.md) — the math, in full.
- [CONTRIBUTING.md](CONTRIBUTING.md) — quick build/test/PR guide.
- [DONATE.md](DONATE.md) — sponsorship.

Deeper dives:

- [docs/baseline.md](docs/baseline.md) — the parameter-matched transformer baseline.
- [docs/comparison.md](docs/comparison.md) — scoring PSSA against that baseline.
- [docs/training-stability.md](docs/training-stability.md) — gradient clipping and the memory cap.
- [docs/paired-eval.md](docs/paired-eval.md) — multi-seed paired evaluation.
- [docs/batching.md](docs/batching.md) — independent-sequence batching.
- [docs/stacked-depth.md](docs/stacked-depth.md) — stacked depth and loops.
- [docs/distill-trainer.md](docs/distill-trainer.md) — Qwen3-to-PSSA distillation trainer.
- [docs/tui-features.md](docs/tui-features.md) — Kaggle, memory inspector, A/B inference, runs browser, alerts, GIF recording.
- [docs/tui-demo.tape](docs/tui-demo.tape) — the dashboard GIF recording tape.
- [assets/math.md](assets/math.md) — layer equations, rendered in the dashboard's F11 math tab.

## What this is not

- These are **1.5M-parameter models** on 12.7M tokens. That is a research
  prototype, not a competitor to anything you have heard of.
- Text quality at this scale is poor for both models. PSSA emits "a barget of
  the Prian Academy", the transformer "a material circulation of the United
  States". The comparison is about learning efficiency, not fluency.
- The speed comparison is CPU-to-CPU, which is fair. The training throughput
  numbers in the write-up are **not** hardware-matched and should not be read as
  an architecture result.
- Two experiments are still unmeasured: retention of earlier skills after a
  corpus switch, and whether ablating the memory bank changes the loss.

## Project status

Current roadmap, the first big training run, depth and loops, WebGPU training
and what is next are tracked in
[docs/contributors.md](docs/contributors.md#project-status-and-roadmap).

## Where the project needs help

### Compute

The whole result above was trained on a free hosted notebook with a single
entry-level GPU, in 200,000-token links, because a session gets cut after a few
hours. Every interesting question left, whether the gap holds at 10x or 100x
these parameters, whether the memory bank matters at scale, how it does against
a modern recurrent baseline, needs one thing: a GPU with real VRAM and
allocations measured in days instead of hours. Anything meaningfully above the
entry-level card this ran on changes what can be asked.

If you have compute to grant, or you work somewhere that does, that is the
single highest-leverage thing anyone can offer this project. The 105M-parameter run is
underway: training checkpoints land on
[sparticle62/pssa-checkpoints-105m](https://huggingface.co/sparticle62/pssa-checkpoints-105m)
as it goes, and the same-scale transformer baseline is still the milestone this
all builds toward.

### Sponsorship

Sponsorship funds compute and nothing else. In return you get named here and in
the write-up of any result your hardware made possible. Get in touch before
sending anything so the details can be agreed.

### Contributing

Issues and pull requests are welcome. The parts most in need of hands: kernel
performance, a modern recurrent baseline to compare against, and evaluation
beyond next-token loss. Validate any branch with `cargo test --release` before
opening a PR. See [docs/contributors.md](docs/contributors.md) for the full
guide.

### Contact

Sparticle62@proton.me

### Donate

Solana: `4XPZ9uAa2BMoth6msoHRxTWL4mUrMfq3LGrxbAGja96h`

## License

See [LICENSE](LICENSE) for the project license.
