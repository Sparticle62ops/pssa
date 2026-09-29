# PSSA: a plastic state-space architecture

PSSA is a small language model that is not a transformer. It reads text one
token at a time through a recurrent state-space layer, keeps a bank of episodic
memories it can look things up in, and rewrites part of its own weights while it
runs. It is written in Rust from scratch, with no PyTorch, no TensorFlow, and no
ML framework of any kind underneath it.

At matched parameters and on the same corpus, it learns faster than a
transformer and generates text about twelve times quicker on the same CPU.

## The result

Two models, same corpus, same tokenizer, same optimizer schedule, same seed,
same number of parameters. One is PSSA, one is a standard transformer. Over
12.7M tokens of cleaned WikiText-103:

![PSSA vs parameter-matched transformer training loss](docs/img/loss-curve.png)

PSSA finished at **3.98** training cross-entropy, the transformer at **4.43**.
That is a gap of **0.45 nats**, perplexity 53.7 against 83.7. The transformer
spent its entire 12.7M-token budget to reach a loss PSSA had already passed
around 2M tokens in.

The two curves never cross, and they never touch:

![Overlap region, second half of training](docs/img/loss-curve-zoom.png)

### It holds on text neither model has seen

Training loss only says a model fit the stream it was fed. So both checkpoints
were scored on a 198,939-token slice cut from a part of the corpus neither run
ever touched:

![Held-out loss per checkpoint on unseen text](docs/img/heldout.png)

Every checkpoint of both runs, 64 PSSA links and 43 transformer links, scored on
a bounded 9,934-token window of that unseen slice. The curves never cross: PSSA
is ahead from the first link and finishes 0.51 nats lower. The table below is the
final checkpoint of each run on the full slice.

| Held-out slice, 198,939 unseen tokens | PSSA | Transformer |
| --- | --- | --- |
| Cross-entropy | **3.997** | 4.429 |
| Perplexity | **54.4** | 83.8 |
| Next-token accuracy | **24.1%** | 18.0% |

The held-out gap, 0.43 nats, is essentially the training gap. PSSA is not
memorizing harder, it is generalizing better.

### And it is much faster to run

Generating 200 tokens on the same CPU, same prompt, same sampler:

| | PSSA | Transformer |
| --- | --- | --- |
| 200 tokens | **226 ms** | 2,735 ms |
| Relative | **12x faster** | baseline |

A recurrent model carries a fixed-size state, so the cost of each new token does
not grow with the length of what came before. A transformer re-reads its whole
context every step.

## What is actually different about it

- **A recurrent state-space core.** Learned continuous state matrices carry
  information forward in a fixed-size state, instead of attention over the full
  context window.
- **An episodic memory bank.** 512 slots with hyperbolic (Poincare-style)
  retrieval and bounded top-4 search, written to and read from during the run.
- **Plastic weights.** Fast updates reinforce what works, novelty drives growth,
  and a refractory gate rate-limits overwrites so repeated contradictory input
  does less damage.
- **Closed-form consolidation.** A ridge-regression step folds the fast plastic
  updates back into the base transition matrix, the way sleep consolidates a
  day's learning.
- **No framework.** Hand-written linear algebra in Rust, with a CUDA path for
  training and a scalar CPU reference that every gradient is checked against
  (max gradient difference 2.98e-8).

## What this is not

Being straight about the scale, because the numbers above are easy to
over-read:

- These are **1.5M-parameter models** on 12.7M tokens. That is a research
  prototype, not a competitor to anything you have heard of.
- Text quality at this scale is poor for both models. PSSA emits "a barget of
  the Prian Academy", the transformer "a material circulation of the United
  States". The comparison is about learning efficiency, not fluency.
- The speed comparison is CPU-to-CPU, which is fair. The training throughput
  numbers further down are **not** hardware-matched and should not be read as an
  architecture result.
- Two experiments are still unmeasured: retention of earlier skills after a
  corpus switch, and whether ablating the memory bank changes the loss.

## Try it

```bash
git clone https://github.com/Sparticle62ops/pssa.git
cd pssa
cargo build --release
./target/release/oxide_ai_pssa
```

Running it with no arguments gives you a home screen listing every command plus
any checkpoint and corpus it finds in the working directory.

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
single highest-leverage thing anyone can offer this project.

### Sponsorship

Sponsorship funds compute and nothing else. In return you get named here and in
the write-up of any result your hardware made possible. Get in touch before
sending anything so the details can be agreed.

### Contributing

Issues and pull requests are welcome. The parts most in need of hands: kernel
performance, a modern recurrent baseline to compare against, and evaluation
beyond next-token loss. Validate any branch with `cargo test --release` before
opening a PR.

### Contact

Sparticle62@proton.me

### Donate

Solana: `4XPZ9uAa2BMoth6msoHRxTWL4mUrMfq3LGrxbAGja96h`

---

# Setup and codebase

Everything below is for running, training, and working on the project.

## Requirements

- Rust toolchain with Edition 2024 support, including Cargo.
- Network access only when using an HTTP/HTTPS dataset or a Hugging Face dataset.
- Enough memory and disk for larger corpora and serialized models.
- Optional: a CUDA device for the GPU training path. The CPU path is the
  reference and always available.

Direct runtime dependencies are [`ureq`](https://crates.io/crates/ureq) for
dataset downloads and [`tokenizers`](https://crates.io/crates/tokenizers) for
byte-level BPE.

## How the comparison was run

### How the two runs were matched

Both chains ran 64 links of 200,000 encoded tokens, each link resuming from the
previous checkpoint, so the learning-rate schedule and optimizer state continue
across the whole run instead of restarting per link.

- Identical corpus: one `clean-wikitext` pass over WikiText-103, reused byte for byte.
- Identical token IDs: the baseline pins `--tokenizer-from` to the PSSA chain's
  own checkpoint, so neither model sees a different vocabulary.
- Identical optimization: 30,000-update cosine horizon, no warm-up restart, 512
  supervised target tokens per update, seed 42.
- PSSA: latent 256, recurrent state 16, 512 memory slots, key width 32, vocab 2,048.
- Baseline: 1,541,120 parameters, 1 layer, width 256, 4 heads, FFN 448, vocab 2,048.

### Per-token learning curve

End-of-link training cross-entropy:

| Link | Tokens seen | PSSA | Transformer |
| --- | --- | --- | --- |
| ck01 | 200,000 | 5.733 | 6.461 |
| ck05 | 1,000,000 | 4.617 | 5.467 |
| ck10 | 2,000,000 | 4.447 | 5.082 |
| ck15 | 3,000,000 | 4.292 | 4.858 |
| ck20 | 4,000,000 | 4.185 | 4.704 |
| ck25 | 5,000,000 | 4.221 | 4.704 |
| ck30 | 6,000,000 | 4.070 | 4.561 |
| ck35 | 7,000,000 | 4.039 | 4.523 |
| ck37 | 7,400,000 | 3.960 | 4.465 |
| ck44 | 8,800,000 | 4.004 | 4.480 |
| ck48 | 9,600,000 | 3.937 | 4.415 |
| ck52 | 10,400,000 | 3.846 | 4.344 |
| ck56 | 11,200,000 | 3.887 | 4.375 |
| ck60 | 12,000,000 | 3.972 | 4.418 |
| ck64 | 12,800,000 | 3.982 | 4.428 |

The baseline's first session was cut at link 43 by the notebook session limit
and its loss CSV did not survive, so links 1 to 43 are read back from that
session's own run log instead. The chain resumed from `ck43` in a second session
and finished all 64 links, and both curves above now cover the full run.

### Throughput is not hardware-matched

PSSA trained on a Kaggle T4 at roughly 900 tokens/second. The baseline is
CPU-only, because `train-transformer` has no GPU path, and held 212
tokens/second. Those two numbers say nothing about the architectures. On the
same CPU-only Kaggle hardware the batched PSSA path measures 375 tokens/second
against the baseline's 212, and the loss comparison above is unaffected either
way, since it is matched on tokens and updates rather than on time.

### What these numbers are, and are not

The losses are end-of-link training cross-entropy on the stream being fit, not
held-out evaluation. For a held-out comparison on an unseen slice, use the
`compare` command described in [docs/COMPARISON.md](docs/COMPARISON.md).
Generation quality at this scale is poor for both models: PSSA emits "a barget
of the Prian Academy", the baseline "a material circulation of the United
States".

Two experiments are not yet measured: retention of earlier skills after a
corpus switch, and whether ablating the 512 memory slots changes loss.

### Reproducing

```bash
bash kaggle/kaggle_continue.sh              # the PSSA chain
bash kaggle/kaggle_transformer_baseline.sh  # the parameter-matched baseline
```

Both read `TOTAL`, `WINDOW` and `FRESH` from the environment and write
`--loss-csv`, so the curve survives a cut session.

## CLI Reference

General form:

```text
oxide_ai_pssa <COMMAND> [OPTIONS]
```

Commands:

| Command | Purpose |
| --- | --- |
| `train [source]` | Fit a checkpoint on a text corpus and write a `.pssa` file. |
| `generate <prompt>` | Continue a prompt with a trained checkpoint. |
| `chat [source]` or `repl [source]` | Interactive prompt loop against a checkpoint. |
| `evaluate [source]` | Cross entropy, perplexity and accuracy as JSON. |
| `status` | Checkpoints and corpora in the working directory. Takes no options. |
| `download <repo>` | Pull a Hugging Face dataset to a local file. |
| `clean-wikitext INPUT -o OUTPUT` | Stream-clean a raw WikiText file into a new UTF-8 corpus. |
| `benchmark` | End-to-end smoke test on the built-in corpus. |
| `gpu-probe` | Check whether a WebGPU compute device is usable. |
| `help` | Print command and option help. |

Options:

| Option | Default | Applies to | Description |
| --- | --- | --- | --- |
| `-d, --data <source>` | `data/downloaded.txt` when present, otherwise `science` | `train`, `chat`, `evaluate` | Dataset source, or a comma-separated list. |
| `-m, --model <path>` | `data/model.pssa` | `chat`, `generate`, `evaluate` | Checkpoint to load. |
| `-o, --out <path>` | Command-specific; required for `clean-wikitext` | `train`, `download`, `clean-wikitext` | Output checkpoint or dataset path. Cleaning requires a new file. |
| `-p, --prompt <text>` | empty | `generate` | Prompt text. Required for generation. |
| `-e, --epochs <n>` | `4` | `train` | Training epochs. |
| `-t, --temp, --temperature <float>` | `0.70` | `chat`, `generate` | Sampling temperature. |
| `--max-new-tokens <n>` | `64` (maximum 100,000) | `generate` | Generation length cap. |
| `--latent <n>` | `256` | `train` | Latent dimension. |
| `--state <n>` | `16` | `train` | Recurrent state dimension. |
| `--key <n>` | `32` | `train` | Memory-key dimension. |
| `--memory <n>` | `512` | `train` | Memory bank capacity. |
| `--chunk <n>` | `64` | `train` | Sequence chunk length. |
| `--lr <float>` | `1e-3` | `train` | Base learning rate. |
| `--accumulate <n>` | `8` | `train` | Chunks per optimizer update. |
| `--warmup-steps <n>` | `0` | `train` | Linear warm-up before cosine decay. |
| `--seed <n>` | `42` | `train` | Initialization seed. |
| `--tokenizer <bpe\|word>` | `bpe` | `train` | Tokenizer family. |
| `--vocab-size <n>` | `2048` | `train` | BPE vocabulary maximum. |
| `--max-tokens <n>` | unset | `train` | Global cap across input documents, not per document. |
| `--skip-tokens <n>` | `0` | `train` | Skip this many tokens before training starts. |
| `--resume <path>` | unset | `train` | Continue from an existing checkpoint. |

Positional arguments and long/short options can be mixed:

```bash
cargo run --release -- train data/downloaded.txt -e 2 -o data/experiment.pssa
cargo run --release -- train --data data/downloaded.txt --epochs 2 --out data/experiment.pssa
```

### Chat commands

Inside the REPL:

- `/exit` or `quit` exits the process.
- `/info` prints the loaded model path, memory slot count, and adapter count.
- `/temp <value>` reports a temperature value but does not apply it to later turns. Pass `--temp` when launching `chat` instead.

## Training over a long corpus

`--skip-tokens`, `--max-tokens` and `--resume` together let a long corpus be trained as a chain of short runs, so a single run never has to survive a session limit. If a window crosses EOF, selection wraps to the beginning of the corpus. Each link trains its own window and hands its optimizer state to the next:

```bash
cargo run --release -- train data/downloaded.txt -e 1 \
  --skip-tokens 0      --max-tokens 200000 -o chain/ck01.pssa
cargo run --release -- train data/downloaded.txt -e 1 \
  --skip-tokens 200000 --max-tokens 200000 --resume chain/ck01.pssa -o chain/ck02.pssa
```

`kaggle/kaggle_continue.sh` drives this pattern end to end: it sets a window size and a link count, walks the corpus offset by offset, and resumes each link from the previous checkpoint. `status` then reports every checkpoint in the chain with its shape and optimizer step count.

## Dataset Sources

`DatasetManager` accepts one or more comma-separated sources:

```bash
cargo run --release -- train science                       # built-in reference corpus
cargo run --release -- train data/downloaded.txt           # local text file
cargo run --release -- train data/                         # every readable file in a directory
cargo run --release -- train https://example.org/corpus.txt
cargo run --release -- train hf:owner/dataset              # Hugging Face repository
cargo run --release -- train science,data/downloaded.txt   # multiple sources
```

Local files and directories are read directly; HTTP(S) URLs and explicit `hf:owner/dataset`
sources are downloaded. Structured responses are reduced using common fields such as
`text`, `content`, `article`, `story`, `instruction`, `output`, `sentence`, and `summary`;
structured responses without a supported text field are rejected.

Byte-level BPE keeps exact UTF-8 case, whitespace, punctuation, and line endings, and has a complete 256-byte fallback alphabet, so valid UTF-8 never collapses to `<unk>`. The previous lowercase word splitter, including its 10,000-word cap and `<unk>` behavior, is available only with `--tokenizer word`.

Download a Hugging Face dataset into a local text file:

```bash
cargo run --release -- download wikimedia/wikipedia --out data/downloaded.txt
```

Network downloads are not validated or curated by Oxide AI. Review licensing, privacy, and content before training on an external corpus.

### Cleaning WikiText raw corpora

Clean extracted `wikitext-103-raw` text **before a fresh training run**:

```bash
./target/release/oxide_ai_pssa clean-wikitext wiki.train.raw --out data/wikitext-clean.txt
./target/release/oxide_ai_pssa train data/wikitext-clean.txt -o data/model.pssa
# Also available: oxide_ai_pssa help clean-wikitext
```

The same command can be used in Kaggle after extracting text from Parquet; it
accepts a local UTF-8 text file, not Parquet itself. `-o` and `--out` are aliases.
The output path is required and must not already exist (including the input
path or a link to it). This protects the original corpus; choose a new output
name for another run. Read, UTF-8, and write failures exit nonzero through the
normal CLI error path, with partial output removed when possible.

The pass:

- Joins `@-@`, `@.@`, and `@,@` to adjacent text: `guest @-@ starring` →
  `guest-starring`, `52 @.@ 9` → `52.9`, `500 @,@ 000` → `500,000`.
- Drops balanced heading lines such as `= Title =` and `= = Section = =`.
- Removes `<unk>` and collapses remaining inline whitespace to single spaces.
- Removes spaces before `.`, `,`, `)` and after `(`; trims each line.
- Retains at most one consecutive blank line, including at the start/end.
  Removing a heading does not introduce a blank line.
- Writes LF line endings, including a newline on the last retained line.

`oxide_ai_pssa::dataset::clean_wikitext(reader, writer)` is the reusable library
API (`BufRead` / `Write`, returning `std::io::Result<()>`). The CLI uses buffered
file I/O, and the cleaner retains only its input/output line buffers: memory is
proportional to the longest line, not the corpus size. Library callers using a
buffered writer must flush it themselves; the CLI explicitly checks the flush.
No new dependencies are required.

Cleaning is opt-in: existing loaders, tokenizers, training commands, and
`kaggle/kaggle_continue.sh` are unchanged. **Do not switch an in-flight resume
chain to a cleaned corpus**: cleaning changes token IDs/counts and the meaning
of `--skip-tokens` offsets. Prepare and consistently reuse one cleaned corpus
for a new chain instead.

## Training Pipeline

The `train` command performs two phases:

1. **Continuous recurrent ingestion:** token transitions are processed through the PSSA layer. The model updates state, memory, adapters, and routing behavior with a cosine learning-rate schedule.
2. **Adapter consolidation:** after each epoch, the plastic adapter's fast coefficients are folded into its consolidated coefficients with the configured EMA rate.

Defaults are latent 256, recurrent state 16, memory-key 32, memory capacity 512, chunk length 64, learning rate 1e-3, 8 chunks per update, and seed 42. The resulting binary holds weights, configuration, memory, adapters, and optimizer state. It is not an interchange format for other ML frameworks and should be loaded through `PSSALayer::import_from_pssa_bytes`.

## Checkpoint compatibility

New saves use **V7**: the full V6 training/resume payload plus a bounded, length-prefixed standard tokenizer JSON. A V7 BPE checkpoint is self-contained and restores its exact ordered vocabulary without access to the training or evaluation corpus. `generate` and `chat` reject `--data` for V7 BPE because retraining a tokenizer on external data would not validate provenance. V7 word checkpoints and V6 checkpoints retain the legacy optional `--data` exact-vocabulary comparison. Checked V5 artifacts remain inference-only and require `--data` because they never contained tokenizer provenance.

## Inference

Generation is autoregressive and uses temperature 0.70, a top-24 candidate limit followed by top-p 0.85 filtering, a 1.25 repetition penalty over a recent 64-token window, immediate self-transition suppression, `<unk>` suppression, and a default cap of 64 new tokens, ending early after two generated periods.

V7 BPE inference restores the exact embedded tokenizer and never rebuilds it from a selected dataset. Evaluation supplies its data only as held-out text to the restored tokenizer.

## Benchmark

```bash
cargo run --release -- benchmark
```

The suite exercises synthetic streams for contradictory facts, MQAR-style distractors, burst repetition, model serialization, and short generation prompts. It prints milestone results, is not wired into Cargo's test harness, and is not a quality evaluation on general language tasks.

## Project Layout

| Path | Responsibility |
| --- | --- |
| `src/main.rs` | Binary entry point; forwards process arguments to the CLI. |
| `src/cli.rs` | Argument parsing, home screen, training, chat, generation, evaluation, status, download, and benchmark orchestration. |
| `src/ui.rs` | Terminal presentation: logo, panels, spinners, progress bars, ANSI-aware width handling. |
| `src/dataset.rs` | Tokenization, vocabulary construction, built-in corpora, local and remote loading, streaming WikiText cleaning. |
| `src/pssa.rs` | PSSA layer, forward pass, plastic learning, consolidation, and `.pssa` serialization. |
| `src/checkpoint.rs` | Checkpoint format versions, resume payloads, and import/export validation. |
| `src/inference.rs` | Autoregressive sampling and generation constraints. |
| `src/backend.rs` | GEMM dispatch, CPU reference kernels, and the WebGPU device probe. |
| `src/memory.rs` | Fixed-capacity hyperbolic memory bank and retrieval/update logic. |
| `src/adapter.rs` | Low-rank modular adapter projections and updates. |
| `src/defense.rs` | Refractory rate-limiter primitives for stable updates and overwrite defense. |
| `src/linalg.rs` | Small allocation-conscious vector, matrix, math, and deterministic RNG utilities. |
| `src/diagnostics.rs` | CLI banner formatting. |
| `kaggle/` | Chained-training driver for long corpora on a hosted notebook. |
| `data/downloaded.txt` | Checked-in corpus used as the default when present. |
| `data/model.pssa` | Checked-in serialized model artifact. |

## Development

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets
cargo test --release
```

Integration tests live in `tests/`: `allocations.rs`, `bpe_repair.rs`, `checkpoint_repair.rs`, `core_repair.rs`, `linalg.rs`, and `runtime_repair.rs`, with shared artifacts under `tests/fixtures/`. They cover tokenizer round trips, checkpoint import/export across versions, linear-algebra kernels, allocation behavior, and CLI runtime output. Clippy is clean of errors; a number of style warnings in the numeric kernels are left in place deliberately, since rewriting indexed loops there would churn code the gradient tests pin down.

## Limitations

- CPU-oriented prototype with hand-written linear algebra. `gpu-probe` verifies a WebGPU device and a GEMM against the CPU reference, but training and inference still run the layer math on the CPU.
- The CLI parser is intentionally minimal: no shell-style quoting, and little validation beyond numeric parsing.
- A missing or unreadable dataset silently falls back to the built-in science corpus in several loading paths.
- Model and tokenizer vocabularies must remain compatible; a size warning does not repair a mismatch.
- Model shape cannot change across a resume chain: latent, state, key, memory and vocabulary must match the checkpoint being resumed.
- Downloaded content can be large and may contain JSON, malformed text, or data unsuitable for training.
- The REPL temperature command acknowledges a value without changing the active configuration.
- Benchmark output is milestone-oriented and does not measure perplexity, factuality, latency, or safety.
- Serialized `.pssa` files are project-specific binary artifacts without version migration tooling.

## License

See [LICENSE](LICENSE) for the project license.
