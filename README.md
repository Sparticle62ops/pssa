# Oxide AI

Oxide AI is an experimental Rust implementation of a **Plastic State-Space Architecture (PSSA)** for continual language-model training and text generation. It combines a recurrent state-space layer with a hyperbolic episodic memory bank, modular low-rank adapters, plastic updates, and a closed-form ridge-regression consolidation step.

The repository is a research prototype rather than a production language model. It runs on the CPU, has no external ML framework dependency, and ships a small reference corpus for bootstrapping.

## Features

- Recurrent sequence processing with learned continuous state-space matrices.
- Token embeddings, unembeddings, normalization, SiLU feed-forward expansion, and autoregressive sampling.
- Hyperbolic/Poincare-inspired memory retrieval with bounded top-4 search.
- Plastic states for stable reinforcement, novelty-driven growth, refractory overwrite protection, and router tuning.
- Modular low-rank adapters for targeted updates.
- Refractory rate limiting intended to reduce damage from repeated contradictory updates.
- Ridge-regression consolidation from fast plastic updates into the base transition matrix.
- Binary `.pssa` model export/import, with resume-from-checkpoint training.
- Dataset loading from local text, directories, URLs, Hugging Face datasets, and the built-in science corpus.
- A verification benchmark covering contradiction adaptation, distractor-gap recall, consolidation, spam mitigation, serialization, and generation.

## Status

Under active development, covered by Cargo integration tests, and still an experimental local research tool. Validate the current branch with `cargo test --release` before relying on a checkpoint.

## Requirements

- Rust toolchain with Edition 2024 support, including Cargo.
- Network access only when using an HTTP/HTTPS dataset or a Hugging Face dataset.
- Enough memory and disk for larger corpora and serialized models.

Direct runtime dependencies include [`ureq`](https://crates.io/crates/ureq) for dataset downloads and the [`tokenizers`](https://crates.io/crates/tokenizers) crate for standard byte-level BPE. No GPU runtime is required; a WebGPU device is only probed by `gpu-probe` and is not yet used for the layer math.

## Quick Start

```bash
git clone https://github.com/Sparticle62ops/oxide-ai.git
cd oxide-ai
cargo build --release
```

The executable is written to `target/release/oxide_ai_pssa`. Run it with no arguments for the home screen, which lists the commands and every checkpoint and corpus it finds in the working directory:

```bash
./target/release/oxide_ai_pssa
```

Train a model on the bundled corpus. The default tokenizer is unnormalized ByteLevel BPE, trained only on the supplied corpus, with a maximum vocabulary of 2,048 entries:

```bash
cargo run --release -- train data/downloaded.txt --epochs 4 --out data/model.pssa
```

Use the legacy word tokenizer only for compatibility experiments:

```bash
cargo run --release -- train data/downloaded.txt --tokenizer word --epochs 4 --out data/word-model.pssa
```

Generate one completion, or start the REPL:

```bash
cargo run --release -- generate "quantum mechanics" --model data/model.pssa
cargo run --release -- chat data/downloaded.txt --model data/model.pssa --temp 0.70
```

If `data/model.pssa` does not exist, `chat` and `generate` train a four-epoch model first. Training time depends heavily on corpus size and CPU speed.

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
| `benchmark` | End-to-end smoke test on the built-in corpus. |
| `gpu-probe` | Check whether a WebGPU compute device is usable. |
| `help` | Print command and option help. |

Options:

| Option | Default | Applies to | Description |
| --- | --- | --- | --- |
| `-d, --data <source>` | `data/downloaded.txt` when present, otherwise `science` | `train`, `chat`, `evaluate` | Dataset source, or a comma-separated list. |
| `-m, --model <path>` | `data/model.pssa` | `chat`, `generate`, `evaluate` | Checkpoint to load. |
| `-o, --out <path>` | `data/model.pssa` | `train`, `download` | Output checkpoint or dataset path. |
| `-p, --prompt <text>` | empty | `generate` | Prompt text. Required for generation. |
| `-e, --epochs <n>` | `4` | `train` | Training epochs. |
| `-t, --temp <float>` | `0.70` | `chat`, `generate` | Sampling temperature. |
| `--max-new-tokens <n>` | `64` | `generate` | Generation length cap. |
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

`--skip-tokens`, `--max-tokens` and `--resume` together let a long corpus be trained as a chain of short runs, so a single run never has to survive a session limit. Each link trains its own window and hands its optimizer state to the next:

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

The loader first tries local paths. A missing non-special source is treated as a Hugging Face repository name. Remote loading probes the datasets server and several conventional raw-file names. JSON-like responses are reduced using common fields such as `text`, `content`, `article`, `story`, `instruction`, `output`, `sentence`, and `summary`.

Byte-level BPE keeps exact UTF-8 case, whitespace, punctuation, and line endings, and has a complete 256-byte fallback alphabet, so valid UTF-8 never collapses to `<unk>`. The previous lowercase word splitter, including its 10,000-word cap and `<unk>` behavior, is available only with `--tokenizer word`.

Download a Hugging Face dataset into a local text file:

```bash
cargo run --release -- download wikimedia/wikipedia --out data/downloaded.txt
```

Network downloads are not validated or curated by Oxide AI. Review licensing, privacy, and content before training on an external corpus.

## Training Pipeline

The `train` command performs two phases:

1. **Continuous recurrent ingestion:** token transitions are processed through the PSSA layer. The model updates state, memory, adapters, and routing behavior with a cosine learning-rate schedule.
2. **Ridge consolidation:** online transition statistics are accumulated and consolidated into the base matrix using closed-form ridge regression.

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
| `src/dataset.rs` | Tokenization, vocabulary construction, built-in corpora, local and remote loading. |
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
