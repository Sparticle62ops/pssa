# Setup

This guide covers installing PSSA, choosing a backend, preparing data and
caches, and training over a corpus that is too large for one session. For the
full command and option reference see [cli.md](cli.md); for the dashboard see
[tui.md](tui.md); for measured results and how they were produced see
[benchmarks.md](benchmarks.md).

## Requirements

- Rust 1.88 or newer (Edition 2024 and let-chain support), including Cargo.
- Network access only when using an HTTP/HTTPS dataset or a Hugging Face dataset.
- Enough memory and disk for larger corpora and serialized models.
- Optional: a CUDA device for the GPU training path. The CPU path is the
  reference and always available.

Direct runtime dependencies are [`ureq`](https://crates.io/crates/ureq) for
dataset downloads and [`tokenizers`](https://crates.io/crates/tokenizers) for
byte-level BPE. A GPU is optional. Built with `--features cuda` the dense
matrix work dispatches through cuBLAS with a device-resident weight cache; a
WebGPU adapter is used for the same stages when CUDA is unavailable, and
software adapters are refused because they are slower than the CPU path.
Everything falls back to the CPU implementation with no feature flags.

## Install and build

```bash
git clone https://github.com/Sparticle62ops/pssa.git
cd pssa
cargo install --path .                    # CPU build
cargo install --path . --features cuda    # NVIDIA GPU build (cuBLAS)
pssa
```

Running it with no arguments opens the dashboard. Everything the dashboard does
is also a plain CLI command, so it works the same over SSH, in a notebook
terminal, or in a script.

## Backends

`train --backend auto|cpu|webgpu|cuda` is additive. Omitting it (or using `auto`)
keeps the original automatic GPU selection and CPU fallback. `cpu` does not
probe GPUs. Explicit `webgpu` and `cuda` fail visibly rather than silently
falling back. CUDA requires a binary built with `--features cuda`; the TUI setup
wizard only offers CUDA in such a build. This is backend selection, not an
individual GPU/device picker. Existing CLI flags, checkpoint formats, and
training math are unchanged.

The training log's `backend=` line establishes which backend actually
initialized; a CUDA-feature build alone does not establish that CUDA is in use.
WebGPU training is not yet verified on hardware; CUDA and CPU are the tested
paths.

## Training from the TUI

Run `pssa tui` in a terminal, then press **Tab** to reach **setup**. The
command-line training and piped-log monitor remain available.

The wizard has four pages: **Dataset**, **Model**, **Training**, and
**Review / launch**. It shows the exact equivalent CLI command, so a configured
run can be copied into a script. Select **START TRAINING** and press Enter; the
app switches to the monitor on success.

The wizard creates `model.pssa` and `train.log` in the chosen directory. It
refuses to overwrite either file, so use a new directory for each run, including
resumed runs, and only one wizard-launched run may be active at a time. The
trainer is a separate process using `--no-tui`, with its own process group on
Unix. **Quitting the TUI leaves the child training.** To stop a detached run,
use normal OS process controls. To reopen its monitor from another terminal:

```sh
tail -f 'runs/my run/train.log' | pssa tui --chain 'runs/my run'
```

Numeric inputs, local-file readability, HF names and output paths are validated
before launch. When resuming, latent/state/depth and chunk must match the
checkpoint; vocabulary comes from that checkpoint. Errors from the child appear
in the monitor and remain in `train.log`. See
[training-setup.md](training-setup.md) for the full wizard reference.

## Datasets

`DatasetManager` accepts one or more comma-separated sources:

```bash
cargo run --release -- train science                       # built-in reference corpus
cargo run --release -- train data/downloaded.txt           # local text file
cargo run --release -- train data/                         # every readable file in a directory
cargo run --release -- train https://example.org/corpus.txt
cargo run --release -- train hf:owner/dataset              # Hugging Face repository
cargo run --release -- train science,data/downloaded.txt   # multiple sources
```

Local files and directories are read directly; HTTP(S) URLs and explicit
`hf:owner/dataset` sources are downloaded. Structured responses are reduced
using common fields such as `text`, `content`, `article`, `story`,
`instruction`, `output`, `sentence`, and `summary`; structured responses without
a supported text field are rejected.

Byte-level BPE keeps exact UTF-8 case, whitespace, punctuation, and line
endings, and has a complete 256-byte fallback alphabet, so valid UTF-8 never
collapses to `<unk>`. The previous lowercase word splitter, including its
10,000-word cap and `<unk>` behavior, is available only with
`--tokenizer word`.

Download a Hugging Face dataset into a local text file:

```bash
cargo run --release -- download wikimedia/wikipedia --out data/downloaded.txt
```

### Hugging Face training and cache

```bash
pssa train --hf-dataset Salesforce/wikitext \
  --hf-config wikitext-103-v1 --hf-split train --hf-field text \
  --max-tokens 200000 -e 1 --no-tui | pssa tui
```

HF downloads use the datasets-server rows API, streaming pages to an atomic disk
cache before the existing in-memory tokenizer/trainer reads the corpus. This is
not online/infinite-dataset training: `--max-tokens` caps training, not the
download. Set `PSSA_HF_CACHE` to a writable cache directory (for example
`/kaggle/working/pssa-hf-cache` on Kaggle, with Internet enabled); otherwise the
cache lives under `$XDG_CACHE_HOME/pssa/huggingface` or
`~/.cache/pssa/huggingface`. Cached data can be reused offline; delete its
cache file to refresh it. Choose a configuration explicitly when discovery
reports multiple choices.

Renamed environment variables still accept their legacy `OXIDE_*` fallback
when the corresponding `PSSA_*` variable is unset (including `OXIDE_PSSA_HF_CACHE`
for `PSSA_HF_CACHE`). A set `PSSA_*` value always takes precedence, even if empty.

The TUI **feed** tab shows the dataset, completed selected-window rows, trained
tokens, and a short decoded preview. The displayed IDs are actual input tokens
from the latest reported training chunk; with batching, the preview is the last
lane in that update. Row counts include repeated epochs/windows, and the
selected row number is after skip/cap selection, not a global Hugging Face row
ID. Older logs and local training still work without sample metadata. The
animation is cosmetic and runs only in the log reader. See [tui.md](tui.md).

HF runs add percent-encoded `feed_*` fields to the existing throttled progress
lines (at most every five seconds when piped, plus first/final updates). These
include a short sample of dataset content; treat saved logs accordingly. Local
runs do not add these fields. Do not combine `--hf-dataset` with a positional
source or `--data`.

### Private and gated datasets

For private/gated datasets, open **HF login** with Tab in `pssa tui`. Enter a
read token (masked) and press **Enter**: the app verifies the account with HF's
`whoami-v2` API and atomically saves `~/.cache/huggingface/token` with mode
`0600`. Existing `HF_TOKEN` credentials take precedence over that file on
startup; plain CLI downloads use the same credentials as a Bearer header. No
token is shown in logs or errors. **Esc** clears entry; **Ctrl+L** logs out and
removes the saved file. Explicit login/logout also applies to subsequently
wizard-launched trainers, without passing tokens in command arguments.
Already-running trainers keep their credentials. Also unset `HF_TOKEN` in your
parent shell to sign out future independently launched CLI runs. Login does not
grant gated access: request/accept access on the dataset's Hugging Face page and
wait for approval. Logout does not remove already downloaded dataset caches.

Network downloads are not validated or curated by PSSA. Review licensing,
privacy, and content before training on an external corpus.

### Cleaning WikiText raw corpora

Clean extracted `wikitext-103-raw` text **before a fresh training run**:

```bash
./target/release/pssa clean-wikitext wiki.train.raw --out data/wikitext-clean.txt
./target/release/pssa train data/wikitext-clean.txt -o data/model.pssa
# Also available: pssa help clean-wikitext
```

The same command can be used in Kaggle after extracting text from Parquet; it
accepts a local UTF-8 text file, not Parquet itself. `-o` and `--out` are
aliases. The output path is required and must not already exist (including the
input path or a link to it). This protects the original corpus; choose a new
output name for another run. Read, UTF-8, and write failures exit nonzero
through the normal CLI error path, with partial output removed when possible.

The pass:

- Joins `@-@`, `@.@`, and `@,@` to adjacent text: `guest @-@ starring` →
  `guest-starring`, `52 @.@ 9` → `52.9`, `500 @,@ 000` → `500,000`.
- Drops balanced heading lines such as `= Title =` and `= = Section = =`.
- Removes `<unk>` and collapses remaining inline whitespace to single spaces.
- Removes spaces before `.`, `,`, `)` and after `(`; trims each line.
- Retains at most one consecutive blank line, including at the start/end.
  Removing a heading does not introduce a blank line.
- Writes LF line endings, including a newline on the last retained line.

`pssa::dataset::clean_wikitext(reader, writer)` is the reusable library API
(`BufRead` / `Write`, returning `std::io::Result<()>`). The CLI uses buffered
file I/O, and the cleaner retains only its input/output line buffers: memory is
proportional to the longest line, not the corpus size. Library callers using a
buffered writer must flush it themselves; the CLI explicitly checks the flush.
No new dependencies are required.

Cleaning is opt-in: existing loaders, tokenizers, training commands, and
`kaggle/kaggle_continue.sh` are unchanged. **Do not switch an in-flight resume
chain to a cleaned corpus**: cleaning changes token IDs/counts and the meaning
of `--skip-tokens` offsets. Prepare and consistently reuse one cleaned corpus
for a new chain instead.

## Training over a long corpus

`--skip-tokens`, `--max-tokens` and `--resume` together let a long corpus be
trained as a chain of short runs, so a single run never has to survive a session
limit. If a window crosses EOF, selection wraps to the beginning of the corpus.
Each link trains its own window and hands its optimizer state to the next:

```bash
cargo run --release -- train data/downloaded.txt -e 1 \
  --skip-tokens 0      --max-tokens 200000 -o chain/ck01.pssa
cargo run --release -- train data/downloaded.txt -e 1 \
  --skip-tokens 200000 --max-tokens 200000 --resume chain/ck01.pssa -o chain/ck02.pssa
```

For opt-in divergence containment, add e.g. `--grad-clip 1.0 --memory-value-cap 512`.
Clipping runs after gradient accumulation, immediately before Adam on CPU and GPU;
progress reports the pre-clip `grad_norm`. A non-finite norm clears gradients and
skips the update without advancing Adam moments, step count, or the LR schedule;
the data group is not retried. Warnings and `skipped_updates` count these skips,
and the **21st consecutive skip** aborts without saving. The memory cap rescales
vectors (not individual coordinates); key norms and other bank metadata are
unchanged. Both flags default off, preserve the checkpoint format, and are
**runtime-only: repeat them on every resumed link**. The example cap is a policy
choice, not a guarantee of stability or a universal model-scale setting.

`kaggle/kaggle_continue.sh` drives this pattern end to end: it sets a window
size and a link count, walks the corpus offset by offset, and resumes each link
from the previous checkpoint. `status` then reports every checkpoint in the
chain with its shape and optimizer step count. The validated behavior behind
these flags is documented in [GPU-CLIP-SPEED.md](GPU-CLIP-SPEED.md).

## Checkpoint compatibility

New saves use **V7**: the full V6 training/resume payload plus a bounded,
length-prefixed standard tokenizer JSON. A V7 BPE checkpoint is self-contained
and restores its exact ordered vocabulary without access to the training or
evaluation corpus. `generate` and `chat` reject `--data` for V7 BPE because
retraining a tokenizer on external data would not validate provenance. V7 word
checkpoints and V6 checkpoints retain the legacy optional `--data`
exact-vocabulary comparison. Checked V5 artifacts remain inference-only and
require `--data` because they never contained tokenizer provenance.

## Inference

Generation is autoregressive and uses temperature 0.70, a top-24 candidate limit
followed by top-p 0.85 filtering, a 1.25 repetition penalty over a recent
64-token window, immediate self-transition suppression, `<unk>` suppression, and
a default cap of 64 new tokens, ending early after two generated periods.

V7 BPE inference restores the exact embedded tokenizer and never rebuilds it from
a selected dataset. Evaluation supplies its data only as held-out text to the
restored tokenizer.

## Next steps

- Full command and option reference: [cli.md](cli.md).
- Dashboard tour and run controls: [tui.md](tui.md).
- Training wizard details: [training-setup.md](training-setup.md).
- Opt-in training features and experiments: [experiments.md](experiments.md).
- Results and methodology: [benchmarks.md](benchmarks.md).
