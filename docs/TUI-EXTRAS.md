# Phase 11: the project dashboard

Start `pssa tui --chain "path to chain"` in a terminal. `Tab` cycles
screens; `Ctrl+K` opens a searchable command palette. `?` opens help outside text
entry; `F1` opens help everywhere (question marks still work in prompts/paths).
`Ctrl+C` quits everywhere. Existing piped / `--no-tui` CLI behavior is unchanged.
Screens retain the CRT green style and compact/narrow-terminal layouts.

## Kaggle: launch and monitor

Install no extra Rust dependencies. Supply **either** `KAGGLE_USERNAME` and
`KAGGLE_KEY` **or** the existing `~/.kaggle/kaggle.json`. Never put credentials in a
notebook, log, demo, screenshot or command argument. The tab does not display keys.

In the Kaggle tab, enter a local notebook folder with `kernel-metadata.json` and
the referenced `.ipynb` or `.py` source. Press Enter for a launch preview and `y`
to confirm pushing/launching a version. It can consume Kaggle quota. Enable the
notebook's Internet/GPU settings as needed in its metadata. Use ordinary training
commands with `--no-tui` inside the notebook so its stdout contains progress fields.
The dashboard sends remote progress through the same monitor and graph parser as
local progress. API access and log freshness are subject to Kaggle availability.

`Esc` in the busy Kaggle tab **detaches monitoring only**; it does not cancel the
remote run. Quitting the TUI also does not cancel it. Stop the notebook on Kaggle
to stop quota use. Credentials and service calls are not needed by offline tests.

## Plastic memory inspector

The memory tab renders occupancy and a Poincare occupancy disk using existing
progress fields. While visible, it reads a changed local `.pssa` checkpoint in
a background worker at most once per five seconds (64 MiB checkpoint size cap
for automatic background loading). `r` refreshes manually;
`PgUp`/`PgDn` scroll slot metadata. It never modifies the checkpoint or model.

The setup wizard also exposes the six optional dreaming fields: cadence
(`--dream-every`, `0` means off), replay entries (`--dream-replay`), mode
(`--dream-mode memory|generate|both`), generated length (`--dream-len`),
rehearsal rate (`--dream-lr`), and rehearsal passes (`--dream-steps`). Their
names, defaults, and validation match the CLI. The monitor parses the trainer's
`dream phase=start` and `dream phase=end` events and shows the active phase,
completed count, mode, and latest rehearsal loss without mixing it into the
training-loss graph.

**Telemetry limits are explicit:** existing checkpoints store vectors and
last-seen steps/confidence, not the source token text. A feed snippet is only feed
context, never claimed as a confirmed write. Exact eviction rates and write
snippets appear only if the producer provides `memory_evictions=N` and
`memory_write_snippet=PERCENT_ENCODED_TEXT`. Changed slot fingerprints between
snapshots are a lower bound on observed overwrites, not an exact eviction count.
No trainer/model-math instrumentation was added. Remote checkpoint paths are not
local files; download one separately to inspect its slot metadata.

## A/B inference

In inference, use `/ab a PATH` and `/ab b PATH`, then enter a prompt. Both PSSA
`.pssa` and transformer `.trfm` checkpoints are accepted; paths may contain spaces.
`/ab on` enables comparison and `/ab off` returns to the untouched single-chat
history. Each panel reports its own generation rate. The same current prompt,
system text and attachments go to both models; previous single-chat history is
not included. A/B replies are temporary, not saved as a conversation.

A and B run **sequentially** so only one checkpoint is loaded at a time on small
machines. Rates exclude model loading/queue time and include prefill. Esc stops;
model loading must finish before cancellation can take effect. This is not a
claim of identical model context windows or equivalent hardware throughput.

## Runs browser

The runs tab scans `--chain`, `data/`, `runs/`, and `comparison/`, to depth two,
with bounded discovery (1,000 displayed files / 10,000 visited entries). Use:

- `↑`/`↓`: select a checkpoint or `.log`; selected checkpoint dimensions, learning
  rate, depth and optimizer steps are loaded lazily in the background, with a
  64 MiB automatic-loading cap. Larger checkpoints remain selectable for explicit
  chat/scoring jobs.
- `Enter`: open the sibling `.log` in the monitor (8 MiB display limit). A
  checkpoint with no saved log is explicitly shown without invented history.
- `c`: open PSSA in chat; a transformer is selected in the A/B inference screen.
- `s`: enter a **genuinely held-out local file**, Enter to score in a cancellable
  child process. Result metrics are saved in a format-specific sibling sidecar
  (for example `model.pssa.score.json` or `model.trfm.score.json`),
  never in the model file. `Esc` stops scoring; `Ctrl+U` clears the path.
- `r`: refresh discovery. Dates are checkpoint/log modification dates in UTC,
  not an inferred training start time.

Wizard output directories are added to discovery automatically; opening their
`model.pssa` restores the corresponding `train.log` history. Active training
cannot be replaced by recorded history or a second Kaggle launch.

The list's perplexity is saved **held-out** perplexity from `.score.json` (legacy
extension-replacing sidecars remain readable); absent
scores show `unscored`. It is never inferred from checkpoint weights or relabelled
training loss. Historical checkpoints cannot prove corpus overlap or all original
training hyperparameters; retain command lines, logs and corpus provenance.

## One-key matched benchmark

Open the benchmark tab or choose it in the palette. Enter the original local
corpus, chain directory, **new** output directory and original link/window/batch/
accumulation settings (`↑`/`↓`, Enter to edit, Ctrl+U to clear). Then press **b**.
After configuration, the palette's **Run matched benchmark** is another one-action
launch. `Esc` cancels the local child, retaining partial files for inspection;
choose a fresh output directory for another run.

This calls the existing `compare` harness, now exposed as a CLI command. It
**does not retrain PSSA**: it replays the PSSA chain's supervised targets, optimizer
updates, tokenizer and schedule in the transformer, then evaluates both. The UI
uses seed 42 and the next 256 unseen encoded tokens after the training budget.
The harness rejects training/evaluation overlap and inconsistent update clocks
before training. It writes `manifest.json`, checkpoints, curves and `results.json`;
the TUI renders the latter as a result card. Parameter counts may differ; it matches
exposure, not wall time, compute or architecture. Larger/custom held-out slices
and mixed link plans remain available through the CLI; see [COMPARISON.md](COMPARISON.md).
This is distinct from the historical `benchmark` smoke-test CLI, which is unchanged.

## Alerts

Completion/error transitions ring the terminal bell and show a persistent status
line. Terminal settings determine whether the bell is audible or visual. An input
stream ending without a completion summary is not labelled a successful run.
Child scoring, comparison, and Kaggle lifecycle events also produce alerts.

## Record a GIF

Install [VHS](https://github.com/charmbracelet/vhs) and its runtime dependencies
(`ttyd`, `ffmpeg`) separately; they are not Rust/default-build dependencies.
Build the binary, put it on `PATH`, and run from the repository root:

```sh
cargo build --release
export PATH="${CARGO_TARGET_DIR:-target}/release:$PATH"
vhs docs/tui-demo.tape
```

The tape records idle animation, palette, memory empty state and help without
launching training, uploads, or credential-entry screens. Existing HF credentials
can still trigger the normal background identity check on startup; unset
`HF_TOKEN` and use an empty temporary `HOME` for a strictly offline capture. It writes
`docs/img/tui-demo.gif` (do not commit a large generated file blindly). Resize the
tape's terminal settings to record other layouts. For real training recordings,
use a separate demo environment with sanitized logs and no private filenames,
credential fields or personal conversations. The tape intentionally does not open
login or launch Kaggle/benchmark jobs.
