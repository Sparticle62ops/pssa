# Paired multi-seed evaluation (opt-in)

Run from the repository root. Nothing runs during normal training; no new Rust
or Python dependencies. `scripts/paired_eval.py` launches a **sequential** pair
of command pipelines per seed, preserves explicit seed order (default
`[11,29,47]`), and alternates control-first/variant-first. It never invokes a shell.
The last command on each side must print **exactly one JSON card**; preceding
commands may print training logs. All stdout/stderr and expanded argv are saved.
The default per-command timeout is 240 seconds (maximum 300); on Linux timeout
kills the process group, and a failure does not prevent the other side/seeds
from running. Use only trusted local manifests/commands.

## Quick three-seed CPU calibration

Compile on the authorized machine, then run the supplied **A/A** smoke manifest:

```sh
cargo build --profile fast --bin pssa
PSSA_BIN="$CARGO_TARGET_DIR/fast/pssa" python3 scripts/paired_eval.py \
  experiments/paired-smoke.json --out /tmp/pssa-paired-smoke
```

For the shared codespace, from this directory:

```sh
/workspace/bin/csrun 'cargo build --profile fast --bin pssa && PSSA_BIN="$CARGO_TARGET_DIR/fast/pssa" python3 scripts/paired_eval.py experiments/paired-smoke.json --out /tmp/pssa-paired-smoke'
```

The manifest uses `experiments/paired_pssa.py` only to resolve `PSSA_BIN` and
exec the binary with an argv array. Training is width 16/state 4/key 4/bank 16,
word-tokenized, batch 1, chunk 16, one epoch, seed-matched, CPU/Rayon 1. The
small authored training/validation documents are disjoint. This is a plumbing
calibration, **not a language-quality benchmark**. CE/PPL deltas must be zero;
throughput need not match. Reserve a new output directory for each run.

## Manifest contract

`schema: 1`; `control` and `variant` each contain nonempty argv arrays (steps).
Each pipeline must use `{seed}` somewhere. Other literal substitutions are
`{side}` and `{run_dir}` (unique absolute directory for each side/seed). Paths
are relative to the **invocation cwd**, not the manifest. No shell expansion,
including `$VAR`, occurs. Optional `env` is a string mapping; Rayon defaults to
one thread. Do not place secrets in manifests: the explicit manifest is saved.
`inputs` lists immutable local files whose SHA256 is recorded before/after.
Record source/toolchain/CPU details alongside the card for serious runs.

Default metric JSON pointers:

- `/metrics/cross_entropy`: lower wins.
- `/metrics/perplexity`: lower wins.
- `/metrics/targets_per_second`: higher wins (scoring, not training/decode).

Override `metrics` with `{ "boxes_accuracy": { "path": "/boxes/accuracy",
"direction": "higher" } }` or any named numeric metrics from a worker. This
supports boxes cards without implementing a new boxes benchmark. Counts and
metric definitions must be in the worker's identity fields.

`match` defaults to `[/identity, /protocol, /model/parameters,
/model/optimizer_updates]` (JSON strings in the manifest). These must exactly
agree within each pair **and across seeds**; null/missing fields fail the pair.
Adapt this nonempty list explicitly for a different card schema (including
round one's two-model card). Do not remove token/split/exposure checks to make
a mismatched experiment pass. If architectures intentionally have different
parameter counts, report actual counts and put a checked matching tolerance in
the worker's identity rather than claiming exact equality.

The runner records command failure counts on both sides, failed pairs, and
per-metric complete-pair counts/failures. Invalid JSON/NaN/Infinity, timeout,
missing metrics and null/overflow metrics never become zero scores. Metrics
with different failure patterns retain only **their own complete seed pairs**.
Reports persist on partial/all failure; exit status is 1. Setup errors exit 2.
Existing output paths (including symlinks) are never overwritten.

## Minimal frozen card port

```sh
pssa eval-card validation.txt --model trained.pssa --skip-tokens 0 --max-tokens 128
```

This round ports **only single-PSSA** scoring from `swarm-evals`, not the
round-one PSSA/transformer CLI. `src/eval_harness.rs::score_model` accepts an
already configured model, tokenizer, raw data and explicit slice; prototype
workers can call it directly. `checkpoint_card`/the CLI load default runtime
settings, with only `--loops` exposed. Do **not** use default checkpoint scoring
for a runtime-only architecture/memory experiment.

Cards record corpus, canonical tokenizer and exact document/token-stream FNV
fingerprints; encoded/target counts; parameters, update count and occupancy;
one warmup and three frozen measured scoring passes (median). I/O/tokenization
are outside timing; teacher-forced scoring/carry resets are inside. Every pass
must have identical scores. `evaluation::evaluate_pssa_documents` restores all
depth/shared-loop carries even on errors (ported round-one fix). FNV detects
accidental drift, not malicious changes. `heldout_verified` remains false:
checkpoints lack training provenance; supply disjoint train/validation files
and fit the tokenizer only on training data.

## Comparing round-one prototypes

Do not merge prototype code merely to use the generic runner. Build a worker
on each isolated prototype branch and replace the manifest argv pipelines.
Preserve train data, tokenizer, target exposure, optimizer schedule, shape,
seed, evaluation window and thread settings.

- `swarm-arch` (`87d3b550`): call
  `PSSALayerV2::set_ssm_output_gate(true)` before training **and scoring**.
  This branch has no output-gate CLI flag. Use a worker calling `score_model`
  on the still-configured model; a checkpoint reload silently disables it.
- `swarm-memory` (`7a3f6e6b`): `--memory-adaptive-bandwidth` on training;
  call `set_memory_adaptive_bandwidth(true)` before card scoring. Occupancy
  must exceed one for a nonvacuous test; record it. The prototype's ordinary
  `score` supports its flag but does not emit this card schema.
- `swarm-stability` (`c38bc4df`): append `--agc 0.01` to training. No scoring
  flag is needed because AGC affects updates, not the forward function.

The public card API allows all three without introducing prototype dependencies
into this harness branch. No prototype has been selected as a winner here.

## Statistics and limits

Every delta is **variant minus control**. Means/medians and side means use
complete pairs. The 95% percentile interval resamples whole seed-pair deltas
with replacement, using specified SplitMix64, 10,000 draws and bootstrap seed
20260501 by default; quantiles interpolate linearly. `bootstrap_resamples` and
`bootstrap_seed` are recorded/overridable. Fewer than two complete pairs yield
no interval. Only scoring metrics/ordering/statistics can be byte-stable;
wall-clock/throughput values are inherently variable.

`interval_signal` merely says whether zero is inside the interval. **Three
seeds are smoke evidence, not a powered significance test**: percentile
bootstrap coverage is poor with tiny n; repeated metrics/variants are multiple
comparisons; survivor bias can make failed training look good. Inspect each
seed, all failures and a practical effect-size threshold. Shared CPU contention
especially undermines throughput claims. Never convert an excluded-zero
three-seed interval into a general language-model superiority claim.

Checks:

```sh
python3 -m unittest discover -s scripts -p 'test_paired_eval.py' -v
/workspace/bin/csrun 'cargo test --profile fast --lib'
```
