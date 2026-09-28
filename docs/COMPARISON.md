# Replaying the Kaggle chain against the transformer

`compare` trains **only the transformer**, then scores it and the finished PSSA
checkpoint on the same unseen slice. Matching is by supervised target tokens and
optimizer updates, not time. No new dependencies or checkpoint format changes.

## One command after ck64 finishes

Keep the original corpus **unchanged** and retain all `ck01.pssa` … `ck64.pssa`.
For the checked-in Kaggle defaults (64 links, 200,000 encoded tokens/link,
epochs=1, batch-size=8, accumulate=1):

```sh
export CC=/workspace/bin/zigcc CXX=/workspace/bin/zigcc AR=/workspace/bin/ar
export CARGO_TARGET_DIR=/workspace/oxide-target-test
cargo build --release
cargo test
BIN="$CARGO_TARGET_DIR/release/oxide_ai_pssa"

"$BIN" compare /kaggle/working/corpus/big.clean.txt \
  --chain-dir /kaggle/working/chain \
  --out /kaggle/working/comparison \
  --links 64 --window 200000 --batch-size 8 --accumulate 1 \
  --eval-skip-tokens 12800000 --eval-tokens 200000 \
  --loss-every 10000
```

On Kaggle use its normal Rust toolchain/build and set
`BIN=./target/release/oxide_ai_pssa` instead of the workspace-specific Zig paths.
The output directory must not already exist; its parent must exist. This avoids
overwriting a previous experiment. Defaults match the flags above; evaluation's
default offset is the sum of the links' encoded-token budgets. Seed defaults to
42. The transformer is a CPU baseline, so this may take substantially longer
than GPU PSSA training.

Outputs:

- `manifest.json`: corpus FNV-1a drift fingerprint, tokenizer checkpoint, actual
  parameter counts, per-link window/lane plan, cumulative targets/updates, LR
  endpoints, horizon/warmup, and evaluation slice.
- `ck01.trfm` … `ck64.trfm`: independently trained transformer checkpoints,
  saved after each corresponding link, with complete optimizer/tokenizer state.
- `transformer.csv`: training curve using the schema below.
- `results.json`: PSSA and transformer held-out cross entropy/perplexity,
  accuracy/OOV/target count, training exposure, updates and parameter counts.
  The same final metrics are printed to stdout.

The command imports the **exact tokenizer** from `ck01.pssa`, reads chunk size
and per-link base LR/horizon/warmup from the PSSA checkpoints, and replays each
window independently. Both models use `sequence_plan`: document lanes, short
chunks, accumulation groups, gradient weighting, and final partial updates are
identical. Transformer lanes are executed serially on CPU before one shared
update; this matches optimizer grouping, not PSSA hardware parallelism.

**Preflight happens before training/output creation.** Every PSSA checkpoint's
cumulative update counter must agree with the replayed plan; tokenizer/shape and
schedule continuity are checked across links. Evaluation must be within the
corpus and disjoint from every training window, including EOF wrapping. A chain
that saw the entire corpus has no held-out slice in that corpus: use a genuinely
separate validation corpus with the standalone evaluation commands instead.
No silent truncated budgets, approximate `chunk * batch * accumulation` counts,
wall-clock stopping, or PSSA retraining.

## Historical chains and limits of checkpoint provenance

Older PSSA checkpoints do **not** store corpus hashes, offsets, batching,
accumulation, epochs, seeds or target-token counts. Therefore you must supply the
actual corpus and settings used. The recorded update counts detect many mistakes
but **cannot prove corpus identity or distinguish plans with equal counts**.
Keep the Kaggle logs, exact corpus and command history. The manifest's FNV hash
is an accidental-drift audit, not a cryptographic provenance guarantee.

- If the whole chain used the historical single-lane configuration, pass
  `--batch-size 1 --accumulate 8` instead of the current defaults.
- For a mixed historical lane/window plan, pass `--link-plan plan.json` with
  exactly one entry per link. This overrides `--window`, `--batch-size` and
  `--accumulate`; `--links` still specifies the checkpoint count. For example:

  ```json
  [
    {"skip_tokens": 0, "max_tokens": 200000, "batch_size": 1, "accumulate": 8},
    {"skip_tokens": 200000, "max_tokens": 200000, "batch_size": 8, "accumulate": 1}
  ]
  ```

  Use `--links 2` for that example, or provide all 64 entries for the real chain.
  Each link is one epoch, as in `kaggle_continue.sh`.
- Horizonless checkpoints still load and replay their legacy **per-link**
  schedules; the command does not replace these with a new whole-chain horizon.
  Fixed horizons and persisted warmup are restored. For a historical fresh link
  whose warmup was not recorded, pass its original `--warmup-steps N` (default
  0); missing warmup metadata on later links retains no-rewarmup semantics.
- If the corpus was cleaned/replaced midway through the chain, there is no
  identical single-corpus replay. Do not pass today's cleaned file and claim it
  reproduces old raw-corpus exposure. Recover the original run history or start
  a new controlled comparison.
- The runner requires a fresh output directory and does not automatically resume
  an interrupted comparison. Completed transformer link checkpoints remain
  usable with `train-transformer --resume`, the next link's manifest settings,
  and the same CSV. Archive/truncate any CSV tail beyond the last checkpoint
  before resuming. Do not restart against existing output and accidentally mix
  two runs.

## Logging PSSA and transformer curves

Both `train` and `train-transformer` accept:

```sh
--loss-csv /path/to/model.csv --loss-every 10000
```

Add these flags to **each** training link, always reusing that model's own CSV.
For example, the PSSA commands corresponding to the current Kaggle defaults are:

```sh
"$BIN" train "$DATA" -o "$CHAIN/ck01.pssa" -e 1 \
  --max-tokens 200000 --skip-tokens 0 --batch-size 8 --accumulate 1 \
  --total-updates 30000 --loss-csv "$CHAIN/pssa.csv" --loss-every 10000
"$BIN" train "$DATA" -o "$CHAIN/ck02.pssa" -e 1 \
  --resume "$CHAIN/ck01.pssa" --max-tokens 200000 --skip-tokens 200000 \
  --batch-size 8 --accumulate 1 --total-updates 30000 \
  --loss-csv "$CHAIN/pssa.csv" --loss-every 10000
```

Continue with `skip=(link-1)*200000`. `compare` produces the transformer's CSV
automatically. To train it manually with identical lane grouping, the existing
`train-transformer` command now accepts `--batch-size` too (default remains 1).

```csv
tokens_seen,updates,loss,tokens_per_second
```

| Column | Meaning |
| --- | --- |
| `tokens_seen` | Global count of **scored next-token targets** processed, including repeated exposure; not the encoded input-window budget. No targets cross document/window boundaries. |
| `updates` | Global completed optimizer updates, including previous links. |
| `loss` | Target-weighted mean **pre-update** cross entropy in nats over the interval since the previous row in this invocation. |
| `tokens_per_second` | Targets in that interval / elapsed training seconds. Loading/tokenization, checkpoint saving and evaluation are excluded; reporting overhead is included. Hardware/backend must be reported separately. |

Rows are appended/flushed at the first completed update crossing each global N
threshold, plus the final partial interval at each invocation/link end. Updates
are never split; one large update produces one row, not fabricated intermediate
samples. Loss is an interval mean, not an epoch running mean.

Reusing a CSV restores its token offset and verifies that its last update equals
the resumed checkpoint's counter. Starting a **new** CSV from a checkpoint needs
`--tokens-seen N`, the actual previous target count (sum the training logs' epoch
`tokens=` or use a verified manifest), because old checkpoints cannot provide it.
Corrupt/stale files fail rather than silently append. Use one writer per file.

A completed chain trained without logging has **no recoverable historical
per-token curve**. The harness still compares its final held-out metrics without
retraining it. You can log future resumed training, but must not relabel that as
the missing historical curve.

## Evaluate existing checkpoints without training

```sh
"$BIN" evaluate "$DATA" -m "$CHAIN/ck64.pssa" \
  --skip-tokens 12800000 --max-tokens 200000
"$BIN" evaluate-transformer "$DATA" -m /kaggle/working/comparison/ck64.trfm \
  --skip-tokens 12800000 --max-tokens 200000
```

Both print the same JSON fields: `cross_entropy` (mean loss in nats),
`perplexity` (`exp(cross_entropy)`), `perplexity_overflow`, `oov_rate`,
`token_count` (scored targets), and `next_token_accuracy`. The standalone commands
validate slice bounds and never wrap, but cannot verify training overlap because
a checkpoint lacks training-window provenance. Omit slice flags to evaluate an
entire separate held-out file. Evaluation never updates or saves parameters,
optimizer state, or the trained memory bank.

PSSA retains its trained episodic memory and within-document recurrent carry;
the transformer remains chunk-local. The default models match parameter counts
closely at vocabulary 2048, not compute, available context or hardware. Inspect
actual counts in the manifest, especially for non-default PSSA shapes. Keep
unselected fixed-prompt generations alongside loss/perplexity: these numbers
alone do not establish coherent language or a statistically replicated win.
