#!/usr/bin/env bash
# Parameter-matched decoder-only transformer baseline, trained on the SAME corpus,
# the same 200k-token windows, the same tokenizer and the same update schedule as
# the PSSA chain, so the two per-token loss curves can be plotted against each other.
#
# Differences from kaggle_continue.sh on purpose:
#   * train-transformer instead of train (CPU only; no CUDA feature needed)
#   * chunk 64 * accumulate 8 = 512 tokens per optimizer update, matching the
#     PSSA chain's batch-size 8 * chunk 64 * accumulate 1
#   * --tokenizer-from an existing PSSA checkpoint so both models see IDENTICAL
#     token ids; without that the loss numbers are not comparable
#   * writes a loss CSV so the curve survives the notebook session
set -euo pipefail

REPO="${REPO:-https://github.com/Sparticle62ops/oxide-ai.git}"
WORK="${WORK:-/kaggle/working}"
BRANCH="${BRANCH:-main}"
TOTAL="${TOTAL:-64}"
WINDOW="${WINDOW:-200000}"
CORPUS_MB="${CORPUS_MB:-64}"
TOTAL_UPDATES="${TOTAL_UPDATES:-30000}"
CHUNK="${CHUNK:-64}"
ACC="${ACC:-8}"
CHAIN="${CHAIN:-$WORK/chain-trfm}"
LOSS_CSV="${LOSS_CSV:-$WORK/transformer_loss.csv}"
LOSS_EVERY="${LOSS_EVERY:-10000}"
CORPUS_URL="${CORPUS_URL:-https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet}"
BIG="${BIG:-$WORK/corpus/big.txt}"

source "$HOME/.cargo/env" 2>/dev/null || true
if ! command -v cargo >/dev/null 2>&1; then
  echo "### 0. Rust toolchain (clean container)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
fi
cargo --version

echo
echo "### 1. Update checkout to $BRANCH"
if [ -d "$WORK/oxide-ai/.git" ]; then
  cd "$WORK/oxide-ai"
  git fetch --quiet origin "$BRANCH"
  git checkout --quiet -B "$BRANCH" "origin/$BRANCH"
else
  cd "$WORK"
  git clone --quiet --branch "$BRANCH" "$REPO"
  cd oxide-ai
fi
git log --oneline -1

echo
echo "### 1b. Corpus (identical to the PSSA chain)"
NEED=$((CORPUS_MB * 1024 * 1024))
HAVE=0
[ -f "$BIG" ] && HAVE=$(wc -c < "$BIG")
if [ "$HAVE" -lt "$NEED" ]; then
  mkdir -p "$(dirname "$BIG")"
  echo "fetching ~${CORPUS_MB}MB of text"
  curl -sL -o "$WORK/corpus.parquet" "$CORPUS_URL"
  python3 - "$WORK/corpus.parquet" "$BIG" "$NEED" <<'PYCONV'
import sys
import pyarrow.parquet as pq

src, dst, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
written = 0
with open(dst, "w", encoding="utf-8") as out:
    for batch in pq.ParquetFile(src).iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not text or not text.strip():
                continue
            line = text.strip()
            if line.startswith("=") and line.endswith("="):
                continue
            out.write(line + "\n")
            written += len(line) + 1
        if written >= target:
            break
print("corpus_bytes=%d" % written)
PYCONV
  rm -f "$WORK/corpus.parquet"
fi
if [ -s "$BIG" ]; then
  DATA="$BIG"
else
  echo "WARNING: corpus fetch failed, falling back to the small checked-in file"
  DATA="data/downloaded.txt"
fi

echo
echo "### 2. Build (CPU: the transformer baseline has no GPU path)"
cargo build --release
HELP_TEXT="$(./target/release/oxide_ai_pssa help train-transformer 2>&1 || true)"
case "$HELP_TEXT" in
  *--tokenizer-from*) echo "--tokenizer-from present" ;;
  *) echo "ERROR: this checkout has no train-transformer --tokenizer-from, stopping"; exit 1 ;;
esac

echo
echo "### 2b. Clean the corpus (same cleaner, same output path)"
CLEAN="${CLEAN:-$WORK/corpus/big.clean.txt}"
if [ "$DATA" = "$BIG" ]; then
  if [ ! -s "$CLEAN" ]; then
    rm -f "$CLEAN"
    ./target/release/oxide_ai_pssa clean-wikitext "$BIG" -o "$CLEAN"
  fi
  if [ -s "$CLEAN" ]; then
    DATA="$CLEAN"
  else
    echo "WARNING: cleaning produced nothing, staying on the raw corpus"
  fi
fi
echo "data=$DATA bytes=$(wc -c < "$DATA")"

echo
echo "### 2c. Locate the PSSA tokenizer"
# The comparison is only honest if both models tokenize identically. Prefer the
# FIRST checkpoint of the PSSA chain: its BPE table is the one every later link
# inherited. Attach the pssa-model notebook output as an input to get it.
TOKENIZER_CK=""
TOK_MIN=999999
while IFS= read -r ck; do
  [ -n "$ck" ] || continue
  n="$(basename "$ck" .pssa)"
  n="${n#ck}"
  n="$((10#${n:-0}))"
  if [ "$n" -gt 0 ] && [ "$n" -lt "$TOK_MIN" ]; then
    TOK_MIN="$n"
    TOKENIZER_CK="$ck"
  fi
done <<EOF
$(find -L /kaggle/input "$WORK/chain" -maxdepth 8 -name 'ck*.pssa' 2>/dev/null)
EOF
if [ -n "$TOKENIZER_CK" ]; then
  echo "tokenizer imported from $TOKENIZER_CK"
else
  echo "WARNING: no PSSA checkpoint found under /kaggle/input."
  echo "The baseline will fit its own BPE table on the same corpus, which is close"
  echo "but NOT token-identical. Attach the pssa-model notebook output for an exact"
  echo "comparison (Add Input -> Notebook Output -> pssa-model)."
fi

echo
echo "### 3. Train the chain"
mkdir -p "$CHAIN"
PREV=""
START=1
for i in $(seq 1 "$TOTAL"); do
  CK="$CHAIN/ck$(printf '%02d' "$i").trfm"
  if [ -f "$CK" ]; then PREV="$CK"; START=$((i + 1)); fi
done
if [ -n "$PREV" ]; then
  echo "resuming from $PREV, next is ck$(printf '%02d' "$START")"
  if [ "$START" -gt "$TOTAL" ]; then
    echo "chain already complete through ck$(printf '%02d' "$TOTAL"), nothing to do"
    exit 0
  fi
fi

for i in $(seq "$START" "$TOTAL"); do
  OUT="$CHAIN/ck$(printf '%02d' "$i").trfm"
  SKIP=$(( (i - 1) * WINDOW ))
  echo "--- ck$(printf '%02d' "$i") (corpus offset $SKIP) ---"
  if [ -z "$PREV" ]; then
    EXTRA=""
    [ -n "$TOKENIZER_CK" ] && EXTRA="--tokenizer-from $TOKENIZER_CK"
    # shellcheck disable=SC2086
    ./target/release/oxide_ai_pssa train-transformer "$DATA" -o "$OUT" \
      --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1 \
      --chunk "$CHUNK" --accumulate "$ACC" --total-updates "$TOTAL_UPDATES" \
      --loss-csv "$LOSS_CSV" --loss-every "$LOSS_EVERY" --tokens-seen 0 $EXTRA
  else
    ./target/release/oxide_ai_pssa train-transformer "$DATA" -o "$OUT" \
      --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1 \
      --chunk "$CHUNK" --accumulate "$ACC" --resume "$PREV" \
      --loss-csv "$LOSS_CSV" --loss-every "$LOSS_EVERY"
  fi
  PREV="$OUT"
done

echo
echo "### 4. Sample"
./target/release/oxide_ai_pssa generate-transformer -m "$PREV" -p "The sun is"
./target/release/oxide_ai_pssa generate-transformer -m "$PREV" -p "Anarchism is"

echo
echo "### 5. Loss curve"
echo "csv=$LOSS_CSV"
tail -5 "$LOSS_CSV" 2>/dev/null || true
