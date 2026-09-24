#!/usr/bin/env bash
# Update an existing Kaggle checkout to main and continue the chain from the
# newest checkpoint already in /kaggle/working/chain, instead of retraining
# from scratch. Safe to run after kaggle_gpu_setup.sh died partway.
set -euo pipefail

REPO="${REPO:-https://github.com/Sparticle62ops/oxide-ai.git}"
WORK="${WORK:-/kaggle/working}"
BRANCH="${BRANCH:-main}"
TOTAL="${TOTAL:-32}"
# Each link reads a different WINDOW-sized slice instead of the same prefix, so the
# chain walks the whole corpus. The offset wraps around at the end of the file.
WINDOW="${WINDOW:-200000}"
# The checked-in corpus is only ~3 MB, so a 16-link chain wraps and sees the same
# text three times over, which is what drift looks like. Pull a much larger corpus
# once into the working directory so every link reads text the model has not seen.
CORPUS_MB="${CORPUS_MB:-48}"
CORPUS_URL="${CORPUS_URL:-https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet}"
BIG="${BIG:-$WORK/corpus/big.txt}"

source "$HOME/.cargo/env" 2>/dev/null || true
if ! command -v cargo >/dev/null 2>&1; then
  echo "### 0. Rust toolchain (clean container)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
fi
cargo --version

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
echo "### 1b. Corpus"
# Kept outside the git checkout so `git checkout -B` never has to clobber it, and
# so it survives between notebook runs.
NEED=$((CORPUS_MB * 1024 * 1024))
HAVE=0
[ -f "$BIG" ] && HAVE=$(wc -c < "$BIG")
if [ "$HAVE" -lt "$NEED" ]; then
  mkdir -p "$(dirname "$BIG")"
  echo "fetching ~${CORPUS_MB}MB of fresh text"
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
echo "data=$DATA bytes=$(wc -c < "$DATA")"

echo
echo "### 2. Build"
cargo build --release
HELP_TEXT="$(./target/release/oxide_ai_pssa help 2>&1 || true)"
case "$HELP_TEXT" in
  *--resume*) echo "--resume present" ;;
  *) echo "ERROR: this checkout has no --resume, stopping"; exit 1 ;;
esac

echo
echo "### 3. Continue the chain"
mkdir -p "$WORK/chain"
# A new Kaggle session starts with an empty /kaggle/working, so checkpoints from a
# previous saved version arrive under /kaggle/input instead. Copy them back in so
# the chain resumes instead of silently restarting at ck01.
if ! ls "$WORK/chain"/ck*.pssa >/dev/null 2>&1; then
  SEED="$(find /kaggle/input -maxdepth 8 -name 'ck*.pssa' 2>/dev/null | sort | tail -1)"
  if [ -n "$SEED" ]; then
    echo "seeding chain from $(dirname "$SEED")"
    cp "$(dirname "$SEED")"/ck*.pssa "$WORK/chain"/
    ls -1 "$WORK/chain" | tail -3
  else
    echo "no checkpoints found under /kaggle/input. what is mounted:"
    ls -R /kaggle/input 2>/dev/null | head -40
    echo "starting a fresh chain"
  fi
fi
PREV=""
START=1
for i in $(seq 1 "$TOTAL"); do
  CK="$WORK/chain/ck$(printf '%02d' "$i").pssa"
  if [ -f "$CK" ]; then PREV="$CK"; START=$((i + 1)); fi
done
if [ -n "$PREV" ]; then
  echo "resuming from $PREV, next is ck$(printf '%02d' "$START")"
  if [ "$START" -gt "$TOTAL" ]; then
    echo
    echo "the chain is already complete through ck$(printf '%02d' "$TOTAL") (TOTAL=$TOTAL),"
    echo "so there is nothing to train. to keep going, rerun with a bigger cap, e.g.:"
    echo "  TOTAL=$((TOTAL + 32)) bash kaggle/kaggle_continue.sh"
    exit 0
  fi
else
  echo "no existing checkpoints, starting fresh"
fi

for i in $(seq "$START" "$TOTAL"); do
  OUT="$WORK/chain/ck$(printf '%02d' "$i").pssa"
  SKIP=$(( (i - 1) * WINDOW ))
  echo "--- ck$(printf '%02d' "$i") (corpus offset $SKIP) ---"
  if [ -z "$PREV" ]; then
    ./target/release/oxide_ai_pssa train "$DATA" -o "$OUT" --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1
  else
    ./target/release/oxide_ai_pssa train "$DATA" -o "$OUT" --max-tokens "$WINDOW" --skip-tokens "$SKIP" -e 1 --resume "$PREV"
  fi
  PREV="$OUT"
done

echo
echo "### 4. Sample"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "The sun is"
./target/release/oxide_ai_pssa generate -m "$PREV" -p "Anarchism is"
