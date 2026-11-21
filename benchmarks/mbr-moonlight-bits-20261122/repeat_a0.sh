#!/usr/bin/env bash
# Ticket 03 diagnostic: is an arm reproducible against itself? Runs ARM (default A0) at one k and
# seq N times,
# each into its own folder, so every run keeps its saved tensors for compare_repeats.py.
#
# Usage: [ARM=F] repeat_a0.sh SEQ K RUNS [math]
#   math: add --math-attention (SDPA math backend only, in deterministic mode)
#
# Inputs, overridable from the environment:
#   VENV, MOONLIGHT_ROOT, OUT  (same defaults as run_bits.sh)
#
# Refuses to run when a target folder already exists. No `set -e`: a failed run is recorded.
set -u

if [ $# -lt 3 ]; then
  echo "usage: $0 SEQ K RUNS [math]" >&2
  exit 2
fi
SEQ=$1
K=$2
RUNS=$3
ATTENTION=${4:-default}

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$MOONLIGHT_ROOT/results/bits}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
TAG=""
ARM=${ARM:-A0}
PREFIX=""
if [ "$ARM" != "A0" ]; then PREFIX="$ARM-"; fi
EXTRA=()
if [ "$ATTENTION" = "math" ]; then
  TAG="math-"
  EXTRA=(--math-attention)
fi

for i in $(seq 1 "$RUNS"); do
  if [ -e "$OUT/${PREFIX}${TAG}rep$i-s$SEQ" ]; then
    echo "refusing to run: $OUT/${PREFIX}${TAG}rep$i-s$SEQ exists. Move it away first." >&2
    exit 1
  fi
done

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

for i in $(seq 1 "$RUNS"); do
  dir="$OUT/${PREFIX}${TAG}rep$i-s$SEQ"
  "$VENV/bin/python" -u "$REPO/benchmarks/harness/mbr_measure.py" run --arm "$ARM" --k "$K" --seq "$SEQ" \
    --kind bits --deterministic "${EXTRA[@]}" --steps 3 \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --out "$dir" \
    >> "$dir.jsonl" 2>> "$dir.err"
  echo "run $i exit=$?"
done
echo "repeat done"
