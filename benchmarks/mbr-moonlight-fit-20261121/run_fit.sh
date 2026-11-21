#!/usr/bin/env bash
# Ticket 02: which MBR windows fit Moonlight on a 4 GiB laptop card. Runs one window at a time.
#
# Inputs, all overridable from the environment:
#   VENV            virtualenv with torch 2.14 cu130, transformers 5.20.2 and nvidia-cuda-nvcc 13
#                   (default: $HOME/.venvs/soup-rnd-cuda)
#   MOONLIGHT_ROOT  folder written by p72_prepare_moonlight_cache.py, with source/ and cache/
#                   (default: $HOME/moonlight-local)
#   OUT             results folder (default: $MOONLIGHT_ROOT/results/fit)
#
# Refuses to run when $OUT/fit.jsonl already exists, so a rerun cannot double the rows.
# No `set -e`: a failed window is recorded in fit.err and the next window still runs, on purpose.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$MOONLIGHT_ROOT/results/fit}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

if [ -e "$OUT/fit.jsonl" ]; then
  echo "refusing to run: $OUT/fit.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"

run() {
  local arm=$1 k=$2 seq=$3
  echo "== arm=$arm k=$k seq=$seq start $(date +%T)"
  PYTHONPATH="$REPO/src:$REPO" "$VENV/bin/python" -u "$REPO/benchmarks/harness/mbr_measure.py" run \
    --arm "$arm" --k "$k" --seq "$seq" --kind speed \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" --out "$OUT" \
    --lora-targets "$TARGETS" \
    >> "$OUT/fit.jsonl" 2>> "$OUT/fit.err"
  echo "== arm=$arm k=$k seq=$seq exit=$? $(date +%T)"
}

run C 8 64
run A0 8 64
run C 8 512
run A0 8 512
run C 64 64
echo "all done $(date +%T)"
