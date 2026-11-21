#!/usr/bin/env bash
# Ticket 03: the S=512 ladder again, with the SDPA math attention backend in every run.
# The default backend gave A0 different gradients from itself at this window (see
# bits-summary.md), so this ladder is the one to read for S=512. Same arms, k and model as
# run_bits.sh; only the attention backend and the output folder differ.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default
# $MOONLIGHT_ROOT/results/bits/s512-math). Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$MOONLIGHT_ROOT/results/bits/s512-math}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"

if [ -e "$OUT/rows.jsonl" ]; then
  echo "refusing to run: $OUT/rows.jsonl exists. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

echo "== lever-bits seq=512 math attention start $(date +%T)"
"$VENV/bin/python" -u "$REPO/benchmarks/harness/mbr_measure.py" lever-bits --seq 512 --steps 3 \
  --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
  --lora-targets "$TARGETS" --math-attention --out "$OUT" \
  > "$OUT/sweep.log" 2>&1
echo "== lever-bits seq=512 math attention exit=$? $(date +%T)"
echo "all done $(date +%T)"
