#!/usr/bin/env bash
# Ticket 04: the bits ladder at k=64 on Moonlight. A0 does not run above k=8, so the ladder takes
# C as the reference (mbr_measure.reference_arm). Only C and E run here, on the two k=64 windows.
#
# Window (64,64) fits in 4 GiB (ticket 02: C 2.03 GiB NVML). Window (64,512) probably does not:
# an OOM is written as a row with its peaks and is a result, so the run is not forced or retried.
# Deterministic mode. The default SDPA backend, like the S=64 ladder of ticket 03; the S=512
# window of ticket 04 only matters if it fits, and then the backend is stated in the summary.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default
# $MOONLIGHT_ROOT/results/bits). Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$MOONLIGHT_ROOT/results/bits}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"

for seq in 64 512; do
  if [ -e "$OUT/k64-s$seq/rows.jsonl" ]; then
    echo "refusing to run: $OUT/k64-s$seq/rows.jsonl exists. Move it away first." >&2
    exit 1
  fi
done

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

# Before the redirect below: a missing $OUT parent would fail both runs while the script
# still reports success (the same guard as run_bits.sh).
mkdir -p "$OUT"

for seq in 64 512; do
  echo "== lever-bits k=64 arms=C,E seq=$seq start $(date +%T)"
  "$VENV/bin/python" -u "$HARNESS" lever-bits --seq "$seq" --ladder-ks 64 --arms C,E \
    --steps 3 --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --lora-targets "$TARGETS" --out "$OUT/k64-s$seq" \
    > "$OUT/k64-s$seq.log" 2>&1
  echo "== lever-bits k=64 seq=$seq exit=$? $(date +%T)"
done
echo "all done $(date +%T)"
