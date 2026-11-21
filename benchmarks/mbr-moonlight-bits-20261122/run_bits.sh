#!/usr/bin/env bash
# Ticket 03: exact GEMM bits on Moonlight-16B-A3B, RTX 3050 Laptop (sm_86, 4 GiB).
#
# 1. Environment record (detect_environment, GPU query) into $OUT/environment.txt.
# 2. Ladder k = 1, 2, 4, 8 for A0, C, E, F, G in deterministic mode, at seq 64 and seq 512
#    (lever-bits sweep, one fresh process per run, ArmWatch around each run).
# 3. One E run at k = 64, seq 64, deterministic: the python_forward counter only, no ladder.
#
# Inputs, all overridable from the environment:
#   VENV            virtualenv with torch 2.14 cu130 and nvidia-cuda-nvcc 13
#                   (default: $HOME/.venvs/soup-rnd-cuda)
#   MOONLIGHT_ROOT  folder written by p72_prepare_moonlight_cache.py, with source/ and cache/
#                   (default: $HOME/moonlight-local)
#   OUT             results folder, outside git: loss and grad tensors are about 1.3 GB
#                   (default: $MOONLIGHT_ROOT/results/bits)
#
# Refuses to run when a previous sweep is in $OUT, so a rerun cannot double the rows.
# No `set -e`: a failed run is recorded and the next one still runs, on purpose.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$MOONLIGHT_ROOT/results/bits}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
MODEL=(--source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache")

if [ -e "$OUT/s64/rows.jsonl" ] || [ -e "$OUT/s512/rows.jsonl" ]; then
  echo "refusing to run: a sweep already wrote rows under $OUT. Move it away first." >&2
  exit 1
fi
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

{
  "$VENV/bin/python" -c "from soup_cli.utils.expert_gemm import detect_environment; print(detect_environment())"
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.limit,driver_version --format=csv
} > "$OUT/environment.txt" 2>&1

for seq in 64 512; do
  echo "== lever-bits seq=$seq start $(date +%T)"
  "$VENV/bin/python" -u "$HARNESS" lever-bits --seq "$seq" --steps 3 "${MODEL[@]}" \
    --lora-targets "$TARGETS" --out "$OUT/s$seq" \
    > "$OUT/sweep-s$seq.log" 2>&1
  echo "== lever-bits seq=$seq exit=$? $(date +%T)"
done

echo "== counter E k=64 seq=64 start $(date +%T)"
"$VENV/bin/python" -u "$HARNESS" run --arm E --k 64 --seq 64 --kind bits --deterministic \
  --steps 3 "${MODEL[@]}" --lora-targets "$TARGETS" --out "$OUT/k64" \
  >> "$OUT/k64.jsonl" 2>> "$OUT/k64.err"
echo "== counter E k=64 exit=$? $(date +%T)"
echo "all done $(date +%T)"
