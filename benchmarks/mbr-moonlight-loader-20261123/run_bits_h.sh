#!/usr/bin/env bash
# Ticket 05 part 2: the bits ladder of arm H (cpp_async_2slot) against the reference.
#
# Reference rule of the campaign: A0 at k <= 8, C above it (mbr_measure.reference_arm).
#   1. S=64, default backend, k = 1,2,4,8: A0 and H.     -> verdicts on (8,64)
#   2. S=64, default backend, k = 64:      C and H.      -> (64,64), not a blocking gate
#   3. S=512, math backend, k = 8:         A0 and H.     -> (8,512) math
# Deterministic mode, three steps per run (one warm-up). torch.equal on the saved loss and
# the 216 LoRA gradients decides; the harness repeats nothing by itself, a differs cell is
# repeated by hand in a new folder.
#
# Inputs, overridable from the environment: VENV, MOONLIGHT_ROOT, OUT (default this campaign
# folder: the raw rows, arm records and tensors stay inside the repository, as in the earlier
# campaigns), NATIVE_EXT (default /tmp/opencode/native-ext).
# Refuses to run over existing rows.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-loader-20261123}"
NATIVE_EXT="${NATIVE_EXT:-/tmp/opencode/native-ext}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
HARNESS="$REPO/benchmarks/harness/mbr_measure.py"
MODEL=(--source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache")

for folder in s64 k64 s512-math; do
  if [ -e "$OUT/$folder/rows.jsonl" ]; then
    echo "refusing to run: $OUT/$folder/rows.jsonl exists. Move it away first." >&2
    exit 1
  fi
done
mkdir -p "$OUT"

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$NATIVE_EXT:$HOME/moonlight-local/native-ext:$REPO/src:$REPO"

{
  "$VENV/bin/python" -c "from soup_cli.utils.expert_gemm import detect_environment; print(detect_environment())"
  "$VENV/bin/python" -c "import torch; from soup_cli.utils.expert_loader import detect_loader_environment; print(torch.__version__, detect_loader_environment())"
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.limit,driver_version --format=csv
} > "$OUT/environment.txt" 2>&1

echo "== H ladder S=64 k=1,2,4,8 start $(date +%T)"
"$VENV/bin/python" -u "$HARNESS" lever-bits --seq 64 --ladder-ks 1,2,4,8 --arms A0,H \
  --steps 3 "${MODEL[@]}" --lora-targets "$TARGETS" --out "$OUT/s64" \
  > "$OUT/s64.log" 2>&1
echo "== H ladder S=64 exit=$? $(date +%T)"

echo "== H k=64 S=64 against C start $(date +%T)"
"$VENV/bin/python" -u "$HARNESS" lever-bits --seq 64 --ladder-ks 64 --arms C,H \
  --steps 3 "${MODEL[@]}" --lora-targets "$TARGETS" --out "$OUT/k64" \
  > "$OUT/k64.log" 2>&1
echo "== H k=64 exit=$? $(date +%T)"

echo "== H S=512 math k=8 start $(date +%T)"
"$VENV/bin/python" -u "$HARNESS" lever-bits --seq 512 --ladder-ks 8 --arms A0,H \
  --math-attention --steps 3 "${MODEL[@]}" --lora-targets "$TARGETS" --out "$OUT/s512-math" \
  > "$OUT/s512-math.log" 2>&1
echo "== H S=512 math exit=$? $(date +%T)"
echo "all done $(date +%T)"
