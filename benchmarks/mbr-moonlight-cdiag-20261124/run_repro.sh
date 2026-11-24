#!/usr/bin/env bash
# Ticket 14, step 1: reproduce C against A0 at k=2, S=64 in two fresh processes.
#
# Deterministic mode (strict: the SDPA backward of ticket 12 raises instead of drifting), AC online
# checked by the driver before each run, systemd-inhibit around each process. Each run writes the
# harness loss and all 216 LoRA gradients under <out>/*/tensors/ and the boundary capture under
# <out>/boundaries/. compare_bits.py then counts the differing gradients.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-cdiag-20261124}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
ACAD=/sys/class/power_supply/ACAD/online

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 14 C-vs-A0 diagnosis")
command -v systemd-inhibit > /dev/null || INHIBIT=()

if [ "$(cat "$ACAD" 2>/dev/null)" != 1 ]; then
  echo "refusing to run: AC power is not online" >&2
  exit 1
fi

{
  "$VENV/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
  nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.limit,driver_version --format=csv
} > "$OUT/environment.txt" 2>&1

failed=0
for arm in A0 C; do
  tag="$(echo "$arm" | tr 'A-Z' 'a-z')-k2-s64"
  echo "== $arm k=2 S=64 start $(date +%T)"
  "${INHIBIT[@]}" "$VENV/bin/python" -u "$REPO/benchmarks/mbr-moonlight-cdiag-20261124/capture_boundaries.py" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --out "$OUT" --arm "$arm" --k 2 --seq 64 --steps 3 --lora-targets "$TARGETS" \
    --tag "$tag" > "$OUT/$tag.log" 2>&1
  status=$?
  echo "== $arm exit=$status $(date +%T)"
  [ "$status" -eq 0 ] || (( failed += 1 ))
done

"$VENV/bin/python" -u "$OUT/compare_bits.py" \
  --left "$OUT/tensors/A0-k2-s64.pt" --right "$OUT/tensors/C-k2-s64.pt" \
  --labels A0,C --out "$OUT/repro" || (( failed += 1 ))

echo "repro done, failed passes: $failed $(date +%T)"
exit "$failed"
