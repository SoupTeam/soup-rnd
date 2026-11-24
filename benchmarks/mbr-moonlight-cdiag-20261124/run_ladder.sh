#!/usr/bin/env bash
# Ticket 14, step 5: C with the segmented native expert GEMM against A0, k = 1, 2, 4, 8.
#
# Windows (8,64) on the default SDPA backend and (8,512) on the math backend. Strict deterministic
# mode, three steps per run, a fresh process per run, AC online and systemd-inhibit around it. Both
# arms get --expert-gemm segmented; for A0 the segment plan stays empty (one micro-batch per call),
# so it is a passthrough and the run doubles as the no-op proof. Verdict per window: torch.equal on
# the loss and all 216 LoRA gradients, from the saved tensors.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HOME/.venvs/soup-rnd-cuda}"
MOONLIGHT_ROOT="${MOONLIGHT_ROOT:-$HOME/moonlight-local}"
OUT="${OUT:-$REPO/benchmarks/mbr-moonlight-cdiag-20261124}"
SEQS="${SEQS:-64 512}"
KS="${KS:-1 2 4 8}"
TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
ACAD=/sys/class/power_supply/ACAD/online

export CUDA_HOME=$("$VENV/bin/python" -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
INHIBIT=(systemd-inhibit --what=sleep --why="ticket 14 segmented ladder")
command -v systemd-inhibit > /dev/null || INHIBIT=()

exec 9> "$OUT/.ladder.lock"
if ! flock -n 9; then
  echo "refusing to run: another ladder holds $OUT/.ladder.lock" >&2
  exit 1
fi
if [ "$(cat "$ACAD" 2>/dev/null)" != 1 ]; then
  echo "refusing to run: AC power is not online" >&2
  exit 1
fi

failed=0
run_one() {
  local arm="$1" k="$2" seq="$3" backend="$4"
  local tag
  tag="$(echo "$arm" | tr 'A-Z' 'a-z')-seg-k${k}-s${seq}"
  if [ "$(cat "$ACAD" 2>/dev/null)" != 1 ]; then
    echo "refusing $tag: AC power is not online" >&2
    return 1
  fi
  echo "== $tag start $(date +%T)"
  local backend_args=()
  [ "$backend" = "math" ] && backend_args=(--backend math)
  "${INHIBIT[@]}" "$VENV/bin/python" -u \
    "$REPO/benchmarks/mbr-moonlight-cdiag-20261124/capture_boundaries.py" \
    --source "$MOONLIGHT_ROOT/source" --cache "$MOONLIGHT_ROOT/cache" \
    --out "$OUT" --arm "$arm" --k "$k" --seq "$seq" --steps 3 --lora-targets "$TARGETS" \
    --expert-gemm segmented --tag "$tag" "${backend_args[@]}" > "$OUT/$tag.log" 2>&1
  local status=$?
  echo "== $tag exit=$status $(date +%T)"
  [ "$status" -eq 0 ] || (( failed += 1 ))
}

for seq in $SEQS; do
  backend="default"
  [ "$seq" = "512" ] && backend="math"
  for k in $KS; do
    for arm in A0 C; do
      run_one "$arm" "$k" "$seq" "$backend"
    done
  done
done

for seq in $SEQS; do
  for k in $KS; do
    "$VENV/bin/python" -u "$OUT/compare_bits.py" \
      --left "$OUT/tensors/A0-k${k}-s${seq}.pt" --right "$OUT/tensors/C-k${k}-s${seq}.pt" \
      --labels A0,C-segmented --out "$OUT/ladder/k${k}-s${seq}" || (( failed += 1 ))
  done
done

echo "ladder done, differing cells or failed passes: $failed $(date +%T)"
exit "$failed"
