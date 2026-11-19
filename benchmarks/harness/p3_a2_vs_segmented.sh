#!/usr/bin/env bash
# PR #73 (Triton batch-invariant expert GEMM) against PR #75 (segmented cuBLAS) on ONE card, ONE
# harness. Moonlight-16B-A3B, window 8x64, MBR group 8, packed + prefetch + host cache 12 GiB.
#   OUT=/root/cmp nohup bash benchmarks/harness/p3_a2_vs_segmented.sh >/dev/null 2>&1 &
# 1. setup: run_p3_a2_stage2.sh SETUP_ONLY=1 (venv, model, qualified cache).
# 2. digests: one bridge call per arm (2 timed steps, --dump-grads): A0 L2L native,
#    At L2L batch_invariant, B0 MBR native, B0s MBR segmented, Bt MBR batch_invariant.
#    Expected: B0s == A0 and Bt == At bit for bit; B0 != A0.
# 3. F1 A-B-B-A (p72_f1_blocks.py, 5-block pilot + rule top-up): b0-b0s, b0-b0t, b0s-b0t.
# Output: OUT/{stage2,series,f1-*}, OUT/summary.json, OUT/DONE (PASS/FAIL).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P) || exit 2
H="$REPO/benchmarks/harness"
B5="$REPO/benchmarks/protocols/b5-moonlight-20261109"
OUT=${OUT:-/root/cmp}
mkdir -p "$OUT" && cd "$OUT" || exit 2
S="$OUT/stage2"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$OUT/series.log"; }
fail() { log "FAIL $*"; printf 'FAIL\n' >"$OUT/DONE"; exit 1; }

log "setup start"
OUT="$S" SETUP_ONLY=1 SPEED_GEMMS=native bash "$H/run_p3_a2_stage2.sh" || true
[[ $(cat "$S/DONE" 2>/dev/null) == PASS ]] || fail "setup (see $S/status.tsv)"
log "setup done"

export VIRTUAL_ENV="$S/.venv" PATH="$S/.venv/bin:$PATH" PYTHONPATH="$REPO/src:$REPO:$H"
export HOME="$S/home" TMPDIR="$S/tmp" HF_HOME="$S/hf" HF_HUB_OFFLINE=1
export SOUP_LAYER_STREAM_CACHE_DIR="$S/layer-cache" TRITON_CACHE_DIR="$S/triton-cache"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_DISABLE_PROGRESS_BARS=1
CACHE=$(<"$S/results/cache-path.txt")
COMMON=(--root "$S" --source "$S/source" --cache "$CACHE" --data "$B5/data.json")
COMPLETE=(--expert-transfer packed --expert-host-cache-mib 12288)

declare -A ARGS=(
    [A0]="--profile-kinds l2l"
    [At]="--profile-kinds l2l --expert-gemm batch_invariant"
    [B0]="--profile-kinds mbr --expert-prefetch"
    [B0s]="--profile-kinds mbr --expert-prefetch --expert-gemm segmented"
    [Bt]="--profile-kinds mbr --expert-prefetch --expert-gemm batch_invariant"
)
mkdir -p series
for run in A0 At B0 B0s Bt; do
    log "series $run start"
    # shellcheck disable=SC2086
    python "$H/p3_route_preserving_mbr_soup_bridge.py" "${COMMON[@]}" --output "series/$run" \
        --mbr-group-size 8 --phase-profile-only --phase-profile-blocks 0 --timed-steps 2 \
        --dump-grads "series/$run.grads.pt" "${COMPLETE[@]}" ${ARGS[$run]} \
        >"series/$run.log" 2>&1 || fail "series $run"
    log "series $run done"
done

for arms in b0-b0s b0-b0t b0s-b0t; do
    log "f1 $arms start"
    python "$H/p72_f1_blocks.py" --root "$S" --source "$S/source" --cache "$CACHE" \
        --data "$B5/data.json" --output "f1-$arms" --arms "$arms" --host-cache-mib 12288 \
        --timed-steps 3 --max-blocks 10 \
        --label "RTX 4060 Ti (vast.ai), Moonlight MBR g8, $arms" >"f1-$arms.log" 2>&1 \
        || fail "f1 $arms"
    log "f1 $arms done"
done

python - "$OUT" <<'PY' || fail "summary"
import json
import sys
from pathlib import Path

import torch
from rich.console import Console

out = Path(sys.argv[1])
runs = ("A0", "At", "B0", "B0s", "Bt")
series = {}
for run in runs:
    result = json.loads((out / "series" / run / "result_phase_profile_only.json").read_text())
    series[run] = {
        "grad_sha256": result["grad_sha256"],
        "loss": [r["loss"] for r in result["measured_uninstrumented"]],
        "model_step_seconds": [r["model_step_seconds"] for r in result["measured_uninstrumented"]],
        "arm": result["arm"],
    }
grads = {run: torch.load(out / "series" / f"{run}.grads.pt", map_location="cpu",
                         weights_only=True) for run in runs}


def bits_equal(a, b):
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
        and torch.equal(a[k].contiguous().view(torch.uint8), b[k].contiguous().view(torch.uint8))
        for k in a)


def rel_l2(a, b):
    num = sum(float((a[k].double() - b[k].double()).pow(2).sum()) for k in a)
    den = sum(float(a[k].double().pow(2).sum()) for k in a)
    return (num / den) ** 0.5 if den else None


pairs = {f"{x}_vs_{y}": {"bit_equal": bits_equal(grads[x], grads[y]),
                         "grad_rel_l2": rel_l2(grads[x], grads[y])}
         for x, y in (("B0", "A0"), ("B0s", "A0"), ("Bt", "At"), ("At", "A0"), ("Bt", "A0"))}
repeat = {run: len(set(series[run]["grad_sha256"])) == 1 for run in runs}
f1 = {}
for arms in ("b0-b0s", "b0-b0t", "b0s-b0t"):
    report = json.loads((out / f"f1-{arms}" / "f1_report.json").read_text())
    medians = {"A": [], "B": []}
    for row in report["runs"]:
        medians[row["arm"]].append(row["value"])
    f1[arms] = {"summary": report["summary"],
                "median_step_seconds": {k: sorted(v)[len(v) // 2] for k, v in medians.items()},
                "digests_by_arm": {arm: sorted({tuple(r["extra"]["grad_sha256"])[0]
                                                for r in report["runs"] if r["arm"] == arm})
                                   for arm in ("A", "B")}}
expected = (pairs["B0s_vs_A0"]["bit_equal"] and pairs["Bt_vs_At"]["bit_equal"]
            and not pairs["B0_vs_A0"]["bit_equal"] and all(repeat.values()))
summary = {"status": "PASS" if expected else "FAIL",
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "series": series, "pairs": pairs, "repeat_within_run": repeat, "f1": f1}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
Console().print_json(json.dumps({"status": summary["status"], "pairs": pairs,
                                 "f1": {k: v["summary"].get("mean_diff_percent")
                                        for k, v in f1.items()}}))
PY
log "summary done"
printf 'PASS\n' >"$OUT/DONE"
