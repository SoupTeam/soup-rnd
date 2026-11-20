#!/bin/bash
# Diagnosis feedback loop: B0 against an A0-exact arm on Moonlight, one process each.
# Prints the median step of each, the ratio B/B0 and whether the arm's gradients equal A0.
# Red when the ratio is above 1.10 or the bits differ. Needs results/series/A0.grads.pt
# (remote-f1-direct.sh series with RUNS=A0).
# Usage: diag-loop.sh TAG "ARM FLAGS" [STEPS]
#   e.g. diag-loop.sh bltd "--expert-gemm lt-merged --expert-direct-backward"
R=/workspace/p3-baseline
TAG=$1
ARM=$2
STEPS=${3:-3}
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch"
COMMON="--phase-profile-only --phase-profile-blocks 0 --timed-steps $STEPS"
OUT=$R/results/loop/$TAG
rm -rf $OUT && mkdir -p $OUT
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
$R/venv/bin/python $B --root $R --output $OUT/b0 $COMMON $MBR > $OUT/b0.log 2>&1 || { echo "b0 failed"; exit 2; }
$R/venv/bin/python $B --root $R --output $OUT/arm --dump-grads $OUT/arm.grads.pt $COMMON $MBR $ARM \
  > $OUT/arm.log 2>&1 || { echo "arm failed"; tail -5 $OUT/arm.log; exit 2; }
$R/venv/bin/python - "$OUT" <<'EOF'
import json, statistics, sys, torch
out = sys.argv[1]
def med(run):
    d = json.load(open(f"{out}/{run}/result_phase_profile_only.json"))
    return statistics.median(m["seconds"] for m in d["measured_uninstrumented"])
b0, arm = med("b0"), med("arm")
a = torch.load("/workspace/p3-baseline/results/series/A0.grads.pt", map_location="cpu")
g = torch.load(f"{out}/arm.grads.pt", map_location="cpu")
bits = all(torch.equal(a[k], g[k]) for k in a)
ratio = arm / b0
verdict = "GREEN" if ratio <= 1.10 and bits else "RED"
print(f"LOOP {verdict}: B0 {b0:.2f} s, arm {arm:.2f} s, ratio {ratio:.3f}, bits==A0 {bits}")
EOF
