#!/usr/bin/env bash
# A2 Stage 2 / Moonlight. On a fresh vast.ai Ubuntu 24.04 + CUDA 13 host, as root,
# check out this A2 branch and run (detach so an SSH drop does not end the run):
#   nohup bash benchmarks/harness/run_p3_a2_stage2.sh >/dev/null 2>&1 &
#   tail -f /workspace/a2-stage2/out/status.tsv
# Optional subsets: ARMS='control_native mbr_invariant' SEEDS='42 43' bash ...
# SPEED_GEMMS='native batch_invariant' selects the bridge backends. Use OUT=/new/path
# for a new attempt; an existing output directory is never reused. One arm alone trains
# but cannot provide a paired estimate. Nothing rents, destroys, or powers off the VM.
#
# Hardware admission policy: 1x RTX 4060 Ti 16 GB, 120 GiB (128849018880 bytes) free on
# the work volume at start, 48 GiB (51539607552 bytes) usable host RAM after install;
# choose local NVMe. Soup checks storage itself; do not bypass an SSD/HDD rejection.
# These are provisioning floors, NOT measured minimum requirements. For the B5 reference
# snapshot only: 31920879456 bytes of weights, 9947958701-byte cache, 18494839040-byte
# sharding free-space bound, 4463886336-byte preparation peak RSS. A 42.5 GiB cgroup
# worked for historical CLI quality, not proof of a 32 GiB limit. Reserve ~200 GB total
# for CUDA wheels, venv, Hub/data scratch, Arrow, checkpoints, page cache and logs.
# results/{preflight,cache}.json, resource-usage.log and status.tsv capture this host.
#
# Planning estimates (NOT 4060 Ti measurements): old 3060 Ti 8x64 bridge 98.27s control,
# 18.37s MBR; 3 of each per backend incl. diagnostic ~= 5.8 min, ~12 min for both,
# before JIT/startup. Allow 15-60 min for system/packages/model/data (31.9 GB weights;
# historic download 239s, cache build 16s + ~63s B5 qualification). CLI reference
# step times 90.52-116.61s vs 17.76-25.29s imply 60-78 vs 12-17 min training per seed,
# 4.2-5.6h for 3 controls + 6 MBR runs. Longer <=256-token rows, I/O and the new kernel
# can increase these substantially. Validation: 9 x 2 x 64 old-dev rows; fresh: 6 paired
# evaluations x 2 x 128 rows (control rescored for each contrast). At ~6s/forward (rough
# 98s/(2 reads x 8), NOT a measured eval time), ~2h + ~2.6h extra. Budget an overnight run.
#
# Data: seeded RECONSTRUCTION of a 160/64 MetaMathQA split, not the private historical
# corpus. Two epochs, attention-LoRA r=8, 40 optimizer steps; compare only OUR arms.
# The original fresh selector excludes the reconstructed groups to select another 64 GSM
# + 64 MATH. Each seed's evaluation uses a common native operator, signed task-paired
# bootstrap CIs, not a non-inferiority gate or confidence interval over training seeds.
# Output: speed/{native,batch_invariant}/result.json, data/{split.json,reference.pt,
# fresh.pt,fresh.manifest.json}, config/, quality/ARM/SEED/checkpoint-40, eval/*.json,
# logs/, exit-codes/, status.tsv, DONE (PASS/FAIL). An arm failure never cancels a peer.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P) || exit 2
H="$REPO/benchmarks/harness"
B5="$REPO/benchmarks/protocols/b5-moonlight-20261109"
RECIPE="$REPO/benchmarks/protocols/p3-a2-stage2/moonlight.yaml"
OUT=${OUT:-/workspace/a2-stage2/out}
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT" && cd "$OUT" || exit 2
OUT=$PWD
mkdir -p logs exit-codes results speed config quality eval home tmp || exit 2
printf 'step\tstart_utc\tend_utc\texit_code\n' >status.tsv
failed=0
step() {
    local label=$1 start rc
    shift
    start=$(date -u +%FT%TZ)
    printf '%s START %s\n' "$start" "$label"
    "$@" >"logs/$label.log" 2>&1; rc=$?
    printf '%s\n' "$rc" >"exit-codes/$label"
    printf '%s\t%s\t%s\t%s\n' "$label" "$start" "$(date -u +%FT%TZ)" "$rc" >>status.tsv
    if (( rc )); then failed=1; fi
    return "$rc"
}
finish() {
    { df -B1 .; du -sb source layer-cache data quality eval speed .venv 2>/dev/null; } \
        >resource-usage.log
    if (( failed )); then printf 'FAIL\n' >DONE; exit 1; fi
    printf 'PASS\n' >DONE
    exit 0
}
retry() {
    local n
    for n in 1 2 3; do
        "$@" && return 0
        if (( n < 3 )); then sleep $((n * 15)); fi
    done
    return 1
}
read -ra arms <<<"${ARMS:-control_native mbr_native mbr_invariant}"
read -ra seeds <<<"${SEEDS:-42 43 44}"
read -ra gemms <<<"${SPEED_GEMMS:-native batch_invariant}"
validate_inputs() {
    local x free
    declare -A seen=()
    [[ $(id -u) == 0 ]] || { echo 'run as root'; return 2; }
    free=$(df -B1 --output=avail . | tail -1)
    (( free >= 120 * 1024**3 )) || { echo "insufficient free disk: $free bytes"; return 2; }
    (( ${#arms[@]} && ${#seeds[@]} && ${#gemms[@]} )) || return 2
    for x in "${arms[@]}"; do
        [[ $x == control_native || $x == mbr_native || $x == mbr_invariant ]] || return 2
        [[ ! ${seen[$x]+x} ]] || return 2; seen[$x]=1
    done
    for x in "${seeds[@]}"; do
        [[ $x =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( x <= 4294967295 )) || return 2
        [[ ! ${seen[$x]+x} ]] || return 2; seen[$x]=1
    done
    for x in "${gemms[@]}"; do
        [[ $x == native || $x == batch_invariant ]] || return 2
        [[ ! ${seen[$x]+x} ]] || return 2; seen[$x]=1
    done
}
step admission validate_inputs || finish
export HOME="$OUT/home" TMPDIR="$OUT/tmp" HF_HOME="$OUT/hf" UV_CACHE_DIR="$OUT/uv-cache"
export SOUP_LAYER_STREAM_CACHE_DIR="$OUT/layer-cache" TRITON_CACHE_DIR="$OUT/triton-cache"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$REPO/src:$REPO:$H" TORCHDYNAMO_DISABLE=1 WANDB_DISABLED=true
step apt-update retry apt-get update -qq || finish
step apt-install retry apt-get install -y python3.12-dev python3.12-venv build-essential git \
    ca-certificates || finish
step uv-bootstrap /usr/bin/python3.12 -m venv bootstrap || finish
step uv-install retry bootstrap/bin/pip install uv || finish
UV="$OUT/bootstrap/bin/uv"
step venv "$UV" venv --python /usr/bin/python3.12 .venv || finish
export VIRTUAL_ENV="$OUT/.venv" PATH="$OUT/.venv/bin:$PATH"
step torch retry "$UV" pip install --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.11.0+cu130' || finish
# [train] is the editable CLI + training portion of [dev]. No testing/MCP/UI extras.
step train-deps retry "$UV" pip install -e "$REPO[train]" 'torch==2.11.0+cu130' \
    transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2 \
    safetensors==0.8.0 tokenizers==0.23.3 huggingface-hub==2.4.0 tiktoken==0.14.0 \
    blobfile==3.1.0 regex==2026.11.9 numpy==2.1.3 psutil==7.2.2 trl==0.29.1 || finish
step environment "$UV" pip freeze || finish
step repo-revision git -C "$REPO" rev-parse HEAD || finish
PROTO="$H/p3_a2_stage2_protocol.py"
step preflight python "$PROTO" preflight || finish
step snapshot retry python "$PROTO" download || finish
SOURCE="$OUT/source"
# Resolve the same shard path that soup train computes from the absolute local base.
step cache-path python - "$SOURCE" <<'PY' || finish
import sys
from pathlib import Path
from soup_cli.utils.layer_shard import resolve_shard_dir
Path("results/cache-path.txt").write_text(resolve_shard_dir(sys.argv[1]) + "\n")
PY
CACHE=$(<results/cache-path.txt)
MODEL=moonshotai/Moonlight-16B-A3B
REVISION=2a9a177f4841649f849ba941e41d2ee73c042314
step b5-build env HF_HUB_OFFLINE=1 python "$H/b5_real_moe.py" --kind real \
    --config "$B5/soup.yaml" --source "$SOURCE" --cache "$CACHE" --output results/b5 \
    --data "$B5/data.json" --model "$MODEL" --revision "$REVISION" --steps 4 --shard-threads 6 || finish
step cache-gate python "$PROTO" cache-gate "$CACHE" || finish
for gemm in "${gemms[@]}"; do
    step "speed-$gemm" env HF_HUB_OFFLINE=1 python "$H/p3_route_preserving_mbr_soup_bridge.py" \
        --root "$OUT" --source "$SOURCE" --cache "$CACHE" --data "$B5/data.json" \
        --output "speed/$gemm" --mbr-group-size 8 --expert-gemm "$gemm" --deterministic
done
step reference-data python "$H/p3_a2_stage2_data.py" --model "$SOURCE" --output data || finish
# Independent selector with a different seed, refusing overlap with either reference split.
step fresh-data python "$H/p3_prepare_fresh_metamath_eval.py" --source-data data/reference.pt \
    --model "$SOURCE" --output data/fresh.pt --per-family 64 --max-length 256 --seed 20261118
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

for seed in "${seeds[@]}"; do
    declare -A trained=()
    for arm in "${arms[@]}"; do
        stem="$arm-$seed"
        step "config-$stem" python "$PROTO" config "$RECIPE" "$arm" "$seed" || continue
        step "train-$stem" soup train --config "config/$stem.yaml" --yes || continue
        adapter="quality/$arm/$seed/checkpoint-40"
        step "adapter-$stem" python "$PROTO" check-trained "$adapter" || continue
        trained[$arm]=$adapter
    done
    # Prefer native L2L as baseline regardless of requested execution order. If only
    # the two MBR arms survived/were requested, report that explicitly named contrast.
    baseline=
    for arm in control_native mbr_native mbr_invariant; do
        if [[ ${trained[$arm]+yes} ]]; then baseline=$arm; break; fi
    done
    if [[ ! -f data/fresh.pt || ${#trained[@]} -lt 2 ]]; then
        printf 'No paired evaluation: fresh data or two completed adapters unavailable.\n' \
            >"eval/$seed-unpaired.txt"
        continue
    fi
    for arm in "${arms[@]}"; do
        [[ $arm != "$baseline" && ${trained[$arm]+yes} ]] || continue
        pair="$arm-minus-$baseline-$seed"
        step "eval-$pair" python "$H/p3_cli_fresh_quality_eval.py" \
            --source "$SOURCE" --cache "$CACHE" --data data/fresh.pt --count 128 \
            --control "${trained[$baseline]}" --mbr "${trained[$arm]}" --output "eval/$pair.json"
    done
done
finish
