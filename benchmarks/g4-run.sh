#!/usr/bin/env bash
# G4 GPU runbook: every arm of benchmarks/probe-g4-l2l-activation-compression.md §7, in the
# order the rule names, on a Linux box with one CUDA GPU (24 GB is enough).
#
# The environment is a prerequisite, not something this script installs: a Python with the
# CUDA torch build and the pinned packages of §5, Soup's sources on PYTHONPATH (set below).
# G4_PYTHON names that interpreter (default: python3 on PATH).
#
#   bash benchmarks/g4-run.sh check            # CUDA, CPU tests, CUDA codec round trip
#   bash benchmarks/g4-run.sh fixtures         # model + Dolly at pinned revisions (online)
#   bash benchmarks/g4-run.sh arms <results>   # every arm, offline, then the verdict
#
# "arms" is resumable: a finished step leaves <results>/logs/<step>.done and is skipped on a
# re-run. A failed step leaves no marker and stops the script; its partial JSON is kept,
# and the step refuses to start again until that JSON is renamed (the record keeps every
# attempt). The rule in §3 is fixed before the GPU run; commit it before the first arm.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
G4="${G4_HOME:-$ROOT/.soup/g4}"
MODEL_ID=unsloth/mistral-7b-instruct-v0.3
REVISION=60b99dcbc7f853b7cd18472a607e8d045b96f03e
MODEL_DIR="$G4/mistral-7b-instruct-v0.3"
SHARDS="$G4/shards"
DOLLY_PATH_FILE="$G4/dolly-path.txt"
PY="${G4_PYTHON:-python3}"
export PYTHONPATH="$ROOT/src" HF_HUB_DISABLE_TELEMETRY=1

# SHA-256 of the pinned model files (Hub LFS oids at $REVISION, and the small files hashed
# from a download of the same revision).
MODEL_SHA256="ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19  model-00001-of-00003.safetensors
8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0  model-00002-of-00003.safetensors
905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca  model-00003-of-00003.safetensors
9262d15f830e511429ccd175a71680e51c656ef3a4aa7460f3e9596508d32487  config.json
60b945759e27a63c3c5c0ca675881f5a73b4aa38b5d1d6818570308d4f1a3c59  tokenizer.json
b0c776216a54c6d031866d1dff0b31715bd73f5ba87f8a30eb35e8c603dff95d  tokenizer_config.json"

stamp() {  # box state before a step: GPU peers, free RAM, load
  {
    date -u +%FT%TZ
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.used,memory.total,clocks.sm,temperature.gpu,power.draw \
      --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    free -b
    cat /proc/loadavg
  } >> "$1"
}

step() {  # step <name> <output json> <command...>
  local name=$1 out=$2
  shift 2
  if [ -f "$OUT/logs/$name.done" ]; then
    echo "skip $name (done)"
    return
  fi
  if [ -e "$out" ]; then
    echo "refusing $name: $out exists from an attempt that did not finish; rename it" >&2
    exit 1
  fi
  echo "=== $name: $*"
  stamp "$OUT/logs/$name.box"
  "$@" 2>&1 | tee "$OUT/logs/$name.log"
  stamp "$OUT/logs/$name.box"
  [ -s "$out" ] || { echo "$name exited 0 but wrote no $out" >&2; exit 1; }
  touch "$OUT/logs/$name.done"
}

need_cuda() {
  "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "CUDA is not available to $PY; the arms need one CUDA GPU" >&2; exit 1; }
}

case "${1:-}" in
check)
  nvidia-smi
  need_cuda
  mkdir -p "$G4"  # pytest's --basetemp needs its parent; on a fresh clone only fixtures made it
  "$PY" -m pytest tests/test_g4_codec.py tests/test_g4_harness.py tests/test_g4_rule.py \
    tests/test_l2l_probe_harness.py tests/test_l2l_session.py -q -o addopts='' \
    --basetemp "$G4/check-test-tmp"
  # The codecs only cast (no FP8 GEMM), so any CUDA card must round-trip them on the device.
  PYTHONPATH="$ROOT/src:$ROOT/benchmarks/harness" "$PY" -c "
import torch, g4_activation_codec as c
x = torch.randn(4, 512, 4096, device='cuda', dtype=torch.bfloat16)
for name in ('int8', 'fp8_e4m3', 'int4'):
    err = c.restore_error(x, c.round_trip(x, name))
    print(name, 'cuda round trip rms row rel', round(err['rms_row_rel'], 5))
"
  ;;
fixtures)
  mkdir -p "$G4"
  "$PY" - "$MODEL_ID" "$REVISION" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
model_id, revision, target = sys.argv[1:]
snapshot_download(repo_id=model_id, revision=revision, local_dir=target, token=False,
                  allow_patterns=["config.json", "generation_config.json", "*.safetensors",
                                  "model.safetensors.index.json", "tokenizer.json",
                                  "tokenizer_config.json", "special_tokens_map.json",
                                  "tokenizer.model"])
EOF
  (cd "$MODEL_DIR" && printf '%s\n' "$MODEL_SHA256" | sha256sum -c -)
  PYTHONPATH="$ROOT/src:$ROOT/benchmarks/harness" "$PY" - "$DOLLY_PATH_FILE" <<'EOF'
import sys
import g4_data
path = g4_data.source_path()
g4_data.load_splits(path)  # refuses a file or a split that is not the pinned one
open(sys.argv[1], "w", encoding="utf-8").write(path + "\n")
print("dolly", path, "sha256 and C5 split digests ok")
EOF
  ;;
arms)
  OUT="$(realpath -m "${2:?usage: g4-run.sh arms <results dir>}")"
  ADAPTERS="$G4/adapters/$(basename "$OUT")"  # ~55 MB each: kept out of the results dir
  mkdir -p "$OUT/logs" "$ADAPTERS"
  export HF_HUB_OFFLINE=1
  need_cuda
  DOLLY="$(cat "$DOLLY_PATH_FILE")"
  (cd "$MODEL_DIR" && printf '%s\n' "$MODEL_SHA256" | sha256sum -c --quiet -)
  git rev-parse HEAD > "$OUT/logs/git-head.txt"
  git status --porcelain > "$OUT/logs/git-status.txt"
  "$PY" -m pip freeze > "$OUT/logs/pip-freeze.txt"
  COMMON=(--weights "$MODEL_DIR" --model-revision "$REVISION" --shards "$SHARDS"
          --dolly "$DOLLY" --label g4)
  PROBE="$ROOT/benchmarks/harness/g4_probe.py"
  # 1. Error experiment: restore error, one-recompute propagation, gradient agreement,
  #    controls, per-chunk micro-benchmark. One process, deterministic.
  step error "$OUT/error.json" "$PY" "$PROBE" --mode error "${COMMON[@]}" --out "$OUT/error.json"
  # 2. Training arms: 394 steps = the 4000-row train split packed into 512-token rows, once.
  for arm in bf16:1 int8:1 fp8_e4m3:1 bf16:2; do
    codec=${arm%%:*}
    seed=${arm##*:}
    name="train-$codec-seed$seed"
    step "$name" "$OUT/$name.json" "$PY" "$PROBE" --mode train "${COMMON[@]}" \
      --codec "$codec" --seed "$seed" --steps 394 --adapter-out "$ADAPTERS/$name" \
      --out "$OUT/$name.json"
  done
  # 3. Gate suites on each arm's saved adapter (resident NF4 base, greedy, 64 new tokens),
  #    each preceded by the base-parity check V13 (resident vs streamed NLL, 16 panel rows).
  for arm in bf16:1 int8:1 fp8_e4m3:1 bf16:2; do
    codec=${arm%%:*}
    seed=${arm##*:}
    name="suites-$codec-seed$seed"
    step "$name" "$OUT/$name.json" "$PY" "$PROBE" --mode suites "${COMMON[@]}" \
      --arm "$OUT/train-$codec-seed$seed.json" --out "$OUT/$name.json"
  done
  # 4. Timing: bf16 store against each codec's store, A-B-B-A blocks, fresh processes,
  #    ArmWatch (suspend and power required), repeats rule. Informational.
  step timing "$OUT/timing.json" "$PY" "$ROOT/benchmarks/harness/g4_timing.py" \
    --codecs int8,fp8_e4m3 --out "$OUT/timing.json" --label g4 --weights "$MODEL_DIR" \
    --model-revision "$REVISION" --shards "$SHARDS" --dolly "$DOLLY"
  # 5. The verdict, from the JSON alone.
  "$PY" "$ROOT/benchmarks/harness/g4_rule.py" "$OUT" --json "$OUT/verdict.json" \
    | tee "$OUT/logs/verdict.log"
  (cd "$OUT" && find . -type f ! -name 'sha256.txt' -print0 | sort -z | xargs -0 sha256sum > sha256.txt)
  ;;
*)
  sed -n '2,16p' "$0"
  exit 2
  ;;
esac
