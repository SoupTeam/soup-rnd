#!/usr/bin/env bash
# Run AFTER the Stage 2 quality screen releases its GPU. No download, cache build or resume.
# OUT=/workspace/a2-pair bash benchmarks/harness/p3_a2_bitwise_cli_pair.sh
# STAGE2 points to the original screen's OUT. Use a fresh OUT for every attempt.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
H="$REPO/benchmarks/harness"
STAGE2=$(realpath -e "${STAGE2:-/workspace/a2-stage2/out}")
OUT=$(realpath -m "${OUT:-/workspace/a2-pair}")
[[ -f $STAGE2/DONE ]] || { echo "Stage 2 has not finished: missing $STAGE2/DONE" >&2; exit 2; }
PY="$STAGE2/.venv/bin/python"
CACHE=$(realpath -e "$(<"$STAGE2/results/cache-path.txt")")
[[ -x $PY && -f $CACHE/index.json && -f $STAGE2/data/train.jsonl && -d $STAGE2/data/arrow ]]
# Soup requires real Arrow paths under cwd and its cache root under cwd/HOME/TMPDIR.
# Work from the shared workspace, not OUT with a symlink escaping its containment guard.
WORKSPACE=$("$PY" -c 'import os,sys; print(os.path.commonpath(sys.argv[1:]))' "$STAGE2" "$OUT" "$CACHE")
[[ $WORKSPACE != / ]] || { echo 'STAGE2, cache and OUT must share a workspace' >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT"/{config,quality,logs,home,tmp}
cd "$WORKSPACE"
export STAGE2 OUT CACHE
export HOME="$OUT/home" TMPDIR="$OUT/tmp" HF_HOME="$STAGE2/hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SOUP_LAYER_STREAM_CACHE_DIR="$(dirname "$CACHE")" TRITON_CACHE_DIR="$OUT/triton-cache"
export VIRTUAL_ENV="$STAGE2/.venv" PATH="$STAGE2/.venv/bin:$PATH" PYTHONPATH="$REPO/src:$REPO:$H"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export TORCHDYNAMO_DISABLE=1 WANDB_DISABLED=true

"$PY" - "$REPO/benchmarks/protocols/p3-a2-stage2/moonlight.yaml" <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml
from p3_route_preserving_mbr_soup_bridge import sha256_file, source_revision
from soup_cli.config.schema import SoupConfig
from soup_cli.utils.layer_shard import resolve_shard_dir

stage, out, cache = (Path(os.environ[k]) for k in ("STAGE2", "OUT", "CACHE"))
source = str(stage / "source")
(out / "revision.txt").write_text((source_revision() or "unknown") + "\n")
receipt = json.loads((stage / "results/cache.json").read_text())
if (receipt["status"] != "PASS" or Path(resolve_shard_dir(source)).resolve() != cache
        or sha256_file(cache / "index.json") != receipt["index_sha256"]):
    raise ValueError("not the original qualified Stage 2 cache")
# Explicit experiment cells: MBR flag, GEMM, grouped microbatches (unset for L2L).
arms = {"control_invariant": (False, "batch_invariant", None),
        "mbr_invariant": (True, "batch_invariant", 8)}
for arm, (mbr, gemm, group) in arms.items():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    cfg.update(base=source, output=str(out / "quality" / arm / "42"))
    cfg["data"].update(train=str(stage / "data/train.jsonl"),
                       tokenized_path=str(stage / "data/arrow"))
    cfg["training"].update(seed=42, data_seed=42, l2l_route_preserving_mbr=mbr,
                           stream_expert_gemm=gemm, l2l_mbr_group_size=group)
    SoupConfig.model_validate(cfg)
    (out / "config" / f"{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
for arm in control_invariant mbr_invariant; do
    if ! "$PY" "$H/p3_a2_deterministic_train.py" --config "$OUT/config/$arm.yaml" --yes \
            >"$OUT/logs/$arm.log" 2>&1; then
        printf '{"status":"ERROR","arm":"%s","log":"logs/%s.log"}\n' "$arm" "$arm" >"$OUT/bitwise.json"
        exit 1
    fi
done

"$PY" - <<'PY'
import json
import math
import os
from pathlib import Path

import torch
import yaml
from rich.console import Console
from safetensors.torch import load_file

out = Path(os.environ["OUT"])
arms = ("control_invariant", "mbr_invariant")
roots = [out / "quality" / arm / "42" for arm in arms]
report = {"scope": "seed 42; raw SFT window and AdamW trajectory for steps 1-40; checkpoints 20/40",
          "soup_revision": (out / "revision.txt").read_text().strip(), "arms": arms,
          "checkpoints": {}}


def tensor_bits(a, b):
    # Comparing dtype, shape and bytes distinguishes signed zero and does not cast BF16.
    result = {}
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            result[key] = False
            continue
        left, right = a[key], b[key]
        result[key] = (left.dtype == right.dtype and left.shape == right.shape and
                       torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                   right.contiguous().reshape(-1).view(torch.uint8)))
    return result


def optimizer(directory):
    path = directory / "optimizer.pt"  # HF Trainer, single-device adamw_torch
    if not path.is_file():
        return None
    opt = torch.load(path, map_location="cpu", weights_only=True)
    slots = {}
    for pid, state in opt["state"].items():
        if not {"exp_avg", "exp_avg_sq", "step"} <= state.keys():
            raise ValueError(f"{path}: incomplete AdamW state for {pid}")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{path}: unsupported non-tensor state {pid}/{name}")
            slots[f"{pid}/{name}"] = value
    if len(opt["state"]) != 216:
        raise ValueError(f"{path}: expected AdamW state for all 216 trainable adapter tensors")
    return slots, opt["param_groups"]


def losses(directory, step):
    state = json.loads((directory / "trainer_state.json").read_text())
    rows = [(r["step"], r["loss"]) for r in state["log_history"] if "loss" in r]
    if state["global_step"] != step or [s for s, _ in rows] != list(range(1, step + 1)):
        raise ValueError(f"{directory}: incomplete per-step training loss history")
    if not all(math.isfinite(loss) for _, loss in rows):
        raise ValueError(f"{directory}: non-finite logged training loss")
    return [float(loss).hex() for _, loss in rows]


def first_diff(left, right, path=""):
    """Return the first different leaf, not two potentially enormous state dicts."""
    if isinstance(left, dict) and isinstance(right, dict):
        for k in dict.fromkeys([*left, *right]):
            field = f"{path}/{k}"
            if k not in left or k not in right:
                return {"field": field, "missing_from": arms[0] if k not in left else arms[1]}
            diff = first_diff(left[k], right[k], field)
            if diff:
                return diff
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for i, (a, b) in enumerate(zip(left, right)):
            diff = first_diff(a, b, f"{path}/{i}")
            if diff:
                return diff
    elif left != right:
        return {"field": path, arms[0]: left, arms[1]: right}
    return None


def trajectory():
    mismatched_steps = []
    first = None
    with (roots[0] / "trajectory.jsonl").open() as a, (roots[1] / "trajectory.jsonl").open() as b:
        for step in range(41):
            rows = [json.loads(stream.readline()) for stream in (a, b)]
            for row in rows:
                if row["step"] != step or row["deterministic_algorithms"] is not True:
                    raise ValueError(f"invalid deterministic trajectory at step {step}")
                if step == 0:
                    if len(row["initial_adapters"]) != 216:
                        raise ValueError("missing initial adapters")
                elif (len(row["micro_loss_bits"]) != 8 or len(row["input_ids"]) != 8
                      or len(row["labels"]) != 8 or len(row["raw_lora_gradients"]) != 216
                      or len(row["post_adapters"]) != 216 or len(row["adamw"]) != 3 * 216):
                    raise ValueError(f"incomplete trace at step {step}")
            diff = first_diff(*rows)
            if diff:
                mismatched_steps.append(step)
                if first is None:
                    first = {"step": step, **diff}
        if a.readline() or b.readline():
            raise ValueError("unexpected rows after step 40")
    return {"initial_state_checked": True, "training_steps_checked": 40,
            "deterministic_algorithms": True, "mismatched_steps": mismatched_steps,
            "first_difference": first}


try:
    recipes = {}
    for arm, root in zip(arms, roots):
        cfg = yaml.safe_load((out / "config" / f"{arm}.yaml").read_text())["training"]
        recipe = json.loads((root / "l2l_recipe.json").read_text())
        expected = {"expert_gemm": "batch_invariant", "effective_k": 8, "two_read_experts": True,
                    "grouped_expert_routes": True, "route_preserving_mbr": cfg["l2l_route_preserving_mbr"],
                    "mbr_group_size": cfg["l2l_mbr_group_size"]}
        if any(recipe.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{arm}: runtime recipe does not match requested experiment cell")
        recipes[arm] = recipe
    report["resolved_recipes"] = recipes
    # The pair is only meaningful if exactly the control is L2L and the other arm really is MBR.
    if [r.get("route_preserving_mbr") for r in recipes.values()] != [False, True]:
        raise ValueError("expected control_invariant without MBR and mbr_invariant with MBR")
    same_fields = [{k: v for k, v in r.items() if k not in ("route_preserving_mbr", "mbr_group_size")}
                   for r in recipes.values()]
    if same_fields[0] != same_fields[1]:
        raise ValueError("resolved L2L recipes differ beyond MBR enablement and group size")
    report["trajectory"] = trajectory()
    passed = not report["trajectory"]["mismatched_steps"]
    first = report["trajectory"]["first_difference"]
    for step in (20, 40):
        dirs = [root / f"checkpoint-{step}" for root in roots]
        adapters = tensor_bits(*(load_file(str(d / "adapter_model.safetensors")) for d in dirs))
        if len(adapters) != 216:
            raise ValueError(f"step {step}: expected 216 attention LoRA tensors")
        left, right = (losses(d, step) for d in dirs)
        diff = [i for i, (a, b) in enumerate(zip(left, right), 1) if a != b]
        row = {"adapter_tensor_equal": adapters, "logged_losses_checked": step,
               "logged_loss_mismatch_steps": diff}
        a, b = (optimizer(d) for d in dirs)
        if a is None or b is None:
            # This experiment claims AdamW equality, so both checkpoints must carry the state.
            row["optimizer"] = {"status": "missing_from_a_checkpoint"}
            optimizer_ok = False
        else:
            equal = tensor_bits(a[0], b[0])
            groups_equal = a[1] == b[1]
            row["optimizer"] = {"status": "compared", "tensor_equal": equal,
                                "param_groups_equal": groups_equal}
            optimizer_ok = groups_equal and all(equal.values())
        row["passed"] = all(adapters.values()) and not diff and optimizer_ok
        if not row["passed"]:
            failure_step = diff[0] if diff else step
            if first is None or failure_step < first["step"]:
                field = "trainer_state/log_history/loss"
                if not diff:
                    field = next((f"adapter/{n}" for n, ok in adapters.items() if not ok), None)
                    if field is None:
                        field = next((f"optimizer/{n}" for n, ok in
                                      row["optimizer"].get("tensor_equal", {}).items() if not ok),
                                     "optimizer/presence_or_param_groups")
                first = {"step": failure_step, "field": field, "source": f"checkpoint-{step}"}
        report["checkpoints"][str(step)] = row
        passed = passed and row["passed"]
    report["first_difference"] = first
    report["status"] = "PASS" if passed else "FAIL"
except Exception as exc:
    report.update(status="ERROR", error=str(exc))
(out / "bitwise.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
Console().print(f"{report['status']}: {out / 'bitwise.json'}")
raise SystemExit(0 if report["status"] == "PASS" else 1)
PY
