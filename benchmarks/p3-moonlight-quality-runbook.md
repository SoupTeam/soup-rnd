# Moonlight-16B-A3B quality runbook

Authority: [the amended plan](p3-moonlight-quality-plan.md) and the user's $15 authorization. This is a quality comparison, not the DeepSeek pilot. Two methods × two retained lengths means **four independent 64-step training runs**; omitting L4096 under the authorized budget rule leaves two. Never evaluate the base model.

**DeepSeek-V3 671B: отложен.** Historical tiny E3 remains **FAIL 0.05028628354277727 > 0.05**. Neither result is replaced or rerun here. PR #61 and the `deepseek-v3` policy are unchanged.

## Admission before any rental

Required, with complete retained evidence:

1. Exact fast-tokenizer/tiktoken parity on the frozen corpus; generated tokenizer and pack hashes.
2. The amended no-numerical-consumer proof on Transformers 5.19.0, matching wheel RECORD: exact FP32/BF16 loss, logits, ordered expert choices and every LoRA gradient. Keep the unchanged Oracle guards. Both runtime configurations omit only the admitted auxiliary pair before either native construction or Oracle attachment.
3. Two real checkpoint layers and 64 pack tokens pass the isolated FP32 reference comparison. Expert choices must agree exactly; layer-1 output max-absolute error must not exceed `1e-4 * reference_max_abs`. A mismatch stops before rental; no threshold changes or retry for PASS.
4. Remote CPU integration passes, including all original PR #61 tests, the new tests, Ruff, CLI refusals and live Hub metadata. Workstation project execution is prohibited.
5. Reviewed code, immutable protocol SHA256, complete upper-bound cost estimate, evidence destination and deletion permissions are recorded before rental.
6. Qualify the separate remote statistics environment with `scipy==1.15.3` and exercise the summary APIs below before freeze. This is the planned pin in [the v2 plan](p3-stage2-v2-plan.md), not an installed-environment receipt. The observed `integration-reference-v2/evidence/packages.log` has no SciPy. Retain the new interpreter/package inventory and API evidence before calling analysis executable.

### Unresolved freeze dependency

The plan requires protocol freeze **before rental** and actual source/cache/initial-adapter identities in that protocol. The production preparation path creates the CUDA NF4 cache and shared initial adapter in stage b, after rental. No admitted production cache or initial adapter exists yet. CPU/tiny artifacts do not satisfy this requirement.

Until this dependency is resolved explicitly, **do not create an instance or volume**. Do not put a recipe hash in place of an artifact hash, insert a placeholder, or rewrite the frozen protocol after preparation. The procedure below is conditional on a valid pre-rental protocol; it does not authorize moving freeze past stage b. See the current [result status](results/p3-moonlight-quality/README.md).

The initial reference admission downloaded and hash-verified 2.006757520 GB, then stopped before either model worker because the declared legacy `inv_freq` shape was wrong. [The 27 range headers](results/p3-moonlight-quality/source-headers-v1.json) show 5344 BF16 tensors and 27 legacy `[64]` buffers (1728 elements); excluding them leaves 15960110208 checkpoint tensor elements, including persistent router biases. After correcting that declaration, the first actual numerical comparison [passed](results/p3-moonlight-quality/reference-admission-v1.json): all ordered top-6 choices matched; max error `2.086162567138672e-7` was below `0.00011039664745330811`. Neither constructor's native nonpersistent `[32]` buffers were overwritten. No numerical comparator retry or tolerance change occurred; shape-incomparable buffer diagnostics are not a parity criterion.

## Budget and manual supervision

One Vast rental only. Include reference checking, GPU time, root storage, local-volume storage, ingress, egress and deletion in the **$15 total**. Compute stops at **$14 accounted/accrued expense**; reserve **$1** for preservation and deletion. Include billing lag conservatively. The process wrapper cannot stop billing or delete provider resources.

Use a fresh on-demand offer and a matching local-volume offer on the same machine. Requirements: Ampere or newer, 24 GB VRAM, effective host/cgroup RAM at least 32 GiB, compatible CUDA-13 driver and at least 150 GB free **physical NVMe**, not overlay. Provider descriptions are not execution proof. Price root and volume separately; `search offers --storage` does not include a separately rented volume. [Offer pricing](https://docs.vast.ai/cli/reference/search-instances) and [volume pricing](https://docs.vast.ai/cli/reference/search-volumes) use the requested storage amount.

Preparation itself binds the target's `st_dev` through `/proc/self/mountinfo` and `/sys/dev/block` to a writable ext4/xfs mount and a direct PCIe NVMe controller with `rotational: 0`. It refuses overlay/network storage, device mapper/RAID, hidden sysfs and ambiguous mounts. It also bounds physical RAM by visible cgroup limits. If the offer cannot expose that evidence for `$SOURCE` and `$CACHE`, stop rather than treating a provider label or throughput number as admission.

If the upper estimate exceeds $15, first reduce the frozen-order HellaSwag prefix, then omit the complete L4096 pair if still necessary. Record each before/after estimate and reduction in the protocol **before hashing**. Never change 64 steps, thresholds, scoring or the decision rule. If the permitted reductions still exceed $15, do not rent.

Record a single absolute UTC compute deadline and stage allowances in the cost ledger. Every command must fit its full allowance, remaining traffic and closeout reserve. Include raw-gate readmission, pairing, setup, evidence transfers and teardown, not just training. `--seconds` bounds the child process, then the helper allows up to ten seconds to reap it; price that cleanup margin too. Evaluation also checks the frozen `loglik.deadline_seconds` from its own start. Inspect provider charges before/after commands and transfers, and at least every 15 minutes during long commands. If supervision, billing visibility or evidence transfer is lost, stop. Do not install a rental controller, watchdog, uploader or automatic deleter.

Never touch unrelated instance `60034978` or any resource not created and recorded for this run.

## Fixed paths and common operations

Production commands in a through f run on the admitted Linux rental, never the workstation. Pre-rental admission and post-pair statistics may run in the recorded remote CPU environment. Use the reviewed source packet, including `src/`, `benchmarks/harness/`, the frozen v2 scoring protocol and its data files. Do not substitute an unreviewed checkout.

The observed environment is Python 3.12.3, Torch distribution `2.11.0`, runtime `2.11.0+cu130`, Transformers 5.19.0, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, safetensors 0.8.0, tokenizers 0.23.3, NumPy 2.1.3, Rich 13.9.4, psutil 7.2.2 and Hugging Face Hub 2.4.0. The full inventory and installation inputs are [packages.log](results/p3-moonlight-quality/prelaunch-review-v1/integration-reference-v2/evidence/packages.log) and [bootstrap.py](results/p3-moonlight-quality/prelaunch-review-v1/integration-reference-v2/evidence/bootstrap.py). They also pin `tiktoken==0.14.0`, `regex==2026.11.9` and `blobfile==3.1.0` for tokenizer conversion/parity. Do not guess newer versions or copy the old pilot's setup. Record the rental image, driver, interpreter, wheel/source hashes and complete installed inventory; keep them unchanged through gate, training and evaluation. The separate SciPy analysis environment remains subject to admission item 6.

Set these operator variables from the frozen cost ledger. `RUN_ID`, allowances and receiver coordinates are operator inputs, not artifact identities. `PROTOCOL_SHA256` must identify an existing admitted file. `protocol-v1.json` below is its required deployment destination, not a file that this runbook has created. The missing production cache/initial hashes still block rental.

```bash
set -u
umask 077
: "${RUN_ID:?Recorded unique run identifier}"
: "${PROTOCOL_SHA256:?Immutable pre-rental SHA256 from the ledger}"
: "${A_SECONDS:?Frozen stage-a allowance}"
: "${B_SECONDS:?Frozen stage-b allowance}"
: "${C_SECONDS:?Frozen stage-c allowance}"
: "${D_SECONDS:?Frozen six-pass gate allowance}"
: "${READMIT_SECONDS:?Frozen raw-evidence readmission allowance}"
: "${PAIR_SECONDS:?Frozen pairing allowance}"
: "${TRAIN_SECONDS:?Frozen allowance for the selected complete 64-step arm}"
: "${EVAL_SECONDS:?Frozen allowance for the selected evaluation shard}"
: "${VIRTUAL_ENV:?Admitted Python 3.12.3 environment, already installed}"
: "${EVIDENCE_SSH_TARGET:?Approved user@host evidence receiver}"
: "${EVIDENCE_REMOTE_DIR:?Existing private absolute receiver directory without spaces}"
: "${DATA_ARTIFACT_ROOT:?Restored admitted data directory containing candidate-pack and generated}"
export PROFILE=moonlight-16b-a3b
export ROOT=/data/p3-moonlight-quality
export REPO="$ROOT/source"
export EVIDENCE="$ROOT/evidence/$RUN_ID"
export EXPORTS="$ROOT/exports"
export SOURCE="$ROOT/sources/476b36a473d4467f94469414bef6cee75c9c8172"
export CACHE="$ROOT/cache/476b36a473d4467f94469414bef6cee75c9c8172"
export TOKENIZER="$ROOT/tokenizer/476b36a473d4467f94469414bef6cee75c9c8172"
export PACK="$ROOT/pack"
export PACK_SHA256=2d142c8f1641bd55173910311744455a79c4be5c9152cf181673d3965247cee4
export PROTOCOL="$REPO/benchmarks/results/p3-moonlight-quality/protocol-v1.json"
export SCORING_PROTOCOL="$REPO/benchmarks/results/p3-stage2-v2/p3-stage2-v2-protocol.json"
export DATA="$REPO/benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=23
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
mkdir -p "$ROOT/sources" "$ROOT/cache" "$ROOT/tokenizer" "$EXPORTS"
mkdir -p "$EVIDENCE/setup"
cd "$REPO"
python --version
python -m pip freeze > "$EVIDENCE/setup/packages.txt"
printf '%s  %s\n' "$PROTOCOL_SHA256" "$PROTOCOL" | sha256sum -c -
printf '%s  %s\n' \
  685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249 "$SCORING_PROTOCOL" \
  07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81 "$DATA/manifest.json" \
  | sha256sum -c -
```

Do not enable shell `errexit` around a stage: inspect its exit code, preserve partial evidence, then stop on technical failure. Do not chain a failed stage into later scientific work. The process helper writes `stdout.log`, `result.json` and `manifest.json` in its `--output` directory. Inspect `result.json.status`, `returncode`, `cleanup.child_reaped`, `cleanup.errors` and `evidence_snapshot_only`. Require COMPLETE, a reaped child, no cleanup errors and no snapshot-only evidence. Exit 124 means timeout; 130 means interruption. A zero wrapper exit is process completion, not scientific admission.

The admitted data artifact's source directory is `prelaunch-review-v1/data-v1/moonlight-preflight-evidence/data/`. Restore its `candidate-pack/` and `generated/476b36a473d4467f94469414bef6cee75c9c8172/` without regeneration, retaining the archive/member verification receipt. After verifying those exported bytes, deploy them once:

```bash
if test -e "$PACK" || test -e "$TOKENIZER"; then
    printf 'Pack/tokenizer destination already exists; stop and inspect.\n' >&2
    exit 1
fi
cp -a "$DATA_ARTIFACT_ROOT/candidate-pack" "$PACK" || exit 1
cp -a "$DATA_ARTIFACT_ROOT/generated/476b36a473d4467f94469414bef6cee75c9c8172" "$TOKENIZER" || exit 1
printf '%s  %s\n' "$PACK_SHA256" "$PACK/manifest.json" | sha256sum -c -
python - "$PACK" "$TOKENIZER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
pack, tokenizer = map(Path, sys.argv[1:])
manifest = json.loads((pack / 'manifest.json').read_bytes())
assert manifest['profile_id'] == 'moonlight-16b-a3b'
assert manifest['test_only'] is False
for descriptor in manifest['files'].values():
    path = pack / descriptor['path']
    assert path.stat().st_size == descriptor['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256']
for name, descriptor in manifest['tokenizer']['files'].items():
    path = tokenizer / name
    assert path.stat().st_size == descriptor['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256']
PY
```

Use this generated fast tokenizer explicitly in evaluation. Omitting `--tokenizer-dir` defaults to `$SOURCE`, which has only the source tiktoken files and does not satisfy this profile. Both source and tokenizer directory basenames must be the pinned revision. All profile CLI defaults are `deepseek-v3`; keep every displayed `--profile`. Preparation, gate, training and evaluation require explicit source/cache paths. Evaluation's default scoring file is the displayed repository v2 protocol, but the commands pass it explicitly. The frozen protocol's original Kaggle `data.dataset_manifest_path` is historical, not the rental data directory. The deployed data root above contains `manifest.json` and `<endpoint>/tasks.jsonl`, with each task path/hash/size under `manifest['endpoints'][endpoint]['files']['tasks']`.

The frozen Moonlight protocol must pass both `train._protocol` and `eval.admit_protocol`; no generation command or specimen protocol is supplied here. Bind these fields from admitted evidence before hashing:

| Fields | Required contract |
| --- | --- |
| `schema`, `status`, `profile_id` | `p3-deepseek-quality-protocol-v1`, `FROZEN`, `moonlight-16b-a3b` |
| `model`, `arms` | Pinned Moonlight model/revision; ordered `['bf16_l2l', 'oracle_block64_e4m3']` |
| `include_base_model_evaluation`, `numerical_gate_blocks_quality` | Both `false` |
| `source`, `cache`, `aux_admission`, `tokenizer` | Complete production session identities, admitted auxiliary binding and pack tokenizer identity, including `profile_id`; not recipe hashes. Stage b emits source/cache under `manifest['session']`. |
| `initial_adapter_sha256`, `pack_sha256` | Shared production adapter file SHA256 and the admitted pack manifest SHA256 |
| `attention_policy` | Exact `p3_deepseek_quality_attention.policy_identity('moonlight-16b-a3b')` object |
| `training` | `seed: 23`, `steps: 64`, `input_tokens_per_step: 4096`, `micro_batch_size: 1`, `lengths: [1024,4096]` or budget-admitted `[1024]`; exact optimizer and scheduler below |
| `training.optimizer` | `{'name':'AdamW','lr':0.0002,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.01,'clip_max_norm':1.0}` |
| `training.scheduler` | `{'name':'cosine','warmup_steps':2,'horizon':64}` |
| `data.manifest_sha256`, `scoring`, `loglik` | Data hash `07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81`; `scoring` equals v2 verbatim; `loglik` has the same keys and values as v2 except positive integer frozen `hellaswag_n`, `max_tokens_per_batch`, `deadline_seconds`. No truncation or scoring changes. |
| `budget` | Integer `limit_usd: 15`, `compute_stop_usd: 14`, `closeout_reserve_usd: 1`, finite `upper_bound_total_usd <= 15`, ordered `adjustments` list |

With no cuts, `adjustments` is empty and HellaSwag N is 10042. A `hellaswag_prefix` adjustment records `before_n: 10042`, integer `after_n`, `upper_bound_before_usd > 15` and non-increasing `upper_bound_after_usd`. Omitting L4096 requires that adjustment first, followed by `omit_L4096_pair`; successive estimates must join exactly, and the final estimate must equal `upper_bound_total_usd`. Include the source packet, software/hardware admission, analysis environment, shard schedule, UTC deadline, complete cost ledger and evidence/deletion destinations in the pre-rental record. Never fill an identity with a placeholder to pass structural validation.

Unmodified v2 `loglik` is `{'add_special_tokens': false, 'deadline_seconds': 2700, 'hellaswag_n': 10042, 'logits_memory_limit_bytes': 536870912, 'logprob_dtype': 'float32', 'max_tokens_per_batch': 8192, 'nll_label_policy': 'p3_stage2_dataset._encode_document', 'padding': 'right', 'prepend_bos': false, 'reduction_chunk_tokens': 128}`. The frozen Moonlight values for the three scheduling fields control execution; this list does not supply a new allowance. Keep `scoring` equal to v2 as a parsed JSON object, including `row_drop: false` and `truncation: false`.

The source config remains untouched. Runtime-normalization receipts must contain each field's original value, `runtime: absent`, and `reason: no numerical consumer in pinned native deepseek_v3; generic config validation read only`.

Record the instance and volume IDs immediately after creation. Use only the evidence-export function and manual teardown procedure from [the old pilot runbook](p3-deepseek-quality-pilot-runbook.md#evidence-export-after-every-stage), not its setup, PASS-only gate, capacities, prices, deadlines, pilot policy or commands. The function below is the same named-path transport, with Moonlight variables:

```bash
export_evidence() {
    local label="$1"
    shift
    test ! -e "$EXPORTS/$label.tar.gz" || return 1
    tar -czf "$EXPORTS/$label.tar.gz" -C "$ROOT" -- "$@" || return 1
    (cd "$EXPORTS" && sha256sum "$label.tar.gz" > "$label.tar.gz.sha256") || return 1
    scp "$EXPORTS/$label.tar.gz" "$EXPORTS/$label.tar.gz.sha256" \
        "$EVIDENCE_SSH_TARGET:$EVIDENCE_REMOTE_DIR/" || return 1
    ssh "$EVIDENCE_SSH_TARGET" \
        "cd '$EVIDENCE_REMOTE_DIR' && sha256sum -c '$label.tar.gz.sha256'" || return 1
}
export_evidence "${RUN_ID}-setup" "evidence/$RUN_ID/setup"
```

Confirm that harmless setup transfer and receiver SHA256 before scientific work. Record transfer bytes, UTC start/end, archive hash, receiver path and traffic charges. Export complete manifests, raw gate tensors, initial/final adapters, every checkpoint, step records and resource receipts after each stage; preserve partial evidence on failure before teardown. Keep archive files outside scientific output directories. Weights/cache payloads need not be exported, but their identities and inventories must be retained. Never prune raw proof tensors to fit the budget.

## a → b: source and cache

Run one block at a time, retain the wrapper exit code and child receipt, export, then decide whether to proceed.

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/a-process" --seconds "$A_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_prepare download \
  --profile "$PROFILE" --source "$SOURCE" \
  --output "$EVIDENCE/a-download" --approved-pilot
stage_rc=$?
printf 'stage_a_exit=%s\n' "$stage_rc"
export_evidence "${RUN_ID}-a" "evidence/$RUN_ID/a-process" "evidence/$RUN_ID/a-download"
```

Require download `a-download/manifest.json.status: PASS`, exactly 27 BF16 shards totalling 31920879456 bytes and five pinned metadata files. This path does not execute Hub model/tokenizer code; it does use the Hub client. Inspect `a-download/hub-inventory.json`, each `<filename>.receipt.json`, `source`, `checkpoint_identity_sha256` and `$SOURCE/checkpoint-identity.json`. Source metadata, shapes, all 26 correction biases, index/header coverage and absence of quantization/scale tensors must be admitted. Check the pinned 27 `[64]` legacy rotary buffers against the range-header evidence, not a `[32]` constructor buffer. A partial transport resume uses `download --resume`, a fresh wrapper/receipt output directory and only the remaining original stage-a allowance; complete files are reverified and partial files discarded. Integrity failures are not retried.

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/b-process" --seconds "$B_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_prepare convert \
  --profile "$PROFILE" --source "$SOURCE" --cache "$CACHE" \
  --output "$EVIDENCE/b-convert" --approved-pilot
stage_rc=$?
printf 'stage_b_exit=%s\n' "$stage_rc"
export INITIAL="$EVIDENCE/b-convert/initial-adapter.safetensors"
export_evidence "${RUN_ID}-b" "evidence/$RUN_ID/b-process" "evidence/$RUN_ID/b-convert"
```

Require `b-convert/manifest.json.status: PASS`, `session_closed: true`, NF4/noDQ/per-expert/disk, BF16 compute and FP32 rank-2/alpha-4 adapters. The manifest records `session.source`, `session.cache`, `session.precision`, `initial_adapter`, `resources.memory` and `resources.cgroup`; the adapter payload is `b-convert/initial-adapter.safetensors`. Compare the production identities with the already frozen protocol, including the adapter's file SHA256; mismatch stops, never updates the protocol. This is one shared initial adapter, not a separately regenerated initializer for each arm. `$CACHE` must not exist before convert, and its parent must already exist on the same admitted `/data` NVMe as source.

## c: technical smoke

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/c-process" --seconds "$C_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_smoke \
  --profile "$PROFILE" --output "$EVIDENCE/c-smoke"
stage_rc=$?
printf 'stage_c_exit=%s\n' "$stage_rc"
export_evidence "${RUN_ID}-c" "evidence/$RUN_ID/c-process" "evidence/$RUN_ID/c-smoke"
```

Require `c-smoke/manifest.json.status: COMPLETE` and clean teardown for every required technical attempt. Smoke probes the selected Moonlight attention dimensions but trains a separate tiny `deepseek-v3` qualification fixture, three steps per arm and length. Its `qualification_only: true`, `test_fixture: true`, `quality_evaluated: false` and `full_model_parity_evaluated: false` must remain explicit. It does not load `$SOURCE`/`$CACHE`, evaluate Moonlight quality or replace historical tiny E3. Do not relax dimensions, native-BF16 admission, routing or attention fallback restrictions.

## d: all six numerical passes, diagnostic only

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/d-process" --seconds "$D_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_gate \
  --profile "$PROFILE" --source "$SOURCE" --cache "$CACHE" \
  --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
  --initial-adapter "$INITIAL" --output "$EVIDENCE/d-gate"
stage_rc=$?
printf 'stage_d_exit=%s\n' "$stage_rc"
export GATE="$EVIDENCE/d-gate/manifest.json"
export GATE_SHA256="$(sha256sum "$GATE" | cut -d ' ' -f 1)"
export_evidence "${RUN_ID}-d" "evidence/$RUN_ID/d-process" "evidence/$RUN_ID/d-gate"
```

Order: GA, mandatory GA repeat, BF16 L2L, independent MATH GA floor, FP8 block64 E4M3 + Oracle L2L, INT4 + Oracle control. All use B1/L64/k8, the same prefix and initial adapter, and no optimizer update. Main attention stays strict EFFICIENT; MATH is floor-only.

For `moonlight-16b-a3b`, the gate CLI returns **0 after `load_gate` verifies complete production raw evidence**, for PASS, FAIL or NO_VERDICT with finite raw tensors. It returns 1 for incomplete/qualification-only execution; a loader exception also fails the process. The 0/PASS, 1/FAIL, 2/NO_VERDICT mapping applies only to the unchanged `deepseek-v3` branch. Do not ignore a Moonlight process failure as a numeric diagnostic. Export the full closure and retain a bounded independent readmission receipt:

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/d-readmit-process" --seconds "$READMIT_SECONDS" -- \
  python - "$GATE" "$GATE_SHA256" <<'PY'
import sys
from pathlib import Path
from rich.console import Console
from benchmarks.harness.p3_deepseek_quality_gate import load_gate
result = load_gate(Path(sys.argv[1]), sys.argv[2], profile_id='moonlight-16b-a3b')
Console().print({'diagnostic_verdict': result['verdict']})
PY
stage_rc=$?
printf 'gate_readmit_exit=%s\n' "$stage_rc"
export_evidence "${RUN_ID}-d-readmit" "evidence/$RUN_ID/d-readmit-process"
```

The loader accepts complete Moonlight FAIL/NO_VERDICT diagnostics with finite raw tensors without changing their computed verdict. Undefined derived ratios remain `null` with exact diagnostic paths independently reconstructed from raw evidence; they are not raw NaN/Inf failures. An accepted numeric FAIL continues to training/evaluation. Retain `d-gate/manifest.json`, `result.json`, `gate-input.json`, all six `<pass>/pass.json` files and every manifest-listed raw tensor/archive file. Missing or corrupt evidence, incomplete passes, raw NaN/Inf, absent Oracle VJP, RoutingGuard refusal or OOM are technical failures. No gate repeat for PASS. Training re-admits this manifest and matches model, cache, initial adapter, pack, execution source and environment; a summary alone is insufficient.

## e: independent full training arms

Run the following block separately for `(bf16_l2l,1024)`, `(oracle_block64_e4m3,1024)`, then the corresponding L4096 pair if retained in the frozen protocol. Each invocation owns a fresh model/session/optimizer and loads the same initial adapter. It never continues from the other arm's final adapter.

```bash
: "${ARM:?Select a frozen arm}"
: "${LENGTH:?Select a retained length}"
: "${TRAIN_SECONDS:?Frozen allowance for this arm and length}"
export TRAIN_OUT="$EVIDENCE/train-$ARM-L$LENGTH"
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/train-$ARM-L$LENGTH-process" --seconds "$TRAIN_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_train \
  --profile "$PROFILE" --source "$SOURCE" --cache "$CACHE" \
  --pack "$PACK" --pack-sha256 "$PACK_SHA256" --initial-adapter "$INITIAL" \
  --arm "$ARM" --length "$LENGTH" --steps 64 --device cuda \
  --protocol "$PROTOCOL" --protocol-sha256 "$PROTOCOL_SHA256" \
  --gate "$GATE" --gate-sha256 "$GATE_SHA256" --output "$TRAIN_OUT"
stage_rc=$?
printf 'train_%s_L%s_exit=%s\n' "$ARM" "$LENGTH" "$stage_rc"
export_evidence "${RUN_ID}-train-$ARM-L$LENGTH" \
  "evidence/$RUN_ID/train-$ARM-L$LENGTH-process" "evidence/$RUN_ID/train-$ARM-L$LENGTH"
```

B1, L1024/k4 or L4096/k1 means 4096 input tokens per update. AdamW lr 2e-4, betas .9/.999, eps 1e-8, weight decay .01, clip 1.0; HF cosine, warmup 2, horizon 64 and LR0 first update. Preserve cyclic pack order and final cutoff. Require `manifest.json.status: COMPLETE` and `completed_steps: 64`. Preserve `RUNNING.json`, `steps.jsonl`, every `checkpoints/step-000001/` through `step-000064/`, `step-1-gradients.safetensors`, `step-1-gradients.json`, `adapter-delta.json` and `final_adapter/manifest.json` with its config/weights. Step rows contain losses, input/target counts, gradient evidence/norms, adapter-before/after hashes, `adapter_changed` and resources. Only step 1 saves raw gradient tensors; `adapter-delta.json` records the final-versus-initial norm, not a per-step delta norm. Finite zero gradients or zero adapter change are diagnostics, not invented stop conditions.

A technical failure invalidates the affected arm and its pair. Do not retry a scientific/technical failure, shorten the run or substitute a partial checkpoint. Only a recorded infrastructure-loss event and a complete admitted checkpoint can use the existing `--resume PRIOR_OUTPUT --infrastructure-event EVENT_JSON` interface with a fresh `--output`. The one-rental, same-execution-identity and original money/deadline limits still apply. A terminal COMPLETE/FAILED prior output cannot resume. Do not add `--pilot-prefreeze`, use `--prepare` to rebuild cache or change the 64-step horizon.

## f: evaluate every final adapter and assemble v2 decisions

Only use `final_adapter` from a COMPLETE 64-step run. Execute each frozen endpoint in contiguous, disjoint shards for every arm and retained length. Full endpoint counts: HellaSwag 10042 (or its pre-frozen prefix), ARC 3548, BoolQ 3270, LAMBADA 5153, GSM8K 1288 and MBPP-test 494. Do not select examples after observing scores.

```bash
: "${ARM:?Select this adapter's frozen arm}"
: "${LENGTH:?Select this adapter's retained length}"
: "${ENDPOINT:?One of hellaswag arc boolq lambada gsm8k mbpp}"
: "${START:?Frozen shard start index}"
: "${STOP:?Frozen shard exclusive stop index}"
: "${EVAL_SECONDS:?Frozen allowance for this endpoint interval}"
export TRAIN_OUT="$EVIDENCE/train-$ARM-L$LENGTH"
export ADAPTER="$TRAIN_OUT/final_adapter"
export ADAPTER_SHA256="$(sha256sum "$ADAPTER/manifest.json" | cut -d ' ' -f 1)"
export EVAL_OUT="$EVIDENCE/eval-$ARM-L$LENGTH-$ENDPOINT-$START-$STOP"
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVAL_OUT-process" --seconds "$EVAL_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_eval run-endpoint \
  --profile "$PROFILE" --data-root "$DATA" --source "$SOURCE" --cache "$CACHE" \
  --tokenizer-dir "$TOKENIZER" --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA256" \
  --protocol "$PROTOCOL" --protocol-sha256 "$PROTOCOL_SHA256" \
  --scoring-protocol "$SCORING_PROTOCOL" --length "$LENGTH" --device cuda \
  --endpoint "$ENDPOINT" --start "$START" --stop "$STOP" --output "$EVAL_OUT"
stage_rc=$?
printf 'eval_%s_L%s_%s_%s_%s_exit=%s\n' "$ARM" "$LENGTH" "$ENDPOINT" "$START" "$STOP" "$stage_rc"
export_evidence "${RUN_ID}-eval-$ARM-L$LENGTH-$ENDPOINT-$START-$STOP" \
  "evidence/$RUN_ID/eval-$ARM-L$LENGTH-$ENDPOINT-$START-$STOP-process" \
  "evidence/$RUN_ID/eval-$ARM-L$LENGTH-$ENDPOINT-$START-$STOP"
```

Scoring is unchanged v2: no BOS, `add_special_tokens=False`, FP32 log probabilities and frozen prompt/choice/target rules. Require each shard's `manifest.json.status: COMPLETE`, `selected_shard_complete: true` and matching `binding.start`, `binding.stop`, `row_count`, `arm`, `length` and adapter identity. Its payload files are `rows.jsonl` and `batches.jsonl`. Both trained methods are evaluated through the same BF16 streamed inference path, with `fp8: false` and `oracle: false` in evaluation receipts. GSM8K and MBPP are teacher-forced secondary NLL only, not generation/pass@k. No base checkpoint is admissible. Export every shard before proceeding.

Populate shell arrays `BF16_SHARDS` and `FP8_SHARDS` with only the completed shard directories for all six endpoints and every retained length, then:

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
  --output "$EVIDENCE/pair-process" --seconds "$PAIR_SECONDS" -- \
  python -m benchmarks.harness.p3_deepseek_quality_eval pair \
  --profile "$PROFILE" --baseline "${BF16_SHARDS[@]}" --candidate "${FP8_SHARDS[@]}" \
  --output "$EVIDENCE/paired" --data-root "$DATA" \
  --protocol "$PROTOCOL" --protocol-sha256 "$PROTOCOL_SHA256" \
  --scoring-protocol "$SCORING_PROTOCOL"
stage_rc=$?
printf 'pair_exit=%s\n' "$stage_rc"
export_evidence "${RUN_ID}-paired" "evidence/$RUN_ID/pair-process" "evidence/$RUN_ID/paired"
```

Require `paired/manifest.json.status: COMPLETE`. Missing/duplicate/reordered tasks, mixed adapters, changed policies or incomplete pairs block the decision; never report a subset as the frozen study. Pair writes `pairs.jsonl`, `nll-pairs.jsonl` and `manifest.json`. The manifest supplies `expected_tasks`, `expected_secondary_tasks`, `lengths`, `seeds: [23]`, `primary_rows`, `secondary_rows`, `input_manifests`, `files` and `manifest_payload_sha256`. Its `expected_tasks` is primary-only; secondary task IDs come from `expected_secondary_tasks`.

On the separately admitted remote CPU analysis environment, restore the exported paired directory and scoring file with byte verification, use the reviewed source on `PYTHONPATH`, and set `$EVIDENCE`/`$SCORING_PROTOCOL` to those restored paths. Do not assume the rental paths exist on that host. Require the recorded Python/package inventory with `scipy==1.15.3`; that pin remains unqualified until remote evidence is retained. The frozen chance mapping is top-level `scoring['chance']`, not `scoring['data']['chance']`: ARC `0.2501597143930853`, BoolQ `0.6217125382262997`, HellaSwag `.25`, LAMBADA `0.0`. Pass the recorded Moonlight protocol hash too, so the paired evidence cannot come from a different study:

```bash
: "${PROTOCOL_SHA256:?Recorded immutable Moonlight protocol hash}"
python - "$EVIDENCE/paired" "$SCORING_PROTOCOL" "$PROTOCOL_SHA256" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from rich.console import Console
from benchmarks.harness.p3_stage2_statistics import summarize_pairs
from benchmarks.harness.p3_stage2v2_decision import apply_informativeness_gate
from benchmarks.harness.p3_stage2v2_results import ARMS, summarize_nll
root, scoring_path = map(Path, sys.argv[1:3])
protocol_sha256 = sys.argv[3]
scoring_bytes = scoring_path.read_bytes()
assert hashlib.sha256(scoring_bytes).hexdigest() == (
    '685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249')
scoring = json.loads(scoring_bytes)
manifest_path = root / 'manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
assert manifest['status'] == 'COMPLETE'
assert manifest['profile_id'] == 'moonlight-16b-a3b'
assert manifest['include_base_model_evaluation'] is False
assert manifest['seeds'] == [23]
assert manifest['protocol_sha256'] == protocol_sha256
assert manifest['scoring_protocol']['sha256'] == hashlib.sha256(scoring_bytes).hexdigest()
payload = {key: value for key, value in manifest.items() if key != 'manifest_payload_sha256'}
canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False).encode()
assert hashlib.sha256(canonical).hexdigest() == manifest['manifest_payload_sha256']
assert set(manifest['files']) == {'pairs.jsonl', 'nll-pairs.jsonl'}
root_real = os.path.realpath(root)
for name, descriptor in manifest['files'].items():
    path = root / name
    assert os.path.commonpath((root_real, os.path.realpath(path))) == root_real
    assert path.stat().st_size == descriptor['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256']
assert (scoring['margin'], scoring['one_sided_alpha'], scoring['floor_offset']) == (0.01, 0.025, 0.05)
rows = [json.loads(line) for line in (root / 'pairs.jsonl').read_text().splitlines()]
assert len(rows) == manifest['primary_rows']
raw = summarize_pairs(rows, expected_tasks=manifest['expected_tasks'],
                      seeds=manifest['seeds'], lengths=manifest['lengths'],
                      margin=scoring['margin'], one_sided_alpha=scoring['one_sided_alpha'])
quality = apply_informativeness_gate(raw, chance=scoring['chance'],
                                    floor_offset=scoring['floor_offset'])
nll_rows = []
assert ARMS == ('bf16_l2l', 'oracle_block64_e4m3')
paired_nll = (root / 'nll-pairs.jsonl').read_text().splitlines()
assert all(row['profile_id'] == 'moonlight-16b-a3b' for row in rows)
assert len(paired_nll) == manifest['secondary_rows']
for line in paired_nll:
    row = json.loads(line)
    assert row['profile_id'] == 'moonlight-16b-a3b'
    for label, arm in zip(('baseline', 'candidate'), ARMS, strict=True):
        nll_rows.append({
            'endpoint': row['benchmark'], 'task_id': row['task_id'],
            'length': row['length'], 'seed': row['seed'], 'arm': arm,
            'n_targets': row['n_targets'], 'nll_sum': row[label + '_nll_sum'],
            'nll_mean': row[label + '_nll_mean'],
        })
secondary = summarize_nll(nll_rows, expected_tasks=manifest['expected_secondary_tasks'],
                          seeds=manifest['seeds'], lengths=manifest['lengths'])
assert all(cell['status'] == 'COMPLETE' for cell in secondary)
result = {'quality': quality, 'secondary_nll': secondary,
          'gated_quality_decision': quality['decision'],
          'statistical_quality_decision': quality['statistical_decision'],
          'include_base_model_evaluation': False,
          'pair_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
          'scoring_protocol_sha256': hashlib.sha256(scoring_bytes).hexdigest()}
with (root.parent / 'quality-summary.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
Console().print({'gated': quality['decision'], 'statistical': quality['statistical_decision']})
PY
```

This writes `$EVIDENCE/quality-summary.json` once, never overwriting an earlier summary. Preserve its hash and the analysis environment receipt with the paired evidence. `quality['decision']` is the gated verdict; `quality['statistical_decision']` retains the paired statistical verdict. Other v2 fields include `floor_cells`, per-cell `informative`, `coverage.complete`, `observed_margin_breaches`, `pointwise_bounds` and `simultaneous_bounds`. The primary rule is unchanged: NONINFERIOR requires every pointwise upper bound strictly below .01; EXCESSIVE_DEGRADATION requires at least one union-bound-adjusted lower bound strictly above .01. Incomplete coverage blocks; excessive degradation takes precedence over the floor veto. Otherwise any BF16 accuracy strictly below chance + .05 yields INCONCLUSIVE_UNINFORMATIVE_FLOOR. Secondary NLL is descriptive and cannot override the primary verdict. The snippet converts each paired NLL row into the two per-arm `endpoint` rows required by `summarize_nll`, with candidate-minus-BF16 units. No future-seed or 671B generalization is supported by this single-seed rank-2 study.

## Closeout, including failure or budget stop

1. Stop the active bounded command and verify child cleanup. Preserve complete and partial receipts, losses/checkpoints, raw gate closure and resource observations. Verify exported archive hashes on the control machine before deletion.
2. Record provider charges and conservative unbilled storage/traffic. Delete **only this run's instance ID**, retain the credentials-free response and timestamp, and confirm absence in a fresh instance listing.
3. Delete **only this run's volume ID** and confirm absence in a fresh volume listing. Stop/detach is not deletion. Retain no paid resource by default.
4. Reconcile final charges, or record billing lag and the required later reconciliation. Never raise $15 or consume the reserved $1 for more computation.
5. Report separate rows for the gated quality verdict, statistical quality verdict and diagnostic gate A3/E1/E2/E3/INT4 result. Include per-step loss, gradient norms and adapter-change flags/hashes; final-versus-initial adapter norm; recorded pass/step/evaluation resources; all-in cost and deletion/absence proof; unchanged tiny FAIL; **DeepSeek-V3 671B: отложен**. Report incomplete coverage as incomplete, not noninferiority or degradation.
