# DeepSeek-V3 671B diagnostic pilot runbook

## Authority and immutable results

This is the operator procedure for the approved DeepSeek-V3 671B pilot. On
2026-11-11 the user approved running it on the full 671B checkpoint. This PR
itself performs no rental, model download, native qualification or full-model
execution. Before creating resources, obtain team-lead approval of the complete
PR and the user's confirmation of a fresh selected offer, its total price,
spending cap and elapsed deadline. The separate small Ampere smoke approval does
not cover this pilot. `--approved-pilot` is only an operator acknowledgement. It
cannot grant permission or prove human approval.

The proposal in `results/p3-deepseek-quality/pilot-cost-proposal-v1/` was quoted
at 2026-11-10T21:19:04.112718+00:00. It proposes USD 90 and 96 hours from the
first paid resource, with computation stopped by USD 87 or hour 95. Reserve USD 3
and one hour for evidence and teardown. A quote is not a booking promise. Refresh
availability, machine/volume pairing, rates, traffic terms and capacity before
confirming the price. A missing or changed offer requires a new decision, not a
silent substitution.

The historical tiny E3 is permanently **FAIL: 0.05028628354277727 > 0.05**. Do not
rerun its gate, recalculate its E3, alter its fixture or replace its evidence.
Stage c is a separate technical smoke, not a new scientific tiny verdict. Report
any full 671B verdict on a separate row.

Read this together with `p3-deepseek-quality-prelaunch-review.md` and
`p3-deepseek-quality-plan.md`. The cost proposal preserves the quote-time state of
the code. The commands below use the new fail-fast gate and 5400-second timing
contract. Do not execute the older plan's tiny-gate rerun instructions or its
historical base-model quality-evaluation language.

## Manual provider admission and accounting

Use the authenticated provider console or direct Vast API requests from the
operator's control machine. There is no provider controller, scheduler, rental
watchdog, automatic uploader or automatic deletion in this procedure. Do not
install or build one. The process wrapper only bounds a child process group.

Before creating anything, record in a private, access-controlled operator ledger:

- Team-lead PR URL, reviewed commit, approval time and any explicitly outstanding
  native checks. Remote CPU evidence is not proof of native CUDA kernels.
- User's approval message and time, selected GPU offer, machine and local-volume
  offer, image digest, requested 64 GiB root and 1400 GiB local volume, all rates,
  traffic terms, total cap and deadline. Capture credentials-free quote fields.
- Evidence destination, SSH host-key verification and a successful harmless
  transfer to that destination. Check operator permissions to delete both assets.
- First billable UTC time, provider instance ID and volume ID immediately after
  creation. Never touch instance `60034978` or any unrelated resource.

The quote's candidate combinations are RTX 3090 offer `58275924`, machine
`139376`, volume offer `58685188`, and RTX 4090 offer `58067853`, machine `155483`,
volume offer `58067860`. Quoted GPU+root+volume rates were USD 0.769037/h and
USD 0.865926/h, respectively, excluding traffic. The proposed all-in hourly ceiling
is USD 0.87/h. These are historical quote facts, not current offers or measured
performance. Verify the mounted physical storage, not just the offer description.

Track compute, root storage, local-volume storage, ingress, egress and any other
billable charges separately. At creation, before and after every bounded command,
at least every 15 minutes during long commands and after every upload, record
elapsed hours, metered bytes, provider charges and conservative accrued-but-unbilled
charges. Include all paid setup time. Before starting a command, reserve its full
remaining wall allowance, likely transfer cost and the teardown reserve. Shorten
the command allowance if
needed, never the scientific workload. Do not start if the reserve cannot fit.
A responsible operator must remain available to stop computation at USD 87 or
hour 95 even if a stage deadline has not expired. The wrapper cannot see billing.
If supervision, billing visibility or the upload destination is lost, stop.

## Fixed paths and source package

All following shell blocks are for the approved Linux rental, unless marked
control machine. Nothing here is a workstation project-execution instruction.
The source package must be the exact team-lead-reviewed commit, including this
runbook and the new smoke module. Do not copy a dirty checkout or fetch code
mid-pilot. Use a new run ID and fresh output directories. No overwrites or retries.

On the control machine, package only the reviewed commit and record its identity:

```bash
: "${REVIEWED_COMMIT:?Set the full team-lead-approved Git commit}"
git archive --format=tar.gz --output=p3-pilot-source.tar.gz "$REVIEWED_COMMIT"
printf '%s\n' "$REVIEWED_COMMIT" > p3-pilot-source.commit
sha256sum p3-pilot-source.tar.gz > p3-pilot-source.tar.gz.sha256
```

Transfer those three files with the operator's existing SSH/SCP connection. Pin
the approved image by digest in the ledger. Use Python 3.12.3 on Linux x86_64 and
a driver compatible with CUDA 13.0, matching the qualified environment's Python
and CUDA family. A driver/image change is a recorded admission change, not an
excuse to accept unsupported kernels.

On the rental, first verify the received archive against the control-machine
SHA256 and commit record. Then use these paths for the entire pilot:

```bash
set -u
umask 077
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-pilot-v1"
export ROOT=/root/p3-pilot
export REPO="$ROOT/source"
export EVIDENCE="$ROOT/evidence/$RUN_ID"
export EXPORTS="$ROOT/exports/$RUN_ID"
export REV=e815299b0bcbac849fa540c768ef21845365c9eb
export SOURCE="/data/sources/$REV"
export CACHE=/data/nf4-noDQ-per_expert-v1
export PACK="$ROOT/inputs/training-pack-v1"
export DEV_DATA="$REPO/benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data"
export SOURCES_JSONL="$REPO/benchmarks/results/p3-stage2-quality/admission-kaggle-v3/evidence/dataset/training-sources.jsonl"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=23
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT" "$EVIDENCE/setup" "$EXPORTS" "$ROOT/inputs"
test ! -e "$REPO" && mkdir "$REPO"
sha256sum -c p3-pilot-source.tar.gz.sha256
tar -xzf p3-pilot-source.tar.gz -C "$REPO"
cp p3-pilot-source.commit p3-pilot-source.tar.gz.sha256 "$EVIDENCE/setup/"
cd "$REPO"
```

Keep source, cache and receipts in separate directories. Source and cache must
remain on the same admitted `/data` physical NVMe. Do not relocate, touch or
retimestamp weights after conversion. Source/cache admission includes metadata
and source fingerprints; a cache-only copy cannot reproduce that identity.

### Reproduce the qualified package set

The recorded full package list is
`benchmarks/results/p3-deepseek-quality/efficient-vast-v1/setup/pip-freeze.txt`.
It includes torch 2.11.0+cu130 at runtime, transformers 5.19.0, peft 0.21.2,
bitsandbytes 0.50.2, accelerate 1.15.0, safetensors 0.8.0 and tokenizers 0.23.3.
The freeze's `torch==2.11.0` omits the CUDA local-version suffix. Install the CUDA
wheel explicitly. Its historical editable `/root/p3-efficient-smoke/source` entry
must not install the old checkout. Build a requirements copy excluding those two
entries, then install this source without dependency resolution:

```bash
python3.12 -m venv "$ROOT/env"
. "$ROOT/env/bin/activate"
python -m pip install pip==26.3 setuptools==81.0.0
python - "$REPO" "$EVIDENCE/setup/requirements-pinned.txt" <<'PY'
import sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
freeze = root / 'benchmarks/results/p3-deepseek-quality/efficient-vast-v1/setup/pip-freeze.txt'
rows = [line for line in freeze.read_text().splitlines()
        if line and not line.startswith('-e ') and not line.startswith('torch==')]
with output.open('x') as stream:
    stream.write('\n'.join(rows) + '\n')
PY
python -m pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$EVIDENCE/setup/requirements-pinned.txt"
python -m pip install --no-deps -e "$REPO"
python -m pip check
python -m pip freeze > "$EVIDENCE/setup/pip-freeze.txt"
python -VV > "$EVIDENCE/setup/python.txt"
nvidia-smi -q > "$EVIDENCE/setup/nvidia-smi.txt"
uname -a > "$EVIDENCE/setup/uname.txt"
```

Retain installation stdout/stderr in the private operator terminal transcript.
If pins cannot be installed or `pip check` refuses, stop and report. Do not
silently upgrade dependencies. The old package list is a reproducibility input,
not evidence that this revision has executed. Record the new image, driver,
Python, freeze and source archive hashes. Keep the environment unchanged from
gate through e/f. No tests, lint or tiny scientific gate run as part of stage c.

## Evidence export after every stage

Configure an already-approved evidence SSH account and an existing private
absolute directory on the control machine. These are operator connection values,
not repository secrets. Do not include tokens, private keys or raw account API
responses in evidence or Git. Set these connection variables on the control
machine before rental, then pass them to the rental's operator shell:

```bash
: "${EVIDENCE_SSH_TARGET:?Set the approved user@host evidence receiver}"
: "${EVIDENCE_REMOTE_DIR:?Set its existing absolute directory without spaces}"
```

Use ordinary `tar`, `scp`, `ssh` and SHA256 verification, not a nonexistent
project upload CLI. The following shell function packages only named evidence
paths under `$ROOT`. Each label is used once. Run it after setup, a, b, pack, c,
d, gate readmission, each of the four e steps and f, also on failure before
teardown. If a receipt never finalized, include its partial files and wrapper
logs. The archive and its hash live outside all scientific output directories.

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

Record the remote path, bytes, SHA256, transfer start/end, verification result and
metered egress in the ledger. Use an operator transcript for this export, not the
gate directory. A summary JSON or README cannot replace raw evidence. Preserve
all manifest-listed files with unchanged relative paths. Gate raw gradients and
boundary tensors can be large. The proposal's 6/10 GB egress allowance is
[INFERENCE], not a measured gate-archive size. Measure `du -sb` and the archive
size, allow time and traffic charges for the full raw closure, and upload completed
pass/partial evidence during a long stage if needed. A live snapshot is partial
and non-admitting; only a later finalized complete directory can be re-admitted.
If export cannot fit the remaining allowance, stop computation early and use the
reserve for preservation and deletion. Do not discard raw tensors to fit a quote.

Weights and the NF4 cache are not evidence uploads. Upload their receipts,
inventories, fingerprints and hashes, not roughly 1 TB of regenerable payload.
Do upload the shared initial adapter, training pack, all scientific receipts and
raw gate artifacts. Preserve final checkpoint/adapter files from the one-step
pilot as resource evidence; they are not approved phase 3 trained models.

## Bounded stage allocations

Run a, b, c, d in order. Only full gate PASS admits e and f. Inspect the child
receipt after every command; wrapper exit 0 means process completion only.
Wrapper exit 124 means deadline, 130 interruption, 1 failure, 0 successful child
exit. The gate itself returns 0 PASS, 1 FAIL, 2 NO_VERDICT; the wrapper maps both
nonzero gate exits to 1 and preserves the child return code in `result.json`.
A timeout can leave only partial scientific evidence. Never synthesize a manifest
or call a partial receipt PASS.

Execute one stage block at a time. Record the wrapper's exit code immediately
after its invocation, before the export command replaces `$?`. Export evidence
even after failure, then inspect the child receipt and make the manual admission
decision. A later shell block is never authorization to advance after refusal.
These commands deliberately provide no automatic stage scheduler. Confirm each
export remotely before starting the next stage.

| Allocation | Hard process allowance | Admission before moving on |
| --- | ---: | --- |
| Setup, hardware admission, pack, uploads and accounting | 3 h total operator allowance | All fixed prerequisites and upload confirmed |
| a, download | 18900 s, 5.25 h | Download PASS, complete pinned source |
| b, conversion | 9000 s, 2.5 h | Conversion PASS and shared initial adapter |
| c, technical smoke | 1800 s, 0.5 h | Smoke COMPLETE, never tiny E3 |
| d, full gate | 108000 s, 30 h | Full six-pass COMPLETE/PASS |
| d, raw gate readmission | 1800 s, 0.5 h | `load_gate` accepts raw closure |
| e, each L1024 arm | 36000 s, 10 h | One COMPLETE step at B1/k4 |
| e, each L4096 arm | 50400 s, 14 h | One COMPLETE step at B1/k1 |
| f, resource timing | 5400 s, 1.5 h | COMPLETE resource sample only |

These chosen maxima total 91.25 h including the 3 h operator allowance, leaving
3.75 h before the hour-95 computation stop and one further hour to the absolute
96 h deadline. They are limits, not predictions. Setup and uploads consume elapsed
rental time even outside a child process. The original quote's a-f upper scenario
was [INFERENCE] 87.5 h for the 3090 and [INFERENCE] 89.25 h for the 4090, not a
promise that these allocations or a full pilot will finish. Every invocation must
also fit the live USD 87/hour-95 stop. Do not extend a deadline after seeing a
failure. The wrapper kills the original process group, not the rental or its bill.
Its commands must not detach children into new sessions. Cleanup errors/unreaped
children are failed, potentially still-changing snapshots, not confirmed termination.

## Before a, prove the rented machine

Save `findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS`, `df -B1 /data`,
`lsblk -b -o NAME,TYPE,SIZE,ROTA,TRAN,MODEL,MOUNTPOINTS`, relevant
`/proc/self/mountinfo` and NVMe sysfs ancestry, `/proc/meminfo`, cgroup memory
limits/peaks, GPU/driver/BF16 evidence and network observations in setup evidence.
The prepare helper independently refuses overlay/non-NVMe storage. Physical
NVMe ancestry must be provable, with at least 1,400,000,000,000 free bytes before
download. A 1400 GiB order is not proof of free space. Effective host/cgroup RAM
must be at least 64 GiB. Conversion checks its remaining cache-space budget too.

Before downloading, run the existing strict attention API on native BF16 MLA
shapes. This is an attention execution check, not a tiny gradient gate or an E3
measurement. It must record executed forward/backward kernels without fallback.
The shapes below keep B1/H128/QK192/V128 and check both long-step lengths. No
model weights are loaded. Run it within the setup allowance:

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/preflight-process" --seconds 300 -- \
    python - "$EVIDENCE/setup/attention-admission.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from rich.console import Console
from benchmarks.harness.p3_deepseek_quality_attention import (
    MAIN_BACKEND, attention_execution, require_execution_evidence,
)
rows = []
for length in (1024, 4096):
    with attention_execution(MAIN_BACKEND, device='cuda') as observation:
        q = torch.randn(1, 128, length, 192, device='cuda', dtype=torch.bfloat16,
                        requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn(1, 128, length, 128, device='cuda', dtype=torch.bfloat16,
                        requires_grad=True)
        F.scaled_dot_product_attention(q, k, v, is_causal=True).float().sum().backward()
        torch.cuda.synchronize()
    row = observation.report()
    require_execution_evidence(row, backend=MAIN_BACKEND, dtype='bfloat16', backward=True)
    rows.append(row)
    del q, k, v
with Path(sys.argv[1]).open('x') as stream:
    json.dump({'scope': 'attention_only_not_scientific_gate', 'observations': rows}, stream,
              indent=2, allow_nan=False)
Console().print('Strict MLA attention admission complete')
PY
export_evidence "${RUN_ID}-preflight" "evidence/$RUN_ID/setup" \
    "evidence/$RUN_ID/preflight-process"
```

If admission fails, preserve the wrapper log and hardware evidence and delete the
rental and volume. Do not download weights to a suspect mount, use CPU/T4, enable
MATH fallback for main passes or try a smaller shape for PASS.

## a, download the exact fresh source

The helper requires a nonexistent source directory whose basename is the full
revision. It verifies pinned Hub metadata, HEAD commit/ETag/size, all 163 weight
shards totalling 688586727753 bytes, source configuration and tokenizer metadata.
Metadata files are additional to the weight count/byte total. The reviewed
`benchmarks/results/p3-deepseek-quality/source-inventory-v1.json` pins the Hub
LFS SHA256 values and metadata Git blobs. The four required metadata files are
`config.json`, `model.safetensors.index.json`, `tokenizer.json` and
`tokenizer_config.json`. This revision has no `generation_config.json`; do not
invent or copy one from another revision.

```bash
mkdir -p /data/sources
test ! -e "$SOURCE" && test ! -e "$CACHE"
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/a-process" --seconds 18900 -- \
    python -m benchmarks.harness.p3_deepseek_quality_prepare download \
    --source "$SOURCE" --output "$EVIDENCE/a-download" --approved-pilot
export_evidence "${RUN_ID}-a" "evidence/$RUN_ID/a-process" "evidence/$RUN_ID/a-download"
```

Require child `manifest.json` status PASS and all per-file receipts. Preserve
`hub-inventory.json`, per-file hashes/ETags and `checkpoint-identity.json` identity
via the receipt. Do not reuse a partial download, substitute R1 or change revision.
A failed attempt ends this pilot.

## b, convert once and use its initial adapter

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/b-process" --seconds 9000 -- \
    python -m benchmarks.harness.p3_deepseek_quality_prepare convert \
    --source "$SOURCE" --cache "$CACHE" --output "$EVIDENCE/b-convert" --approved-pilot
export INITIAL="$EVIDENCE/b-convert/initial-adapter.safetensors"
export INITIAL_SHA256="$(sha256sum "$INITIAL" | cut -d ' ' -f 1)"
export_evidence "${RUN_ID}-b" "evidence/$RUN_ID/b-process" "evidence/$RUN_ID/b-convert"
```

Require conversion manifest PASS, `session_closed: true`, eight shard threads,
NF4, `double_quant: false`, `expert_layout: per_expert`, BF16 compute, disk tier,
source/cache fingerprints and the initial adapter's matching file SHA256. Seed 23
writes this FP32 rank-2 attention LoRA initial file once. Every gate/arm loads this
same file. Never regenerate it with the separate training initial-adapter command.
The measured B6 cache size 380276447824 bytes and 2568.94-second conversion are
historical references from another host, not this stage's result.

### Build the paired data pack once

Build once from the frozen 7801-row JSONL and the source's pinned tokenizer. Do
not retokenize per arm. Frozen source SHA256 is
`8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7`.
The tokenizer JSON SHA256 is
`621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41` and its
config SHA256 is
`637bcd1a08cf7c772ce6a383196b22930921c79c3c73223c5afc0c7f41545546`.
The existing loader enforces these bytes and the revision basename.

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/pack-process" --seconds 1800 -- \
    python -m benchmarks.harness.p3_deepseek_quality_data \
    --sources "$SOURCES_JSONL" \
    --sources-sha256 8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7 \
    --tokenizer-dir "$SOURCE" --output "$PACK"
export PACK_SHA256="$(sha256sum "$PACK/manifest.json" | cut -d ' ' -f 1)"
printf '%s  %s\n' c7809f47dd44b1f0cb67a01db63406c381872013dc5bc8df58aedc83d68b1936 \
    "$PACK/manifest.json" | sha256sum -c -
printf '%s  %s\n' 07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81 \
    "$DEV_DATA/manifest.json" | sha256sum -c -
export_evidence "${RUN_ID}-pack" "evidence/$RUN_ID/pack-process" "inputs/training-pack-v1"
```

Require pack manifest SHA256
`c7809f47dd44b1f0cb67a01db63406c381872013dc5bc8df58aedc83d68b1936`, the published
pack identity bound by the attention protocol. A mismatch is refusal, not permission
to change that immutable protocol. Require dev manifest SHA256
`07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81`.
The pack uses fixed-order cycling to 131072 tokens per domain, no BOS, one native
EOS per document, response-only labels and common 1024-token masks. Both lengths
have 64 windows with 4096 input tokens per window. Record supervised target counts
beside the Granite 179326-per-arm reference and cycle/final-cutoff identities.
Do not treat repeated MBPP source cycles as additional unique data.

## c, separate technical smoke on this machine

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/c-process" --seconds 1800 -- \
    python -m benchmarks.harness.p3_deepseek_quality_smoke --output "$EVIDENCE/c-smoke"
export_evidence "${RUN_ID}-c" "evidence/$RUN_ID/c-process" "evidence/$RUN_ID/c-smoke"
```

Inspect the smoke's top-level `manifest.json`, `RUNNING.json` and per-attempt
`receipts/*.json`; failure also writes `FAILED.json`. Require top-level COMPLETE
and all required attempts COMPLETE. The smoke has only `--output`; do not invent
device, threshold or gate flags. It runs synthetic three-step fixture execution
for both arms and lengths plus resource-only forward checks. Those fixture steps
are distinct from e's one-step full-model measurements. It does not publish a
scientific PASS/E3 or production fit claim. Never call the tiny gate API, its E3
calculation or tiny-gate tests here. Preserve the immutable historical tiny FAIL
unchanged.

## d, full 671B gate, with immediate refusal

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/d-process" --seconds 108000 -- \
    python -m benchmarks.harness.p3_deepseek_quality_gate \
    --source "$SOURCE" --cache "$CACHE" --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
    --initial-adapter "$INITIAL" --output "$EVIDENCE/d-gate"
export_evidence "${RUN_ID}-d" "evidence/$RUN_ID/d-process" "evidence/$RUN_ID/d-gate"
```

Three primary arms do not mean three execution passes. A successful production
gate has all six, each at B1/L64/k8 from the same initial adapter, no optimizer step:

1. Ordinary streamed GA, strict EFFICIENT_ATTENTION.
2. GA repeat, required noise observation, not a retry for PASS.
3. BF16 L2L, strict EFFICIENT_ATTENTION.
4. Independent MATH GA floor on eight 64-token microbatches.
5. FP8 block64 E4M3 plus Oracle L2L, strict EFFICIENT_ATTENTION.
6. INT4 plus Oracle negative control, strict EFFICIENT_ATTENTION.

Judge A3 immediately after pass 4. FAIL or NO_VERDICT stops before FP8. Judge
candidate E1/E2/E3 immediately after pass 5. Any failed/missing/nonfinite metric or
forward/Oracle refusal stops before INT4. No later control runs after a known
candidate refusal. On the successful path the INT4 control must discriminate by
failing E1, E2 and E3. That expected negative-control failure is not candidate
failure. An unavailable or nondiscriminating control gives no admission.

Keep the original A3 floor rule. E1 RMS <=0.05 and worst row <=0.25, E2 recompute
RMS <=0.10, E3 global relative L2 <=0.05 and minimum per-tensor cosine >=0.99 are
unchanged. Oracle-forced routing is not measured natural route stability. Main
and long-step attention must observe H128/QK192/V128, native BF16, strict
EFFICIENT and deterministic algorithms with `warn_only=False`, no fallback.

Require `status: COMPLETE`, `verdict: PASS`, `qualification_only: false`, all six
passes and complete raw evidence. Early `execution-start.json`, `session-open.json`
and pass observations are non-admitting partial receipts. Do not add upload logs
or annotations inside `d-gate` after its manifest is written. Raw inventory equality
is checked during admission.

### Re-admit the saved full gate without rerunning it

There is no `validate` or `admit` gate CLI subcommand. Use the existing Python
`load_gate` API below, which verifies hashes/inventory and recomputes metrics from
saved raw tensors. It does not execute a new model gate. Keep the raw directory
intact and the same source checkout. Run only after the full gate reported PASS:

```bash
export GATE="$EVIDENCE/d-gate/manifest.json"
export GATE_SHA256="$(sha256sum "$GATE" | cut -d ' ' -f 1)"
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/d-readmit-process" --seconds 1800 -- \
    python - "$GATE" "$GATE_SHA256" <<'PY'
import sys
from pathlib import Path
from rich.console import Console
from benchmarks.harness.p3_deepseek_quality_gate import load_gate
value = load_gate(Path(sys.argv[1]), sys.argv[2])
Console().print({'status': value['status'], 'verdict': value['verdict'],
                 'scope': 'saved full-production gate raw-evidence admission'})
PY
export_evidence "${RUN_ID}-d-readmit" "evidence/$RUN_ID/d-readmit-process"
```

Require API success. e and f independently re-admit this same raw gate and match
live source, cache, initial adapter, attention policy and software/CUDA/GPU-class
identity. e also binds the gate prefix to the admitted training pack. Physical
instance continuity is checked against the operator ledger; these fields are not
a GPU/host UUID attestation. Do not move to another machine. A passed summary
alone, changed environment or missing raw archive is not admissible. Refusal here
ends the pilot, without a fresh gate attempt.

## e, exactly one resource step per arm and length

Execute each block separately with a manual accounting/admission/upload checkpoint
between them. Do not paste the four commands into an unattended batch. Fixed B1,
N4096, seed23, strict EFFICIENT, both arms BF16 compute. There is no OOM resizing.
`--pilot-prefreeze` permits exactly one step and no frozen training protocol.

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/e-L1024-bf16-process" --seconds 36000 -- \
    python -m benchmarks.harness.p3_deepseek_quality_train \
    --source "$SOURCE" --cache "$CACHE" --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
    --initial-adapter "$INITIAL" --gate "$GATE" --gate-sha256 "$GATE_SHA256" \
    --arm bf16_l2l --length 1024 --steps 1 --pilot-prefreeze \
    --output "$EVIDENCE/e-L1024-bf16"
export_evidence "${RUN_ID}-e-L1024-bf16" "evidence/$RUN_ID/e-L1024-bf16-process" \
    "evidence/$RUN_ID/e-L1024-bf16"
```

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/e-L1024-fp8-process" --seconds 36000 -- \
    python -m benchmarks.harness.p3_deepseek_quality_train \
    --source "$SOURCE" --cache "$CACHE" --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
    --initial-adapter "$INITIAL" --gate "$GATE" --gate-sha256 "$GATE_SHA256" \
    --arm oracle_block64_e4m3 --length 1024 --steps 1 --pilot-prefreeze \
    --output "$EVIDENCE/e-L1024-fp8"
export_evidence "${RUN_ID}-e-L1024-fp8" "evidence/$RUN_ID/e-L1024-fp8-process" \
    "evidence/$RUN_ID/e-L1024-fp8"
```

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/e-L4096-bf16-process" --seconds 50400 -- \
    python -m benchmarks.harness.p3_deepseek_quality_train \
    --source "$SOURCE" --cache "$CACHE" --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
    --initial-adapter "$INITIAL" --gate "$GATE" --gate-sha256 "$GATE_SHA256" \
    --arm bf16_l2l --length 4096 --steps 1 --pilot-prefreeze \
    --output "$EVIDENCE/e-L4096-bf16"
export_evidence "${RUN_ID}-e-L4096-bf16" "evidence/$RUN_ID/e-L4096-bf16-process" \
    "evidence/$RUN_ID/e-L4096-bf16"
```

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/e-L4096-fp8-process" --seconds 50400 -- \
    python -m benchmarks.harness.p3_deepseek_quality_train \
    --source "$SOURCE" --cache "$CACHE" --pack "$PACK" --pack-sha256 "$PACK_SHA256" \
    --initial-adapter "$INITIAL" --gate "$GATE" --gate-sha256 "$GATE_SHA256" \
    --arm oracle_block64_e4m3 --length 4096 --steps 1 --pilot-prefreeze \
    --output "$EVIDENCE/e-L4096-fp8"
export_evidence "${RUN_ID}-e-L4096-fp8" "evidence/$RUN_ID/e-L4096-fp8-process" \
    "evidence/$RUN_ID/e-L4096-fp8"
```

L1024 uses k4 and L4096 k1. Do not add another step, alter the horizon/warmup or
replace the shared initial file. The first scheduled optimizer call uses LR0.
Report step loss, gradients, adapter movement and checkpoint identities as
observed; this is execution/memory measurement, not evidence of learning or
improvement. A long-step OOM/timeout/refusal does not retroactively change a
valid d gate PASS, but it ends further pilot execution and blocks completion.

## f, bounded resource-only development sample

Only after d PASS and successful e, with a fresh output and enough remaining
allowance, run:

```bash
python -m benchmarks.harness.p3_deepseek_quality_stage \
    --output "$EVIDENCE/f-process" --seconds 5400 -- \
    python -m benchmarks.harness.p3_deepseek_quality_timing \
    --source "$SOURCE" --cache "$CACHE" \
    --initial "$INITIAL" --initial-sha256 "$INITIAL_SHA256" \
    --data-root "$DEV_DATA" --gate "$GATE" --gate-sha256 "$GATE_SHA256" \
    --max-tokens-per-batch 8192 --deadline-seconds 5400 --device cuda \
    --output "$EVIDENCE/f-timing"
export_evidence "${RUN_ID}-f" "evidence/$RUN_ID/f-process" "evidence/$RUN_ID/f-timing"
```

Both deadlines are at most 5400 seconds. The outer POSIX process deadline bounds
an in-flight CUDA call; the internal deadline checks alone cannot. Preserve
execution-start and per-batch partial receipts on timeout. A timed-out sample is
incomplete, not a measured full-dev throughput.

The runner takes the first frozen development task per endpoint for HellaSwag,
ARC and BoolQ, ten choice requests total. It uses input IDs/positions, native
selected logits and unlabelled FP32 log-softmax scratch, with no gold gather,
base accuracy, NLL, predictions, paired outcome or evaluation-split rows. It does
not cover a full dev sweep, LAMBADA or secondary NLL. Record task IDs, requests,
input/padded tokens, batches, native forward calls, position chunks, wall time,
logits bytes and peaks. These limited resource samples alone cannot admit the
complete phase 3 evaluation budget.

## Stop, preserve and delete

A failed prerequisite, process failure, FAIL, NO_VERDICT, OOM, routing refusal,
missing Oracle VJP, nonfinite value, unavailable required control, evidence loss,
USD 87 computation limit or hour-95 limit stops execution. Do not run the next
stage, retry for PASS, change thresholds or alter L/B/k. Mandatory GA repeat and
INT4 control are only the specified gate passes. There is no default resume or
held instance. An infrastructure-loss resume would need its own recorded event,
valid complete checkpoint and explicit decision within approval; this runbook
never silently starts one. A different GPU requires a new offer/price approval
and new native admission, not dimension changes on this rental.

On the control machine, finish preservation and provider teardown without waiting
for phase 3 review:

1. Interrupt any still-running bounded command. Inspect the wrapper's cleanup
   result; an unreaped child or signaling error is not confirmed termination.
   Preserve logs, partial outputs and resource
   observations. Upload all completed/failed stage evidence and verify transport
   hashes. If the machine has died, preserve the last uploaded evidence and
   report the exact missing interval rather than inventing receipts.
2. Record current provider charges and traffic. Use the direct provider console
   or authenticated API to delete only this recorded instance ID. Capture the
   credentials-free deletion response, UTC time and provider request/deletion ID.
3. Confirm the instance is absent from a fresh provider listing. Then delete only
   this pilot's recorded local-volume ID. Confirm it is absent from a fresh volume
   listing. Stopping an instance or detaching a volume is not deletion.
4. Confirm no associated billable resource remains. Record final charges or the
   billing lag and a later reconciliation time. Verify billing stops, not merely
   that a process exited. Save deletion/absence/billing evidence on the control
   machine, since the rental filesystem is gone. Add it to the final evidence
   inventory and publish a redacted closeout.

The default is deletion of both instance and volume immediately after completion
or refusal. Keeping either needs prior explicit user price approval for its own
retention period; pilot approval does not authorize holding. A volume stays tied
to its physical host and does not reserve its GPU. Source plus cache would both
be needed for later fingerprint admission. The proposal's holding/re-download
break-even numbers are [INFERENCE], omit new smoke/gate admission costs from the
a+b reprepare subtotal, and do not justify unapproved retention. Stopped-instance
storage may have a different rate. Never substitute stop for destroy.

## After the pilot, freeze a proposal, not an execution

Only after the pilot, prepare Task 2.2's complete protocol and cost forecast from
the preserved evidence. Bind model revision, source/cache identity, exact input
and pack hashes, shared initial adapter, frozen task manifest, source package,
attention policy, passed same-machine raw gate, software/hardware identity,
lengths, seed, step count, scheduler horizon, batch tokens, shard/deadline plan,
quality rule and `include_base_model_evaluation: false`. The immutable attention
policy SHA256 is
`fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6`.
Do not turn the pilot's one-step adapters into the phase 3 starting point.

Separate measured time/traffic from every forecast marked [INFERENCE]. Price the
four planned training runs, repeated expert reads, complete primary and secondary
evaluation, missing timing coverage, setup/readmission, evidence and teardown.
Use observed tokens and forward calls, not gate diagnostic throughput as training
speed. Do not extrapolate the old B6 USD 3.60 bill to this pilot or phase 3.

Present explicit cost/statistical-headroom alternatives before any outcomes:
fewer training steps, frozen-order HellaSwag prefix, dropping the L4096 pair,
another approved native-BF16 GPU, or an expert-reuse branch applied to both arms.
Record the user's chosen scope, freeze protocol bytes, compute their SHA256, then
obtain the user's explicit approval of both that exact hash and a separate phase 3
budget. An incomplete gate/timing sample supplies no permission to proceed. This
runbook contains no phase 3 training or evaluation execution command.

## Final report template

Use `benchmarks/results/p3-deepseek-quality/pilot-v1/README.md` for the later
report, with fresh immutable evidence paths and a redacted closeout. Fill observed
values from receipts, or state NOT RUN / NO_VERDICT and the recorded reason.
Never fill missing measurements with forecasts.

| Result | Value to report | Evidence identity |
| --- | --- | --- |
| Historical tiny E3 | FAIL, 0.05028628354277727 > 0.05, unchanged | Existing immutable tiny receipt path/hash |
| Full DeepSeek-V3 671B gate | PASS / FAIL / NO_VERDICT; A3, E1, E2, E3 and control discrimination separately | d manifest SHA256 and complete raw artifact location |
| Later long-step/timing status | Separate from gate verdict, including NOT RUN after refusal | Each child and wrapper manifest SHA256 |
| Authorization and source | User price approval, lead PR approval, source commit/archive hash | Redacted approval references, setup receipt |
| Model/data identity | Revision, 163 shards, 688586727753 B, source/cache fingerprint, initial/pack/dev manifest hashes | a/b/pack receipts |
| Platform | GPU/VRAM, driver/CUDA, native BF16, RAM/cgroup limits, physical NVMe/free bytes, Python/freeze | Hardware and setup evidence |

Add one row per a-f stage and per gate pass, including GA repeat, MATH floor and
INT4, with start/end UTC, wall seconds, stop allowance, status and evidence path.
For each long step report arm/L/B/k/N, LR, loss, gradient norm, tokens/s, adapter
change norm and checkpoint digest. Gate timing includes raw diagnostic disk I/O
and is not training throughput.

Report sampled RSS, cgroup lifetime `memory.peak`, CUDA allocated peak and CUDA
reserved peak separately, with sampling/scope limits. Report owned boundary-store
payload/scales/padding and Oracle bytes separately. Those store bytes are not the
whole decoder/autograd activation peak. Never sum overlapping store, process and
allocator peaks. Include expert fetch counts/logical bytes without claiming they
are physical NVMe reads. Report Oracle forced-route/VJP counts without a natural
route-stability claim.

For f include covered tasks/requests, batches/chunks, input and padded tokens,
native forward calls, wall seconds, input-token/request/forward rates where
measured, logits memory and the explicit LAMBADA/NLL/full-dev coverage gap. State
that no base quality outcome or evaluation split was scored. Put [INFERENCE]
phase 3 hours/USD and cut options in a separate forecast table.

Close with compute/root/volume/ingress/egress charges, total paid and any pending
billing reconciliation, total elapsed time, stop reason, instance ID, volume ID,
delete request/confirmation IDs, UTC deletion times, fresh absence-check evidence
and billing-closeout identity. List each raw evidence bundle's remote path, bytes
and SHA256 separately from the human summary. State that phase 3 remains blocked
until the user approves the exact frozen protocol hash and budget.
