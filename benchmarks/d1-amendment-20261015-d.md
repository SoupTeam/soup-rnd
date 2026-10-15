# D1 amendment D: complete RTX 3090 cohort

Dated 2026-10-15, committed before the new CUDA cohort. The completed pilot and
all earlier rules/attempts remain unchanged. User authorized a **$10 cumulative
D1 budget**, including approximately $0.47 already charged for the pilot.
This is a separate physical-GPU/source/environment cohort; never pool pilot
timings with it. Pin bitsandbytes 0.50.2, torch 2.11.0 CUDA 12.6,
transformers 5.17.0 and peft 0.20.0, retaining the full resolved environment.

## Complete measurement plan

The original geometry lists FOUR unique (N,K) classes: q/o=(4096,4096),
k/v=(1024,4096), gate/up=(14336,4096), down=(4096,14336).
Earlier prose saying "five-shape" was a counting error, not a fifth geometry.
With six M values, two dtypes and two pairs this is **96 comparison cases**.
Run five complete valid fresh-process ABBA blocks initially for every case;
then follow the existing F1 `blocks_to_add` pilot/N/one-top-up rule until zero.
Do not drop SAME_PATH cases, void attempts, failed workers or extra repeats.

Before performance work, repeat both large-alias/streaming correctness fixtures
and the existing #842/#968 GPU tests on the new GPU. Then perform five valid
ABBA blocks of the unchanged RAM-tier cached-zero ablation, followed by five
valid ABBA blocks of the unchanged BF16 real QA study (20 complete 276-update
training/evaluation runs initially). These precede the complete projection grid.
The ablation measures an upper bound with garbage math; it is not a kernel gain.
The QA split, initialization, optimizer, evaluation and equality rules stay fixed.

## Warm-up and clock policy fixed before new timings

Lock the dedicated rented GPU's SM clock to 1500 MHz using
`nvidia-smi -lgc 1500,1500`, retaining the successful command output and the
provider's unchanged power limit. If locking fails, stop before timing and
review the environment; no silent stock-clock fallback or changed clock policy.
Record actual DURING-work clocks and apply the original <=5% spread requirement.
Reset the clock with `nvidia-smi -rgc` on normal completion, error and cleanup.
No hardware GPU reset is used. Official command reference:
https://docs.nvidia.com/deploy/nvidia-smi/index.html

For frozen PROJECTION arms only, warm up at least 20 steps AND 5 seconds,
synchronizing additional warm-up steps. Retain actual warm-up count/time.
Use 200 synchronized timed steps instead of the original 100. This replaces
the pilot subset's expensive 5000-step thermal warm-up under an explicitly
fixed clock, and increases timed observations. No optimizer is present here.
Train and ablation remain exactly 20 warm-up optimizer updates and their original
256/100 timed steps. Never use time-based extra optimizer updates.

## Execution, evidence and bounds

`d1_full_study.py` freezes the plan/source, resumes completed cases, assigns
distinct global block IDs to additional attempts, and preserves every worker
output. It stops on numerical/worker failures or cohort-wide clock/identity
violations instead of silently retrying a changed setup. Only ordinary invalid
blocks receive F1 replacements. An explicit deadline reserves export time;
independent rental guards enforce the cumulative spending limit. A deadline or
budget stop means incomplete measurements, never acceptance.

Temporary sharded weights are cleaned only after the runtime closes, outside
timing. Source, numeric records, arrays, tensor SHA-256, telemetry and all
failed/void/interrupted attempts remain. Exact duplicate files may be stored
as verified archive hard links; this changes storage, never evidence bytes.
Audit EVERY accepted QA block and all its four training runs against the pinned
gold and the same initial/final adapter tensors, loss sequence and predictions.
Record accepted block IDs explicitly so added repetitions cannot be overlooked.

Completion of measurements yields a report and reproducible handoff for the
team lead. It does not impersonate independent review, close upstream #842,
claim a new fused backward implementation, or send messages to other people.
