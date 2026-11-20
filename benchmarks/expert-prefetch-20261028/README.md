# Packed NF4 expert transfer and one-expert lookahead

**Verdict: CONFIRMED_PRACTICAL_WIN.** Selected **packed**: four NEW balanced AB/BA pairs give **17.747%** paired step-time reduction, bootstrap 95% interval **[17.236%, 18.255%]**, **21.577%** throughput increase. Selection samples are excluded. The unchanged >=2% lower-confidence-bound gate passes.

## What changed

The opt-in prototype packs an expert's NF4 weights/sidecars into one aligned
pinned byte arena and one H2D copy. `prefetch` also stages the next routed expert
on a separate CUDA stream before the current expert's MLP. Both modes use the
same dequantization/math. Two bounded slots have separate copy-ready and
post-dequantization consumer fences; dense decoded weights own their storage.
Initial GPU allocation happens on the copy stream. No background CPU reader,
new CPU thread policy, CLI/config/default change or automatic placement.

Companion source [f5670242](https://github.com/SoupTeam/soup-rnd/commit/f5670242e13b2b480a78e8fc1e3c797d6bb13800)
and [portable patch ZIP](prototype.patch.zip) against immutable `43cf0b34`. The production
PR diff is separate; this report does not declare the prototype shipped.

## Same-cohort protocol

Kanana-2-30B-A3B, 30.67B total parameters, revision
`0d31365acf656e5f9a593cacc4913e9c0d36cb1c`, frozen NF4 base/BF16 compute,
FP32 attention LoRA rank4/alpha8, context512, batch1, accumulation2.
Physical RTX4060Ti 8188MiB, native RAM <=32GiB, workload swap disabled.
Torch2.10.0+cu130 / CUDA13 / Triton3.6.0 / bitsandbytes0.50.2, hash-locked
runtime wheels. Native allocator and asynchronous launches are matched.
All three arms already use grouped routes, two-read and static 8GiB RAM/2GiB GPU
expert-weight cache ceilings, with identical inputs/adapters/cache policy.
`native` is the original synchronous path on THIS source/runtime. Older timing
cohorts and ordinary three-read streaming are not denominators.

## Speed and memory

Exploration: fixed packed/native/prefetch order, one fresh process per arm,
two warmup and six measured accumulated AdamW updates, no discarded updates.
The table is exploratory, not a confirmed ranking. Confirmation, when selected,
uses four NEW fresh-process balanced pairs, 5000 paired bootstrap draws/seed29.

| Mode | Median step, s | Eight-update entry, s | Allocated VRAM, GiB | Reserved VRAM, GiB | RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 16.6828 | 185.444 | 3.6443 | 3.8320 | 11.9458 |
| packed | 13.5057 | 162.003 | 3.6496 | 3.8574 | 11.9766 |
| prefetch | 13.7692 | 164.305 | 3.6496 | 3.8574 | 11.9739 |

Held-out confirmation only (four workers per mode): step and entry are medians of worker summaries; memory is the maximum across these workers.

| Mode | Step, s | Eight-update entry, s | Allocated GiB | Reserved GiB | RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 16.5315 | 182.229 | 3.6443 | 3.8320 | 11.9762 |
| packed | 13.6245 | 157.868 | 3.6496 | 3.8574 | 11.9839 |


Entry includes model/cache setup, warmup, timing and worker startup/validation.
Common checkpoint download/NF4 preparation is separate. RSS is sampled every50ms,
excludes filesystem/kernel charges and is not total system RAM. Added pinned/GPU
staging has two <=16MiB slots per tier and is separately recorded in raw counters.
All actual timing arrays, setup costs, cache/transfer counters and confirmation
results are in [summary.json](summary.json) and [the evidence packet](evidence.zip).
No whole-training speed projection or claim of optimal memory allocation.

## Accuracy first

All ten actual small CUDA controls passed without skips before checkpoint
download, including delayed consumers, ring reuse, initial allocation ordering,
negative missing-fence controls and complete tiny NF4/BF16 optimizer trajectories.
Then all three separate full real-model 2-AdamW-update/4-microbatch captures
matched **bitwise** before any timing. All288 trainable LoRA tensors/144
attention projections receive gradients and actually update. Original reference-first
rtol=.01/atol=1e-5, finite and three-decimal loss gates were not changed.
The standalone audit reloaded complete captures and replayed the comparisons.

Scope: deterministic algorithms, TF32 off, eval mode with differentiable LoRA
updates and dropout disabled. This is same-NF4 numerical/resource qualification,
not downstream quality/convergence or NF4-vs-BF16 equivalence. Full final timed
tensors and unrecorded first-micro scalar losses retain NO VERDICT. No full-model
profile was taken in this qualification; the previous narrow transfer profile
motivates the hypothesis but does not prove a full-step bottleneck.

## Verification, retention and cleanup

149 scoped CPU source controls passed (10 GPU cases deselected locally), 14 private
runner controls and 81 operator controls passed. The later GPU-only allocation
fix was included in all ten executed GPU controls. Ruff and applicable source
pre-commit passed. Full-suite/hosted-CI success is not inferred.
The pre-data frozen standalone auditor is [audit_transfer.py](audit_transfer.py.txt);
it verifies all original archives/chunks/files, source/input/worker identities,
resource boundaries, full numerical captures, raw timing selection/confirmation,
budget and cleanup. No reported pass flag alone is accepted as numerical proof.
These review snapshots normalize line endings only; exact runnable `.py` originals are in the
packet's `operator/` entries (resolve `aliases.json` when extracting).

All **228 originals / 3,533,513,136 bytes**
were retained and hash-verified from the **626,278,400-byte**
lossless archive. Compact public evidence restores exact bytes using aliases;
large tensors/model/source archives are explicitly hash-indexed as local-only.
The full61GB checkpoint is not retained; its official expected/observed SHA receipts
are. Physical NVMe backing is unverified. Numerical/archive audit and secret scans
passed before publication. [Evidence manifest](evidence-manifest.json).

Owned instance58130492 and persistent volume58130491 were deleted and freshly
verified absent. Shared-account colleague resources were untouched.
Conservative new-task cost **$0.605370 / $5**, remainder
**$4.394630**; these are ledger estimates, not
invoices. The older unused $4.909114 remains a separate allowance, not spent here.
