# Qwen3-32B NF4: bounded-RAM disk streaming

Outcome: **COMPLETE_INDEPENDENTLY_REPLAYED**.

Physical 8 GB GPU; real physical/native-cgroup RAM upper bound <=32 GiB.
Pinned Qwen3-32B, batch 1, accumulation 2, context 512, ordinary PEFT LoRA
rank/alpha 4 on seven decoder projections. Fixed double-quantized NF4 weights
and identical nonzero native adapters across cache arms. No compile, Liger,
FastLoRA or explicit FlashAttention change. This is numerical/resource research,
not a 32B task-quality, convergence or 70B/120B capacity result.

The unchanged numerical/timing harness enables deterministic algorithms in all arms.
PyTorch 2.8 can fill otherwise uninitialized allocations in that mode; the profile's
allocation/fill costs must not be presented as a proven production kernel opportunity.
See [PyTorch's deterministic-memory behavior](https://docs.pytorch.org/docs/2.8/deterministic.html).
No determinism setting or numerical gate was changed during this study.

Actual GPU: NVIDIA GeForce RTX 4060 Ti.
Physical RAM 30.9970 GiB; effective bound
29.7559 GiB.
Host swap may exist; model stages reject observed container swap use.
No kernel-disabled-swap claim is made.

Capacity selection before accuracy: 7 GPU layers,
equal RAM-cache ceilings 14336 MiB in RAM/both arms.
GPU-resident layers are excluded from the RAM hotset. Selection is a
capacity policy, not a claim of globally optimal time or maximum useful occupancy.

| Metric | Ordinary disk | RAM cache | GPU cache | Both |
| --- | ---: | ---: | ---: | ---: |
| Optimizer step, s | 9.2608 | 8.6317 | 9.1622 | 8.5955 |
| Native setup/preload, s | 21.3162 | 28.4580 | 35.6367 | 39.8500 |
| Whole fresh six-update entry, s | 82.5211 | 84.8784 | 93.7806 | 94.1461 |
| Full staged process wall, s | 84.6188 | 88.0947 | 96.0281 | 97.3577 |
| Peak allocated VRAM, GiB | 3.6444 | 3.6444 | 5.4083 | 5.4083 |
| Peak reserved VRAM, GiB | 3.8184 | 3.8184 | 5.7812 | 5.7812 |
| Peak process RSS, GiB | 5.8270 | 19.8284 | 5.9035 | 19.8955 |

Four balanced fresh-process blocks; 16 processes, 96 complete optimizer
updates, two warmup plus four measured per process. All arrays and losses
retained. Shared preparation/capacity selection is outside these entries;
complete entry includes native setup/preload/training/saving.
The entry clock begins after CLI imports; full staged-process wall also includes
imports and the memory-scope wrapper. All four process-wall values are retained.
No full-task or time-to-convergence speedup is inferred.

- baseline_to_ram: paired step-time reduction **+6.794%**, 95% whole-block bootstrap [+6.476%, +8.102%]; unchanged >=2% practical gate: **True**.
- baseline_to_vram: paired step-time reduction **+1.085%**, 95% whole-block bootstrap [+0.710%, +2.467%]; unchanged >=2% practical gate: **False**.
- baseline_to_both: paired step-time reduction **+7.153%**, 95% whole-block bootstrap [+6.827%, +8.593%]; unchanged >=2% practical gate: **True**.
- ram_to_both: paired step-time reduction **+0.386%**, 95% whole-block bootstrap [+0.375%, +0.534%]; unchanged >=2% practical gate: **False**.

## What the measurements support

The combined arm keeps 56 decoder layers in
RAM and 7 different layers in VRAM: 63/64
decoder layers avoid recurring disk-source reads. The remaining decoder and
large vocabulary matrices still use streaming. This is not the whole model in VRAM.

On this fixed workload, RAM caching passes the practical step-speed gate;
adding GPU residency on top does not. Occupying more GPU memory is therefore
not a demonstrated additional practical win here.

The next placement-policy candidate is opt-in RAM-cache selection with existing
physical/cgroup headroom and startup-cost break-even for the planned training length.
A short job can finish later despite faster individual steps. A speed-aware selector
is not implemented or validated by these four capacity-selected arms, and the
measured cache size is not claimed globally optimal. Compare smaller/larger budgets
and intended production determinism before implementing a kernel or automatic policy.


## Diagnostic profile

One active profiled step: 9.2754 s;
source.get interval union 7.2246 s,
including 0.4352 s
without observed GPU kernel/copy activity. These overlap and are not additive
or pure SSD service-time buckets. The diagnostic run is separate from all
ordinary timing processes; its step times are not speed evidence.

The trace contains matching CPU and GPU `ProfilerStep#2` annotations for one
optimizer update. Analysis uses the single CPU window and requires matching CUDA
annotations; five postprocessing checks reject missing/ambiguous/mismatched markers.

Eight complete numerical captures were independently replayed before profiling
and ordinary timing: original reference-first rtol=.01, atol=1e-5, finite
and .3f loss gates. All final adapters and six-update trajectories replayed.
Actual frozen answer-only batches and every model-stage RAM/swap proof audited.
Native training masks are all ones; padding follows both supervised positions.

## Storage, controls, retention and spending

The actual container disk mount and loop-device details are retained. Physical
NVMe backing is not assumed; disk-source or Linux I/O counters do not establish
physical NVMe service time or eliminate host filesystem caching effects.

- local_protocol_and_retention_controls: 32 tests; 0 failures, 0 errors, 0 skips.
- context_refusal_control: 1 tests; 0 failures, 0 errors, 0 skips.
- profile_interval_control: 1 tests; 0 failures, 0 errors, 0 skips.
- profile_marker_postprocessing_controls: 5 tests; 0 failures, 0 errors, 0 skips.
- native_profile_and_stop_protocol_controls: 3 tests; 0 failures, 0 errors, 0 skips.
- shared_account_ownership_controls: 4 tests; 0 failures, 0 errors, 0 skips.
- gpu_host_controls: 150 tests; 0 failures, 0 errors, 0 skips.

No full-repository or hosted-CI success is inferred from these controls.
Current-stage conservative accounting **$0.771675**;
aggregate since the 4B pilot **$3.927052**;
combined remainder **$9.416553**. Not an invoice.
All prior pilot/capacity retries are included once; earlier unrelated studies
were already deducted from the starting allowance. Exact instance and persistent
volume independently confirmed absent by exact IDs. Unrelated resources were preserved;
fresh listing counts and unrelated instance IDs are retained in cleanup evidence.

[Machine-readable results](result-summary.json), [evidence manifest](evidence-manifest.json),
[full raw-file index](raw-artifact-index.json.gz). Concatenate numbered evidence
parts in manifest order; the ZIP round-tripped byte-for-byte. Complete original
outputs remain in the stable SoupResearch directory identified in the raw index.

BoolQ: Clark et al., NAACL 2019, CC BY-SA 3.0 (data only). Model weights and
credentials are not committed. All outcomes and original tolerances are preserved.

Identical tokenizer JSON copies share one canonical ZIP entry.
`duplicate-tokenizer-map.json` maps each original path to those exact bytes; every
logical original payload was verified before/after repacking. The original review
ZIP and all raw outputs remain unchanged in SoupResearch.
