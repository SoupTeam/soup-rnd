# Residency + torch.compile on physical 8 GiB GPUs — blocked preflight

**No model accuracy, training-speed, training-memory or model-capacity result was obtained.**
All paid attempts stopped before dependency setup, model download, GPU tests or training.
This is an infrastructure blocker and a prepared research protocol, not a failed numerical
qualification, a negative compiler-performance result, or completion of D2.

## What was prepared

The primary plan pins Qwen3-8B/14B/32B. A smaller-RAM fallback pins Qwen3-8B/14B and
Mistral-Small-24B-Base-2501. Pinned revisions/configs are in the planned archive.
BF16 frozen base, ordinary PEFT LoRA rank/alpha 4, no quantization, batch 1,
accumulation 2 and context 1024 use the same synthetic arithmetic fixture. No task-quality claim.

Comparisons are ordinary streaming versus resident layers, then **resident eager versus
resident + Inductor compilation**. Only resident decoder inner forwards are compiled;
scheduling, pooled-weight ownership and checkpointing stay eager. Fullgraph is required,
CUDA graphs and error suppression are disabled. A separately labeled operation-preserving
compiler candidate is planned only if default compilation fails accuracy; it is not an
eager fallback presented as compiled speed.

Before timing: original reference-first `rtol=0.01`, `atol=1e-5`, exact inputs/initial
adapters, finite tensors, loss to three decimals, and causal/padding checks. Only the
architecture-specific adapter-count assertion changes from 448 to 14 times layer count.
Then four fresh-process AB/BA/BA/AB pairs, six warmup and eight measured updates, final
14-update adapter/loss checks, no compilation in measured windows, synchronized step times,
allocated/reserved VRAM, whole-run peaks, CPU RSS and GPU/peer telemetry. Planned practical
gate: median paired time reduction >= 2%, >= 75% positive pairs, bootstrap 95% lower > 0.

Research cache selection leaves an explicit workspace reserve below a 7 GiB allocator cap
on a real 8 GiB GPU. It is not a shipped automatic planner or a promise to occupy all VRAM.
Largest successfully tested model would be reported, not a universal model-size maximum.

The frozen intended baseline is upstream `d500ae0dca7c4da905f8d59c9866af2f8aaa824b`; 602 production files
are unchanged. The intended source archive is `2b3ce103921a5713849d120b6bd3d74fc5c895124b65bbea93f06a2658bc3f6a`.
The older PR branch is not that baseline. **This source was not GPU-executed in this task.**

## Why no benchmark ran

| Attempt | Physical GPU confirmed | Usable RAM preflight | Retention blocker | Model work |
| --- | --- | ---: | --- | --- |
| A | RTX 3070, 8192 MiB | 164.03 GiB | Direct HTTP 0.480 MiB/s, below 2 MiB/s gate | None |
| D | RTX 3060 Ti, 8192 MiB | 58.47 GiB | Fresh HTTPS relay hostname DNS resolution failed | None |
| E | RTX 3070, 8192 MiB | 163.33 GiB | DNS fallback worked; HTTPS 0.229 MiB/s, below gate | None |
| F | RTX 4060 Laptop, 8188 MiB | 55.60 GiB | Single SSH 32 MiB probe timed out after 90 s | None |

The SSH timeout does not establish an exact transfer rate or a zero-byte transfer.
The DNS fallback was tested with injected local DNS failure while retaining TLS hostname
and certificate verification; Windows DNS/pagefile/system settings were not changed.
Two intervening requests received `no_such_ask` despite an advertised offer; no instances
were created. Another fallback preparation found no qualifying whole-host offer and never rented.
The 2 MiB/s result-retention threshold was not lowered. No new rentals continue after F.

## Checks, cost and cleanup

Local regression checks: **14 passed, 2 CUDA skips**. CPU AOT-eager checks cover selected-layer
checkpointed updates and selection reserves; they do not validate CUDA Inductor, pretrained
models or performance. Full repository Ruff and diff checks passed before publication.
GPU tests run: **zero**. CI is separate and must not be described as passed before completion.

All four paid instances `57086947, 57088446, 57089132, 57089727` were deleted, then independently
verified absent. Other instances were untouched. New preflight provider-observed charges
at this snapshot: **USD 0.142**, subject to settlement.
The conservative additional estimate is **USD 0.719**;
it includes time margins and per-attempt reserves. Including prior research, conservative
series usage is **USD 4.996/10**, leaving approximately
**USD 5.004**. Account balance differences are not cost evidence.

[Machine-readable blocked record](streaming-residency-compile-8gb-20261021-results.json) and
[planned harness, diagnostics and reproduction notes](evidence/streaming-residency-compile-8gb-20261021/README.md).
Private provider responses/credentials remain outside Git. There are no new training tensors
to retain because no capture ran. The earlier 7B residency result remains a separate experiment.

## Resume condition

First qualify a result-retention path on a suitable physical 8 GiB host, preserving hashes
and cleanup guards. Then execute the frozen per-model numerical gates before any timing.
Do not infer model fit, compiler correctness or an acceleration percentage from these preflights.
