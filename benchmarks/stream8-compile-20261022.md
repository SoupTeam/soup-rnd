# Physical 8 GiB residency and Inductor study: retention blocked

The 2026-10-22 standalone **USD 5** attempt stopped before model work. It does
not establish numerical parity, a speedup, a VRAM reduction, or a model-size limit.
All four arms and all model sizes remain untested on CUDA in this attempt.

## What was verified

- CPU residency/checkpoint and selected-compilation checks: **14 passed, 2 CUDA
  skips**, Python 3.10, torch 2.13.0+cpu. The compilation test uses AOT eager;
  this is not evidence for CUDA Inductor.
- Fresh source audit: **602 production files** match upstream
  `d500ae0dca7c4da905f8d59c9866af2f8aaa824b` byte for byte. The comparator AST
  matches the previously committed planned overlay. Source archive SHA-256:
  `2b3ce103921a5713849d120b6bd3d74fc5c895124b65bbea93f06a2658bc3f6a`.
- Provider instance **57114362**, label `stream8-compile-20261022-a`, reported
  **NVIDIA GeForce RTX 3050, 8192 MiB**. The host/cgroup available-RAM lower
  bound was **64,547,737,600 bytes (60.11 GiB)**. Three idle checks found no
  compute peers, 0% GPU utilization and 1 MiB device memory in use.
- Both local deadline cleanup and the remote instance-scoped budget guard
  were installed. The remote guard was observed before setup.

An earlier creation request for advertised RTX 5060 offer `56602875` returned
`no_such_ask`. An independent label lookup found no created instance. That request
was not silently retried against a different lease.

## New transport diagnosis

The previous study had failed on HTTP/relay throughput and an OpenSSH timeout.
This attempt tested a different binary implementation: authenticated Paramiko SSH,
a **64 MiB receive window**, **1 MiB maximum packet**, and a **16 MiB random
payload**. It trusts the host key recorded by the initial OpenSSH connection.

All **16,777,216 bytes** arrived and matched the remote SHA-256. Transfer time,
including SSH connection establishment, was **42.2274 seconds**, or
**0.378901 MiB/s**. The unchanged admission threshold was 2 MiB/s. The failure
therefore demonstrates a completed, hash-verified but insufficiently fast binary
transfer, rather than another unmeasured 32 MiB timeout.

The prospective first-model bound was **12 GiB of raw capture artifacts**.
At the observed rate this requires **32,430.6 seconds (9.01 hours)** of transfer
before training/setup/cleanup. This is a conservative artifact-size bound, not
an observed archive size or a claim that compression cannot help. It does not
fit the selected eight-hour lease policy. The controller rejected model work;
the numerical tolerance was not changed and no further rental repeated the probe.

## Planned experiment and outstanding work

The first target remains pinned `Qwen/Qwen3-8B`, dense frozen BF16, ordinary
PEFT rank/alpha 4 LoRA on all seven projections, dropout 0, seed 792, nonzero
adapter generator 123, context 1024, batch 1, accumulation 2, the common 32-row
arithmetic fixture, and a 7 GiB allocator cap with an additional workspace reserve.
Neither this model nor any 14B/24B/32B model was downloaded or trained.

The comparisons remain ordinary streaming versus persistent decoder weights,
then resident eager versus resident `inner.forward` compiled with fullgraph
Inductor, with eager ownership/checkpoint scheduling and disabled CUDA graphs and
error suppression. A separately labeled precision-preserving compiler candidate
is allowed only if the default candidate fails qualification. No compiled graph
was executed; startup, recompiles, device peaks and throughput are unmeasured.

Before any timing, replay the original reference-first `rtol=0.01`, `atol=1e-5`
comparator, exact initial adapters/inputs, finite tensors, three-decimal loss and
causal/padding checks. Qualified comparisons then need four fresh AB/BA/BA/AB
pairs, six warmup plus eight measured optimizer updates, final fourteen-update
loss/adapter checks and synchronized time/memory measurements. The planned
practical speed gate remains median paired reduction >=2%, >=75% positive pairs
and a positive 95% bootstrap lower bound (20,000 resamples, seed 792).

Larger models require measured host/cgroup headroom, disk, time and budget;
the largest successfully tested model must not be called a universal maximum.
This attempt has **no largest tested model**. A next attempt first needs an
artifact-size-aware retention plan that fits its reserved time/cost and passes
an authenticated hash check. CPU tests alone do not justify a performance claim.

## Cost and cleanup

Instance **57114362 was deleted** in the controller's `finally` block and verified
absent again by a separate provider-list query. Unrelated instances were untouched.
No model captures exist to retain; all required preflight diagnostics and source
hashes were saved before deletion.

The conservative standalone expense is **USD 0.144429 / 5**, including a padded
GPU/disk rate, all observed container network traffic, 120 seconds of time margin
and USD 0.12 reserve. Provider-observed charges at this snapshot were **USD 0.007**;
settlement can lag. Historical study spending and the other chat's USD 5 share
are excluded. The unspent allowance is not evidence that model work completed.

Full repository Ruff and `git diff --check` passed. CI was not awaited.

[Machine-readable record](stream8-compile-20261022-results.json) ·
[Diagnostics, hashes and attempted controllers](evidence/stream8-compile-20261022/README.md)
