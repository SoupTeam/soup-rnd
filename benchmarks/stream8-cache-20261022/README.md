# Physical 8 GiB cache study: blocked before model work

No Qwen3-8B numerical, training-speed, training-memory, copy-time or model-fit verdict
was obtained. One rental stopped at the prospective artifact-retention gate, before
dependency installation, model download, CUDA tests or training. This is a transport
blocker, not a numerical rejection, OOM, or negative performance result.

[Machine-readable result](results.json), [compact evidence](evidence/manifest.json).

| Arm | Research implementation | Physical GPU qualification / timing |
| --- | --- | --- |
| Baseline streaming | Original two rotating decoder slots and checkpointing | Not run |
| Persistent decoder weights | Dedicated immutable slots, checkpointing unchanged | Not run |
| Retained activations | Selected checkpoints skipped; private copies of every saved rotating-weight operand | Not run |
| Both | Selected resident layers retain activations and borrow their immutable slots | Not run |

The independent retained arm is implemented; it is not replaced by the resident arm.
Its weight copies are necessary autograd storage, charged against the same cap.
It does not claim activation retention is free of additional frozen-weight operands.
The combined arm avoids those copies by keeping selected base weights resident.

## Fixed experiment and ownership

Prepared source is upstream `d500ae0dca7c4da905f8d59c9866af2f8aaa824b` plus explicit
research harnesses. Independent Git-blob verification found all **602 production files
byte-identical**. The PR branch's production code is not the frozen benchmark source.
The original comparator is taken without changes from the preserved legacy overlay.

The pinned Qwen/Qwen3-8B revision is
`b968826d9c46dd6066d109eabc6255188de91218`: dense frozen BF16, ordinary PEFT LoRA
rank/alpha 4, all seven projections, dropout 0, seed 792, nonzero adapter generator 123,
context 1024, batch 1, accumulation 2, and the common 32-row arithmetic fixture.
[Protocol](protocol.json) records all numerical and performance gates.
These are prepared settings; native BF16 and the model were not GPU-executed.

All arms retain a 7 GiB allocator cap on a physical 8 GiB card. A prospective selection
uses the baseline's measured reserved peak minus a 0.75 GiB workspace reserve.
Resident-only selection charges a decoder slot per layer. Retained/both additionally
charge 512 MiB/layer as a conservative activation allowance, so selections may differ.
The hard allocator cap remains authoritative; any OOM must be retained and reported.
No selection or actual training-memory measurement occurred in this attempt.

Only qualified arms may be timed: original reference-first `rtol=.01`, `atol=1e-5`,
exact initial adapters/inputs, finite values, loss to three decimals, gradients
captured before optimizer reset, final adapters, and causal/padding checks.
The architecture-only capture count is 14 times the actual decoder-layer count.
Timing is four fresh-process AB/BA/BA/AB comparisons against the same baseline,
six warmup plus eight measured optimizer updates, checked 14-update trajectories,
CUDA synchronization, allocated/reserved/device peaks, CPU RSS, logical copies and
decoder forward/recompute counts. No numerical tolerance was changed.
The >=2% median improvement, >=75% positive pairs, bootstrap 95% lower-bound>0 gate
remains unchanged. No performance statistic can be computed from this attempt.

## CPU evidence and limits

**27 passed, 2 CUDA skips**, including 16 new cache tests, on Python 3.10 / Torch 2.13.0+cpu
with the reconstructed frozen runtime imported explicitly via `PYTHONPATH`.
This is a small synthetic floating-point fixture, not a pretrained-model gate and
not validation of the planned Torch 2.8.0+cu128 environment.
The new cases preserve outputs, losses, input gradients, every adapter gradient
before reset and final parameters bitwise over four updates in all four arms.

Three deliberately broken variants are caught:

- Blind checkpoint bypass without ownership protection.
- Returning mutable weight aliases instead of snapshots for adjacent retained layers;
  this produces different training tensors despite the repaired scheduler.
- Removing the missing-backward-prefetch repair; the ownership guard raises.

The last repair matters because the original backward prefetch relies on every
layer recomputing. A retained neighbour no longer executes its body. The candidate
restores missing loads with the original pool's copy/event ordering before `wait`.
CUDA asynchrony, full-sized activation lifetime and performance remain unverified.
No production configuration, automatic planner, 10Cache integration, torch.compile,
NF4 replication or unrelated feature was added.

The lead's Mistral7B NF4/context 512 RTX 5070 Laptop quote (0.77s→0.60s,
1.3→4.7GB,bitwise gradients) remains **unverified quoted context**.
Public upstream main `7f6f5a032d1d7fce5fd95bbf2501fdc6370be93e` was inspected;
the targeted runtime/source search did not identify the quoted retention implementation
or its raw comparison. This isolated candidate is not presented as that implementation.
No result here independently confirms the quoted speed, memory or gradients.

## One bounded rental and the blocker

Owned instance **57114865**, label `stream8-cache-20261022`, used an RTX 3060 Ti with
**8192 MiB** reported by `nvidia-smi`. It was idle: 0% GPU utilization, 1 MiB device memory,
no compute processes. The host/current cgroup-visible headroom check reported
160,395,608,064 usable bytes; no memory or paging settings were changed.

Before any model work, an authenticated direct SSH stream wrote 8 MiB random data
straight to a local binary file. The complete transfer matched its remote SHA256:
**18.469 s, 0.433158 MiB/s**. This improves diagnosis over the old timed-out 32 MiB SSH
probe: binary file streaming completed, but the route remained too slow.

The prospective gate was explicitly changed before rental from the historical 2 MiB/s
to **1 MiB/s**, based on an upper planning allowance of 12 GiB compressed raw evidence
and a four-hour transfer reserve (3.42 h at 1 MiB/s). Numerical gates were unaffected.
The observed rate projects 7.88 h for that upper allowance, leaving no safe compute,
setup and cleanup window inside the bounded 8 h lease. The gate failed; it was not
lowered again after observing the result. There was no repeated rental.
The 12 GiB allowance is prospective, not measured model output. A future compact,
losslessly reconstructable archive protocol could be designed and qualified separately.

All scientific scripts and policies were prepared locally. None were uploaded or
executed on the GPU after preflight failure. There are no new model tensors to retain.
The raw 8 MiB probe, original diagnostics, planned source archive and private controllers
remain in ignored `output/stream8-cache-20261022`; compact evidence includes hashes.

## Cost, cleanup and verification

This half's fresh cap was **USD 5**, with prior usage 0. Local and instance-scoped remote
guards bounded one 8 h lease and included network/storage/cleanup allowances.
Provider-observed own-instance usage was **USD 0.005**, subject to settlement.
The conservative recorded spend is **USD 0.15**, including network, extra storage
margin, teardown and guard reserve; no historical residual budget is added.

`finally` deleted instance 57114865. A fresh independent paginated provider listing
then verified it absent. Other chats' and pre-existing instances were untouched.
Checks completed: source reconstruction / 602-file audit, targeted CPU regressions,
Ruff on `src/soup_cli/ tests/` and this research directory, and `git diff --check`.
Full CI was not awaited and is not claimed.

## Reproduce preparation without renting

The previous committed `planned-harness.zip` supplies original capture/comparator,
timing/semantics code, dependency constraints, dataset/model preparation and fixtures.
Every ZIP member is verified against its stored SHA256 manifest before use.

```powershell
python benchmarks/stream8-cache-20261022/prepare.py --output output/cache-repro
$env:PYTHONPATH = "$PWD/output/cache-repro/repo/src"
python -m pytest tests/test_stream8_cache_20261022.py tests/test_streaming_residency_candidate.py --no-cov -q -o addopts=''
```

The preparation command neither rents hardware nor downloads models. It creates an
isolated upstream tree, research overlay and scripts; Windows Git archive conversion
is disabled to preserve exact source bytes. It refuses an existing output directory.

On an already qualified Linux CUDA host, use `/root/stream8-cache/repo` for that tree
and `/root/stream8-cache` for generated inputs/scripts. Install the pinned constraints,
including Torch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.20.0 and TRL 0.29.1, then `pip check`.
Set the same deterministic environment as the legacy reproduction notes, replacing
`/root/liger` with `/root/stream8-cache`; preserve affinity and idle/peer checks.
Prepare the pinned model with `python prepare_model.py 8`. Run `bench.py --help` for
individual fresh-process capture/semantics/timing commands. `worker.py` describes the
four-arm orchestration but requires an explicit local time policy; it does not rent.
First qualify result retention, then repeat every unchanged numerical gate.
