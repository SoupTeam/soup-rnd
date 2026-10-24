# Streaming research plan

Updated 2026-10-25. Destination: SoupTeam/soup-rnd PR #5.

## Goal and checklist

Reduce real LoRA training time for models exceeding GPU memory. Coordinate GPU,
RAM and disk with memory headroom and measured quality. Maximum occupancy alone
is not the objective.

- [x] Qwen3-4B BF16/NF4 BoolQ pilot: matched training and independent held-out
  accuracy/NLL replay before speed. This does not establish equivalent quality.
- [x] Qwen3-32B NF4 on a physical 8 GB GPU and a real <=32 GiB RAM bound:
  qualify ordinary streaming, GPU residency, RAM cache and both before timing.
- [x] Matched disk-tier four-arm study with retained backing-storage evidence,
  a separate CUDA diagnostic and all 16 timing processes / 96 optimizer updates.
  Disk overflow was required by the unchanged RAM-source admission policy;
  physical NVMe backing and physical RAM infeasibility were not established.
- [ ] Speed-aware automatic placement: compare additional cache budgets and
  intended production settings; include setup break-even and memory headroom.
  Existing automatic GPU residency is a capacity policy, not a speed optimizer.
- [ ] Larger-model capacity and candidate Triton kernels need their own evidence.
  No 70B/120B run or capacity claim has been completed.

## New opportunity screen on current upstream, 2026-10-25

The GPU results below remain frozen on the older research source. Upstream Soup
0.76.0 at `51777d12077df022db8ee8a57cf46ed2f76909ef` already includes the improved
pinned-arena packer and `stream_recompute`. Do not rediscover either as a new feature
or transfer old timings to the changed code. The next measurement needs a fresh baseline.

[CPU screening data](streaming-impact-20261025/candidate-screen.json) and its
[self-contained evidence](streaming-impact-20261025/evidence-manifest.json) retain
the exact inspected source, real 32B tensor headers and reproduction script.
The extracted archive reproduced its JSON byte-for-byte. No GPU was rented or used.
These are mechanism/placement checks, not new training-speed or capacity results.

1. **First candidate: cache-aware activation retention.** The current native
   `stream_recompute: auto` returns immediately for `tier='disk'`, without probing
   retention. An extracted-method control confirms this; the RAM positive control
   reaches the planner. Our combined cache covers 63/64 decoder layers, so the tier
   name alone is insufficient to classify that workload as read-bound. Rebase the
   cache prototype on current upstream, then compare RAM cache alone, RAM plus GPU
   weights, RAM plus retained activations, and their combination. Share one VRAM
   cap and include optimizer/setup reserves, lifetime guards and setup break-even.
   The changed decision is a hypothesis until full correctness and GPU timing pass.
2. **Second candidate: avoid duplicate CPU copies of persistent GPU weights.**
   Current `RamSource` keeps the full base. With hypothetical exclusion of four
   centred decoder layers, real 32B headers and the current upstream packer give
   18.205 GB CPU payload and 17 GiB pinned arenas. Both pass the same host's 55%
   physical-RAM term. Seven exclusions give 17.450 GB payload / 16.5 GiB arenas.
   Without exclusions, 19.211 GB payload / 18 GiB arenas still fail that term.
   Placement bounds/non-overlap were checked for counts 0 through 12. This makes
   training without recurring disk weight loads worth testing on 8 GB VRAM / 32 GB
   RAM, without raising the safety fraction. It is not a whole-run memory proof:
   extras, runtime, transient allocations, pinned allocator caching and cgroup
   availability still matter. Bootstrap, source replacement and rollback need work.
3. **Third candidate: frozen embedding row gather with one persistent head slot.**
   An untied Qwen3-32B vocabulary matrix is 1,555,824,640 BF16 bytes. Four current
   large loads move about 6.22 GB/update. Gathering only the input rows on CPU and
   keeping the head in the existing GPU slot would instead transfer at most 10 MiB
   of embedding activations for two 512-token microbatches, after the initial head
   preload. Decoder-only host staging arithmetic falls from 4 GiB to 0.5 GiB.
   Small CPU float32/BF16 lookup and downstream LoRA-gradient controls are bitwise
   equal, and a wrong-row negative control is detected. This is not streamed GPU
   qualification: row I/O, device synchronization, hooks, predictor and lifetimes
   remain untested. Existing large-copy prefetch overlaps compute, so fewer bytes
   do not predict a step-time percentage. Scope starts with frozen untied weights;
   this does not propose the extra GPU slot declined in issue #1174.
4. **Secondary candidate: direct prequantized NF4 input/import.** The current
   source gate explicitly refuses a bitsandbytes NF4 checkpoint; a CPU control
   reproduces that refusal and accepts the unquantized positive control. Supporting
   packed weights and their complete quantization state could reduce cold download,
   conversion and scratch-space costs. This is a startup/storage opportunity,
   not an established step-speed gain. Never bypass the protective source gate.

Relevant upstream work and ownership were checked read-only:

- [#2152](https://github.com/MakazhanAlpamys/Soup/pull/2152) is merged: native
  activation retention exists and differs from the earlier conservative prototype.
- [#1702](https://github.com/MakazhanAlpamys/Soup/issues/1702) is closed: improved
  arena packing already shipped. Its speed result does not establish a gain.
- [#1693](https://github.com/MakazhanAlpamys/Soup/issues/1693), the chunked head/loss
  work, is claimed; do not duplicate it as an unowned task.
- [#2161](https://github.com/MakazhanAlpamys/Soup/pull/2161) remains under review;
  [#2162](https://github.com/MakazhanAlpamys/Soup/issues/2162) explicitly awaits
  maintainer scope/triage before a claim. Do not start a competing QKVO feature.
- [#842](https://github.com/MakazhanAlpamys/Soup/issues/842) already has merged
  fused-forward code and substitution caching; the issue remains for GPU measurement.
  A new transposed kernel is not a small unimplemented forward patch.

The useful 10Cache direction is profile-guided joint placement and prefetch using
actual reuse/waits. Soup already has the basic buffers/prefetch; the paper's speedups
are not a forecast. New kernel work follows an intended-production-setting profile.

## Completed 4B quality pilot

[Full report and evidence](nf4-boolq-4b-20261024/README.md).
Physical RTX 3060 Ti 8 GB, RAM-backed streaming, context 512, ordinary PEFT LoRA,
explicitly matched FP32 adapters, one seed, 256 training and 256 held-out examples.
Each format trained for 128 optimizer updates.

Post-training BoolQ accuracy: BF16 86.719%, NF4 85.547%; NF4 minus BF16 -1.172 pp,
paired 95% interval [-3.906, +1.172] pp. No acceptable degradation margin was agreed.
Keep BF16 as a control; this pilot cannot establish larger-model quality or convergence.
NF4 paired step-time reduction: 17.230%; peak process RSS 3.626 versus 9.045 GiB.
All four pilot instances and volumes, including retries, were deleted.

## Completed 32B NF4 cache study

[Full report, compact results and exact evidence](nf4-ssd-32b-20261024/README.md).
Physical RTX 4060 Ti 8 GB, physical RAM 30.997 GiB, effective native cgroup bound
29.756 GiB. Batch 1, accumulation 2, context 512; ordinary native LoRA rank/alpha 4
on seven decoder projections. Fixed double-quantized NF4 base and identical native
adapter initialization. No compile, Liger, FastLoRA or explicit FlashAttention change.

| Result | Ordinary disk | RAM cache | GPU residency | Both |
| --- | ---: | ---: | ---: | ---: |
| Optimizer step, s | 9.2608 | 8.6317 | 9.1622 | 8.5955 |
| Fresh six-update entry, s | 82.5211 | 84.8784 | 93.7806 | 94.1461 |
| Peak allocated VRAM, GiB | 3.6444 | 3.6444 | 5.4083 | 5.4083 |
| Peak process RSS, GiB | 5.8270 | 19.8284 | 5.9035 | 19.8955 |

Four balanced blocks: paired step-time reductions RAM 6.794%, GPU 1.085%,
both 7.153%. Adding GPU residency over RAM alone saves 0.386%; this and GPU-only
fail the unchanged >=2% practical gate. All intervals and arrays remain in the report.
The short complete entries are slower with caches; startup costs must be amortized.
Full staged-process wall clocks, including imports and monitoring, are also reported.

Seven different layers stay in VRAM and 56 in the 14 GiB RAM allocation. Together
63/64 decoder layers avoid recurring disk-source reads; vocabulary matrices still stream.
All eight numerical captures match bitwise, independently replayed before timing.
All 96 updates, final adapters, losses, actual frozen batches and 29 model-stage
RAM/swap proofs were independently audited. No container swap use was observed.
The original reference-first rtol=.01, atol=1e-5, finite and .3f loss gates are unchanged.

The separate diagnostic step lasted 9.2754 s: GPU kernel union 8.0085 s, copy union
5.2522 s; these overlap and are not additive. Source-get time without observed GPU
activity was 0.4352 s, not a pure SSD service-time estimate. Physical NVMe backing is
unverified. The controlled harness enables deterministic algorithms; allocation/fill
costs need a production-setting control before they justify kernel work.

Validation: 41 prelaunch local controls, 150 GPU-host controls including the real CUDA
promotion regression, and five profile-marker postprocessing checks passed. No full
suite or hosted-CI success is inferred. Raw retention: 796 files, 12,096,731,174 logical
bytes under C:/Users/esenb/Documents/SoupResearch/nf4-ssd-32b-20261024-c.

## What was repaired, and what comes next

The first NF4 joint-cache attempt exposed a genuine ownership bug: a RAM-promoted
layer incorrectly borrowed its obsolete rotating slot. The fix preserves checks for
genuine cold loans and cached-arena CUDA events. CPU red/green and two killed mutations
are retained; the real-CUDA regression and full 32B comparison now pass.

The next feature candidate is opt-in RAM-cache selection that accounts for actual
arena allocation, physical/cgroup headroom and expected training length. Smaller and
larger budgets, plus production-setting controls, precede a claim of optimal placement.
Do not unconditionally spend spare VRAM when its measured incremental benefit fails.

[10Cache](https://arxiv.org/html/2511.14124v1) motivates profiling reuse/order and
prefetch. Soup already has async reads, lookahead and reusable pinned arenas. The
paper's DeepSpeed gains are not Soup LoRA results. The old measured source predicted
19.33 GB pinned after seven GPU exclusions, above the host's 55% RAM term. The fresh
upstream packer changes that arithmetic as shown in the new opportunity screen above;
the original GPU results are unchanged. A GPU-exclusive full-RAM source still needs
implementation and real qualification. Payload/arena arithmetic is not a whole-run proof.

## Funding, history and cleanup

Starting remainder was $3.343605273574782; the user authorized up to $10 more if needed.
Combined allowance: $13.343605273574782. Charge every own retry, disk and transfer once.
The API key is shared by the group: unrelated rentals are neither adopted nor deleted,
and colleague spending is outside this task's ledger.

| Completed stage | Conservative USD | Evidence |
| --- | ---: | --- |
| Four 4B pilot leases, including retries | 1.842130 | [Pilot](nf4-boolq-4b-20261024/README.md) |
| 32B full-RAM admission refusal before training | 0.492097 | [Policy refusal](nf4-capacity-32b-20261024/README.md) |
| First NF4 disk run: staging-loan failure | 0.508311 | [Original failure and fix](nf4-ssd-32b-20261024-blocked/README.md) |
| CUDA fixture precondition failure before download | 0.312839 | [Fixture correction](nf4-ssd-32b-20261024-fixture-check/README.md) |
| Successful bounded NF4 four-arm retry | 0.771675 | [Completed study](nf4-ssd-32b-20261024/README.md) |

Aggregate since the 4B pilot: $3.927052; remaining $9.416553. Additional $10 allowance
used: $0.583447. Conservative accounting includes reserves; it is not an invoice.
Earlier studies were already deducted from the starting remainder and are not charged again.

Every instance and persistent volume created by this task is deleted. The final
instance 57539954 and volume 57539953 are independently absent by exact IDs.
Unrelated colleague instance 57533867 was preserved. All original results and failed
attempts remain retained; the public PR evidence uses lossless archives and hashes.
