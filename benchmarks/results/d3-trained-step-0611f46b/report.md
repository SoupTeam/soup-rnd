# D3 complete trained OLMoE step study: 84/84 passed

Completed on 2026-10-23 (Asia/Qyzylorda), under prospective protocol revision 8
and actual source commit `0611f46b4407a238ec549b0d8ce16ed4bd3b4c8c`. The single session
`67c826b7c5ed4327899952fe05376b1b` consumed exactly 18 correctness cells, 60 timed cells
and six CUDA profiles. Every cell exited successfully with required F1 OK.
No replacements or formal reruns were used. The earlier six incomplete sessions
remain separate; the rejected revision-7 preflight was never a formal session.

This is a complete-model forward/backward/SGD experiment with a **frozen base
and the declared rank-4 post-embedding residual adapter**. It is not full-weight
fine-tuning or public Soup trainer integration.

## Recipe and hardware

Pinned trained model: `allenai/OLMoE-1B-7B-0924`, revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`. All sixteen native OLMoE decoder
layers execute, with eager native routing/attention and deterministic synthetic
token IDs (batch 1, 512 tokens). Four SGD steps, one warm-up and three measured
steps per timing arm; five fresh-process ABBA blocks per stored format.

One RTX 4090 / 24 GB on a KVM VM, driver 580.95.05, Python 3.12.15,
torch 2.14.1+cu130, Triton 3.8.0, transformers 5.19.0, bitsandbytes 0.50.2
and safetensors 0.8.0. Exact GPU, cuBLAS, checkpoint, bundle, source and F1
instrument identities were frozen before the session. Guest-wide diskstats
were validated on all 48 prepared shards. Physical-host storage, cache and
isolation are not certified.

Both timed arms use the same bounded grouped weight pool and GEMM policy:
native OLMoE GEMM for rows 1–128 and 481+, grouped GEMM for rows 129–480.
The control uses native SiLU/multiply; the candidate uses the optional Triton
activation kernel. Native BF16, no TF32 and no reduced/split-K accumulation
remain fixed. The corrected Ada dispatch profile is distinct from historical
RTX 3090 and cutoff-16 studies.

## Exactness and controls

All 18 correctness cells passed across dense BF16, NF4 and nested NF4:
native, sequential, grouped resident/streamed, and Triton resident/streamed.
Full saved native step tensors and adapter states matched exactly; A/A,
effective active-expert negative and restored-positive controls passed.
All timing/profile steps retained their required native-oracle checks and
positive actual grouped dispatch coverage. Twelve complete native tensor
files are retained in the verified full supplemental archive.

## Descriptive timing results

The seconds columns are medians of the five within-block averages, each
combining both occurrences of that mode. The change column is the median of
the five **paired relative ABBA contrasts**; it is not a ratio of the two
seconds columns. Negative means the Triton candidate used less time.

| Stored format | Native activation, s/step | Triton, s/step | Paired median change | Paired block range |
|---|---:|---:|---:|---:|
| dense | 5.829516 | 5.798648 | -1.3891% | -4.8537% to +8.7208% |
| nf4 | 3.118096 | 3.184708 | +0.8272% | +0.6040% to +5.5095% |
| nf4-dq | 3.083363 | 3.159585 | +1.8686% | -2.7676% to +5.7447% |

These small, mixed differences do not establish a general training speedup.
There is no declared significance or convergence claim and no production
default change. Descriptive outcomes remain valid even without a speedup.

## Whole-step allocator peaks

Medians of the ten timing-arm peaks per mode; MiB = 2^20 bytes.
Reserved memory is allocator-reserved storage, not total device usage.

| Stored format | Native allocated MiB | Triton allocated MiB | Native reserved MiB | Triton reserved MiB |
|---|---:|---:|---:|---:|
| dense | 2852.212 | 2852.212 | 2976.000 | 2976.000 |
| nf4 | 2867.975 | 2867.975 | 2988.000 | 2988.000 |
| nf4-dq | 2870.741 | 2870.741 | 2988.000 | 2988.000 |

**No reduction of the observed whole-step allocated/reserved peak was measured.**
The earlier isolated activation-tensor saving does not imply lower total model
VRAM. Process RSS and all individual timing/memory samples remain in the raw
records and summary. Weight-arena and temporary budgets are not total VRAM/RAM.

## Retention and CPU replay

`raw-records.zip` contains all 84 result/log records, session, launcher receipts
and the frozen source. Six separate profile ZIPs retain the original CUDA
traces; `payload-locations.json` maps every published payload to its archive.
`analysis.json.gz` preserves the frozen analyzer's exact uncompressed output.
Archives were separated only to keep each Git payload below the file limit.

The full supplemental ZIP is **931,258,444 bytes**, SHA-256
`101f57588dcac92f515225251250260fcbc5c2e22f24a4a9ae6d1efcbc24e7ba`. Its verified local copy is retained outside Git
under `C:/ml/soup/work/`. Only the twelve native tensor files are omitted from
the public packages; their sizes/hashes and full-backup identity remain in
`native-oracle-retention.json`. Do not discard that full backup.

Local CPU verification checked all 857 payload hashes,
all 659 source blobs against actual Git commit
`0611f46b`, all twelve native tensor-file hashes and ZIP CRCs. The archived
analyzer regenerated the original analysis bytes. This was an archive replay,
not another CUDA execution. The 346 passed selected preflight regressions and
one disclosed root-permission skip are recorded separately in the VM
preparation record; a new full repository/platform test run is not claimed.

With Python 3.12 and a checkout containing the measured commit:

```text
python verify.py --full /path/to/d3-trained-step-0611f46b-full.verified.zip --checkout /path/to/soup-rnd --extract /path/to/NEW-directory --out replay-verification.json
```

The output directory must not already exist. No GPU or model download is
required for this hash/analyzer replay. Independent review and final D3 task
acceptance remain separate. This study does not establish language quality,
convergence, useful-data learning, full-weight training or a supported trainer
integration. No new rental or rerun is implied by those scope limits.
