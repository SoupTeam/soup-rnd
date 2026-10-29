# H3 additional direction: two-stage pipeline GPU validation

**The selected two-stage Qwen2 + LoRA pipeline smoke passes its predeclared
correctness/save/reload gates on two real RTX 3090 GPUs.** Two fresh runs each
completed four optimizer steps, with two accumulated native GPipe schedule calls
per step. Both independent adapter reloads had zero output error. This supplies
the additional selected-direction evidence requested in PR #42's H3 review.

See the [complete H3 acceptance map](h3-acceptance-20261029.md),
[pre-rental protocol](h3-pipeline-protocol-20261029.md) and
[raw artifacts/ledger](evidence/h3-pipeline-20261029/hardware/README.md).
This is a standalone research recipe; the general Soup CLI pipeline route and
upstream #39's 70B/DeepSeek/4xA100 requirements are not claimed complete.

## Configuration and actual execution

Measured clean source: **`55f675e17a20812d548054956d5af7edc7815d09`**.
The later result/evidence commits do not change the measured training code.
Executed harness SHA256:
`3ef3a96e45ad1a70baf2b69d167a89e47a1d8e4f09c1d3738d0e8df8679233f5`.

| Control | Value |
|---|---|
| Base | Real pretrained Qwen/Qwen2.5-0.5B-Instruct |
| Immutable revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Precision | IEEE FP32 base/adapters/gradients, eager attention, highest matmul, TF32/AMP/compile disabled |
| LoRA | q_proj/v_proj, r8, alpha 16, dropout 0; 96 tensors / 540,672 trainable elements |
| Data | 64 synthetic seeded random token rows, sequence length 128, labels=input_ids; not quality evidence |
| Seeds | Train/init42, token generator314159 |
| Partition | Stage0: embedding + blocks 0..11; stage1: blocks 12..23 + final norm/head |
| Actual schedule | Native `PipelineStage` + `ScheduleGPipe`, two CUDA ranks, no data-parallel fallback |
| Batch | Schedule batch4 = two microbatches of2; two accumulated schedule calls => **global8**, not16 |
| Optimizer | AdamW lr0.0002, betas0.9/0.999, eps1e-8, weight_decay0, global clip norm1 |
| Repeats | Two fresh four-step runs; eight distinct samples/step, rows 0..31 per run |
| Reference | Ordinary full model on each GPU, same init/data/microbatch order; adapters reset before pipeline |

Each micro-loss is normalized once by microbatches × accumulation.
`scale_grads=False` avoids repeatedly scaling earlier accumulated gradients.
Stage gradient norms are combined before global clipping. Every step compares
unclipped stage gradients and updated adapters against the ordinary control at
the unchanged **atol1e-5, rtol0.02**. Mean loss uses atol=rtol1e-5.
Resolved Pydantic configuration, prepared data, base-file hashes and the executed
commands are public in the artifact folder. Step count is an explicit smoke-loop
control, not an HF Trainer epoch schedule.

Actual parameter placement is measured from each local stage's parameters:

| Rank | Decoder blocks | Local parameter elements | Trainable elements | LoRA tensors |
|---|---|---:|---:|---:|
| 0 / cuda:0 | 0..11 | 315,353,600 | 270,336 | 48 |
| 1 / cuda:1 | 12..23 | 315,354,496 | 270,336 | 48 |

Trainable ownership is disjoint and its union is exactly the 96 expected keys.
Frozen tied embedding/head weights have a full copy on each stage. These local
counts therefore are **not FSDP shard counts** and must not be summed as unique
model parameters. The earlier FSDP2 evidence separately measures DTensor local
shards; pipeline does not replace or reinterpret that accounting.

Hardware: one Linux server, two NVIDIA GeForce RTX 3090, 24 GB each; AMD EPYC 7452,
PCIe `NODE` topology without NVLink. Driver 595.58.03 supports CUDA 13.2; the actual
Torch runtime is CUDA 13.0, NCCL 2.29.7, Torch 2.13.0+cu130, Transformers 5.17.0,
PEFT 0.20.0, Accelerate 1.14.0. Python 3.12 is identified by runtime paths; its patch
version was not separately captured. Full package freeze and hardware inventory
are retained. NCCL all-reduce and GPU P2P passed before model execution.

## Results

| Run | Optimizer steps/stage | Actual forwards/backwards per stage | Max gradient difference from control | Max updated-adapter difference | Reload logit error |
|---|---:|---:|---:|---:|---:|
| repeat-1 | 4 | 16 / 16 | 6.966292858e-7 | 1.057727786e-6 | 0.0 |
| repeat-2 | 4 | 16 / 16 | 6.966292858e-7 | 1.057727786e-6 | 0.0 |

All observed losses and adapter gradients are finite. Frozen stage weight hashes
are unchanged in all four rank records. The first step changes 48 B tensors and
leaves initial A unchanged, as allowed by LoRA initialization; all 96 adapter
tensors change by the final step. Final adapters from both fresh runs are
bitwise identical. Recorded mean losses were 14.048529, 13.960103, 13.822834, 13.781522;
this short synthetic smoke does not establish fine-tuning quality.

Each stage's 48-key adapter shard is saved independently, then merged with strict
duplicate/missing/unexpected-key checks. Both final 96-key safetensors match the
stage tensors bit-for-bit, with no base weights. A fixed 1x32 input is evaluated
through the actual two stage modules, with activation P2P, before serialization.
Separate fresh processes load the adapter over the same hashed base and compare
all logits at the predeclared **atol=rtol=1e-5**: both observed errors are 0.0.
Loaded adapter values/keys also match exactly. Training/reload PIDs are distinct
and raw eval inputs/references are public. This proves final adapter reload,
not periodic pipeline checkpoints or optimizer/scheduler resume.

Offline revalidation checks every archive/member SHA, first-gradient and
first/final-adapter comparisons, disjoint ownership, serialized values,
data/source identity and process separation. AST comparison confirms that the
launcher repair left all numerical training/configuration/reload functions and
tolerances unchanged from the pre-rental preparation. See `validation.json`.

## Memory and measurement boundaries

| Run | Rank | Peak allocated MiB | Peak reserved MiB | RSS before/after GiB | Host used RAM after GiB |
|---|---:|---:|---:|---|---:|
| repeat-1 | 0 | 1813.75 | 1886 | 3.094 / 3.146 | 45.715 |
| repeat-1 | 1 | 2832.39 | 3068 | 2.566 / 2.663 | 45.190 |
| repeat-2 | 0 | 1813.75 | 1886 | 3.058 / 3.109 | 44.809 |
| repeat-2 | 1 | 2832.39 | 3068 | 2.539 / 2.636 | 44.536 |

CUDA peaks are reset before pipeline training, after the full-model controls and
stage construction, and read before final eval/save/reload. They include probes
and the allocator's current stage buffers, not a minimum-memory configuration.
RSS and host-used RAM are point samples, **not peaks**; shared mappings mean RSS
must not be summed as unique RAM. System-wide host-used RAM is not job-owned RAM.

The diagnostic pipeline windows last 4.34–5.08 seconds/rank and include correctness
probes. **No pipeline throughput or speedup is admitted.** The H3 scaling
requirement remains covered by the separate matched FSDP2 campaigns, with their
original sequence length 512/precision/backend controls and warmup/timing boundaries.

## Failed preparation, correction and resource accounting

The first new rental 58210461, machine 48612, failed SSH authentication because
the server rejected authorized_keys ownership/modes. One scoped reboot did not
repair it; it was deleted before training. Its startup error and cleanup are
retained as infrastructure diagnostics.

Replacement 58212110 on machine 108411 passed 4 MiB transfers in both directions,
NCCL all-reduce and GPU P2P. Its initial entry-point attempt at `e74469aa` failed
to import Rich before loading the model: system `torchrun` used Python outside
the active venv. The repaired launcher uses
`sys.executable -m torch.distributed.run`. A real-subprocess regression with a
shadowed console entry point covers this defect. **14 targeted pipeline tests**,
full-scope Ruff and shell syntax passed before the repaired GPU attempt. The
earlier expanded 350-test selection remains separate; counts are not summed.
Failed entry-point logs/prepared inputs are retained and excluded from success.

The replacement kept the original overall one-hour deadline, without extension.
Only these two newly created IDs/labels were mutated. Original and refreshed
existing-machine exclusions were combined; no SSH command or mutation targeted
an existing instance. One unrelated instance disappeared between read-only
account snapshots, so no claim is made that every account-wide change was ours.

| Task rental | Outcome | Wall rental duration | Latest provider charge |
|---|---|---:|---:|
| 58210461 | SSH preparation rejected, no training | 668.69s | $0.115 |
| 58212110 | Two passing pipeline runs/reloads after launcher repair | 1054.05s | $0.126 |

New-stage reported total: **$0.241**, below the authorized $1. All H3 rentals:
**$2.589** reported, below the cumulative $4 cap. The conservative new-stage
estimate with reserve is $0.40; using the previous $2.60 conservative upper bound
gives **$3.00 cumulative**. These are provider snapshots/estimates, not a final
invoice. Both new rentals were deleted after collection, and API absence was
verified for all nine task-created rental IDs. There are no ongoing H3 rentals.

## Acceptance status and limits

The selected H3 technical deliverables now have direct evidence: real two-GPU
FSDP2+PEFT, per-rank/shard/memory accounting, adapter/checkpoint save/reload,
matched 1→2 scaling, this additional two-stage pipeline smoke and the acceptance
mapping. Final acceptance remains with the reviewer; the card was not edited.

4/8 GPUs, 70B/DeepSeek large recipes, physical multi-node, other parallel backends,
quality and optimizer/scheduler/RNG resume remain unmeasured/outside this chosen
slice. General CLI pipeline integration and the complete upstream issues remain
separate work. H1 organizational completion is unknown; its authoritative card
was not supplied. No CPU result is relabelled as hardware proof.
