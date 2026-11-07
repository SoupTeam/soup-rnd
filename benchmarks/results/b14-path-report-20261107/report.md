# B1.4 ticket 09: the per_expert path on an 8 GB GPU

This is a capacity report. It says what fits and runs, how much memory it takes, how much it
reads from disk and how many GPU kernels one pass launches. It states no speed. The measurement
sections of the result files hold no duration. Only bookkeeping carries time (`environment.uptime`,
the log timestamps, the rental length below), and no throughput is computed from it.

## Result

On a physical NVIDIA GeForce RTX 3070 (8277196800 bytes = 7.71 GiB, vast.ai), one forward and
backward pass of the per_expert streamed path ran to completion, with all trainable LoRA
gradients non-zero, for two models:

| Model | Weights in bf16 | Largest routed-expert layer | Peak VRAM allocated / reserved at length 2048 | Result |
| --- | ---: | ---: | ---: | --- |
| Real: `ByteDance-Seed/academic-ds-9B` (9.37 B params, 15 MoE layers x 64 experts) | 18.73 GB | 1.11 GB | 4.82 GB / 5.79 GB | passes, 160 of 160 LoRA gradients non-zero |
| Synthetic, random weights (11.27 B params, 2 MoE layers x 640 experts) | 22.55 GB | 11.07 GB | 1.31 GB / 1.79 GB | passes, 30 of 30 LoRA gradients non-zero |

- The real model is 2.3 times the card's memory. The synthetic expert layer (11.07 GB) is larger
  than the whole card (8.28 GB). The path holds one expert slot of 17 301 504 bytes in VRAM per
  pass for both; the rest of the weights stay on disk (experts) or in host RAM (other layers).
- No out-of-memory error occurred at any measured length (16, 512, 2048). The limit of the card
  was not searched: lengths above 2048 were not run, so "fits at 2048" is the claim, not "2048 is
  the maximum".
- Peak host RAM during a pass of the real model was 5.7 to 6.4 GB on the 3070 host (idle 3.3 GB).
  The setup step that writes the per_expert cache peaked higher, at 10.7 GB.
- One pass of the real model at length 2048 launched 86 730 CUDA kernels and read 49.7 GB of
  expert weights from storage (2.65 times the checkpoint size).

## What was measured and how

One pass is one forward and one backward of `build_streamed_model` (device `cuda`, bfloat16,
`tier=ram`, `quant=none`, `buffers=2`, pinned host buffers), batch of one row of seeded random
token ids, labels equal to the ids, LoRA r=2 on the MLA projections
(`q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj`) with non-zero B factors. One
warm-up pass at the shortest length runs first and is not measured. Lengths 16, 512, 2048 run in
order. The shortest length runs twice and its counts must be identical (they were, on all six
runs; `repeat.identical` in each JSON).

| Quantity | How |
| --- | --- |
| VRAM, idle | `torch.cuda.memory_allocated` and `memory_reserved` after the model is built, before any pass |
| VRAM, peak | `torch.cuda.max_memory_allocated` and `max_memory_reserved` after `reset_peak_memory_stats` before the pass |
| RAM, idle | process RSS (`psutil`) after the model is built |
| RAM, peak in a pass | largest RSS sample of a thread that polls every 20 ms. It can miss a shorter spike, so it is a lower bound |
| RAM, whole-process peak | `ru_maxrss` after cache creation (`rss_peak_after_shard_bytes`) |
| Disk, logical | `expert_bytes_read` from the expert pool (stored bytes of each read) and `expert_fetches` |
| Disk, physical | `read_bytes` from `/proc/self/io` (psutil), the bytes that reached storage |
| Layer loads | `layer_loads`, `layer_restores`, `large_loads` from the runtime stats, as the difference over the pass |
| GPU kernel launches | `torch.profiler` with CUPTI: events of category `kernel` in the exported Chrome trace, forward and backward profiled separately and added |
| Cross-check of launches | the `cudaLaunchKernel` family of runtime events in the same trace (`runtime_launch_calls_total`), and the ATen operation count from `TorchDispatchMode` |

### How launches are counted (card checkbox 4, plan step 22)

Chosen method: kernel events of `torch.profiler`. This is the quantity the card names ("number of
GPU operation launches"). Two cross-checks are recorded next to it:

- The `cudaLaunchKernel` runtime calls in the same trace. On torch 2.11 (Kaggle) they equal the
  kernel count exactly on every pass. On torch 2.7.1 (vast) they are 1.5 to 2.9 percent lower on the CUDA runs of the large models and equal on the tiny one
  (for example 55 260 against 56 871 on the real model at length 16). I did not find which launch
  call the missing 1 611 kernels use; the kernel events are the count reported.
- The ATen operation count (`TorchDispatchMode`). It counts dispatcher operations, not kernels,
  and it differs with the torch version (see below), so it is not used for the launch claim. It
  is recorded because it also works on CPU, which is where the counter is tested.

The counter itself is `OpCounter` in `benchmarks/harness/b14_path_report.py`, tested on CPU in
`tests/test_b14_path_report.py` (a known number of `mm` and `add` operations, backward counted,
no CUDA fields on CPU, a refusal when CUDA profiling is asked on a host without CUDA). The CUDA
count has no unit test on CPU. It was checked by running the harness on the tiny synthetic
checkpoint on two GPUs before the large runs (the first Kaggle attempt failed on a CPU batch
and is not kept). Two sources agree on it: kernels against runtime launch calls, above.

Expert-attributed counts (forward only): hooks on each `mlp.experts` module count the ATen
operations run inside its forward. They give 77.0 (torch 2.7.1) or 68.9 (torch 2.11) operations
per visited expert on the real model. The backward of an expert runs inside an autograd node with
no module boundary, so the backward launches are not attributed to experts; only totals are
reported for the full pass.

## Results by machine

Visited experts are the distinct (layer, expert) pairs the router sent tokens to in the pass,
counted by a forward hook on the router. Each visited expert is read three times (forward,
recompute during backward, backward reload): `expert_fetches` is exactly 3 x visited on every
row below. That matches the B1.3 report ("exactly three reads per expert visit") and is also
asserted by the unit tests.

Columns: VRAM and RAM in GB (10^9 bytes). Loads are `layer_loads+layer_restores+large_loads`.
Kernels are CUDA kernel launches of the pass (forward + backward).

### Real checkpoint `academic-ds-9B` (revision `2a205727f616995ad4aba57806f8e2d0d2f28743`)

RTX 3070, 8 GB (physical card of the card's constraint), torch 2.7.1+cu126:

| Length | VRAM peak alloc | VRAM peak reserved | RAM peak (pass) | Visited | Fetches | Expert bytes | Storage read | Loads | Kernels fwd + bwd | Kernels per visited |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 16 | 0.99 | 1.03 | 5.66 | 617 | 1851 | 32.03 | 32.04 | 29+0+2 | 10 729 + 46 142 = 56 871 | 92.2 |
| 512 | 1.93 | 2.05 | 6.30 | 945 | 2835 | 49.05 | 49.07 | 29+0+2 | 15 438 + 68 384 = 83 822 | 88.7 |
| 2048 | 4.82 | 5.79 | 6.40 | 957 | 2871 | 49.67 | 49.70 | 29+0+2 | 16 525 + 70 205 = 86 730 | 90.6 |

Idle: VRAM 0.95 GB allocated, 0.96 GB reserved; RAM 3.29 GB. Cache on disk 18.73 GB. Setup RAM
peak 10.69 GB. Fixed buffers: 0.94 GB in total on the device (includes the 0.53 GB large-layer pool and the
0.017 GB expert slot), 2.15 GB pinned host memory.

Tesla T4, 16 GB (Kaggle, a different capacity, not the card's constraint), torch 2.11.0+cu130:

| Length | VRAM peak alloc | VRAM peak reserved | RAM peak (pass) | Visited | Fetches | Storage read | Kernels fwd + bwd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.99 | 1.03 | 5.36 | 616 | 1848 | 31.99 | 11 019 + 46 400 = 57 419 |
| 512 | 1.93 | 2.09 | 6.64 | 944 | 2832 | 49.02 | 15 691 + 68 456 = 84 147 |
| 2048 | 4.82 | 6.37 | 7.89 | 957 | 2871 | 49.70 | 16 830 + 70 396 = 87 226 |

On the real model, peak allocated memory is identical on both GPUs at every length (993 306 112;
1 927 735 808; 4 822 503 936 bytes). Reserved memory differs. On the synthetic model it does not
hold at length 2048 (1.31 GB on the 3070, 1.71 GB on the T4); the two hosts also run different
torch versions (2.7.1 and 2.11), and I did not test which one causes the difference. The router chose 36 to 51 of the 64 experts per layer
at length 16 and 60 to 64 at 512 and 2048, so the two longer rows read nearly every expert once
per pass over three reads. A real-text batch would hit a similar number (ticket 08 saw 44 to 62
on 15 tokens). The visited counts differ by at most 1 between the machines (the same routing
difference ticket 08 saw, cause not tested).

### Synthetic oversized checkpoint (random weights, no language, no training)

Written by `b14_synthetic_checkpoint.py`: 3 layers (1 dense, 2 MoE), 640 routed experts per MoE
layer, hidden 2048, expert width 1408, bfloat16. One layer of routed experts is 11 072 962 560
bytes, 1.34 times the card's total memory. Disk: 22.55 GB checkpoint plus 22.55 GB cache.

RTX 3070, 8 GB, torch 2.7.1+cu126:

| Length | VRAM peak alloc | VRAM peak reserved | RAM peak (pass) | Visited | Fetches | Storage read | Kernels fwd + bwd | Kernels per visited |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 | 0.35 | 0.36 | 11.62 | 62 | 186 | 3.22 | 1 260 + 5 009 = 6 269 | 101.1 |
| 512 | 0.58 | 0.70 | 11.70 | 159 | 477 | 8.26 | 2 642 + 11 590 = 14 232 | 89.5 |
| 2048 | 1.31 | 1.79 | 11.72 | 187 | 561 | 9.71 | 3 195 + 13 611 = 16 806 | 89.9 |

Tesla T4, 16 GB (Kaggle, other capacity): peak allocated 0.36, 0.58, 1.71 GB; kernels 6 388,
14 179, 16 920; visited 62, 158, 187; same fetches = 3 x visited.

The RAM figures of this model (idle 11.3 GB on the 3070 host, 14.5 GB on Kaggle) are an upper
bound and not the cost of the path: the runtime store holds only 0.40 GB. The cause is not
verified; the likely one is memory of the checkpoint writer and the cache step that the same
process did not return to the operating system. The real-model run, which downloads instead of
writing, idles at 3.3 GB. Do not read the synthetic RAM as a requirement.

### Tiny synthetic checkpoint (counter check, not a capacity result)

The 8-expert fixture of `b13_moe_fixture`, float32, on the RTX 3070 (torch 2.7.1) and the T4
(torch 2.11): 2 185 and 2 167 kernels at length 16 (source `3c215ba7` and `9c3533f9`; the second
differs only in the shortened kernel names of the evidence, a later commit). Files:
`vast-rtx3070/result-tiny.json`, `kaggle-t4-tiny/result-tiny.json`.

## Operations per expert (input for ticket 10, the D3 discussion)

- Total kernels of a pass divided by visited experts is 88.7 to 93.2 on the real model and 89.5 to
  103.0 on the synthetic one. This includes everything that is not an expert (embedding, MLA
  attention, norms, LoRA, shared expert, the loss), so it overstates the per-expert cost, most at
  short lengths where the fixed part is divided by fewer experts.
- The marginal cost is lower and steadier. Between length 16 and 512 the visited set grows and
  the sequence-independent kernels stay, so the difference of kernel counts divided by the
  difference of visited experts estimates the kernels of one more visited expert (forward,
  recompute and backward together): 82.2 (real, RTX 3070), 81.5 (real, T4), 82.1 (synthetic,
  RTX 3070), 81.2 (synthetic, T4). These come from two points per run, not a fit, and the
  attention kernels also grow with length. The same estimate between 512 and 2048 is not valid
  (the visited count barely changes while the sequence grows) and is not used. The tiny fixture
  gives 65 and 95 on its two GPUs, too few experts for a stable difference.
- The fixed part that does not scale with experts is about 6 000 kernels on the real model
  (56 871 minus 617 x 82.2) and about 1 200 on the synthetic one.
- Read per visited expert: 3 reads and 17 301 504 bytes of slot, as above. 12 to 14 device copy
  events per visited expert (`memcpy_events_total`: 8 092 / 617 on the real model at length 16).
- ATen operations per added visited expert: 139.0 on torch 2.7.1 and 116.0 on torch 2.11
  (identical for the real and the synthetic checkpoint on one torch), while kernel totals are within 2 percent across the two
  versions (1.0 percent on the real model at length 16, 1.9 percent on the synthetic one). Quote kernels, not ATen operations, across torch versions.

P3 estimates grouped routes at 2 to 10 percent and names small matrix multiplications as a risk.
This report gives D3 the baseline count; it does not test grouping.

## Limits

- One forward and one backward pass per length, batch of one row. This is not an optimizer step,
  not the L2L trainer, and not a training run. The public L2L launch guard for MoE is unchanged.
- Random token ids, not text. Loss values (13.8 to 16.7 on the real model) mean nothing.
- Correctness was not compared on CUDA in this report. Ticket 08 proved bitwise equality with a
  resident model on CPU only; the only CUDA evidence here is that the LoRA gradients are non-zero
  and the counts repeat. No CUDA resident reference was run (it does not fit 8 GB).
- bf16 and `quant=none` only. No NF4, no fp32, no disk tier (`tier=ram`; with it the non-expert
  weights sit in host RAM: store 2.12 GB for the real model).
- Lengths above 2048 and batch sizes above one were not run. Capacity above that is unknown.
- The expert-attributed counts cover forward only. The cost of a backward expert is inside the
  pass total.
- Physical reads: `read_bytes` equals `expert_bytes_read` within 0.05 percent, so page-cache hits
  did not reduce it in these runs on a host with more RAM than the checkpoint. I did not check why
  repeated passes over the same files were not served from cache; treat `read_bytes` as measured
  on these hosts, not as a rule.
- The layer-load counter reads 29 loads for 16 layers in every pass of the real model (16 forward
  loads plus 13 in backward by arithmetic). The synthetic runs read 3 loads for 3 layers, so no
  backward reloads. I did not find why the two differ.
- Peak RSS is a sample every 20 ms, a lower bound. The whole-process `ru_maxrss` is the other
  bound and contains the cache step.
- Two torch versions and two hosts: RTX 3070 host (torch 2.7.1+cu126, i7-11700KF, 109 GiB RAM,
  cgroup limit 80 GB) and Kaggle (torch 2.11.0+cu130, 2 x T4 with device 0 used, 31 GiB RAM).
  The Kaggle GPU is 16 GB, so its rows do not speak for the 8 GB constraint.
- The board steps of the ticket (assign, move to "В работе" and "На проверке") are not done here;
  they need a person.

## Environment and cost

- vast.ai instance 59466949, offer 58572966 (RTX 3070 8192 MiB, driver 615.71.09, CUDA runtime
  12.6), image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, 200 GB disk. Ran 1 111 s
  according to `vastai show instance`, at 0.151 USD/h, so about 0.05 USD (ceiling set at 2 USD).
  Destroyed after the files were copied. The team credit moved from 65.29 to 65.22 USD over the
  rental, which includes other team use.
- Python 3.11.13, torch 2.7.1+cu126, transformers 5.20.2, peft 0.21.2, safetensors 0.8.0,
  psutil 7.0.0 (`vast-rtx3070/pip-versions.txt`, `environment` in each JSON).
- Kaggle script kernel `aldiyarsagat/soup-b14-path-big` (stages `oversized` then `real`) and
  `aldiyarsagat/soup-b14-path-tiny`, source from the private dataset
  `aldiyarsagat/soup-b14-source`. Free.
- Measured source: `3c215ba7cfc5cb643fda1f2fab2274673e126d1f` for every result except the Kaggle
  tiny run (`9c3533f9`, see above). `source_sha` is in each JSON.

## Files

- `vast-rtx3070/`: `result-{tiny,oversized,real}.json`, `run-*.log`, `stages.txt`,
  `remote.sh` (the launch script), `pip-versions.txt`
- `kaggle-t4/`: `result-{oversized,real}.json`, `soup-b14-path-big.log`, `run.py`,
  `kernel-metadata.json`
- `kaggle-t4-tiny/`: `result-tiny.json`, log, `run.py`, `kernel-metadata.json`
- `SHA256SUMS`

Harness: `benchmarks/harness/b14_path_report.py` (counters, `measure_path`),
`b14_synthetic_checkpoint.py` (writer), `b14_path_run.py` (remote entry).
Tests: `tests/test_b14_path_report.py` (10 tests, CPU).

## Reproduce

The Kaggle tiny run used lengths 16 and 256 (`B14_SEQS=16,256`); every other run used 16, 512,
2048.

```text
# on a CUDA host, from the checkout root, deps as in pip-versions.txt
export B14_SRC=$PWD B14_OUT=out B14_WORK=/root/work B14_SEQS=16,512,2048 B14_TIER=ram
export PYTHONPATH=$PWD/src
B14_STAGE=tiny      python benchmarks/harness/b14_path_run.py
B14_STAGE=oversized python benchmarks/harness/b14_path_run.py   # needs ~45 GB disk
B14_STAGE=real      python benchmarks/harness/b14_path_run.py   # downloads 18.7 GB, ~40 GB disk
python -m pytest tests/test_b14_path_report.py --no-cov -v       # CPU tests of the counters
```
