# FlashAttention with persistent streaming layers: frozen reproduction

[Report](../../streaming-residency-flashattention-20261020.md) and
[results](../../streaming-residency-flashattention-results-20261020.json) describe
this experiment. `frozen-harness.zip` preserves the tested overlay, exact configs,
dataset, model manifest, dependency constraints, environment, GPU test record and
external-build provenance. Per-file sizes and SHA-256 hashes are in `manifest.json`
both beside and inside the archive. No provider controller, credentials, model
weights or captured training tensors are included in this small Git artifact.

`analysis/` contains the full detailed machine observations and independent
tensor-replay/environment-audit scripts. Public JSON is a compact review summary;
the detailed records remain losslessly archived rather than expanding the PR diff.
Local replay requires restoring those scripts to the retained study directory
with its receipts, protocol, remote progress and frozen `source-tree/` layout.

Use a **separate clean checkout** of upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`, with the archive's `overlay/` copied over
it. All 602 upstream production files remain unchanged. The older PR branch is
not the measured production baseline. Backend selection is explicit and confined
to a research constructor context; it does not enable a production configuration
flag or change checkpointing, optimizers, loss or residency selection.

The original tensor capture and comparator are byte-identical to the preceding
study. The existing semantic function retains every original AST node/check and
adds the same profiled unmasked forward to all arms. Semantic profiling is
excluded from the performance samples.

## External build

This study uses the authenticated previously built FA 2.7.4 wheel with SHA-256
`f7f5d839264e24505604c3d50bc2b0d6d00c001cfa596edf7601ca9e060a5582` and fresh
numerical qualification on this lease. It is **not a stock-wheel result**.
The build uses FA commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`, removes
the sole `--use_fast_math` flag, and enables `UNFUSE_FMA`,
`FLASHATTENTION_DISABLE_ALIBI`, and `FLASHATTENTION_DISABLE_SOFTCAP` to match the
pinned native Torch arithmetic. Kernel sources are unchanged.

`external-build/` retains the build script and provenance, authenticated recovery
record, and this lease's installation receipt. The recovery record explicitly
retains the earlier interrupted-export history; it authenticates the recovered
wheel, not an entire incomplete archive. The current study's exports have their
own complete verification receipts.

The wheel remains in the larger local retained archives rather than Git. To
rebuild it on an already available compatible Linux CUDA host, use the archived
build script in an otherwise fresh `/root/liger` layout with Python 3.12, Torch
2.8.0+cu128, CUDA 12.9 and the recorded dependencies. Rebuilding is optional;
do not mistake matching source/flags for an identical binary. Requalify any
rebuilt binary. No command in this evidence package rents hardware.

## Run on an already available host

The original layout is `/root/liger/repo` for the separate checkout,
`/root/liger/model` for the pinned model, and `/root/liger/results` for observations.
Use a fresh directory with that layout or explicitly adapt every affected path.
Place archived `inputs/` contents in `/root/liger`. Install the recorded package
versions using `constraints.txt` and run `pip check`. Install the qualified wheel
with `--no-deps --target /root/liger/fa274-nofast`.

Before any Torch import:

```bash
export PYTHONPATH=/root/liger:/root/liger/repo:/root/liger/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_DISABLED=true
export FLASH_ATTENTION_DETERMINISTIC=1
mkdir -p /root/liger/results
cd /root/liger/repo
python /root/liger/prepare_7b.py
```

Preparation downloads the pinned checkpoint, converts its original `.bin` shards
with `weights_only=True`, and preserves tensor names/dtypes. Check model/data
hashes against the archived manifests. Use the archived `b1.yaml`/`b2.yaml`.
The dataset is a synthetic numerical/performance fixture, not task-quality data.

Run the archived GPU test entry first:

```bash
python /root/liger/pytest_gpu_entry.py tests/test_streaming_residency_candidate.py tests/test_streaming_residency_attention_bench.py --no-cov -q
```

Every capture/timing invocation is a fresh process. `pin_entry.py` selects four
distinct GPU-local physical CPU cores before Torch imports and records the choice.
Compare the **same residency** with SDPA versus the candidate:

```bash
python /root/liger/pin_entry.py /root/liger/entry.py --attention sdpa --indices 12,13,14,15,16,17,18,19 --config /root/liger/b1.yaml --output /root/liger/results/check-native --capture
python /root/liger/pin_entry.py /root/liger/entry.py --attention hybrid --indices 12,13,14,15,16,17,18,19 --config /root/liger/b1.yaml --output /root/liger/results/check-hybrid --capture
```

The default allocator cap is 10 GiB. `--attention external` selects pure external
FA; `hybrid` preserves native prepared masks and uses external FA only for unmasked
inputs. Constructor selection is verified, and hybrid route counters are retained.
Compare saved directories with
`benchmarks.harness.liger_streaming_parity.compare(reference, candidate)` and
separately check `torch.equal` to assess the stronger bitwise result. Preserve
the original tolerances. Repeat both microbatches, indices 8 through 23 for
resident16, a repeated native control, and the `--semantics` captures.

Only configurations passing every numerical/semantic gate proceed to timing.
Replace `--capture` with `--warmup 6 --steps 8`, using four fresh-process pairs in
AB/BA/BA/AB order for each residency/microbatch. Compare all final adapter tensors
and loss histories after 14 updates. Retain synchronized step windows, allocated
and reserved peaks, host memory, weight-copy accounting, CPU affinity, GPU clocks
and peer telemetry. Use the archived protocol's practical gate and bootstrap;
never pool the preceding lease's timings.

The larger raw tensor archives and all original-file manifests remain locally at
`output/residency-fa-vast-20261020/bundles/`; the public results list their export
receipts and hashes. Local replay commands require that retained layout. The
small committed package supports a fresh experiment and source review; it does
not replace the raw evidence.
