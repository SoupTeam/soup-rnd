# Fast-PCIe lease B: VOID performance observations

**All speed and VRAM comparisons from this lease are VOID for performance
interpretation.** Final telemetry audit found up to three GPU compute processes,
with pre-existing activity before this study's model work. Numerical captures,
raw timing observations and the failed environmental audit are retained. No B
timing is pooled with A or used as a clean performance result. Lease C repeats
the protocol with an idle-GPU preflight and abort-on-peer checks; see the main report.

[Combined report](../../streaming-residency-flashattention-20261020.md) and
[this lease's results](../../streaming-residency-flashattention-fast-pcie-results-20261020.json)
record a separate confirmation on faster CPU-to-GPU transfers. It was planned
after the first lease's timings started, because that offer advertised only
6.3 GB/s PCIe bandwidth. The first lease was completed and deleted before this
lease was rented. Its results remain preserved; no timings are pooled.

This confirmation requires >=20 GB/s advertised PCIe and >=16 GB/s measured
pinned H2D median before model work. It tests resident16 (decoder layers 8–23),
microbatches 1/2, with fresh native controls, native repetition and complete
numerical/semantic qualification before timing. The highest residency was
selected to reduce transfer masking, not to select a favorable attention outcome.

The production and GPU research sources are identical to lease A: upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`, all 602 production files unchanged,
and frozen full source archive SHA-256
`fc843cff5c56bfd875b5f8f3808f9d18972a3583cffdc92c152715722a85e6c7`.
Use a separate checkout with this ZIP's `overlay/`; the older PR branch is not
the measured baseline.

`frozen-harness.zip` and its per-file SHA-256 `manifest.json` preserve tested
sources, exact inputs, model/data manifests, package constraints, GPU test
record and external-build provenance. `analysis/` holds detailed machine
observations and independent replay/environment-audit code. Public JSON is a
compact review summary; no detailed observations are discarded. Raw training
tensors remain in the larger local archives listed by the export receipts.

The qualified FA 2.7.4 wheel SHA-256 is
`f7f5d839264e24505604c3d50bc2b0d6d00c001cfa596edf7601ca9e060a5582`.
Native Torch arithmetic macros and removal of `--use_fast_math` are required;
this is not a stock-wheel result. Build provenance is included. The
[shared reproduction guide](../streaming-residency-flashattention-20261020/README.md)
describes the pinned stack, build and directory layout. Requalify a rebuilt binary.

## Run on an already available Linux CUDA host

No command here rents hardware. Use the fresh `/root/liger/repo`,
`/root/liger/model`, `/root/liger/results`, `/root/liger/fa274-nofast` layout
and environment from the shared guide. Copy archived `inputs/` into
`/root/liger`; install the recorded stack and external wheel. Before preparing
the model, select four GPU-local physical CPU cores and run the hardware probe:

```bash
python /root/liger/pin_entry.py /root/liger/h2d_probe.py
python /root/liger/prepare_7b.py
python /root/liger/pytest_gpu_entry.py tests/test_streaming_residency_candidate.py tests/test_streaming_residency_attention_bench.py --no-cov -q
```

Use frozen `b1.yaml`/`b2.yaml`. Every capture/timing invocation is a fresh process:

```bash
python /root/liger/pin_entry.py /root/liger/entry.py --attention sdpa --indices 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --config /root/liger/b1.yaml --output /root/liger/results/check-native --capture
python /root/liger/pin_entry.py /root/liger/entry.py --attention hybrid --indices 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --config /root/liger/b1.yaml --output /root/liger/results/check-hybrid --capture
```

Compare saved directories with the archived unchanged comparator; separately
check `torch.equal` for the stronger observed agreement. Repeat batch 2, native
batch-1 repetition, and both semantic captures with `--semantics`. Preserve the
10 GiB allocator cap and original numerical tolerances. Only after every gate
passes, replace `--capture` with `--warmup 6 --steps 8`, use four AB/BA/BA/AB
pairs per microbatch and compare all final adapters/losses after 14 updates.
Retain memory/environment records and apply the unchanged practical speed gate.

Raw evidence remains at `output/residency-fa-fast-vast-20261020/bundles/`.
For local replay, restore archived analysis scripts into that study directory
with its receipts, `remote-progress.json`, `protocol.json` and `source-tree/`.
Replay reads original tensors and does not provision hardware. The separate
hardware probe uses 32 warmup copies and 32 measured copies; all samples remain
in the evidence. Its bandwidth is not an attention or training throughput result.
