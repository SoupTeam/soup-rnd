# Lease D: final fast-PCIe repetition

[Combined report](../../streaming-residency-flashattention-20261020.md) and
[results](../../streaming-residency-flashattention-final-pcie-results-20261020.json)
cover resident16 batch1/2. D was registered after C passed accuracy, encountered
local disk exhaustion before timing, was automatically deleted, and had every
original capture recovered and verified. B's peer-contaminated measurements remain
VOID for performance. No timings are pooled across leases.

The full GPU source archive remains
`fc843cff5c56bfd875b5f8f3808f9d18972a3583cffdc92c152715722a85e6c7`.
All 602 production files match clean upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`. Numerical tolerances, original capture
and comparator, inputs, resident layers, warmup and timing method are unchanged.

`frozen-harness.zip` and its internal/external per-file SHA-256 manifests retain
the tested overlay, exact inputs, GPU tests, environment, custom external build
provenance, detailed observations and independent replay/environment-audit code.
The public JSON is a compact review summary. No raw observations are discarded.
Larger raw tensor archives remain locally under
`output/residency-fa-final-vast-20261020/bundles/`, with verified export receipts.

Result retention uses streaming extraction: hash every payload against its
manifest, verify any identical existing file, then preserve the new original
path with a hardlink. Every canonical archive remains unchanged and retained.
Lossless gzip level 1 reduces packaging time. These transport/storage changes
run outside measured training; they do not modify GPU model arithmetic or gates.
The registered preallocation reserve is >=8 GiB local free space, and the
result-transfer preflight requires >=2 MiB/s. All original paths and bytes remain.
No provider credentials, rental controller or model weights are published.

Use the [shared reproduction instructions](../streaming-residency-flashattention-20261020/README.md)
on an already available compatible Linux CUDA host. The pinned custom FA 2.7.4
wheel and native arithmetic flags require fresh qualification if rebuilt; this
is not a stock-wheel result. No command here rents hardware.

Before Torch imports, retain three idle-GPU samples five seconds apart: no compute
processes, <=10% GPU usage and <=256 MiB occupied device memory. Record the UUID.
Abort if sampled compute processes exceed one during a study stage. This does
not prove whole-host CPU exclusivity. Record the separate H2D qualification:

```bash
python /root/liger/pin_entry.py /root/liger/h2d_probe.py
python /root/liger/prepare_7b.py
python /root/liger/pytest_gpu_entry.py tests/test_streaming_residency_candidate.py tests/test_streaming_residency_attention_bench.py --no-cov -q
```

Use the frozen `b1.yaml`/`b2.yaml`, indices 8 through 23, and 10 GiB allocator cap.
Capture native/hybrid for both batches, repeat native batch1, and run both semantic
captures. Require all original numerical gates before timing. Then run four
fresh-process pairs per batch in AB/BA/BA/AB order, six warmup and eight measured
updates per process. Compare every final adapter tensor and logged loss after all
14 updates; apply the unchanged practical speed gate. The H2D probe is a separate
hardware measurement, not an attention or training throughput result.

For local replay, restore archived `analysis/` scripts into the retained study
directory with its protocol, receipts, progress and frozen `source-tree/`.
The small public package supports source review and a fresh experiment; stored
observation replay requires the retained raw tensor archives.
