# Lease C: accuracy-only interrupted run

**C passed all numerical qualification captures but started no timing processes.**
Local disk filled during extraction of its second capture archive. Automatic
cleanup deleted the instance. Both canonical archives were already downloaded;
all 57 original files in the affected package were recovered and hash-verified.
The interrupted state, extraction failure and recovery remain explicitly recorded.
No completed speed/VRAM comparison is claimed for C. D repeats the same GPU source
after fixing result retention; see the combined report.

[Combined report](../../streaming-residency-flashattention-20261020.md) and
[results](../../streaming-residency-flashattention-clean-pcie-results-20261020.json)
record the resident16 batch1/2 numerical qualification. It was registered after lease B's
deletion and failed environmental audit. B's complete numerical/timing evidence
remains archived with a VOID performance status; no samples are pooled.

The full GPU source archive is unchanged:
`fc843cff5c56bfd875b5f8f3808f9d18972a3583cffdc92c152715722a85e6c7`.
All 602 production files match clean upstream
`d500ae0dca7c4da905f8d59c9866af2f8aaa824b`. The original comparator and numerical
tolerances are unchanged. Use a separate checkout, not the older production
source on this PR.

`frozen-harness.zip` and the internal/external `manifest.json` retain the tested
overlay, exact inputs, GPU tests, environment, custom FA build provenance,
detailed machine observations and independent replay/environment-audit code.
The compact public JSON does not replace the detailed observations in the ZIP.
The larger raw tensor archives remain locally under
`output/residency-fa-clean-vast-20261020/bundles/`, with verified export receipts.
All original paths and bytes are preserved; identical files may use verified
hardlinks. No provider credentials, controller or model weights are published.

Use the [shared reproduction instructions](../streaming-residency-flashattention-20261020/README.md)
and pinned stack/build. This is a custom FA 2.7.4 wheel result, not a stock-wheel
qualification. Requalify any rebuilt binary. None of these commands rents hardware.

On an already available, compatible Linux CUDA GPU, before any Torch import,
require three idle samples five seconds apart: no compute processes, GPU usage
at most 10%, and device memory at most 256 MiB. Record the UUID and raw samples.
Abort rather than continuing if more than one compute process appears during
a study stage. This still does not prove whole-host CPU exclusivity.

Copy the frozen `inputs/` into the separate `/root/liger` layout from the shared
guide, and run the separate H2D qualification before model work:

```bash
python /root/liger/pin_entry.py /root/liger/h2d_probe.py
python /root/liger/prepare_7b.py
python /root/liger/pytest_gpu_entry.py tests/test_streaming_residency_candidate.py tests/test_streaming_residency_attention_bench.py --no-cov -q
```

Use `b1.yaml`/`b2.yaml`, indices 8 through 23, and the same 10 GiB allocator cap.
Capture fresh native/hybrid batch1/2, a native batch1 repetition, and both semantic
captures. Preserve every original numerical gate, including padded/causal checks.
Only then time four fresh-process pairs per batch in AB/BA/BA/AB order, using
six warmup plus eight measured updates. Check final adapters and logged losses
after all 14 updates. Apply the original practical gate; no cross-lease pooling.
The hardware copy probe is not an attention or training throughput measurement.

For local replay, restore archived `analysis/` scripts into the retained study
directory with its protocol, receipts, progress and frozen `source-tree/`.
The small public archive supports source review and a fresh experiment; the
local raw tensor archives are required for replaying the stored observations.

After `analyze.py`, run archived `mark_accuracy_only.py` to reproduce C's explicit
interrupted classification and recovery metadata. Its original remote progress
remains unchanged; no completed timing run is manufactured.
