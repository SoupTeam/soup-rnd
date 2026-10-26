# Diagnostic profile of two-read expert streaming

This companion to `two_read_cache.py` does not rent hardware or download models.
Run the existing full numerical qualification before using it. Diagnostic times
include profiler overhead and never establish an unprofiled speed improvement.

```bash
python benchmarks/harness/expert_cache_profile.py \
  --spec /results/prepared/spec.json --arm two_read \
  --out /results/profiles/two_read
python benchmarks/harness/expert_cache_profile.py \
  --spec /results/prepared/spec.json --arm two_read_cache \
  --out /results/profiles/two_read_cache
```

Both arms reuse the native checkpoint/store/input/environment-bound spec, build,
adapters, AdamW and resource gates. They perform two warmup updates and one
unprofiled context update, followed by one accumulated diagnostic update. Only
the first microbatch's complete forward/backward is profiled; the second
microbatch and optimizer step run outside the profiler. Frozen random token
inputs test numerical preservation rather than downstream task quality.

Reversible `record_function` wrappers label expert fetches, storage reads, file
identity checks, quant-state reconstruction and NF4 dequantization. No decoder,
MoE or expert forward hooks are registered. Shapes, stacks, allocation tracing
and FLOP tracing are disabled to reduce profiler overhead and retained state;
see the [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html).
The diagnostic retains every key-average row, including scalar conversions,
copies, nested-scale kernels and matrix multiplication. Inclusive/nested CPU
and concurrent CUDA durations cannot be added as a whole-step breakdown.

The harness requires actual CUDA kernel and host-to-device activity, with no
CPU-only fallback. Before downloading a large checkpoint, execute the cheap
availability control and require a real pass, alongside the native CUDA
numerical controls:

```bash
python -m pytest -q -o addopts= \
  tests/test_expert_cache_profile.py::test_real_cuda_profiler_collects_kernel_and_h2d_events
```

Setup, warmup/context, diagnostic execution and trace processing/export keep
the existing native RAM/VRAM/swap gates. Inputs, adapters, gradients and AdamW
state receive canonical hashes. Raw traces are retained as a verified lossless
gzip with both encoded and decoded SHA-256. The default and maximum encoded
trace size is 128 MiB per arm; an overflow retains a failure and the original
bytes, rather than silently truncating the evidence.

Local preparation controls passed seven CPU cases, with one CUDA availability
case skipped on the CPU-only machine. They include bitwise complete CPU NF4
trajectory parity with the markers enabled, and a separate accumulated-update
control proving that only the first microbatch is profiled while both losses,
adapter gradients and optimizer updates remain unchanged. Actual CUDA
availability, profiling capacity and a real-model result require hardware.
