# G3 Windows DirectStorage arm: result on an RTX 4060 laptop

Protocol: `benchmarks/g3/directstorage/PROTOCOL.txt`, committed in `05ace662` (2026-11-04 16:45
+05:00) before the series ran (20:02-20:03 +05:00). Analysis: `analysis.json`, reproduced from the
retained files by `python benchmarks/g3/directstorage/analyze_ds.py
benchmarks/g3/reports/20261104_directstorage/evidence`. `evidence/` is the unzipped
`results_20261104_200157.zip` returned from the laptop (SHA256 `dc61c0af...87c2`), every file kept.

**Capability: DS_TO_CUDA_VERIFIED.** DirectStorage delivered all 1,073,741,824 bytes of the fixture
into a D3D12 buffer imported into CUDA, correct byte for byte, in every one of the 10 B runs (and
the selftest); the corruption control was detected every time.

**Performance: B_SLOWER.** On this laptop DirectStorage into CUDA memory reads at **2.89 GB/s**,
against **3.53 GB/s** for the #974-style path (unbuffered read into 64 MiB of pinned staging, then
copy to the GPU). Mean difference -0.645 GB/s, 95% interval [-0.675, -0.616] over 5 valid blocks;
ratio 0.82, descriptive.

## Machine and stack

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, driver 573.22 (CUDA driver API 12.8), plus Intel Iris Xe |
| CPU, RAM | Intel Core i5-12500H, 16 GB |
| Disk | GIGABYTE AG4512G-SI B10, NVMe, 512 GB, NTFS, fixture on C: |
| OS | Windows 11 Pro, build 26200 |
| Runtime | DirectStorage 1.3.0 from the NuGet package checked by SHA256 |
| Probe | `ds_probe.exe` SHA256 `f1967bd7...1cc98`, `run_g3.ps1` `a334d9c5...0611` (both as built) |

## Series

5 pilot blocks A B B A, 20 fresh processes, each 2 warm-up and 3 timed reads of 1 GiB. Every run:
`ok`, power on mains in every sample, no wall-clock gap above 2.1 s, largest read by another
process 0 MB (133 system processes could not be opened and are counted). The METHOD repeats rule
needs 5 blocks for a 10% effect at this spread, so the pilot is the whole series; 0 void blocks.

| | A: unbuffered + pinned + H2D | B: DirectStorage into imported CUDA buffer |
|---|---|---|
| Useful GB/s, median of 10 runs | 3.52 | 2.89 |
| Range of timed reads | 0.300-0.309 s | 0.369-0.374 s |
| CPU seconds per 1 GiB read (median) | 0.039 | 0.016 |
| Process private bytes after reads (median) | 1,343 MB | 531 MB |
| Process peak working set (median) | 179 MB | 385 MB |
| Explicit pinned host bytes | 64 MiB | none from the probe (runtime staging 32 MiB) |
| Setup (CUDA, D3D12, DirectStorage) | 0.06-0.08 s | 0.31-0.33 s |

B additionally needs a device-to-device copy if the data must live in an ordinary CUDA
allocation: 0.033 s per GiB (about 33 GB/s), reported, not part of B's number.

## Reading the numbers

- **Bandwidth.** The DirectStorage path is 18% slower than Soup's existing path on this machine, in
  every block. A specific DirectStorage configuration (staging buffer size, request size) was frozen
  at the defaults; a larger staging buffer is a separate question, not tested here.
- **CPU.** B uses less CPU per read, but Windows reports process CPU time in 15.6 ms ticks, so 0.039
  vs 0.016 s is about 2.5 ticks vs 1 tick: direction only.
- **Memory.** B's process holds less private memory (531 vs 1,343 MB) and more working set (385 vs
  179 MB). Under WDDM, private bytes and working set include driver-side accounting for device
  allocations and for DirectStorage's staging, so these are process-level observations, not a
  measured host-RAM saving.
- **Route.** DirectStorage on Windows stages through runtime-managed memory; this probe cannot
  attribute a peer-to-peer NVMe-to-GPU route and does not claim one.

## What this answers for G3

On a consumer Windows laptop, DirectStorage can feed CUDA memory correctly, but it does not beat
Soup's current unbuffered pinned-staging path in bandwidth on this machine (0.82x), and the route is
not shown to bypass host RAM. On Linux, GeForce cards support GDS only in compatibility mode
(NVIDIA release notes) and no rented container reached a native GDS route (8 attempts,
`reports/20261103/STATUS.md`). No integration into Soup's streaming is proposed from this result.

Not measured: two drives; other laptops, drivers or DirectStorage versions; DirectStorage GPU
decompression; staging-buffer or request-size tuning; a training step.
