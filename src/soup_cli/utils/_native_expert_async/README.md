# `_native_expert_async`: the #77 async expert-stream extension

These three files are the C++17/CUDA extension from PR #77 (commit
[`dc0f0e0a`](https://github.com/SoupTeam/soup-rnd/commit/dc0f0e0a9a5beff3e281970e5152f85140d871d3),
author kok-o). They are vendored here so the portability work has a source to build and test.
The opt-in `cpp_async_2slot` loader arm wires them into the per-expert expert pool: every lease
is read through the compiled two-slot stream instead of the Python reader. The default loader
stays `async_lease`.

Changes from the original:

- the POSIX file identity uses `st_mtimespec` on macOS and `st_mtim` elsewhere. The
  original used `st_mtim` unconditionally, which does not compile with clang on macOS.

Not supported: Windows (the extension has a `_WIN32` branch, but the MBR/unified
track is untested there and no Windows path is offered). macOS is supported on CPU
paths only; the extension is CUDA-only and the CUDA kernels do not build or run on
macOS.

Build, without importing anything at package import time:

```bash
python src/soup_cli/utils/_native_expert_async/build_ext.py --build-lib /tmp/native-ext
# add --with-cuda-nf4 for the decode kernel (needs nvcc)
```

The module name is `_native_expert_stream_async`; put the build directory on
`PYTHONPATH` to import it.

Build note: CUDA 13 with GCC 15 (glibc 2.43) stops inside nvcc on glibc's `noexcept`
`rsqrt`/`rsqrtf` declarations before it reaches this source. That is a host-toolchain pairing
(`nvcc` versus the C library headers), not a file from #77. `build_ext.py` diagnoses it before
nvcc runs and prints the documented two-line patch (llama.cpp's): add `noexcept (true)` to the
`rsqrt(double x)` and `rsqrtf(float x)` declarations in
`$CUDA_HOME/include/crt/math_functions.h` (lines 629 and 653 in CUDA 13.0; keep a
`math_functions.h.soup-backup` beside it). The script never edits the header itself. Roll back by
restoring that backup or with `pip install --force-reinstall nvidia-cuda-crt`. The C++-only build
imports on PyTorch 2.14.1+cu130 (CUDA 13).
