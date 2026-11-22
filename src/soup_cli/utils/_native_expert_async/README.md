# `_native_expert_async`: the #77 async expert-stream extension

These three files are the C++17/CUDA extension from PR #77 (commit
[`dc0f0e0a`](https://github.com/SoupTeam/soup-rnd/commit/dc0f0e0a9a5beff3e281970e5152f85140d871d3),
author kok-o). They are vendored here so the portability work has a source to build
and test; no loader imports or wires them yet. The loader gates (`cpp_async_2slot`)
stay in ticket 04.

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

Build note: CUDA 13 with GCC 15 fails inside nvcc on glibc's `noexcept` `rsqrt`
declarations before it reaches this source. That is a host-toolchain pairing
(`nvcc` versus the C library headers), not a file from #77; use a GCC that the
toolkit supports. The C++-only build imports on PyTorch 2.14.1+cu130 (CUDA 13).
