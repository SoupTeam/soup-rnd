"""Build the C++17 async expert stream pybind11 extension."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup


def _absolutize_flag_paths(argv: list[str]) -> list[str]:
    out = list(argv)
    for idx, arg in enumerate(out):
        if arg in ("--build-lib", "--build-temp") and idx + 1 < len(out):
            out[idx + 1] = str(Path(out[idx + 1]).resolve())
        elif arg.startswith("--build-lib="):
            out[idx] = f"--build-lib={Path(arg.split('=', 1)[1]).resolve()}"
        elif arg.startswith("--build-temp="):
            out[idx] = f"--build-temp={Path(arg.split('=', 1)[1]).resolve()}"
    return out


def main() -> None:
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    here = Path(__file__).resolve().parent
    sys.argv = _absolutize_flag_paths(sys.argv)
    # Run from the extension source directory so setuptools does not parse the repo's pyproject.toml
    os.chdir(str(here))

    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    name = "_native_expert_stream_async"
    sources = [str(here / "native_stream_async.cpp")]
    if "--with-cuda-nf4" in sys.argv:
        sys.argv.remove("--with-cuda-nf4")
        sources.append(str(here / "nf4_decode.cu"))
        extension = CUDAExtension(
            name=name,
            sources=sources,
            define_macros=[("SOUP_WITH_CUDA_NF4", "1")],
            extra_compile_args={"cxx": flags, "nvcc": ["-O3", "--ftz=false"]},
        )
    else:
        extension = CppExtension(name=name, sources=sources, extra_compile_args=flags)
    setup(
        name=name,
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=False)},
    )


if __name__ == "__main__":
    main()
