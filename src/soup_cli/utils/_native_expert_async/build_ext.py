"""Build the C++17 async expert stream pybind11 extension."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from setuptools import setup

#: glibc added ``noexcept`` to its ``rsqrt`` declarations in 2.43.
GLIBC_CONFLICT_FLOOR = (2, 43)
#: CUDA 13.0's ``crt/math_functions.h`` still declares them without it.
CUDA_CONFLICT_FLOOR = (13, 0)
#: The two declarations the llama.cpp workaround patches, by their source spelling.
PATCHED_SUFFIX = " noexcept (true)"

_GLIBC = re.compile(r"glibc\s+(\d+)\.(\d+)")
_VERSION = re.compile(r"(\d+)\.(\d+)")


class ToolchainConflictError(RuntimeError):
    """The installed CUDA header cannot compile against this glibc; nothing was changed."""


def parse_glibc_version(text: str) -> Optional[Tuple[int, int]]:
    """``(2, 43)`` from ``os.confstr('CS_GNU_LIBC_VERSION')``; None for musl or junk."""
    found = _GLIBC.search(text or "")
    if found is None:
        return None
    return int(found.group(1)), int(found.group(2))


def _installed_glibc_version() -> Optional[Tuple[int, int]]:
    try:
        return parse_glibc_version(os.confstr("CS_GNU_LIBC_VERSION"))
    except (AttributeError, OSError, ValueError):
        return None


def _cuda_home() -> Optional[Path]:
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            return Path(value)
    try:
        from torch.utils import cpp_extension
    except Exception:  # noqa: BLE001 - no torch simply means no toolkit to inspect
        return None
    home = getattr(cpp_extension, "CUDA_HOME", None)
    return Path(home) if home else None


def _cuda_version(home: Optional[Path]) -> Optional[Tuple[int, int]]:
    """The toolkit version, from ``version.json`` and then from torch's own report."""
    if home is not None:
        version_file = home / "version.json"
        try:
            payload = json.loads(version_file.read_text(encoding="utf-8"))
            version = str(payload["cuda"]["version"])
        except (OSError, KeyError, TypeError, ValueError):
            pass
        else:
            found = _VERSION.search(version)
            if found is not None:
                return int(found.group(1)), int(found.group(2))
    try:
        import torch
    except Exception:  # noqa: BLE001 - a build without torch is a different failure
        return None
    reported = getattr(torch.version, "cuda", None)
    if not reported:
        return None
    found = _VERSION.search(str(reported))
    return (int(found.group(1)), int(found.group(2))) if found is not None else None


def _unpatched_declarations(header: Path) -> list:
    """``(line number, text)`` of each rsqrt declaration still missing ``noexcept``."""
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    unpatched = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("extern "):
            continue
        if "__device_builtin__" not in stripped:
            continue
        if " rsqrt(" not in stripped and " rsqrtf(" not in stripped:
            continue
        if PATCHED_SUFFIX in stripped:
            continue
        unpatched.append((number, stripped))
    return unpatched


def diagnose_rsqrt_conflict(
    header: Path,
    *,
    glibc: Optional[Tuple[int, int]],
    cuda: Optional[Tuple[int, int]],
) -> Optional[str]:
    """The actionable message for an unpatched header, or None when nothing conflicts."""
    if glibc is None or cuda is None:
        return None
    if glibc < GLIBC_CONFLICT_FLOOR or cuda < CUDA_CONFLICT_FLOOR:
        return None
    unpatched = _unpatched_declarations(Path(header))
    if not unpatched:
        return None
    lines = "\n".join(f"  line {number}: {text}" for number, text in unpatched)
    patched = "\n".join(
        f"  {text[:-1]}{PATCHED_SUFFIX};" for _number, text in unpatched
    )
    backup = f"{header}.soup-backup"
    return (
        f"CUDA {cuda[0]}.{cuda[1]} and glibc {glibc[0]}.{glibc[1]} disagree about "
        "rsqrt in the toolkit's own header, and nvcc stops inside it before it compiles "
        "this source:\n\n"
        f"  glibc {glibc[0]}.{glibc[1]} declares rsqrt/rsqrtf with noexcept;\n"
        f"  CUDA {cuda[0]}.{cuda[1]}'s crt/math_functions.h still declares them without it.\n\n"
        "This script did not modify anything: the check only read the header. The documented "
        "patch (the one llama.cpp ships) adds `noexcept (true)` to the two declarations:\n\n"
        f"{lines}\n\n"
        "so they read:\n\n"
        f"{patched}\n\n"
        f"Header: {header}\n"
        f"Back it up before editing (the usual location keeps one beside the header):\n"
        f"  cp '{header}' '{backup}'\n\n"
        "Roll back by copying that backup over the header, or restore the shipped file:\n"
        "  pip install --force-reinstall nvidia-cuda-crt\n\n"
        "This is a host-toolchain pairing (nvcc against the C library headers), not a file "
        "from #77."
    )


def check_toolchain(
    *,
    header: Optional[Path] = None,
    glibc: Optional[Tuple[int, int]] = None,
    cuda: Optional[Tuple[int, int]] = None,
) -> None:
    """Refuse the glibc 2.43 / CUDA 13 rsqrt pairing before nvcc is started.

    Reads the installed header and nothing else. Every value can be supplied for a test
    fixture; the defaults come from this machine.
    """
    home = None if header is not None else _cuda_home()
    if header is None and home is not None:
        header = home / "include" / "crt" / "math_functions.h"
    if header is None:
        return
    message = diagnose_rsqrt_conflict(
        header,
        glibc=_installed_glibc_version() if glibc is None else glibc,
        cuda=_cuda_version(home) if cuda is None else cuda,
    )
    if message is not None:
        raise ToolchainConflictError(message)


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
    # Before setuptools and torch's CUDAExtension: an unpatched header stops nvcc inside
    # itself, and the traceback names the toolkit rather than anything the operator set.
    if "--with-cuda-nf4" in sys.argv:
        check_toolchain()

    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    here = Path(__file__).resolve().parent
    sys.argv = _absolutize_flag_paths(sys.argv)
    # Run from the extension source directory so setuptools does not parse the repo's
    # pyproject.toml
    previous = Path.cwd()
    os.chdir(str(here))

    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    name = "_native_expert_stream_async"
    sources = [str(here / "native_stream_async.cpp")]
    try:
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
    finally:
        os.chdir(str(previous))


if __name__ == "__main__":
    try:
        main()
    except ToolchainConflictError as failure:
        from rich.console import Console

        Console(stderr=True).print(str(failure))
        sys.exit(1)
