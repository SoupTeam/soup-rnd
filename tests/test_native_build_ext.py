"""The CUDA 13 / glibc 2.43 ``rsqrt`` conflict is diagnosed before ``nvcc`` runs.

CUDA 13.0's ``crt/math_functions.h`` declares ``rsqrt(double)`` and ``rsqrtf(float)``
without the ``noexcept (true)`` that glibc 2.43 gives the same declarations. ``nvcc`` stops
inside the header before it reaches ``nf4_decode.cu``, and the message names the toolkit's
header rather than anything the operator set. The build script must explain the conflict and
the documented two-line patch, and it must NOT edit the user's environment: only read the
header and refuse.

These checks feed the pure diagnosis real headers from ``tmp_path`` so they run on every
machine, patched or not, with or without a toolkit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import pytest

BUILD_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "soup_cli"
    / "utils"
    / "_native_expert_async"
    / "build_ext.py"
)

DECLARATION_LINES = {
    629: "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  rsqrt(double x)",
    653: "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   rsqrtf(float x)",
}


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("soup_native_build_ext", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header(tmp_path: Path, *, patched: bool) -> Path:
    """A CUDA-shaped header whose only relevant lines are 629 and 653."""
    lines = [f"/* filler line {number} */" for number in range(1, 700)]
    for number, declaration in DECLARATION_LINES.items():
        suffix = " noexcept (true);" if patched else ";"
        lines[number - 1] = declaration + suffix
    path = tmp_path / "math_functions.h"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_an_unpatched_header_names_both_lines_the_backup_and_the_rollback(
    tmp_path: Path,
) -> None:
    message = _module().diagnose_rsqrt_conflict(
        _header(tmp_path, patched=False), glibc=(2, 43), cuda=(13, 0)
    )
    assert message is not None
    for expected in (
        "629",
        "653",
        "rsqrt(double x)",
        "rsqrtf(float x)",
        "noexcept (true)",
        "math_functions.h.soup-backup",
        "nvidia-cuda-crt",
        "--force-reinstall",
        "glibc 2.43",
        "CUDA 13.0",
    ):
        assert expected in message, expected
    assert "not modify" in message or "did not" in message or "untouched" in message


def test_a_patched_header_needs_no_diagnosis(tmp_path: Path) -> None:
    module = _module()
    assert module.diagnose_rsqrt_conflict(
        _header(tmp_path, patched=True), glibc=(2, 43), cuda=(13, 0)
    ) is None


@pytest.mark.parametrize(
    ("glibc", "cuda"),
    [((2, 42), (13, 0)), ((2, 43), (12, 8)), ((2, 43), None), (None, (13, 0))],
)
def test_only_the_glibc_2_43_and_cuda_13_pairing_is_diagnosed(
    tmp_path: Path, glibc: Tuple[int, int] | None, cuda: Tuple[int, int] | None
) -> None:
    assert _module().diagnose_rsqrt_conflict(
        _header(tmp_path, patched=False), glibc=glibc, cuda=cuda
    ) is None


def test_glibc_version_parsing_reads_the_confstr_spelling() -> None:
    module = _module()
    assert module.parse_glibc_version("glibc 2.43") == (2, 43)
    assert module.parse_glibc_version("glibc 2.39-0ubuntu8.5") == (2, 39)
    assert module.parse_glibc_version("musl libc (x86_64)") is None
    assert module.parse_glibc_version("") is None


def test_the_check_refuses_without_touching_the_header(tmp_path: Path) -> None:
    module = _module()
    header = _header(tmp_path, patched=False)
    before = header.read_bytes()
    before_stat = header.stat().st_mtime_ns
    with pytest.raises(module.ToolchainConflictError) as caught:
        module.check_toolchain(header=header, glibc=(2, 43), cuda=(13, 0))
    assert "629" in str(caught.value)
    assert header.read_bytes() == before
    assert header.stat().st_mtime_ns == before_stat


def test_a_missing_header_is_not_a_conflict(tmp_path: Path) -> None:
    module = _module()
    assert module.diagnose_rsqrt_conflict(
        tmp_path / "absent.h", glibc=(2, 43), cuda=(13, 0)
    ) is None


def test_main_runs_the_check_before_any_extension_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``nvcc`` must never start: the check raises before ``CUDAExtension`` exists."""
    module = _module()
    calls: List[str] = []

    def refuse() -> None:
        calls.append("check")
        raise module.ToolchainConflictError("the header needs two lines")

    monkeypatch.setattr(module, "check_toolchain", refuse)
    from torch.utils import cpp_extension

    monkeypatch.setattr(
        cpp_extension, "CUDAExtension", lambda *args, **kwargs: calls.append("cuda-extension")
    )
    monkeypatch.setattr(
        cpp_extension, "CppExtension", lambda *args, **kwargs: calls.append("cpp-extension")
    )
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: calls.append("setup"))
    monkeypatch.setattr(module.sys, "argv", ["build_ext.py", "build_ext", "--with-cuda-nf4"])
    with pytest.raises(module.ToolchainConflictError):
        module.main()
    assert calls == ["check"]


def test_a_cpu_only_build_does_not_probe_the_cuda_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: List[str] = []

    def unexpected() -> None:
        calls.append("check")

    monkeypatch.setattr(module, "check_toolchain", unexpected)
    from torch.utils import cpp_extension

    monkeypatch.setattr(
        cpp_extension, "CUDAExtension", lambda *args, **kwargs: calls.append("cuda-extension")
    )
    monkeypatch.setattr(
        cpp_extension, "CppExtension", lambda *args, **kwargs: calls.append("cpp-extension")
    )
    monkeypatch.setattr(module, "setup", lambda *args, **kwargs: calls.append("setup"))
    monkeypatch.setattr(
        sys, "argv", ["build_ext.py", "build_ext", "--build-lib", str(Path("/tmp/x"))]
    )
    module.main()
    assert "check" not in calls
    assert "cpp-extension" in calls and "setup" in calls
