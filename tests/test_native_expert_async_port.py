"""Portability of the #77 async expert extension taken into ``src``.

The C++/CUDA source is vendored for the loader arm and is wired into the per-expert pool
only behind the opt-in ``cpp_async_2slot``: importing ``soup_cli`` must not build it, and a
test that needs the compiled module skips instead of failing a machine that never built it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import soup_cli

EXT_DIR = Path(soup_cli.__file__).resolve().parent / "utils" / "_native_expert_async"
SOURCE = EXT_DIR / "native_stream_async.cpp"


def test_the_vendored_source_uses_the_macos_file_time_field() -> None:
    """macOS ``struct stat`` has ``st_mtimespec``; ``st_mtim`` is the Linux spelling."""
    text = SOURCE.read_text(encoding="utf-8")
    block = re.search(r"#if defined\(__APPLE__\)(.*?)#else(.*?)#endif", text, re.S)
    assert block is not None, "the file time fields need an __APPLE__ branch"
    apple, other = block.group(1), block.group(2)
    assert "st_mtimespec.tv_sec" in apple and "st_mtimespec.tv_nsec" in apple
    assert "info.st_mtim." not in apple
    assert "info.st_mtim.tv_sec" in other and "info.st_mtim.tv_nsec" in other


def test_the_build_script_and_cuda_decode_source_are_vendored() -> None:
    assert (EXT_DIR / "build_ext.py").is_file()
    assert (EXT_DIR / "nf4_decode.cu").is_file()


def test_importing_soup_cli_does_not_load_the_native_extension() -> None:
    """A clean checkout has no compiled module; the package must import without one."""
    source_root = Path(soup_cli.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = (
        "import sys, soup_cli\n"
        "assert soup_cli.__version__\n"
        "assert '_native_expert_stream_async' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=str(source_root), env=env, check=True,
        capture_output=True, text=True,
    )


def test_a_test_that_needs_the_compiled_module_skips_when_it_is_absent() -> None:
    try:
        import _native_expert_stream_async as extension
    except ImportError:
        pytest.skip(
            "_native_expert_stream_async is not built; run "
            "python src/soup_cli/utils/_native_expert_async/build_ext.py "
            "(with --with-cuda-nf4 for the decode kernel)"
        )
    assert hasattr(extension, "PackedStreamAsync")
