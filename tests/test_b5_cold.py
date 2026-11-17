"""B5 ticket 06: the page-cache helper of the cold run, on small files in a temporary folder.

What a drop does on a real host (root, ``drop_caches``, an NVMe drive) cannot be tested here and
is confirmed by the run itself from the journal. These tests pin the contract: what is counted,
what the result says, and that nothing is claimed that was not measured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from benchmarks.harness import b5_cold

MIB = 2**20


def _store(folder: Path, files: int = 3, size: int = 2 * MIB) -> Path:
    for number in range(files):
        (folder / f"layer_{number:03d}.safetensors").write_bytes(os.urandom(size))
    (folder / "SYNTHETIC.json").write_text("{}")
    return folder


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux mincore")
def test_residency_counts_every_file_and_sees_a_file_that_was_just_read(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for path in store.glob("*.safetensors"):
        path.read_bytes()

    resident, total = b5_cold.store_residency(store)

    assert total == 3 * 2 * MIB + len("{}")
    assert 0 < resident <= total + 4096


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux mincore")
def test_residency_of_an_empty_folder_is_zero(tmp_path: Path) -> None:
    assert b5_cold.store_residency(tmp_path) == (0, 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux mincore")
def test_a_drop_names_its_method_and_never_claims_more_than_it_measured(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for path in store.glob("*.safetensors"):
        path.read_bytes()

    result = b5_cold.drop_page_cache(store)

    assert result["method"] in b5_cold.METHODS
    assert result["store_bytes"] == b5_cold.store_residency(store)[1]
    assert result["resident_after_bytes"] <= result["resident_before_bytes"]
    assert result["seconds"] >= 0


def test_page_cache_measurements_refuse_unsupported_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b5_cold, "supports_page_cache_residency", lambda: False)

    with pytest.raises(NotImplementedError, match="requires Linux mincore"):
        b5_cold.store_residency(tmp_path)
    with pytest.raises(NotImplementedError, match="requires Linux mincore"):
        b5_cold.drop_page_cache(tmp_path)


def test_cold_run_refuses_unsupported_platform_before_loading_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from argparse import Namespace

    from benchmarks.harness import b5_real_moe

    monkeypatch.setattr(b5_cold, "supports_page_cache_residency", lambda: False)
    with pytest.raises(RuntimeError, match="--cold page-cache measurements require Linux mincore"):
        b5_real_moe.run(Namespace(cold=True))
    with pytest.raises(SystemExit) as error:
        b5_real_moe.main(["--config", "config.yaml", "--output", "out", "--store", "store",
                          "--cold"])
    assert error.value.code == 2
    assert "--cold page-cache measurements require Linux mincore" in capsys.readouterr().err


def test_the_process_read_counter_is_an_int_or_absent() -> None:
    value = b5_cold.process_read_bytes()

    assert value is None or (isinstance(value, int) and value >= 0)
