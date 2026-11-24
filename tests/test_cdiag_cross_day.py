"""Ticket 14, B1: the direct A0-plain against C-seg chain from the saved tensors.

The comparison spans two campaigns: A0 plain is ticket 03's tensor (20261122), C-seg is the
ticket 14 ladder tensor (20261124). All CPU, on synthetic tensor pairs; the real campaign run
is ``benchmarks/mbr-moonlight-cdiag-20261124/compare_cross_day.py``."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks/mbr-moonlight-cdiag-20261124"


@pytest.fixture
def cross(monkeypatch: pytest.MonkeyPatch) -> Any:
    repo = BENCHMARK.parents[1]
    monkeypatch.syspath_prepend(str(repo / "src"))
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.syspath_prepend(str(BENCHMARK))
    sys.modules.pop("compare_cross_day", None)
    return importlib.import_module("compare_cross_day")


def save(path: Path, scale: float, *, tweak: bool = False) -> Path:
    grads = {
        "layer.0.a": torch.full((2, 2), scale, dtype=torch.float32),
        "layer.1.b": torch.full((2,), scale, dtype=torch.float32),
    }
    if tweak:
        grads["layer.0.a"] = grads["layer.0.a"].clone()
        grads["layer.0.a"][0, 0] += 1.0
    torch.save({"loss": torch.tensor([scale], dtype=torch.float64), "grads": grads}, path)
    return path


def test_the_eight_cells_are_the_ladder_windows_in_order(cross: Any) -> None:
    assert cross.CELLS == (
        (1, 64), (2, 64), (4, 64), (8, 64), (1, 512), (2, 512), (4, 512), (8, 512),
    )
    assert cross.window_dir(64) == "s64"
    assert cross.window_dir(512) == "s512-math"


def test_a_direct_pair_is_equal_only_when_the_loss_and_every_gradient_are(
    cross: Any, tmp_path: Path
) -> None:
    left = save(tmp_path / "left.pt", 2.0)
    right = save(tmp_path / "right.pt", 2.0)
    equal = cross.compare_paths((left, right), ("A0 plain", "C-seg"))
    assert equal["loss_bitwise_equal"] is True
    assert equal["grad_differing"] == 0

    tweaked = save(tmp_path / "tweaked.pt", 2.0, tweak=True)
    differing = cross.compare_paths((left, tweaked), ("A0 plain", "C-seg"))
    assert differing["loss_bitwise_equal"] is True
    assert differing["grad_differing"] == 1


def test_a_c_tensor_that_differs_from_a0_seg_is_not_the_ladder_artifact(
    cross: Any, tmp_path: Path
) -> None:
    plain = save(tmp_path / "a0.pt", 2.0)
    segmented = save(tmp_path / "a0seg.pt", 2.0)
    other = save(tmp_path / "c.pt", 2.0, tweak=True)
    assert cross.cell_verdict(plain, segmented, segmented) == "direct"
    assert cross.cell_verdict(plain, segmented, other) == "not-the-ladder-artifact"
    assert cross.cell_verdict(plain, tmp_path / "missing.pt", other) == "missing-tensor"


def test_the_dataset_hash_of_a_window_must_be_one_value(tmp_path: Path, cross: Any) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"kind": "bits", "dataset_sha256": "a" * 64},
        {"kind": "speed", "dataset_sha256": "b" * 64},
        {"kind": "bits", "dataset_sha256": "a" * 64},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert cross.dataset_sha(path) == "a" * 64

    rows[1]["dataset_sha256"] = "a" * 64
    rows[2]["dataset_sha256"] = "c" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="one dataset hash"):
        cross.dataset_sha(path)
