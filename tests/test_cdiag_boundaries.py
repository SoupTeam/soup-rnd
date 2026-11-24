"""Ticket 14: the boundary comparison behind the C-against-A0 divergence diagnosis.

All CPU. The records are the shape ``capture_boundaries.py`` writes: module boundaries matched by
the digest of their input, and expert linear or routed-expert node calls whose rows are matched
across runs by the digest of the input row."""

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
def cdiag(monkeypatch: pytest.MonkeyPatch) -> Any:
    repo = BENCHMARK.parents[1]
    monkeypatch.syspath_prepend(str(repo / "src"))
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.syspath_prepend(str(BENCHMARK))
    sys.modules.pop("cdiag", None)
    return importlib.import_module("cdiag")


def value(seed: int, width: int = 4) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(width, generator=generator, dtype=torch.bfloat16)


def tweak(tensor: Any) -> Any:
    """Flip the lowest bit of the first element."""
    changed = tensor.clone()
    raw = changed.view(torch.uint16)
    raw[0] = raw[0] ^ 1
    return changed


def module(
    cdiag: Any, layer: int, role: str, mb: int, output: Any, inputs: Any = None
) -> dict:
    return cdiag.module_record(
        layer=layer, role=role, mb=mb, inputs=output if inputs is None else inputs,
        outputs=output,
    )


def expert(
    cdiag: Any,
    *,
    layer: int = 0,
    expert: int = 3,
    which: str = "gate_up",
    phase: str = "forward",
    call: int = 0,
    rows: list[tuple[Any, Any]],
) -> dict:
    return cdiag.expert_linear_record(
        layer=layer, expert=expert, which=which, phase=phase, call=call,
        inputs=torch.stack([row[0] for row in rows]),
        outputs=torch.stack([row[1] for row in rows]),
    )


def test_a_row_digest_changes_only_where_a_bit_changes(cdiag: Any) -> None:
    first, second, third = value(0), value(1), value(2)
    rows = cdiag.row_digests(torch.stack([first, second, third]))
    assert len(rows) == 3
    assert len(set(rows)) == 3
    changed = cdiag.row_digests(torch.stack([first, tweak(second), third]))
    assert changed[0] == rows[0] and changed[2] == rows[2]
    assert changed[1] != rows[1]


def test_module_boundaries_compare_by_input_digest(cdiag: Any) -> None:
    tensor = value(0)
    left = [module(cdiag, 0, "self_attn", 0, tensor), module(cdiag, 0, "gate", 0, tensor)]
    right = [module(cdiag, 0, "self_attn", 0, tensor), module(cdiag, 0, "gate", 0, tensor)]
    assert cdiag.match_module_boundaries(left, right)["equal"] is True

    right[1] = module(cdiag, 0, "gate", 0, tweak(tensor), inputs=tensor)
    report = cdiag.match_module_boundaries(left, right)
    assert report["equal"] is False
    assert [(entry["layer"], entry["role"], entry["mb"]) for entry in report["mismatched"]] == [
        (0, "gate", 0)
    ]


def test_module_boundaries_match_across_reversed_micro_batch_order(cdiag: Any) -> None:
    first, second = value(0), value(1)
    left = [module(cdiag, 0, "gate", 0, first), module(cdiag, 0, "gate", 1, second)]
    right = [module(cdiag, 0, "gate", 0, second), module(cdiag, 0, "gate", 1, first)]
    assert cdiag.match_module_boundaries(left, right)["equal"] is True


def test_an_input_no_reference_boundary_has_is_reported(cdiag: Any) -> None:
    left = [module(cdiag, 0, "self_attn", 0, value(0))]
    right = [module(cdiag, 0, "self_attn", 0, value(1))]
    report = cdiag.match_module_boundaries(left, right)
    assert report["equal"] is False
    assert report["missing"][0]["role"] == "self_attn"
    assert report["mismatched"] == []


def test_expert_rows_match_across_row_counts(cdiag: Any) -> None:
    rows = [value(0), value(1)]
    outputs = [value(10), value(11)]
    left = [
        expert(cdiag, call=0, rows=[(rows[0], outputs[0])]),
        expert(cdiag, call=1, rows=[(rows[1], outputs[1])]),
    ]
    right = [expert(cdiag, call=0, rows=[(rows[0], outputs[0]), (rows[1], outputs[1])])]
    report = cdiag.match_expert_rows(left, right)
    assert report["rows_matched"] == 2
    assert report["rows_missing"] == []
    assert report["differing"] == []
    assert report["first_forward"] is None


def test_a_differing_row_names_its_call_and_both_row_counts(cdiag: Any) -> None:
    rows = [value(0), value(1)]
    outputs = [value(10), value(11)]
    left = [
        expert(cdiag, call=0, rows=[(rows[0], outputs[0])]),
        expert(cdiag, call=1, rows=[(rows[1], outputs[1])]),
    ]
    right = [expert(cdiag, call=0, rows=[(rows[0], outputs[0]), (rows[1], tweak(outputs[1]))])]
    report = cdiag.match_expert_rows(left, right)
    assert report["differing"] == [
        {
            "layer": 0, "expert": 3, "which": "gate_up", "phase": "forward", "call": 0,
            "a_rows": [1, 1], "b_rows": 2, "matched_rows": 2, "diff_rows": 1,
        }
    ]
    first = report["first_forward"]
    assert first is not None and first["b_rows"] == 2 and first["a_rows"] == [1, 1]


def test_repeated_row_digests_are_matched_one_for_one(cdiag: Any) -> None:
    row, output = value(0), value(10)
    left = [expert(cdiag, call=0, rows=[(row, output)] * 2)]
    right = [expert(cdiag, call=0, rows=[(row, output)] * 2)]
    report = cdiag.match_expert_rows(left, right)
    assert report["rows_matched"] == 2 and report["differing"] == []


def test_a_row_absent_from_the_reference_is_reported(cdiag: Any) -> None:
    left = [expert(cdiag, call=0, rows=[(value(0), value(10))])]
    right = [expert(cdiag, call=0, rows=[(value(1), value(11))])]
    report = cdiag.match_expert_rows(left, right)
    assert report["rows_matched"] == 0
    assert report["rows_missing"][0]["expert"] == 3
    assert report["first_forward"] is None  # an unmatched input is not a bit difference


def test_the_earliest_forward_divergence_wins(cdiag: Any) -> None:
    row = value(0)
    good, bad = value(10), value(11)
    left = [
        expert(cdiag, layer=0, call=0, rows=[(row, good)]),
        expert(cdiag, layer=1, call=1, rows=[(row, good)]),
    ]
    right = [
        expert(cdiag, layer=0, call=0, rows=[(row, bad)]),
        expert(cdiag, layer=1, call=2, rows=[(row, bad)]),
    ]
    report = cdiag.match_expert_rows(left, right)
    assert report["first_forward"]["layer"] == 0


def test_a_backward_difference_is_apart_from_the_first_forward_one(cdiag: Any) -> None:
    row = value(0)
    left = [
        expert(cdiag, phase="forward", call=0, rows=[(row, value(10))]),
        expert(cdiag, phase="backward", call=2, rows=[(row, value(12))]),
    ]
    right = [
        expert(cdiag, phase="forward", call=0, rows=[(row, value(10))]),
        expert(cdiag, phase="backward", call=2, rows=[(row, value(13))]),
    ]
    report = cdiag.match_expert_rows(left, right)
    assert report["first_forward"] is None
    assert [entry["phase"] for entry in report["differing"]] == ["backward"]


def test_the_verdict_puts_this_layers_modules_before_its_expert_call(cdiag: Any) -> None:
    tensor = value(0)
    modules = cdiag.match_module_boundaries(
        [module(cdiag, 0, "self_attn", 0, tensor)],
        [module(cdiag, 0, "self_attn", 0, tweak(tensor), inputs=tensor)],
    )
    experts = cdiag.match_expert_rows(
        [expert(cdiag, layer=0, rows=[(value(0), value(10))])],
        [expert(cdiag, layer=0, rows=[(value(0), value(11))])],
    )
    assert cdiag.boundary_verdict(modules, experts) == {
        "kind": "module", "layer": 0, "role": "self_attn", "mb": 0,
    }
    equal_modules = cdiag.match_module_boundaries(
        [module(cdiag, 0, "self_attn", 0, tensor)],
        [module(cdiag, 0, "self_attn", 0, tensor)],
    )
    assert cdiag.boundary_verdict(equal_modules, experts)["kind"] == "expert_linear"


def test_an_expert_difference_of_a_lower_layer_beats_a_higher_layers_modules(cdiag: Any) -> None:
    tensor = value(0)
    modules = cdiag.match_module_boundaries(
        [module(cdiag, 1, "gate", 0, tensor)],
        [module(cdiag, 1, "gate", 0, tweak(tensor), inputs=tensor)],
    )
    experts = cdiag.match_expert_rows(
        [expert(cdiag, layer=0, rows=[(value(0), value(10))])],
        [expert(cdiag, layer=0, rows=[(value(0), value(11))])],
    )
    assert cdiag.boundary_verdict(modules, experts)["kind"] == "expert_linear"


def test_equal_boundaries_say_equal(cdiag: Any) -> None:
    tensor = value(0)
    verdict = cdiag.boundary_verdict(
        cdiag.match_module_boundaries(
            [module(cdiag, 0, "gate", 0, tensor)], [module(cdiag, 0, "gate", 0, tensor)]
        ),
        cdiag.match_expert_rows([], []),
    )
    assert verdict == {"kind": "equal", "modules_equal": True, "experts_equal": True}


def test_the_model_revision_falls_back_to_the_cache_record(cdiag: Any, tmp_path: Path) -> None:
    capture = importlib.import_module("capture_boundaries")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "cache.json").write_text(
        json.dumps({"revision": "2a9a177f"}), encoding="utf-8"
    )
    assert capture._model_revision(str(source)) == "2a9a177f"
    (source / "config.json").write_text('{"_commit_hash": "abcd"}', encoding="utf-8")
    assert capture._model_revision(str(source)) == "abcd"
