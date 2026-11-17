"""Complete-step assembly tests; synthetic fixtures are not trained-model evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/harness"))


@pytest.fixture
def prepared(tmp_path):
    from tests.test_d3_full_model import prepared as original_fixture

    return original_fixture.__wrapped__(tmp_path)


@pytest.mark.parametrize("retain", [False, True])
def test_complete_cpu_steps_match_independent_native_and_one_slot(prepared, retain):
    import torch
    from d3_training_model import open_training_model

    _original, folder = prepared
    ids = torch.arange(17).reshape(1, -1)
    with open_training_model(folder, mode="native", device="cpu", library=None,
                             retain_derivatives=retain) as native, open_training_model(
        folder, mode="sequential", device="cpu", library=None, retain_derivatives=retain,
    ) as candidate:
        first = {key: value.clone() for key, value in native.initial.items()}
        for _step in range(2):
            want, got = native.frame(native.step(ids)), candidate.frame(candidate.step(ids))
            assert set(want) == set(got)
            assert all(torch.equal(want[name], got[name]) for name in want)
            assert all(got[f"gradient/{key}"].abs().sum().item() > 0 for key in ("a", "b"))
            details = candidate.check_dispatch()
            assert details["layers"] == 2
            assert candidate.owner.pool.stats()["slot_count"] == 1
            if retain:
                assert len([key for key in got if key.startswith("derivative/")]) == 4
            else:
                assert not any(key.startswith("derivative/") for key in got)
        assert any(not torch.equal(candidate.parameters[key], first[key]) for key in first)
        assert not any(".mlp.experts." in name
                       for name, _ in candidate.owner.model.named_parameters())
        candidate.reset()
        native.reset()
        again = candidate.frame(candidate.step(ids))
        reference = native.frame(native.step(ids))
        assert all(torch.equal(again[key], reference[key]) for key in again)
    with pytest.raises(RuntimeError, match="closed"):
        candidate.step(ids)


def test_old_forward_only_wrapper_remains_guarded(prepared):
    import torch
    from d3_full_model import open_streamed_model

    _original, folder = prepared
    with open_streamed_model(folder, device="cpu", backend="sequential") as old:
        with pytest.raises(RuntimeError, match="forward-only"):
            old.forward(torch.tensor([[2, 3]]))


def test_zero_active_down_is_effective_and_restored_for_complete_derivatives(prepared):
    import torch
    from d3_full_model import validate_bundle
    from d3_full_model_gate import zero_active_down
    from d3_training_model import open_training_model

    _original, folder = prepared
    with open_training_model(folder, mode="sequential", device="cpu", library=None,
                             retain_derivatives=True) as run:
        ids = torch.arange(17).reshape(1, -1)
        positive = run.frame(run.step(ids))
        target = run.routes[0].unique().tolist()[0]
        run.reset()
        with zero_active_down(run.owner, target, validate_bundle(folder)):
            negative = run.frame(run.step(ids))
        assert not torch.equal(negative["logits"], positive["logits"])
        assert any(not torch.equal(negative[f"gradient/{key}"], positive[f"gradient/{key}"])
                   for key in ("a", "b"))
        run.reset()
        restored = run.frame(run.step(ids))
        assert all(torch.equal(restored[name], positive[name]) for name in positive)


def test_training_steps_refuse_disabled_autograd_before_advancing(prepared):
    import torch
    from d3_training_model import open_training_model

    _original, folder = prepared
    with open_training_model(folder, mode="sequential", device="cpu", library=None,
                             retain_derivatives=False) as run:
        with torch.no_grad(), pytest.raises(ValueError, match="autograd"):
            run.step(torch.tensor([[2, 3]]))
        assert run.owner.pool.fetches == 0
