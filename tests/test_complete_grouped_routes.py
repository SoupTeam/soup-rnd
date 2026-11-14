"""Complete lever: grouped expert routes on the streamed routed experts (CPU).

The lever changes how the hit experts and their (slot, token) rows are found, not what is read or
computed. The reference is the same module with the lever off, so every comparison is
``torch.equal`` on output, hidden gradient and routing-weight gradient, and on the expert read
order seen at the public ``reader.read_host`` boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_b1_expert_forward import _experts, _inputs, _store
from tests.test_b13_expert_backward import _ROUTINGS, _assert_exact, _hit, _Loads, _run


def _streamed(module, reader, pool, **kwargs):
    from soup_cli.utils.expert_stream import streamed_experts_class

    return streamed_experts_class()(reader, pool, module.act_fn, 0, **kwargs)


@pytest.mark.parametrize("routing", _ROUTINGS)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_grouped_routes_equal_the_default_path_and_read_in_the_same_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, routing: str
) -> None:
    import torch

    module, _ = _experts()
    module = module.to(dtype=getattr(torch, dtype))
    reader, pool = _store(tmp_path, module)
    try:
        args = _inputs(dtype=getattr(torch, dtype), **_ROUTINGS[routing])
        loads = _Loads(monkeypatch, reader)
        default = _run(_streamed(module, reader, pool), args)
        default_order = list(loads.records)
        loads.records.clear()
        grouped = _run(_streamed(module, reader, pool, grouped_expert_routes=True), args)
        assert default_order and loads.records == default_order
        assert sorted({expert for expert, _ in loads.records}) == _hit(args)
        _assert_exact(default, grouped)
    finally:
        reader.close()
        pool.close()


def test_the_lever_is_off_by_default_and_takes_only_a_bool(tmp_path: Path) -> None:
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        assert _streamed(module, reader, pool).grouped_expert_routes is False
        assert _streamed(module, reader, pool, grouped_expert_routes=True).grouped_expert_routes
        for bad in (1, 0, "true", None):
            with pytest.raises(ValueError, match="grouped_expert_routes must be a bool"):
                _streamed(module, reader, pool, grouped_expert_routes=bad)
    finally:
        reader.close()
        pool.close()
