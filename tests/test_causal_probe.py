"""``causal_probe``: a delay of N us at named points of the step, switched on and off per step.

Causal profiling: if adding D us at point P per call lengthens the step by f * D * calls, then a
fraction f of the CPU time spent at P is on the critical path (f near 1: all of it; near 0: none).
These tests pin the bookkeeping on CPU with a fake spin.
"""

from __future__ import annotations

import pytest

from benchmarks.harness.causal_probe import CausalProbe, parse_causal_spec


def test_parse_spec_reads_name_equals_microseconds_pairs() -> None:
    assert parse_causal_spec("expert_fwd=250,layer=40.5") == {"expert_fwd": 250.0, "layer": 40.5}
    assert parse_causal_spec("") == {}
    with pytest.raises(ValueError):
        parse_causal_spec("expert_fwd")
    with pytest.raises(ValueError):
        parse_causal_spec("expert_fwd=-3")


class Owner:
    def __init__(self) -> None:
        self.calls = 0

    def work(self, value: int) -> int:
        self.calls += 1
        return value * 2


def test_a_wrapped_method_spins_only_while_its_delay_is_set_and_counts_every_call() -> None:
    spins: list[float] = []
    probe = CausalProbe(spin=spins.append)
    probe.wrap("point", Owner, "work")
    owner = Owner()

    assert owner.work(2) == 4 and spins == []        # no delay set: no spin, still counted
    with probe.delays({"point": 120.0}):
        assert owner.work(3) == 6 and owner.work(4) == 8
    assert spins == [120.0, 120.0]
    assert owner.work(5) == 10 and len(spins) == 2             # off again
    assert probe.counts()["point"] == 4
    probe.restore()
    assert not hasattr(Owner.work, "__wrapped__")


def test_a_second_wrap_of_the_same_point_is_refused_and_unknown_delays_too() -> None:
    probe = CausalProbe(spin=lambda us: None)
    probe.wrap("point", Owner, "work")
    with pytest.raises(ValueError):
        probe.wrap("point", Owner, "work")
    with pytest.raises(ValueError):
        with probe.delays({"nope": 10.0}):
            pass
    probe.restore()
