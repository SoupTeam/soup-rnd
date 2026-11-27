"""Route-visit counting for the streamed experts: the pure counter and its log (no torch)."""

from __future__ import annotations

import pytest

from soup_cli.utils import expert_stream as es


def test_each_token_slot_pair_counts_once_for_its_expert() -> None:
    top_k = [[0, 2], [2, 1], [0, 2]]
    assert es.route_visit_counts(top_k, num_experts=4) == {0: 2, 1: 1, 2: 3}


def test_no_tokens_counts_nothing() -> None:
    assert es.route_visit_counts([], num_experts=4) == {}


def test_an_expert_outside_the_reader_is_refused() -> None:
    with pytest.raises(ValueError):
        es.route_visit_counts([[0, 4]], num_experts=4)
    with pytest.raises(ValueError):
        es.route_visit_counts([[-1, 0]], num_experts=4)


def test_the_route_log_is_active_only_inside_its_block() -> None:
    assert es._ROUTE_LOG is None
    with es.route_visit_log() as log:
        assert es._ROUTE_LOG is log
        log[(0, 1)] = 3
    assert es._ROUTE_LOG is None
    assert log == {(0, 1): 3}
