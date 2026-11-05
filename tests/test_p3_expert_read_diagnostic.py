"""P3 B1.4 ticket 03: does the L2L step read a routed expert again for every micro-batch?

Question 2 of #52, in the wording of the P3 phase 0 plan (step 12). The plan's "one read per
window" assumption for experts holds only if the number of expert reads does not grow with
the number of micro-batches k in a window. Test #47 (``test_b13_l2l_moe.py``) already pins
``expert_fetches == 3 * total_hits`` for k = 3; this file measures how that scales at
k = 1, 4, 16, 64 and records the counters.

Seam. ``L2LStep.run`` on the tiny native DeepSeek-V3 of B1.3 (3 decoder layers, 1 dense and 2
MoE, per-expert shards). The oracle for which experts a micro-batch needs is an independent
resident eager model run as plain accumulation (its routers' choices), as in B1.3.

What is counted, per window of k micro-batches:
  * ``layer_loads``: ``LoadCounters.layer_loads`` of the step (decoder layer loads);
  * ``expert_fetches``: ``LoadCounters.expert_fetches``, reads of one expert into the arena;
  * ``expert_bytes_read``: ``LoadCounters.expert_bytes_read``, the LOGICAL bytes: the sum of
    the stored tensor sizes of the experts read, as ``ExpertShardReader.bytes_read`` adds them;
  * ``total_hits``: from the resident router, the (micro-batch, MoE layer, expert) triples
    with at least one routed token. One hit is read three times: forward, recompute, and the
    expert's own backward;
  * ``unique_experts``: the (MoE layer, expert) pairs hit anywhere in the window, which is the
    number of experts a "one read per window" scheme would have to fetch.

Physical reads. The disk cannot be observed from a test without OS tooling (``iostat``,
``fs_usage``), and the page cache sits between a read and the device. The method chosen:
wrap ``safetensors_reader.read_range_into``, the single function every expert read goes
through, and count its calls and the aligned bytes it asks the file handle for, only while an
expert reader runs. This counts the requests the process hands to the OS, after sector
alignment, not the device reads. It separates two things the logical counters mix: the
sector padding of each tensor span, and the number of read calls per expert (one per stored
tensor). On macOS the handle uses ``F_NOCACHE``, on Linux ``O_DIRECT`` where the filesystem
allows it, so on those the requests are close to device reads; elsewhere they are not.

Result (CPU, tiny fixture, measured 2026-11-05 on d272e476 plus tickets 01 and 02):

    k   layer_loads  expert_fetches  logical_bytes  total_hits  unique  phys_calls  phys_bytes
    1        3              39           239616          13        13        78        547536
    4        3             162           995328          54        15       324       2276160
   16        3             621          3815424         207        16      1242       8671128
   64        3            2526         15519744         842        16      5052      35311200

Verdict. Confirmed: the expert is read again for every micro-batch in which the router hits
it. Fetches equal 3 x total_hits at every k and stay near 39 per micro-batch (39.0, 40.5, 38.8,
39.5), while layer loads stay at 3 and the window needs only 16 distinct experts. The plan's
"one read per window" assumption does not hold for experts in the current implementation, so
the x50 estimate for experts needs a new implementation, for example one that groups the
hits of a window per expert. This file claims no speed. Logical bytes are 44 percent of the
physical bytes (sector padding), and every expert fetch is two read requests (gate_up, down).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List

import pytest

from tests.test_b13_l2l_moe import (  # noqa: F401  (make_rig is a fixture)
    Spec,
    ga_windows,
    hits_of,
    l2l_windows,
    make_rig,
    micro_batches,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

K_SWEEP = (1, 4, 16, 64)
#: Reads of one hit expert: no-grad forward, grad recompute, the expert's own backward.
READS_PER_HIT = 3


class PhysicalReads:
    """Calls and aligned bytes of ``read_range_into`` made while an expert reader runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, readers: List[Any]) -> None:
        from soup_cli.utils import safetensors_reader

        self.calls = 0
        self.bytes = 0
        self._active = 0
        original = safetensors_reader.read_range_into

        def counted(handle: Any, start: int, view: Any, expected: int) -> None:
            if self._active:
                self.calls += 1
                self.bytes += int(expected)
            return original(handle, start, view, expected)

        monkeypatch.setattr(safetensors_reader, "read_range_into", counted)
        for reader in readers:
            self._wrap(monkeypatch, reader)

    def _wrap(self, monkeypatch: pytest.MonkeyPatch, reader: Any) -> None:
        original = reader.read_host

        def read_host(expert: int, staging: Any) -> Any:
            self._active += 1
            try:
                return original(expert, staging)
            finally:
                self._active -= 1

        monkeypatch.setattr(reader, "read_host", read_host)


def measure(rig: Any, monkeypatch: pytest.MonkeyPatch, k: int) -> Dict[str, int]:
    """One window of k micro-batches through the L2L step; the counters of that window."""
    specs = [Spec(micro_batches(rig, k))]
    (want,) = ga_windows(rig, specs)
    hits = hits_of(want)
    physical = PhysicalReads(monkeypatch, list(rig.runtime.expert_readers))
    (got,) = l2l_windows(rig, specs)
    counters = got.counters
    return {
        "k": k,
        "layer_loads": counters.layer_loads,
        "expert_fetches": counters.expert_fetches,
        "expert_bytes_read": counters.expert_bytes_read,
        "total_hits": sum(len(seen) for per_mb in hits.values() for seen in per_mb),
        "unique_experts": sum(
            len({expert for seen in per_mb for expert in seen}) for per_mb in hits.values()
        ),
        "physical_calls": physical.calls,
        "physical_bytes": physical.bytes,
    }


@pytest.fixture()
def sweep(
    make_rig: Callable[..., Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int], Dict[str, int]]:
    def run(k: int) -> Dict[str, int]:
        return measure(make_rig(k=k), monkeypatch, k)

    return run


@pytest.mark.parametrize("k", K_SWEEP)
def test_expert_reads_per_window_are_counted_at_every_k(
    sweep: Callable[[int], Dict[str, int]], record_property: Callable[..., None], k: int,
) -> None:
    row = sweep(k)
    for name, value in row.items():
        record_property(name, value)
    assert row["layer_loads"] > 0
    assert row["total_hits"] >= row["unique_experts"] > 0
    # Logical counters against the independent router oracle.
    assert row["expert_fetches"] == READS_PER_HIT * row["total_hits"]
    assert row["expert_bytes_read"] > 0
    # Physical reads, kept apart: one request per stored tensor of each fetched expert, and at
    # least the logical bytes because every span is rounded out to sector bounds.
    assert row["physical_calls"] > 0
    assert row["physical_calls"] % row["expert_fetches"] == 0
    assert row["physical_bytes"] >= row["expert_bytes_read"]


def test_expert_reads_grow_with_the_micro_batches_of_a_window(
    sweep: Callable[[int], Dict[str, int]], record_property: Callable[..., None],
) -> None:
    """The verdict: the number of expert reads follows k, it is not one read per window."""
    rows = {k: sweep(k) for k in K_SWEEP}
    fetches = [rows[k]["expert_fetches"] for k in K_SWEEP]
    loads = Counter(rows[k]["layer_loads"] for k in K_SWEEP)
    record_property("expert_fetches_by_k", fetches)
    record_property("layer_loads_by_k", [rows[k]["layer_loads"] for k in K_SWEEP])
    # Expert reads increase with k at every step of the sweep.
    assert all(after > before for before, after in zip(fetches, fetches[1:])), fetches
    # The reads of one window exceed what one read per hit expert per window would take
    # (forward + recompute + backward of each unique expert) once k > 1.
    for k in K_SWEEP[1:]:
        floor = READS_PER_HIT * rows[k]["unique_experts"]
        assert rows[k]["expert_fetches"] > floor, (k, rows[k])
    # Reads per micro-batch do not fall as k grows: no reuse across micro-batches.
    per_micro_batch = [rows[k]["expert_fetches"] / k for k in K_SWEEP]
    record_property("expert_fetches_per_micro_batch", per_micro_batch)
    assert min(per_micro_batch) > 0.5 * max(per_micro_batch), per_micro_batch
    # Layer loads are the part that does not follow k.
    assert len(loads) == 1, loads

