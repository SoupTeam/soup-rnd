"""The positive control's injections for G7's control arm.

The control arm runs the timer and profile instruments with two known delays added, so a
negative result from the profile means the instrument can see a delay. ``control_wrappers``
returns the two wrappers:

- ``substituted_weights``: a 200 us spin-wait on the nanosecond clock before the original
  runs. The runtime calls it once per visit, in weight preparation.
- ``functional_call``: 50 launches of a trivial kernel before the original runs.

``visit_ranges.install_ranges`` takes them as its ``inner`` wrappers, so both delays fall
inside the ranges they must be found in. ``visit_timer`` and ``visit_ranges`` are not changed.

Benchmark harness code, not shipped. ``torch`` is imported inside ``cuda_launch``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

SPIN_NS = 200_000
LAUNCHES = 50


def cuda_launch() -> Callable[[], None]:
    """A callable that launches one trivial kernel: an in-place add on a one-element tensor."""
    import torch

    buffer = torch.zeros(1, device="cuda")

    def launch() -> None:
        buffer.add_(1)

    return launch


def control_wrappers(
    *,
    spin_ns: int = SPIN_NS,
    launches: int = LAUNCHES,
    clock: Callable[[], int] = time.perf_counter_ns,
    launch: Optional[Callable[[], None]] = None,
) -> Dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """The substitution spin and the ``functional_call`` launches. ``launch`` defaults to
    ``cuda_launch()``, built when this function runs."""
    kernel = launch if launch is not None else cuda_launch()

    def wrap_substituted_weights(orig: Callable[..., Any]) -> Callable[..., Any]:
        def substituted_weights(*args: Any, **kwargs: Any) -> Any:
            end = clock() + spin_ns
            while clock() < end:
                pass
            return orig(*args, **kwargs)

        return substituted_weights

    def wrap_functional_call(orig: Callable[..., Any]) -> Callable[..., Any]:
        def functional_call(*args: Any, **kwargs: Any) -> Any:
            for _ in range(launches):
                kernel()
            return orig(*args, **kwargs)

        return functional_call

    return {
        "substituted_weights": wrap_substituted_weights,
        "functional_call": wrap_functional_call,
    }
