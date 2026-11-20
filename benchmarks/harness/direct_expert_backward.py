"""Harness alias of ``soup_cli.utils.direct_expert_backward`` (moved into ``src``, ADR 0002)."""

from soup_cli.utils.direct_expert_backward import *  # noqa: F401,F403
from soup_cli.utils.direct_expert_backward import (  # noqa: F401
    direct_backward,
    expert_vjp,
    fused_node,
    segment_mm,
    weighted_expert_function,
)
