"""Harness alias of ``soup_cli.utils.a0_exact_lt`` (the extension moved into ``src``, ADR 0002).

Scripts that put ``benchmarks/harness`` on ``sys.path`` keep ``import a0_exact_lt``."""

from soup_cli.utils.a0_exact_lt import *  # noqa: F401,F403
from soup_cli.utils.a0_exact_lt import (  # noqa: F401
    AlgoTable,
    FusedExpert,
    LtExtensionBackend,
    MergedLinear,
    algo_key,
    bucket_segments,
    load_extension,
    merged_matmul,
    native_matmul,
)
