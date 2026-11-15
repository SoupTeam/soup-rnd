"""``stream_source: ram`` on a base that does not fit: one open finding, one guard kept.

Found while checking Qwen3.5-35B-A3B on the RAM tier for B5
(``benchmarks/b5-qwen35-report2-20261109.md`` on PR #60). Nothing in ``src/`` is changed here.

* The first test is a strict expected failure. ``xfail(strict=True)`` turns it into a failure
  the day the estimate is fixed, so the marker is removed together with the fix.
* The second test passes today. B5 report 2 first read ``stream_setup.py:1065`` as a path where
  ``stream_source='ram'`` continues on the disk tier. Running the pre-flight showed that
  ``_validate_qwen4_ngram_ram_fit`` refuses first, from the actual store size, so the test
  keeps that refusal in place.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.layer_stream import estimate_stream_store_bytes
from tests import test_v07203

# Qwen3.5-35B-A3B at revision 59d61f3c, from the safetensors headers (B5 report 2).
QWEN35_SOURCE_BYTES = 71_903_878_016
QWEN35_ROUTED_EXPERT_BYTES_BF16 = 64_424_509_440


@pytest.mark.xfail(
    strict=True,
    reason="the pre-flight estimate charges every parameter the NF4 rate, but "
    "fused experts are 3D Parameters, stay bf16 under NF4, and are 90% of this checkpoint",
)
def test_the_preflight_estimate_is_not_below_the_bf16_expert_bytes_that_nf4_keeps():
    estimate = estimate_stream_store_bytes(
        QWEN35_SOURCE_BYTES, dtype="bfloat16", quant="nf4", double_quant=False
    )

    assert estimate >= QWEN35_ROUTED_EXPERT_BYTES_BF16, (
        f"estimate {estimate / 1e9:.1f} GB is below the {QWEN35_ROUTED_EXPERT_BYTES_BF16 / 1e9:.1f}"
        " GB of bf16 experts the cache will hold (true cache 67.3 GB, estimate 20.2 GB)"
    )


def test_ram_refuses_from_the_actual_store_size_after_sharding(tmp_path, monkeypatch):
    """The early check passes (10 GB free), then the physical-RAM ceiling is exceeded by the
    actual store. ``stream_source: ram`` must still refuse, not continue on the disk tier."""
    monkeypatch.setattr("soup_cli.utils.layer_stream.total_ram_bytes", lambda: 10_000)

    with pytest.raises(ValueError, match="stream_source='ram'"):
        harness = test_v07203.TestAutoTierFallback()
        harness._run(
            tmp_path, monkeypatch,
            free_ram=10_000_000_000, stream_source="ram", disk_kind="nvme",
        )
