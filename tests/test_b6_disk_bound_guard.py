"""The experimental expert path cannot treat a coarse estimate as a disk bound."""

import pytest


def test_per_expert_disk_gate_fails_closed():
    from soup_cli.trainer.stream_setup import _require_known_expert_disk_bound

    with pytest.raises(ValueError, match="validated disk-space upper bound.*before copying"):
        _require_known_expert_disk_bound("per_expert")


def test_fused_disk_gate_remains_unchanged():
    from soup_cli.trainer.stream_setup import _require_known_expert_disk_bound

    assert _require_known_expert_disk_bound("") is None


def test_nonempty_expert_cache_requires_validated_index(tmp_path):
    from soup_cli.trainer.stream_setup import _guard_expert_cache_reuse

    _guard_expert_cache_reuse(tmp_path, None)
    (tmp_path / "incomplete").write_text("data")
    with pytest.raises(ValueError, match="new or validated cache directory"):
        _guard_expert_cache_reuse(tmp_path, None)
    _guard_expert_cache_reuse(tmp_path, object())
