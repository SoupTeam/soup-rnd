"""Atomic replacement reserves space before creating the temporary shard."""

from types import SimpleNamespace

import pytest


def test_insufficient_space_keeps_existing_shard(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils import layer_shard

    output = tmp_path / "layer.safetensors"
    output.write_bytes(b"existing checkpoint")
    monkeypatch.setattr(layer_shard.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="insufficient disk space for atomic shard write"):
        layer_shard._atomic_save({"weight": torch.ones(8, 8)}, str(output))
    assert output.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.iterdir()) == [output]


def test_space_checked_for_complete_new_file_not_size_difference(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils import layer_shard

    output = tmp_path / "layer.safetensors"
    output.write_bytes(b"old" * 10000)
    monkeypatch.setattr(layer_shard.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**20))
    with pytest.raises(OSError, match="including reserve"):
        layer_shard._atomic_save({"weight": torch.ones(8, 8)}, str(output))
    assert output.read_bytes() == b"old" * 10000
