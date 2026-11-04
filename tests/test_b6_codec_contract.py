"""Reject incompatible codec output before writing an expert cache."""

import pytest


@pytest.mark.parametrize("corruption", ["extra", "shape", "dtype", "blocksize"])
def test_codec_contract_checked_before_write(tmp_path, monkeypatch, corruption):
    import torch

    from soup_cli.utils import layer_shard
    from soup_cli.utils.expert_shard import write_expert_shard

    original = layer_shard._quantize_nf4

    def corrupt(*args, **kwargs):
        sidecars, spec, code, nested = original(*args, **kwargs)
        if corruption == "extra":
            sidecars["::unknown"] = torch.zeros(1)
        elif corruption == "shape":
            sidecars[""] = sidecars[""].flatten()
        elif corruption == "dtype":
            sidecars[""] = sidecars[""].float()
        else:
            from dataclasses import replace

            spec = replace(spec, blocksize=32)
        return sidecars, spec, code, nested

    def unexpected(*args, **kwargs):
        pytest.fail("incompatible codec output reached the filesystem writer")

    monkeypatch.setattr(layer_shard, "_quantize_nf4", corrupt)
    monkeypatch.setattr(layer_shard, "_atomic_save", unexpected)
    output = tmp_path / "experts.safetensors"
    with pytest.raises(ValueError, match="NF4 codec returned unexpected"):
        write_expert_shard(
            str(output), [(torch.randn(128, 64), torch.randn(64, 64))],
            quantise=True, double_quant=False,
        )
    assert not output.exists()
