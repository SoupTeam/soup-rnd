"""The narrow header budget covers real serialized synthetic cache files."""

import pytest


def test_header_budget_covers_serialized_expert_cache(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from tests.test_b1_expert_shard import _checkpoint, _shard

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, count=4)
    paths = [str(source / "model.safetensors")]
    budget = new_expert_cache_budget(paths)
    _shard(source, output, expert_layout="per_expert", quant="nf4",
           quant_suffixes=("self_attn.q_proj.weight",), double_quant=False)
    assert sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < budget


def test_budget_refuses_truncated_source_before_using_shapes(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    path = tmp_path / "broken.safetensors"
    path.write_bytes((100).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="truncated"):
        new_expert_cache_budget([str(path)])
