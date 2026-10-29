"""The narrow header budget covers real serialized synthetic cache files."""

import pytest


def test_header_budget_covers_serialized_expert_cache(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from tests.test_b1_expert_shard import _checkpoint, _shard

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, count=4)
    paths = [str(source / "model.safetensors")]
    budget = new_expert_cache_budget(paths)
    selected = new_expert_cache_budget(
        paths, quant_suffixes=("self_attn.q_proj.weight",), n_layers=1,
    )
    _shard(source, output, expert_layout="per_expert", quant="nf4",
           quant_suffixes=("self_attn.q_proj.weight",), double_quant=False)
    assert sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < budget
    assert sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < selected
    assert selected < budget


def test_budget_refuses_truncated_source_before_using_shapes(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    path = tmp_path / "broken.safetensors"
    path.write_bytes((100).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="truncated"):
        new_expert_cache_budget([str(path)])


@pytest.mark.parametrize("n_layers", [0, -1, True, 1.5])
def test_budget_rejects_invalid_explicit_layer_count(n_layers):
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    with pytest.raises(ValueError, match="positive integer"):
        expert_cache_budget_from_headers([], n_layers=n_layers)


def test_dropped_source_layer_still_requires_valid_offsets():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    header = {"model.layers.2.mlp.experts.0.gate_proj.weight":
              {"shape": [64, 64], "dtype": "F32", "data_offsets": [0, 16384]}}
    with pytest.raises(ValueError, match="offsets"):
        expert_cache_budget_from_headers([(1024, 128, header)], n_layers=1)
