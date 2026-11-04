"""Installation failures must release resources and restore the caller's model."""

import pytest

from tests.test_b2_deepseek_v3_stream import _save_fixture, _tiny_deepseek_v3_config
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR


def _edges(model):
    stack, seen, edges = [model], set(), {}
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        edges[id(module)] = {name: id(child) for name, child in module._modules.items()}
        stack.extend(child for child in module._modules.values() if child is not None)
    return edges


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    "failure", ["after_expert_install", "after_source", "after_hook", "cleanup_error"]
)
def test_install_failure_restores_model_and_closes_resources(tmp_path, monkeypatch, failure):
    from soup_cli.utils import layer_stream_runtime as runtime
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights, shards = tmp_path / "weights", tmp_path / "shards"
    config = _tiny_deepseek_v3_config(n_routed_experts=4, n_group=2, topk_group=1)
    _save_fixture(weights, config)
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="deepseek_v3",
        expert_layout="per_expert",
    )
    model = runtime.build_meta_skeleton(str(weights), dtype="float32")
    model.requires_grad_(False)
    owner = runtime.decoder_owner(model)
    before = _edges(model)
    before_hooks = dict(owner._forward_pre_hooks)
    had_device_map = "hf_device_map" in vars(model)
    original_device_map = vars(model).get("hf_device_map")
    monkeypatch.setattr(runtime, "_refuse_expert_storage_cache", lambda _: None)
    prepared, sources = [], []
    original_prepare, original_source = runtime._prepare_expert_streaming, runtime._build_source

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        prepared.append(result)
        if failure == "cleanup_error":
            original_close = result[0].close

            def fail_close():
                original_close()
                raise RuntimeError("injected cleanup failure")

            monkeypatch.setattr(result[0], "close", fail_close)
        return result

    def source(*args, **kwargs):
        result = original_source(*args, **kwargs)
        sources.append(result[0])
        return result

    def fail(*args, **kwargs):
        raise RuntimeError("injected install failure")

    monkeypatch.setattr(runtime, "_prepare_expert_streaming", prepare)
    monkeypatch.setattr(runtime, "_build_source", source)
    if failure == "after_expert_install":
        monkeypatch.setattr(runtime, "_layer_name_map", fail)
    elif failure == "after_source":
        monkeypatch.setattr(runtime, "_streamed_layer_class", fail)
    else:
        monkeypatch.setattr(runtime, "_device_map_value", fail)
    with pytest.raises(RuntimeError, match="injected install failure"):
        runtime.install_streaming(
            model, shard_dir=str(shards), index=index, pin=False, device="cpu", tier="disk",
        )
    assert _edges(model) == before
    assert dict(owner._forward_pre_hooks) == before_hooks
    assert ("hf_device_map" in vars(model)) == had_device_map
    if had_device_map:
        assert vars(model)["hf_device_map"] is original_device_map
    assert len(prepared) == 1 and prepared[0][0].stats()["closed"]
    assert all(reader._handle is None for reader in prepared[0][1])
    if failure != "after_expert_install":
        assert len(sources) == 1 and sources[0]._closed
