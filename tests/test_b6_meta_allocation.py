"""Meta construction must not reserve a giant CPU expert tensor first."""

import pytest


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_native_fused_experts_never_request_cpu_storage_and_buffers_keep_values(tmp_path, quant):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers import AutoModelForCausalLM
    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton
    from tests.test_b2_deepseek_v3_stream import _tiny_deepseek_v3_config

    if not hasattr(modeling_deepseek_v3, "DeepseekV3Experts"):
        pytest.skip("this transformers uses separate experts, not a giant fused allocation")
    config = _tiny_deepseek_v3_config(n_routed_experts=4, n_group=2, topk_group=1)
    config.save_pretrained(tmp_path)
    reference = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    expected_buffers = {name: tensor.clone() for name, tensor in reference.named_buffers()}
    original_init = modeling_deepseek_v3.DeepseekV3Experts.__init__
    requests = []

    class ExpertStorageGuard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.empty.memory_format and len(args[0]) == 3:
                device = torch.device(kwargs.get("device") or "cpu")
                requests.append(device.type)
                assert device.type == "meta", "fused expert storage was requested on CPU"
            return func(*args, **kwargs)

    with ExpertStorageGuard():
        model = build_meta_skeleton(str(tmp_path), dtype="float32", quant=quant, double_quant=False)
    assert requests and set(requests) == {"meta"}
    assert all(parameter.is_meta for parameter in model.parameters())
    actual_buffers = dict(model.named_buffers())
    assert actual_buffers.keys() == expected_buffers.keys()
    assert all(not tensor.is_meta and torch.equal(tensor, expected_buffers[name])
               for name, tensor in actual_buffers.items())
    assert modeling_deepseek_v3.DeepseekV3Experts.__init__ is original_init


def test_native_expert_constructor_is_restored_on_failure(tmp_path, monkeypatch):
    import torch
    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton
    from tests.test_b2_deepseek_v3_stream import _tiny_deepseek_v3_config

    cls = getattr(modeling_deepseek_v3, "DeepseekV3Experts", None)
    if cls is None:
        pytest.skip("separate-expert transformers has no fused constructor")
    config = _tiny_deepseek_v3_config(n_routed_experts=4, n_group=2, topk_group=1)
    config.save_pretrained(tmp_path)
    original = cls.__init__

    def changed_constructor(instance, *args, **kwargs):
        original(instance, *args, **kwargs)
        instance.register_buffer("unexpected", torch.ones(1))

    monkeypatch.setattr(cls, "__init__", changed_constructor)
    with pytest.raises(ValueError, match="unmaterialized buffers"):
        build_meta_skeleton(str(tmp_path), dtype="float32")
    assert cls.__init__ is changed_constructor
    ordinary = cls(config)
    assert ordinary.gate_up_proj.device.type == "cpu"


def test_expert_factory_redirection_is_scoped_to_the_constructing_thread():
    from concurrent.futures import ThreadPoolExecutor

    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    from soup_cli.utils.layer_stream_runtime import _meta_fused_expert_initialization
    from tests.test_b2_deepseek_v3_stream import _tiny_deepseek_v3_config

    cls = getattr(modeling_deepseek_v3, "DeepseekV3Experts", None)
    if cls is None:
        pytest.skip("separate-expert transformers has no fused constructor")
    config = _tiny_deepseek_v3_config(n_routed_experts=4, n_group=2, topk_group=1)
    with _meta_fused_expert_initialization(config):
        meta = cls(config)
        assert all(parameter.is_meta for parameter in meta.parameters()), {
            "kind": config.model_type,
            "parameters": [(name, str(p.device)) for name, p in meta.named_parameters()],
        }
        with ThreadPoolExecutor(max_workers=1) as pool:
            ordinary = pool.submit(cls, config).result()
    assert all(parameter.is_meta for parameter in meta.parameters())
    assert all(parameter.device.type == "cpu" for parameter in ordinary.parameters())
