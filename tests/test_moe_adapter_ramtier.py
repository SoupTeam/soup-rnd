"""CPU checks for benchmarks/harness/moe_adapter_ramtier.py (E1 in Soup's RAM tier).

They pin what a RAM-tier score record leans on: Soup's saved adapter lands in the
measured harness's modules exactly (and anything the harness would compute
differently is refused), and the bytes-per-parameter figure the memory rule reads
is taken from the live tensors and refuses a mixed layout.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_ramtier.py"
HIDDEN, SHARED_WIDTH = 12, 10


@pytest.fixture(scope="module")
def ramtier():
    original_path = sys.path[:]
    spec = importlib.util.spec_from_file_location("_moe_adapter_ramtier", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def measured(ramtier):
    original_path = sys.path[:]
    try:
        yield ramtier.confirm.load_measured_harness()
    finally:
        sys.path[:] = original_path
        sys.modules.pop("_e1_measured_harness", None)


def _tiny_tree():
    """The attribute layout ``attach_adapters`` walks: one linear-attention, one full layer."""
    from torch import nn

    def block(names, dims):
        holder = nn.Module()
        for name, (fan_in, fan_out) in zip(names, dims):
            setattr(holder, name, nn.Linear(fan_in, fan_out, bias=False))
        return holder

    first = nn.Module()
    first.linear_attn = block(("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"),
                              ((HIDDEN, 18), (HIDDEN, 8), (HIDDEN, 2), (HIDDEN, 2), (8, HIDDEN)))
    second = nn.Module()
    second.self_attn = block(("q_proj", "k_proj", "v_proj", "o_proj"),
                             ((HIDDEN, 16), (HIDDEN, 8), (HIDDEN, 8), (16, HIDDEN)))
    for layer in (first, second):
        layer.mlp = nn.Module()
        layer.mlp.shared_expert = block(("gate_proj", "up_proj", "down_proj"),
                                        ((HIDDEN, SHARED_WIDTH), (HIDDEN, SHARED_WIDTH),
                                         (SHARED_WIDTH, HIDDEN)))
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([first, second])
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    return model


def _attached(measured, arm):
    model = _tiny_tree()
    measured.attach_adapters(model, arm, seed=0)
    return model


def _soup_adapter(directory: Path, measured, arm: str, targets, *, alpha: int = 32):
    """A Soup-shaped adapter for the tiny tree: PEFT key spelling, bf16, adapter_config."""
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(7)
    tensors = {}
    for name, module in _attached(measured, arm).named_modules():
        if isinstance(module, measured.LoRALinear):
            for part in ("lora_A", "lora_B"):
                tensors[f"base_model.model.{name}.{part}.weight"] = torch.randn(
                    tuple(getattr(module, part).shape), generator=generator).to(torch.bfloat16)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    config = {"r": 16, "lora_alpha": alpha, "lora_dropout": 0.0, "use_rslora": False,
              "use_dora": False, "bias": "none", "target_parameters": None,
              "modules_to_save": None, "target_modules": list(targets)}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tensors


@pytest.mark.parametrize("arm", ["A", "AS"])
def test_copy_is_exact_and_forward_is_base_plus_scaled_delta(ramtier, measured, tmp_path, arm):
    tensors = _soup_adapter(tmp_path, measured, arm, ramtier.ARM_TARGETS[arm])
    model = _attached(measured, arm)
    summary = ramtier.copy_adapter_into_harness(model, measured, str(tmp_path), arm)
    assert summary["tensors"] == len(tensors) == (30 if arm == "AS" else 18)
    for key, value in tensors.items():
        path, part, _weight = key.removeprefix("base_model.model.").rsplit(".", 2)
        param = getattr(model.get_submodule(path), part)
        assert param.dtype == torch.float32
        assert torch.equal(param.detach(), value.float())
    base = _tiny_tree()
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1))
    paths = ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.self_attn.q_proj"]
    if arm == "AS":
        paths.append("model.layers.1.mlp.shared_expert.gate_proj")
    for path in paths:
        a = tensors[f"base_model.model.{path}.lora_A.weight"].double()
        b = tensors[f"base_model.model.{path}.lora_B.weight"].double()
        with torch.no_grad():
            expected = base.get_submodule(path)(x).double() + 2.0 * (x.double() @ a.T @ b.T)
            got = model.get_submodule(path)(x).double()
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_refuses_a_missing_key(ramtier, measured, tmp_path):
    from safetensors.torch import load_file, save_file

    _soup_adapter(tmp_path, measured, "AS", ramtier.ARM_TARGETS["AS"])
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    tensors.pop("base_model.model.model.layers.1.mlp.shared_expert.down_proj.lora_B.weight")
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="1 missing"):
        ramtier.copy_adapter_into_harness(_attached(measured, "AS"), measured, str(tmp_path),
                                          "AS")


def test_refuses_an_unexpected_key(ramtier, measured, tmp_path):
    # An AS adapter declared as A: its shared-expert tensors have no harness module.
    _soup_adapter(tmp_path, measured, "AS", ramtier.ARM_TARGETS["A"])
    with pytest.raises(ValueError, match="12 unexpected"):
        ramtier.copy_adapter_into_harness(_attached(measured, "A"), measured, str(tmp_path), "A")


def test_refuses_a_shape_mismatch(ramtier, measured, tmp_path):
    from safetensors.torch import load_file, save_file

    _soup_adapter(tmp_path, measured, "A", ramtier.ARM_TARGETS["A"])
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    tensors["base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight"] = torch.zeros(
        16, 17, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        ramtier.copy_adapter_into_harness(_attached(measured, "A"), measured, str(tmp_path), "A")


def test_refuses_a_scale_the_harness_does_not_use(ramtier, measured, tmp_path):
    _soup_adapter(tmp_path, measured, "A", ramtier.ARM_TARGETS["A"], alpha=16)
    with pytest.raises(ValueError, match="harness scale"):
        ramtier.copy_adapter_into_harness(_attached(measured, "A"), measured, str(tmp_path), "A")


def _after_one_adamw_step(dtypes):
    from torch import nn

    layers = nn.ModuleList([nn.Linear(4, 4, bias=False).to(dtype) for dtype in dtypes])
    optimizer = torch.optim.AdamW(layers.parameters(), lr=1e-3)
    out = torch.randn(2, 4, generator=torch.Generator().manual_seed(0))
    for layer in layers:
        out = layer(out.to(layer.weight.dtype))
    out.float().sum().backward()
    optimizer.step()
    return list(layers.named_parameters()), optimizer


@pytest.mark.parametrize(("dtype", "expected"), [(torch.float32, 16), (torch.bfloat16, 8)])
def test_bytes_per_parameter_from_live_tensors(ramtier, dtype, expected):
    named, optimizer = _after_one_adamw_step([dtype, dtype])
    result = ramtier.bytes_per_parameter(named, optimizer)
    assert result["problem"] is None
    assert result["bytes_per_parameter"] == expected
    assert result["breakdown"][0]["state_dtypes"] == {"exp_avg": str(dtype),
                                                      "exp_avg_sq": str(dtype)}
    assert "step" in result["scalar_state"]


def test_bytes_per_parameter_refuses_mixed_dtypes(ramtier):
    named, optimizer = _after_one_adamw_step([torch.float32, torch.bfloat16])
    result = ramtier.bytes_per_parameter(named, optimizer)
    assert result["bytes_per_parameter"] is None
    assert result["problem"] == "2 different per-parameter layouts"
