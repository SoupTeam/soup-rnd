"""The evidence comparator must reject divergent training and unequal inputs."""

import json

import pytest


@pytest.mark.parametrize("group", ["batch/0/input_ids", "initial/lora_A", "gradient/0/lora_B",
                                   "final/lora_B", "logits/0", "loss"])
def test_parity_rejects_each_divergent_group(tmp_path, group):
    import torch
    from safetensors.torch import save_file

    from benchmarks.harness.liger_streaming_parity import compare

    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    trace = {name: torch.tensor([0.01, 0.02]) for name in
             ("initial/lora_A", "gradient/0/lora_B", "final/lora_B", "logits/0")}
    trace["batch/0/input_ids"] = torch.tensor([[1, 2]], dtype=torch.int64)
    trace["loss"] = torch.tensor([1.0004, 1.0004], dtype=torch.float64)
    save_file(trace, str(reference / "trace.safetensors"))
    save_file(trace, str(candidate / "trace.safetensors"))
    assert compare(reference, candidate)["passed"]
    trace[group] = trace[group].clone()
    if group == "loss":
        # Well inside the relative tolerance, but crosses the original .3f gate.
        trace[group][0] = 1.0006
    elif group.startswith("batch"):
        trace[group][0, 0] = 3
    else:
        trace[group][0] = 0.02
    save_file(trace, str(candidate / "trace.safetensors"))
    result = compare(reference, candidate)
    assert not result["passed"], json.dumps(result)


def test_adapter_initialization_is_independent_of_global_rng():
    import torch

    from benchmarks.harness.liger_streaming_parity import initialize_adapters

    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.empty(4, 16)))
    model.register_parameter("lora_B", torch.nn.Parameter(torch.empty(16, 4)))
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(8), requires_grad=False))
    first = {name: value.clone() for name, value in initialize_adapters(model).items()}
    torch.manual_seed(98765)
    second = initialize_adapters(model)
    assert all(torch.equal(value, second[name]) for name, value in first.items())
    assert torch.count_nonzero(model.lora_B) == model.lora_B.numel()
    assert torch.equal(model.frozen, torch.ones(8))
