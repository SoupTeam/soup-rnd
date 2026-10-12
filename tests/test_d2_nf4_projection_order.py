"""NF4 projection arithmetic must preserve the unchanged dense scope guard."""

import copy

import pytest


@pytest.mark.parametrize("path", ["qkv", "mlp"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_nf4_projection_order_keeps_dense_reference_scope_false(path, device):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from benchmarks.harness import fast_lora_probe as probe
    from benchmarks.harness.d2_nf4_probe import make_fixture, make_oracle

    with probe._deterministic(792, device):
        reference = make_fixture(path, torch.bfloat16, device)
        oracle_model = make_oracle(reference, path)
        fast = copy.deepcopy(reference)
        assert probe._patch(fast, path) == 1
        source = torch.randn(2, 3, 8, dtype=torch.bfloat16, device=device)
        inputs = source.clone().requires_grad_(True)
        outputs = probe._outputs(fast, inputs)
        for output in outputs:
            assert type(output.grad_fn).__name__ == probe.EXPECTED_GRAD_FN[path]
            assert output.grad_fn.nf4_projection_order is True
            assert output.grad_fn.reference_order is False
        upstream = tuple(torch.randn_like(value) for value in outputs)
        actual = probe._snapshot(fast, inputs, upstream, path)
        expected = probe._snapshot(reference, source.clone().requires_grad_(True), upstream, path)
        oracle = probe._snapshot(
            oracle_model, source.double().requires_grad_(True),
            tuple(value.double() for value in upstream), path,
        )
        probe._validate_snapshot_keys(path, True, fast=actual, PEFT=expected, float64=oracle)
        for quantity in actual:
            probe._comparison(actual[quantity], expected[quantity], oracle[quantity], "bf16")
