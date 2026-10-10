"""D2: QKV backward must not dequantize frozen bases for an unneeded dX."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("ranks", [(2, 0, 3), (2, 2, 2)])
@pytest.mark.parametrize("needs_input_grad", [False, True])
@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "nf4"])
def test_qkv_backward_only_reads_base_weights_when_dx_is_needed(
    monkeypatch: pytest.MonkeyPatch,
    ranks: tuple[int, int, int],
    needs_input_grad: bool,
    quantized: bool,
) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    from soup_cli.utils import fast_lora_qkv

    torch.manual_seed(41)
    inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=needs_input_grad)
    outputs = (8, 4, 4)
    weights = [torch.randn(width, 8, dtype=torch.float64) for width in outputs]
    kernel_weights = weights
    quant_metas = [None, None, None]
    quant_parts = []
    if quantized:
        bnb = pytest.importorskip("bitsandbytes")
        from soup_cli.utils.fast_lora import _quant_state_parts

        packed_states = [
            bnb.functional.quantize_4bit(
                weight.float(), quant_type="nf4", compress_statistics=True
            )
            for weight in weights
        ]
        kernel_weights = [packed for packed, _state in packed_states]
        weights = [
            bnb.functional.dequantize_4bit(packed, state).double()
            for packed, state in packed_states
        ]
        quant_metas = []
        for _packed, state in packed_states:
            parts, meta = _quant_state_parts(state)
            meta["_count"] = len(parts)
            quant_metas.append(meta)
            quant_parts.extend(parts)
    biases = [torch.randn(width, dtype=torch.float64) for width in outputs]
    pairs = [
        (
            torch.randn(rank, 8, dtype=torch.float64, requires_grad=True),
            torch.randn(width, rank, dtype=torch.float64, requires_grad=True),
        )
        if rank
        else (inputs.new_empty(0), inputs.new_empty(0))
        for rank, width in zip(ranks, outputs)
    ]
    scalings = (1.2, 0.8, 1.7)
    variables = ([inputs] if needs_input_grad else []) + [
        matrix for pair in pairs for matrix in pair if matrix.numel()
    ]
    reference = []
    for weight, bias, (adapter_a, adapter_b), scaling in zip(
        weights, biases, pairs, scalings
    ):
        value = functional.linear(inputs, weight, bias)
        if adapter_a.numel():
            value = value + functional.linear(
                functional.linear(inputs, adapter_a), adapter_b
            ) * scaling
        reference.append(value)
    reference_grads = torch.autograd.grad(
        sum(value.square().mean() for value in reference), variables
    )

    original_dense_weight = fast_lora_qkv._dense_weight
    base_reads = []

    def count_base_reads(*args, **kwargs):
        base_reads.append(args[0])
        return original_dense_weight(*args, **kwargs)

    monkeypatch.setattr(fast_lora_qkv, "_dense_weight", count_base_reads)
    args = [inputs]
    for weight, bias in zip(kernel_weights, biases):
        args.extend((weight, bias))
    for pair in pairs:
        args.extend(pair)
    args.extend((*scalings, *quant_metas, *quant_parts))
    actual = fast_lora_qkv._qkv_function().apply(*args)
    assert len(base_reads) == 3
    base_reads.clear()
    actual_grads = torch.autograd.grad(
        sum(value.square().mean() for value in actual), variables
    )

    for actual_value, reference_value in zip(actual, reference):
        torch.testing.assert_close(actual_value, reference_value)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad)
    assert len(base_reads) == (3 if needs_input_grad else 0)
