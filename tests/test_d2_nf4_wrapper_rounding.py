"""NF4 storage must not substitute Linear4bit's cast seam for dense PEFT."""

import pytest


@pytest.mark.parametrize("wrapper", ["dense", "bnb"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_projection_state_preserves_wrapper_update_rounding(wrapper, dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from benchmarks.harness.d2_nf4_probe import make_fixture
    from soup_cli.utils.fast_lora import _projection_state, _scaled_lora_add

    base_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model = make_fixture("single", base_dtype, "cpu", peft_wrapper=wrapper)
    state = _projection_state(model.o_proj, torch.ones(1, 1, 8, dtype=base_dtype))
    assert state.qmeta["_narrow_lora_update"] is (wrapper == "bnb")
    out = torch.ones(1, 1, dtype=base_dtype)
    hidden = torch.ones(1, 1, dtype=torch.float32)
    # Above the midpoint of an output ulp, below the midpoint of an update ulp.
    # Rounding the update first changes the final value by one output ulp.
    update = 2**-8 + 2**-17 if dtype == "bf16" else 2**-11 + 2**-23
    adapter_b = torch.tensor([[update / 2]], dtype=torch.float32)
    dense_expected = (out.float() + update).to(base_dtype)
    bnb_expected = out + torch.tensor(update, dtype=base_dtype)
    assert not torch.equal(dense_expected, bnb_expected)
    actual = _scaled_lora_add(
        out, hidden, adapter_b, 2.0, quantized=state.qmeta["_narrow_lora_update"]
    )
    torch.testing.assert_close(
        actual, bnb_expected if wrapper == "bnb" else dense_expected, rtol=0, atol=0
    )
