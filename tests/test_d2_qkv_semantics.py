"""D2: shared-X QKV must preserve individual PEFT projection semantics."""

from __future__ import annotations

import copy
from typing import Any

import pytest

_PROJECTIONS = ("q_proj", "k_proj", "v_proj")


def _make_model(targets: tuple[str, ...] = _PROJECTIONS, *, second_adapter: bool = False) -> Any:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, inject_adapter_in_model
    from torch import nn

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 4)
            self.v_proj = nn.Linear(8, 4)

        def forward(self, x: Any) -> tuple[Any, ...]:
            return tuple(getattr(self, name)(x) for name in _PROJECTIONS)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()

        def forward(self, x: Any) -> tuple[Any, ...]:
            return self.attn(x)

    torch.manual_seed(83)
    model = Model()
    inject_adapter_in_model(
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=list(targets),
            rank_pattern={"k_proj": 3, "v_proj": 4},
        ),
        model,
    )
    if second_adapter:
        with pytest.warns(UserWarning, match="Already found a `peft_config`"):
            inject_adapter_in_model(
                LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=list(targets)),
                model,
                adapter_name="other",
            )
        for name in targets:
            getattr(model.attn, name).set_adapter("default")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.2)
    return model


def _assert_gradients_match(actual: Any, reference: Any) -> None:
    torch = pytest.importorskip("torch")
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        wanted = reference_parameters[name].grad
        assert (parameter.grad is None) == (wanted is None), name
        if wanted is not None:
            torch.testing.assert_close(parameter.grad, wanted, msg=name)


@pytest.mark.parametrize("used", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("needs_input_grad", [False, True])
@pytest.mark.parametrize("targets", [_PROJECTIONS, ("q_proj", "v_proj")])
def test_output_subsets_preserve_none_gradients_and_adamw_step(
    used: tuple[int, ...], needs_input_grad: bool, targets: tuple[str, ...]
) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model(targets)
    reference = copy.deepcopy(actual)
    x_ref = torch.randn(2, 3, 8, requires_grad=needs_input_grad)
    x = x_ref.detach().clone().requires_grad_(needs_input_grad)
    assert patch_fast_lora_qkv(actual) == 1
    initial = {name: parameter.detach().clone() for name, parameter in actual.named_parameters()}
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.3)
        for model in (actual, reference)
    ]

    expected = reference(x_ref)
    output = actual(x)
    assert all(type(part.grad_fn).__name__ == "_FastLoraQKVBackward" for part in output)
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got, wanted)
    expected_loss = sum(expected[index].square().mean() for index in used)
    actual_loss = sum(output[index].square().mean() for index in used)
    # A loss of only an unadapted output with a frozen input has no autograd graph in PEFT.
    if expected_loss.requires_grad:
        expected_loss.backward()
    actual_loss.backward()
    _assert_gradients_match(actual, reference)
    if needs_input_grad:
        torch.testing.assert_close(x.grad, x_ref.grad)
    else:
        assert x.grad is x_ref.grad is None

    for optimizer in optimizers:
        optimizer.step()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], msg=name)
        if "lora_" in name and not any(_PROJECTIONS[index] in name for index in used):
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("changed_before", ["k_proj", "v_proj"])
def test_cache_rejects_input_detached_and_reenabled_in_place(changed_before: str) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    leaf_ref = torch.randn(2, 3, 8, requires_grad=True)
    leaf = leaf_ref.detach().clone().requires_grad_(True)
    x_ref, x = leaf_ref * 2, leaf * 2
    for model, inputs in ((reference, x_ref), (actual, x)):
        model.attn.q_proj(inputs)
        if changed_before == "v_proj":
            model.attn.k_proj(inputs)
        version = inputs._version
        inputs.detach_().requires_grad_(True)
        assert inputs._version == version

    expected = getattr(reference.attn, changed_before)(x_ref)
    output = getattr(actual.attn, changed_before)(x)
    torch.testing.assert_close(output, expected)
    expected.square().mean().backward()
    output.square().mean().backward()
    assert x_ref.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert leaf.grad is leaf_ref.grad is None
    _assert_gradients_match(actual, reference)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is None


@pytest.mark.parametrize("needs_input_grad", [False, True])
@pytest.mark.parametrize(
    ("consumed", "subsequent"),
    [("q_proj", "k_proj"), ("q_proj", "v_proj"), ("k_proj", "v_proj")],
)
def test_projection_after_sibling_backward_uses_a_fresh_graph(
    consumed: str, subsequent: str, needs_input_grad: bool
) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x_ref = torch.randn(2, 3, 8, requires_grad=needs_input_grad)
    x = x_ref.detach().clone().requires_grad_(needs_input_grad)
    for model, inputs in ((reference, x_ref), (actual, x)):
        q = model.attn.q_proj(inputs)
        first = q if consumed == "q_proj" else model.attn.k_proj(inputs)
        first.sum().backward()

    expected = getattr(reference.attn, subsequent)(x_ref)
    output = getattr(actual.attn, subsequent)(x)
    torch.testing.assert_close(output, expected)
    expected.sum().backward()
    output.sum().backward()
    _assert_gradients_match(actual, reference)
    if needs_input_grad:
        torch.testing.assert_close(x.grad, x_ref.grad)
    else:
        assert x.grad is x_ref.grad is None
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is None


@pytest.mark.parametrize("retain_old_graph", [False, True])
def test_older_backward_does_not_discard_new_qkv_cache(retain_old_graph: bool) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x_ref = torch.randn(2, 3, 8, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(True)
    old_ref = reference.attn.q_proj(x_ref)
    old = actual.attn.q_proj(x)
    if retain_old_graph:
        old_ref.sum().backward(retain_graph=True)
        old.sum().backward(retain_graph=True)
    expected_q = reference.attn.q_proj(x_ref)
    q = actual.attn.q_proj(x)
    pending = getattr(actual.attn, "_soup_fast_lora_qkv_cache")
    old_ref.square().mean().backward()
    old.square().mean().backward()
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is pending
    expected = (expected_q, reference.attn.k_proj(x_ref), reference.attn.v_proj(x_ref))
    output = (q, actual.attn.k_proj(x), actual.attn.v_proj(x))
    assert all(part.grad_fn is q.grad_fn for part in output)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_last_cache_hits")[0] == 2
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got, wanted)
    sum(part.square().mean() for part in expected).backward()
    sum(part.square().mean() for part in output).backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    _assert_gradients_match(actual, reference)


def test_nonleaf_qkv_preserves_fusion_with_retained_graph() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    leaf_ref = torch.randn(2, 3, 8, requires_grad=True)
    leaf = leaf_ref.detach().clone().requires_grad_(True)
    x_ref, x = leaf_ref * 2, leaf * 2
    expected = reference(x_ref)
    output = actual(x)
    assert all(part.grad_fn is output[0].grad_fn for part in output)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_last_cache_hits")[0] == 2
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got, wanted)
    for index, retain_graph in ((0, True), (1, False)):
        expected[index].square().mean().backward(retain_graph=retain_graph)
        output[index].square().mean().backward(retain_graph=retain_graph)
        torch.testing.assert_close(leaf.grad, leaf_ref.grad)
        _assert_gradients_match(actual, reference)


def test_pending_q_graph_does_not_keep_attention_or_cached_outputs_alive() -> None:
    import gc
    import weakref

    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    assert patch_fast_lora_qkv(actual) == 1
    x = torch.randn(2, 3, 8, requires_grad=True)
    q = actual.attn.q_proj(x)
    attention_ref = weakref.ref(actual.attn)
    cache = getattr(actual.attn, "_soup_fast_lora_qkv_cache")
    cached_refs = [weakref.ref(cache[name]) for name in ("k", "v")]
    del cache, actual
    gc.collect()
    assert attention_ref() is None
    assert all(output_ref() is None for output_ref in cached_refs)
    q.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("q_grad_enabled", [False, True])
@pytest.mark.parametrize("changed_before", ["k_proj", "v_proj"])
def test_cache_respects_current_grad_mode(q_grad_enabled: bool, changed_before: str) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x_ref = torch.randn(2, 3, 8, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(True)
    for model, inputs in ((reference, x_ref), (actual, x)):
        with torch.set_grad_enabled(q_grad_enabled):
            model.attn.q_proj(inputs)
            if changed_before == "v_proj":
                model.attn.k_proj(inputs)
    with torch.set_grad_enabled(not q_grad_enabled):
        expected = getattr(reference.attn, changed_before)(x_ref)
        output = getattr(actual.attn, changed_before)(x)
    assert output.requires_grad == expected.requires_grad
    torch.testing.assert_close(output, expected)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is None
    if expected.requires_grad:
        expected.square().mean().backward()
        output.square().mean().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        _assert_gradients_match(actual, reference)


def _change_projection_state(model: Any, name: str, change: str, inputs: Any) -> None:
    torch = pytest.importorskip("torch")
    projection = getattr(model.attn, name)
    if change == "disable":
        projection.enable_adapters(False)
    elif change == "enable":
        projection.enable_adapters(True)
    elif change == "active":
        projection.set_adapter("other")
    elif change == "scaling":
        projection.scaling["default"] *= 2.5
    elif change == "input_inplace":
        with torch.no_grad():
            inputs.add_(0.5)
    elif change == "input_requires_grad":
        inputs.requires_grad_(False)
    else:
        modules = {
            "a": (projection.lora_A["default"], "weight"),
            "b": (projection.lora_B["default"], "weight"),
            "base": (projection.get_base_layer(), "weight"),
            "bias": (projection.get_base_layer(), "bias"),
        }
        field, operation = change.split("_")
        module, attribute = modules[field]
        parameter = getattr(module, attribute)
        if operation == "inplace":
            with torch.no_grad():
                parameter.add_(0.4)
        elif operation == "replace":
            setattr(module, attribute, torch.nn.Parameter(parameter + 0.4, requires_grad=False))
        elif operation == "freeze":
            parameter.requires_grad_(False)
        else:
            raise AssertionError(change)


@pytest.mark.parametrize("changed_before", ["k_proj", "v_proj"])
@pytest.mark.parametrize(
    "change",
    [
        "disable", "enable", "active", "scaling", "a_inplace", "b_inplace", "base_inplace",
        "bias_inplace", "a_replace", "b_replace", "base_replace", "bias_replace",
        "a_freeze", "input_inplace", "input_requires_grad",
    ],
)
def test_cache_respects_current_projection_state(change: str, changed_before: str) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model(second_adapter=change == "active")
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x_ref = torch.randn(2, 3, 8, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(True)
    for model, inputs in ((reference, x_ref), (actual, x)):
        if change == "enable":
            getattr(model.attn, changed_before).enable_adapters(False)
        model.attn.q_proj(inputs)
        if changed_before == "v_proj":
            model.attn.k_proj(inputs)
        _change_projection_state(model, changed_before, change, inputs)
    expected = getattr(reference.attn, changed_before)(x_ref)
    output = getattr(actual.attn, changed_before)(x)
    torch.testing.assert_close(output, expected)
    assert output.requires_grad == expected.requires_grad
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is None
    assert getattr(actual.attn, "_soup_fast_lora_qkv_last_cache_hits") == (0, ())
    expected.square().mean().backward()
    output.square().mean().backward()
    _assert_gradients_match(actual, reference)
    assert (x.grad is None) == (x_ref.grad is None)
    if x_ref.grad is not None:
        torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.parametrize("changed_before", ["k_proj", "v_proj"])
@pytest.mark.parametrize(
    "dtypes", [(None, "bfloat16"), ("bfloat16", None), ("bfloat16", "float16")]
)
def test_cache_respects_current_autocast(
    dtypes: tuple[str | None, ...], changed_before: str
) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x_ref = torch.randn(2, 3, 8, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=getattr(torch, dtypes[0] or "bfloat16"),
                        enabled=dtypes[0] is not None):
        for model, inputs in ((reference, x_ref), (actual, x)):
            model.attn.q_proj(inputs)
            if changed_before == "v_proj":
                model.attn.k_proj(inputs)
    with torch.autocast("cpu", dtype=getattr(torch, dtypes[1] or "bfloat16"),
                        enabled=dtypes[1] is not None):
        expected = getattr(reference.attn, changed_before)(x_ref)
        output = getattr(actual.attn, changed_before)(x)
    assert output.dtype == expected.dtype
    torch.testing.assert_close(output, expected)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_last_cache_hits") == (0, ())
    expected.float().square().mean().backward()
    output.float().square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    _assert_gradients_match(actual, reference)


@pytest.mark.parametrize("q_inference", [False, True])
def test_cache_respects_current_inference_mode(q_inference: bool) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    x = torch.randn(2, 3, 8)
    initial_context = torch.inference_mode if q_inference else torch.no_grad
    current_context = torch.no_grad if q_inference else torch.inference_mode
    with initial_context():
        actual.attn.q_proj(x)
    with current_context():
        expected = reference.attn.k_proj(x)
        output = actual.attn.k_proj(x)
    assert output.is_inference() == expected.is_inference()
    torch.testing.assert_close(output, expected)
    assert getattr(actual.attn, "_soup_fast_lora_qkv_cache") is None


def test_inference_tensors_delegate_without_requiring_a_version_counter() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    with torch.inference_mode():
        inputs = torch.randn(2, 3, 8)
        expected = reference(inputs)
        output = actual(inputs)
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got, wanted)
        assert got.is_inference()


@pytest.mark.parametrize("fp32_projection", _PROJECTIONS)
@pytest.mark.parametrize("needs_input_grad", [False, True])
def test_heterogeneous_adapter_pair_dtypes_delegate_to_peft(
    fp32_projection: str, needs_input_grad: bool
) -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model().to(dtype=torch.bfloat16)
    projection = getattr(actual.attn, fp32_projection)
    projection.lora_A["default"].float()
    projection.lora_B["default"].float()
    reference = copy.deepcopy(actual)
    x_ref = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=needs_input_grad)
    x = x_ref.detach().clone().requires_grad_(needs_input_grad)
    expected = reference(x_ref)
    sum(part.float().square().mean() for part in expected).backward()
    assert patch_fast_lora_qkv(actual) == 1
    output = actual(x)
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got, wanted)
        assert type(got.grad_fn).__name__ != "_FastLoraQKVBackward"
    assert getattr(actual.attn, "_soup_fast_lora_qkv_last_cache_hits") == (0, ())
    sum(part.float().square().mean() for part in output).backward()
    _assert_gradients_match(actual, reference)
    if needs_input_grad:
        torch.testing.assert_close(x.grad, x_ref.grad)
    else:
        assert x.grad is x_ref.grad is None


def test_q_only_loss_does_not_weight_decay_unused_kv_adapters() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv

    actual = _make_model()
    reference = copy.deepcopy(actual)
    assert patch_fast_lora_qkv(actual) == 1
    inputs = torch.randn(2, 3, 8)
    initial = {name: parameter.detach().clone() for name, parameter in actual.named_parameters()}
    for model in (actual, reference):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.3)
        model.attn.q_proj(inputs).square().mean().backward()
        optimizer.step()
    for name, parameter in actual.named_parameters():
        if "lora_" in name and ("k_proj" in name or "v_proj" in name):
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0, msg=name)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], msg=name)
