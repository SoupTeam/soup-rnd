"""Compare the saved values of repeated benchmark runs."""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_EPS = 1e-30
_LORA = re.compile(r"(?:.*\.)?layers\.(\d+)\.self_attn\.(\w+)\.lora_([AB])\.default\.weight")
ATTENTION_PROJECTION = re.compile(r"(?:.*\.)?layers\.(\d+)\.self_attn\.([^.]+)")
MOONLIGHT_LAYERS = 27
MOONLIGHT_TARGETS = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
SDPA_GRADS = ("dq", "dk", "dv")

SDPA_BACKENDS = {
    "math": "MATH",
    "flash": "FLASH_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "cudnn": "CUDNN_ATTENTION",
}


def projection_error(names: Iterable[str], targets: Sequence[str] | None = None) -> str | None:
    """Check that the capture has one module per expected layer and target."""
    names = list(names)
    coordinates = {
        (int(match[1]), match[2])
        for name in names
        if (match := ATTENTION_PROJECTION.fullmatch(name))
    }
    projections = set(targets) if targets is not None else {name for _, name in coordinates}
    expected = {(layer, name) for layer in range(MOONLIGHT_LAYERS) for name in projections}
    count = MOONLIGHT_LAYERS * len(MOONLIGHT_TARGETS)
    if len(names) != count or len(coordinates) != count or coordinates != expected:
        return f"expected {count} projection gradients covering every layer and target"
    return None


def sdpa_capture_error(calls: Any, path: Path) -> str | None:
    """Check the SDPA call records and the full tensors saved for call zero."""
    if not isinstance(calls, list) or not calls or len(calls) % MOONLIGHT_LAYERS:
        return f"expected a positive multiple of {MOONLIGHT_LAYERS} grad-enabled SDPA calls"
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or call.get("call") != index:
            return f"missing SDPA call {index}"
        shape = call.get("q_shape")
        if not isinstance(shape, list) or not shape:
            return f"SDPA call {index} is missing q_shape"
        for grad in SDPA_GRADS:
            digest = call.get(f"sha_{grad}")
            if not isinstance(digest, str) or not digest:
                return f"SDPA call {index} is missing sha_{grad}"
    for grad in SDPA_GRADS:
        name = f"sdpa_call0_{grad}.pt"
        if not (path / name).is_file():
            return f"missing {name}"
    return None


def tensor_byte_view(tensor: Any) -> Any:
    """Return the native bytes of the tensor in contiguous logical order."""
    import torch

    # A size-one contiguous tensor can still have a non-unit stride.
    return tensor.detach().contiguous().as_strided((tensor.numel(),), (1,)).view(torch.uint8)


def tensor_pair_diff(a: Any, b: Any) -> dict[str, bool | int | float]:
    """Compare values and bytes, with numeric metrics relative to the first tensor."""
    import torch

    if a.shape != b.shape:
        raise ValueError(f"tensor shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    numel = a.numel()
    mismatch = a != b
    reference = a.detach().double().abs()
    delta = (a.detach().double() - b.detach().double()).masked_fill(~mismatch, 0)
    max_abs = reference.max().item() if numel else 0.0
    max_abs_diff = delta.abs().max().item() if numel else 0.0
    sum_diff = delta.sum().item()
    return {
        "equal": torch.equal(a, b),
        "bitwise_equal": (
            a.dtype == b.dtype and torch.equal(tensor_byte_view(a), tensor_byte_view(b))
        ),
        "n_diff": mismatch.count_nonzero().item(),
        "numel": numel,
        "max_abs_diff": max_abs_diff,
        "max_abs": max_abs,
        "rel_max_diff": max_abs_diff / max(max_abs, _EPS),
        "sum_diff": sum_diff,
        "rel_sum_diff": abs(sum_diff) / max(reference.sum().item(), _EPS),
    }


def grad_breakdown(names: Iterable[str], differs: dict[str, bool]) -> dict[str, Any]:
    """Count parsed LoRA differences and find the highest differing layer."""
    per_layer: dict[int, dict[str, int]] = {}
    per_projection: dict[str, dict[str, int]] = {}
    per_ab = {ab: {"differing": 0, "total": 0} for ab in ("A", "B")}
    totals = {"differing": 0, "total": 0}
    unparsed = {"differing": 0, "total": 0}
    first_layer = None
    first_names: list[str] = []
    for name in sorted(set(names)):
        different = differs.get(name, False)
        match = _LORA.fullmatch(name)
        if match is None:
            unparsed["differing"] += different
            unparsed["total"] += 1
            continue
        layer = int(match[1])
        groups = (
            totals,
            per_layer.setdefault(layer, {"differing": 0, "total": 0}),
            per_projection.setdefault(match[2], {"differing": 0, "total": 0}),
            per_ab[match[3]],
        )
        for counts in groups:
            counts["differing"] += different
            counts["total"] += 1
        if different:
            if first_layer is None or layer > first_layer:
                first_layer, first_names = layer, []
            if layer == first_layer:
                first_names.append(name)
    return {
        "totals": totals,
        "per_layer": dict(sorted(per_layer.items())),
        "per_projection": per_projection,
        "per_ab": per_ab,
        "unparsed": unparsed,
        "first_differing_layer_in_backward_order": first_layer,
        "first_differing_names": first_names,
    }


def window_label(k: int, seq: int) -> str:
    """Format the window as recorded in benchmark rows."""
    return f"k={k},S={seq}"


def row_record(
    *,
    model: str,
    model_revision: str,
    card: str,
    window: str,
    arm: str,
    mode: str,
    torch_version: str,
    attention_backend: str,
    tensor_file: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build a canonical row without letting extra fields replace metadata."""
    row = {
        "model": model,
        "model_revision": model_revision,
        "card": card,
        "window": window,
        "arm": arm,
        "mode": mode,
        "torch_version": torch_version,
        "attention_backend": attention_backend,
        "tensor_file": tensor_file,
    }
    collisions = row.keys() & extra.keys()
    if collisions:
        raise ValueError(f"extra fields collide with row metadata: {', '.join(sorted(collisions))}")
    return {**row, **extra}
