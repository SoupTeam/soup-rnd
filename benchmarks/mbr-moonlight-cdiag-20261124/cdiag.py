"""Boundary comparison of the ticket 14 C-against-A0 diagnosis.

The capture (``capture_boundaries.py``) writes two kinds of records:

- module boundaries, keyed by ``(layer, role)`` and matched by the digest of the module input, so
  the micro-batch order of the two arms does not matter (C builds its group from the last
  micro-batch to the first): the forward output of both layer norms, the attention, the router
  gate and the shared expert;
- expert linear calls (``expert_mlp``'s ``gate_up``/``down`` projections in forward and backward)
  and routed-expert node calls, matched row by row through the digest of the input row, because
  A0 and C call the same expert with different row counts.

Pure logic: torch is imported inside the functions, nothing here touches a card.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

MOONLIGHT_LAYERS = 27
MODULE_ROLES = (
    "input_layernorm", "self_attn", "post_attention_layernorm", "gate", "shared_experts",
)
EXPERT_PROJECTIONS = {"gate_up": (2816, 2048), "down": (2048, 1408)}
EXPERT_PHASES = ("forward", "backward")
_EPS = 1e-30


def tensor_byte_view(tensor: Any) -> Any:
    """Native bytes of a tensor in contiguous logical order."""
    import torch

    return tensor.detach().contiguous().as_strided((tensor.numel(),), (1,)).view(torch.uint8)


def digest(tensor: Any) -> str:
    """sha256 over the dtype, shape and bytes of one tensor."""
    hasher = hashlib.sha256()
    hasher.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    hasher.update(tensor_byte_view(tensor).cpu().numpy().tobytes())
    return hasher.hexdigest()


def tensor_pair_diff(first: Any, second: Any) -> dict[str, Any]:
    """Bit and value difference of two tensors; same metrics as the ticket 12 comparison."""
    if first.shape != second.shape:
        raise ValueError(f"shapes differ: {tuple(first.shape)} vs {tuple(second.shape)}")
    mismatch = first != second
    reference = first.detach().double().abs()
    delta = (first.detach().double() - second.detach().double()).masked_fill(~mismatch, 0)
    denominator = max(reference.max().item(), _EPS)
    return {
        "bitwise_equal": bitwise_equal(first, second),
        "n_diff": int(mismatch.count_nonzero().item()),
        "numel": int(first.numel()),
        "max_abs_diff": delta.abs().max().item() if first.numel() else 0.0,
        "max_abs": reference.max().item() if first.numel() else 0.0,
        "rel_max_diff": (delta.abs().max().item() if first.numel() else 0.0) / denominator,
    }


def bitwise_equal(first: Any, second: Any) -> bool:
    """Same dtype, same bytes: NaNs compare equal here, unlike ``torch.equal``."""
    import torch

    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and torch.equal(tensor_byte_view(first), tensor_byte_view(second))
    )


def row_digests(tensor: Any) -> list[str]:
    """One digest per row of a 2-D tensor; a 1-D tensor is one row."""
    if tensor.dim() == 1:
        return [digest(tensor)]
    if tensor.dim() != 2:
        raise ValueError(f"row digests need a 1-D or 2-D tensor, got {tuple(tensor.shape)}")
    return [digest(tensor[index]) for index in range(tensor.shape[0])]


def module_record(
    *, layer: int, role: str, mb: int, inputs: Any, outputs: Any
) -> dict[str, Any]:
    """One module boundary: the digests of its input and output, and the call's micro-batch."""
    if role not in MODULE_ROLES:
        raise ValueError(f"unknown module role {role!r}")
    return {
        "kind": "module", "layer": int(layer), "role": role, "mb": int(mb),
        "shape": list(outputs.shape), "dtype": str(outputs.dtype),
        "numel": int(outputs.numel()), "in_sha": digest(inputs), "sha": digest(outputs),
    }


def expert_linear_record(
    *, layer: int, expert: int, which: str, phase: str, call: int, inputs: Any, outputs: Any,
) -> dict[str, Any]:
    """One expert projection call with the digests of its input and output rows."""
    if which not in EXPERT_PROJECTIONS:
        raise ValueError(f"unknown expert projection {which!r}")
    if phase not in EXPERT_PHASES:
        raise ValueError(f"unknown expert phase {phase!r}")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("input and output row counts differ")
    return {
        "kind": "expert_linear", "layer": int(layer), "expert": int(expert), "which": which,
        "phase": phase, "call": int(call), "rows": int(inputs.shape[0]),
        "in_rows": row_digests(inputs), "out_rows": row_digests(outputs),
        "in_sha": digest(inputs), "out_sha": digest(outputs),
    }


def node_record(*, layer: int, call: int, inputs: Any, outputs: Any) -> dict[str, Any]:
    """One routed-expert node call: the joined input rows and the weighted output rows."""
    return {
        "kind": "expert_node", "layer": int(layer), "expert": -1, "which": "node",
        "phase": "forward", "call": int(call), "rows": int(inputs.shape[0]),
        "in_rows": row_digests(inputs), "out_rows": row_digests(outputs),
        "in_sha": digest(inputs), "out_sha": digest(outputs),
    }


def _row_index(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        for in_sha, out_sha in zip(record["in_rows"], record["out_rows"]):
            key = (record["layer"], record["expert"], record["which"], record["phase"], in_sha)
            index.setdefault(key, []).append({
                "out_sha": out_sha, "rows": record["rows"], "call": record["call"], "used": False,
            })
    return index


def match_expert_rows(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Match the rows of ``right`` (C) to the rows of ``left`` (A0) by the input digest.

    A matched row had the same input to the projection; its output digest says whether the
    projection returned the same bits. The first forward difference is the one in the lowest
    layer, then the lowest call."""
    index = _row_index(left)
    rows_matched = 0
    rows_missing: list[dict[str, Any]] = []
    per_call: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in sorted(right, key=lambda item: item["call"]):
        key_call = (record["layer"], record["expert"], record["which"], record["phase"],
                    record["call"])
        entry = per_call.setdefault(key_call, {
            "layer": record["layer"], "expert": record["expert"], "which": record["which"],
            "phase": record["phase"], "call": record["call"], "b_rows": record["rows"],
            "a_rows": [], "matched_rows": 0, "diff_rows": 0,
        })
        for in_sha, out_sha in zip(record["in_rows"], record["out_rows"]):
            key = (record["layer"], record["expert"], record["which"], record["phase"], in_sha)
            found = next(
                (item for item in index.get(key, []) if not item["used"]), None
            )
            if found is None:
                rows_missing.append({
                    "layer": record["layer"], "expert": record["expert"],
                    "which": record["which"], "phase": record["phase"],
                    "call": record["call"], "in_sha": in_sha,
                })
                continue
            found["used"] = True
            rows_matched += 1
            entry["matched_rows"] += 1
            entry["a_rows"].append(found["rows"])
            if found["out_sha"] != out_sha:
                entry["diff_rows"] += 1
    differing = [entry for entry in per_call.values() if entry["diff_rows"]]
    forward = [entry for entry in differing if entry["phase"] == "forward"]
    first_forward = (
        min(forward, key=lambda item: (item["layer"], item["call"])) if forward else None
    )
    return {
        "calls": len(per_call),
        "rows": sum(entry["b_rows"] for entry in per_call.values()),
        "rows_matched": rows_matched, "rows_missing": rows_missing,
        "differing": differing, "first_forward": first_forward,
    }


def match_module_boundaries(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Match module boundaries by ``(layer, role, input digest)``; compare the output digest.

    The input digest is unique per micro-batch, so the pairs are correct even when the two arms
    visit the micro-batches in opposite orders; a repeated input is matched one for one."""
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in left:
        key = (record["layer"], record["role"], record["in_sha"])
        index.setdefault(key, []).append({"record": record, "used": False})
    mismatched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for record in sorted(right, key=lambda item: item["mb"]):
        key = (record["layer"], record["role"], record["in_sha"])
        found = next((item for item in index.get(key, []) if not item["used"]), None)
        if found is None:
            missing.append({
                "layer": record["layer"], "role": record["role"], "mb": record["mb"],
                "in_sha": record["in_sha"],
            })
            continue
        found["used"] = True
        if found["record"]["sha"] != record["sha"]:
            mismatched.append({
                "layer": record["layer"], "role": record["role"], "mb": record["mb"],
                "left_sha": found["record"]["sha"], "right_sha": record["sha"],
            })
    return {
        "equal": not mismatched and not missing,
        "mismatched": mismatched, "missing": missing,
        "left_count": len(left), "right_count": len(right),
    }


def boundary_verdict(
    modules: dict[str, Any], experts: dict[str, Any], nodes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The first divergent boundary in forward order: lowest layer, modules before expert calls.

    Inside one layer the attention, both norms, the router and the shared expert run first; the
    routed expert node and its linears come after them. A layer's module mismatch is therefore
    reported before that layer's expert call, and any layer's expert call before every module of
    the next layer."""
    layers = {
        *[entry["layer"] for entry in modules["mismatched"]],
        *[entry["layer"] for entry in experts["differing"] if entry["phase"] == "forward"],
        *[entry["layer"] for entry in experts["rows_missing"]],
        *[entry["layer"] for entry in (nodes or {}).get("differing", [])
          if entry["phase"] == "forward"],
        *[entry["layer"] for entry in (nodes or {}).get("rows_missing", [])],
    }
    for layer in sorted(layers):
        module = next(
            (entry for entry in modules["mismatched"] if entry["layer"] == layer), None
        )
        if module is not None:
            return {
                "kind": "module", "layer": layer, "role": module["role"], "mb": module["mb"],
            }
        expert = next(
            (entry for entry in experts["differing"]
             if entry["layer"] == layer and entry["phase"] == "forward"), None
        )
        if expert is not None:
            return {"kind": "expert_linear", **expert}
        node = next(
            (entry for entry in (nodes or {}).get("differing", [])
             if entry["layer"] == layer and entry["phase"] == "forward"), None
        )
        if node is not None:
            return {"kind": "expert_node", **node}
        unmatched = next(
            (entry for entry in experts["rows_missing"] if entry["layer"] == layer), None
        )
        if unmatched is not None:
            return {
                "kind": "unmatched_expert_input", "layer": layer,
                "expert": unmatched["expert"], "which": unmatched["which"],
                "phase": unmatched["phase"], "call": unmatched["call"],
            }
    experts_equal = not experts["differing"] and not experts["rows_missing"]
    nodes_equal = nodes is None or not nodes["differing"] and not nodes["rows_missing"]
    return {
        "kind": "equal", "modules_equal": modules["equal"],
        "experts_equal": experts_equal and nodes_equal,
    }
