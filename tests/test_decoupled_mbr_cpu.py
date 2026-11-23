"""Ticket 06, part 1: decoupled MBR equals A0 on a tiny MoE, CPU only.

Seam: ``L2LStep.run`` over the per-expert streamed DeepSeek-V3 model of
``tests/test_b13_l2l_moe.py``. A0 is plain L2L: attention, router, shared experts and routed
experts once per micro-batch, every expert call with the native GEMM of its own rows. Decoupled
MBR (``mbr_decoupled_subwindow=True``) joins the window's routed experts into one visit and
recomputes attention, router and shared experts one micro-batch at a time, in micro-batch order.

Criterion: ``torch.equal`` on the per-micro-batch losses and on every LoRA gradient at k=1, 2
and 4 with NF4 storage and BF16 weights. The dense FP32 rows keep the known ~9e-8 divergence of
PR #77: the test records what it measured and only holds it inside that envelope.

The M2+ rows pin the zero-materialisation expert backward on its own: ``backward_from_inputs``
against the per-micro-batch ``WeightedFrozenExpert`` autograd, and the segment row counts it hands
to the exact-GEMM adapter (the micro-batch calls A0 makes).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from rich.console import Console

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import Rig, _collect, _zero_grads
from tests.test_b13_moe_runtime import build_moe_streamed, streamed_experts_modules
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

#: The dense FP32 envelope recorded from PR #77 for k >= 2, and the row that holds it.
KNOWN_DENSE_FP32_GAP = 1e-6


def build_rig(
    tmp_path: Path,
    name: str,
    *,
    k: int,
    storage: str = "nf4",
    dtype: str = "bfloat16",
    rows: int = 2,
    seq: int = 6,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
) -> Rig:
    """One streamed tiny MoE with the flags route-preserving MBR validates."""
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    config = fixture.moe_config()
    root = tmp_path / name
    root.mkdir()
    weights = fixture.moe_weights_dir(root, config)
    moe = build_moe_streamed(
        weights,
        root / "shards",
        storage=storage,
        dtype=dtype,
        layout="per_expert",
        lora_config=fixture.moe_lora_config(
            dropout=dropout, targets=fixture.HEAD_SHARED_TARGETS
        ),
        two_read_experts=True,
        grouped_expert_routes=True,
    )
    moe.model.train()
    if attention_dropout:
        for module in moe.model.modules():
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = attention_dropout
            module_config = getattr(module, "config", None)
            if module_config is not None:
                module_config._attn_implementation = "eager"
    first = fixture.moe_batch(config, seed=0, rows=rows, seq=seq)["input_ids"]
    stepper = l2l_step.L2LStep(
        moe.model, moe.runtime, l2l_step.capture_layer_call(moe.model, first)
    )
    store = ActivationStore(
        n_layers=config.num_hidden_layers,
        k=k,
        chunk_shape=(rows, seq, config.hidden_size),
        dtype=getattr(torch, moe.dtype),
        device="cpu",
        pin=False,
    )
    return Rig(config, moe, None, stepper, store, "cpu", rows, seq)


def window(rig: Rig, k: int) -> List[Any]:
    return [
        fixture.moe_batch(rig.config, seed=100 + mb, rows=rig.rows, seq=rig.seq)["input_ids"]
        for mb in range(k)
    ]


def run_window(
    rig: Rig,
    ids: List[Any],
    *,
    decoupled: bool,
    group_size: int | None = None,
    recompute_subwindow: int | None = None,
) -> Dict[str, Any]:
    """One window through ``L2LStep.run``: A0, or decoupled MBR over the whole window."""
    import torch

    _zero_grads(rig.model)
    torch.manual_seed(42)
    if decoupled:
        losses = rig.stepper.run(
            ids,
            rig.store,
            route_preserving_mbr=True,
            mbr_group_size=len(ids) if group_size is None else group_size,
            mbr_decoupled_subwindow=True,
            mbr_recompute_subwindow=recompute_subwindow,
            collect_phase_stats=True,
        )
    else:
        losses = rig.stepper.run(ids, rig.store)
    return {
        "losses": [loss.detach().clone() for loss in losses],
        "grads": _collect(rig.model),
    }


def differing(a0: Dict[str, Any], cand: Dict[str, Any]) -> Tuple[List[str], float]:
    """Names of the losses and gradients that are not ``torch.equal``, and the largest gap."""
    import torch

    assert len(a0["losses"]) == len(cand["losses"])
    assert a0["grads"].keys() == cand["grads"].keys() and a0["grads"]
    bad: List[str] = []
    gap = 0.0
    for i, (left, right) in enumerate(zip(a0["losses"], cand["losses"], strict=True)):
        if not torch.equal(left, right):
            bad.append(f"loss[{i}]")
            gap = max(gap, float((left.float() - right.float()).abs().max()))
    for name in sorted(a0["grads"]):
        left, right = a0["grads"][name], cand["grads"][name]
        if left is None or right is None:
            if left is not right:
                bad.append(name)
                gap = float("inf")
            continue
        if not torch.equal(left, right):
            bad.append(name)
            gap = max(gap, float((left.float() - right.float()).abs().max()))
    return bad, gap


def record(row: str, a0: Dict[str, Any], cand: Dict[str, Any]) -> Tuple[List[str], float]:
    """Print one outcome row (``rich``, like the other bit tests) and return it."""
    bad, gap = differing(a0, cand)
    Console(stderr=True).print(
        f"DECOUPLED_MBR {row}: tensors={len(a0['losses']) + len(a0['grads'])} "
        f"different={len(bad)} max_abs={gap:.3e}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )
    return bad, gap


@pytest.mark.parametrize("k", [1, 2, 4])
def test_decoupled_mbr_equals_a0_bitwise_nf4_bf16(tmp_path: Path, k: int) -> None:
    """The ticket's hard row: losses and every LoRA gradient, ``torch.equal``."""
    rig = build_rig(tmp_path, f"nf4_bf16_k{k}", k=k)
    try:
        ids = window(rig, k)
        a0 = run_window(rig, ids, decoupled=False)
        cand = run_window(rig, ids, decoupled=True)
        bad, gap = record(f"NF4+BF16 k={k}", a0, cand)
        assert not bad, f"{len(bad)} tensors differ at k={k} (max_abs={gap:.3e}): {bad[:5]}"
        stats = cand_stats(rig)
        layers = [
            row
            for row in stats["layers"].values()
            if row.get("init_forward", {}).get("unique_expert_count")
        ]
        assert layers, "no MoE layer ran"
        for layer in layers:
            assert layer["recompute_forward"]["expert_fetches"] == 0
            assert layer["backward_dX"]["expert_fetches"] > 0
            assert layer["route_id_mismatch_slots"] == 0
    finally:
        rig.close()


def cand_stats(rig: Rig) -> Dict[str, Any]:
    stats = rig.stepper.route_preserving_mbr_stats
    assert stats is not None and stats["layers"]
    return stats


def test_decoupled_mbr_equals_a0_with_smaller_groups_and_subwindows(tmp_path: Path) -> None:
    """Groups of 2 and a two-micro-batch recompute sub-window keep A0's bits too."""
    rig = build_rig(tmp_path, "groups", k=4)
    try:
        ids = window(rig, 4)
        a0 = run_window(rig, ids, decoupled=False)
        cand = run_window(rig, ids, decoupled=True, group_size=2, recompute_subwindow=2)
        bad, gap = record("NF4+BF16 k=4 groups=2 sub_w=2", a0, cand)
        assert not bad, f"{len(bad)} tensors differ (max_abs={gap:.3e}): {bad[:5]}"
        stats = cand_stats(rig)
        assert stats["group_size"] == 2
        assert all(
            row["init_forward"]["grouped_expert_dispatch_calls"] == 2
            for row in stats["layers"].values()
            if "init_forward" in row and row["init_forward"].get("unique_expert_count")
        )
    finally:
        rig.close()


def test_decoupled_mbr_dropout_rng_replay_bit_exact(tmp_path: Path) -> None:
    """The no-grad R1 pass and the R2 recompute draw exactly one dropout mask per call."""
    import torch

    rig = build_rig(
        tmp_path, "dropout", k=4, storage="dense", dtype="float32",
        dropout=0.15, attention_dropout=0.1,
    )
    try:
        ids = window(rig, 4)
        torch.manual_seed(99)
        a0 = run_window(rig, ids, decoupled=False)
        rng_a0 = torch.get_rng_state()
        torch.manual_seed(99)
        cand = run_window(rig, ids, decoupled=True)
        assert torch.equal(rng_a0, torch.get_rng_state()), "the RNG stream advanced differently"
        bad, gap = record("dense FP32 dropout k=4", a0, cand)
        assert not [name for name in bad if name.startswith("loss")], bad
        assert gap <= KNOWN_DENSE_FP32_GAP, f"max_abs={gap:.3e}"
    finally:
        rig.close()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_decoupled_mbr_dense_fp32_records_the_known_gap(tmp_path: Path, k: int) -> None:
    """Dense FP32: k=1 is exact; k>=2 records PR #77's ~1e-7 gap and holds the envelope.

    The losses are exact here too: the gap is the R2 multi-root backward adding the routed and
    the non-routed contribution at the residual in another order than one engine call. BF16
    rounds it away (the NF4+BF16 rows are exact), FP32 keeps it.
    """
    rig = build_rig(tmp_path, f"dense_fp32_k{k}", k=k, storage="dense", dtype="float32")
    try:
        ids = window(rig, k)
        a0 = run_window(rig, ids, decoupled=False)
        cand = run_window(rig, ids, decoupled=True)
        bad, gap = record(f"dense FP32 k={k}", a0, cand)
        if k == 1:
            assert not bad, f"k=1 must be exact, {bad[:5]} differ (max_abs={gap:.3e})"
        else:
            assert gap <= KNOWN_DENSE_FP32_GAP, (
                f"the known dense FP32 gap grew to {gap:.3e} at k={k}; "
                f"{len(bad)} tensors differ"
            )
    finally:
        rig.close()


def test_decoupled_mbr_validation_and_dropped_knobs(tmp_path: Path) -> None:
    """Fail closed on flags that do not reach the decoupled path."""
    rig = build_rig(tmp_path, "guards", k=2)
    try:
        ids = window(rig, 2)
        with pytest.raises(ValueError, match="mbr_decoupled_subwindow"):
            rig.stepper.run(ids, rig.store, mbr_decoupled_subwindow=True)
        with pytest.raises(ValueError, match="must be a bool"):
            rig.stepper.run(
                ids, rig.store, route_preserving_mbr=True, mbr_decoupled_subwindow=1
            )
        with pytest.raises(ValueError, match="mbr_recompute_subwindow"):
            rig.stepper.run(ids, rig.store, route_preserving_mbr=True, mbr_recompute_subwindow=2)
        with pytest.raises(ValueError, match="mbr_recompute_subwindow"):
            rig.stepper.run(
                ids,
                rig.store,
                route_preserving_mbr=True,
                mbr_decoupled_subwindow=True,
                mbr_recompute_subwindow=0,
            )

        @l2l_step.mbr_deferred_forward_hook
        def hook(_module: Any, _args: Any, output: Any) -> Any:
            return output

        handle = rig.stepper.layers[1].register_forward_hook(hook)
        try:
            with pytest.raises(ValueError, match="deferred decoder forward hooks"):
                rig.stepper.run(
                    ids,
                    rig.store,
                    route_preserving_mbr=True,
                    mbr_decoupled_subwindow=True,
                )
        finally:
            handle.remove()
    finally:
        rig.close()


def test_decoupled_mbr_refuses_a_patched_expert_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fused expert node (lt-fused) cannot keep its bits here, so it is refused."""
    from soup_cli.utils import direct_expert_backward, expert_stream

    rig = build_rig(tmp_path, "patched", k=2)
    try:
        ids = window(rig, 2)
        node = direct_expert_backward.weighted_expert_function(
            expert_stream.streamed_experts_class().forward
        )

        def fused_forward(*_args: Any, **_kwargs: Any) -> Any:  # a fused node's forward
            raise AssertionError("the fused forward must not run")

        fused_forward.__module__ = "soup_cli.utils.direct_expert_backward"
        monkeypatch.setattr(node, "forward", staticmethod(fused_forward))
        with pytest.raises(ValueError, match="patched expert node"):
            rig.stepper.run(
                ids,
                rig.store,
                route_preserving_mbr=True,
                mbr_decoupled_subwindow=True,
            )
    finally:
        rig.close()


def test_decoupled_mbr_refuses_a_patched_expert_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct_mm backward cannot keep its bits here, so it is refused, not ignored."""
    from soup_cli.utils import direct_expert_backward, expert_stream

    rig = build_rig(tmp_path, "patched_backward", k=2)
    try:
        ids = window(rig, 2)
        node = direct_expert_backward.weighted_expert_function(
            expert_stream.streamed_experts_class().forward
        )

        def direct_backward(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the direct backward must not run")

        direct_backward.__module__ = "soup_cli.utils.direct_expert_backward"
        monkeypatch.setattr(node, "backward", staticmethod(direct_backward))
        with pytest.raises(ValueError, match="patched expert backward"):
            rig.stepper.run(
                ids,
                rig.store,
                route_preserving_mbr=True,
                mbr_decoupled_subwindow=True,
            )
    finally:
        rig.close()


def _expert_inputs(experts: Any, k: int, rows: int, width: int) -> Tuple[List[Any], List[Any]]:
    import torch

    from soup_cli.utils.layer_stream_runtime import _torch_dtype

    dtype = _torch_dtype(str(experts.pool.dtype))
    torch.manual_seed(19)
    hiddens = [torch.randn(rows, width, dtype=dtype) for _ in range(k)]
    indices = []
    for _ in range(k):
        logits = torch.randn(rows, experts.num_experts)
        indices.append(torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).indices)
    return hiddens, indices


def test_streaming_expert_backward_equals_per_microbatch_autograd(tmp_path: Path) -> None:
    """M2+: one streaming expert backward over the window against one node per micro-batch."""
    import torch

    rig = build_rig(tmp_path, "m2", k=2)
    try:
        experts = next(iter(streamed_experts_modules(rig.model).values()))
        k, rows, width = 3, 4, experts.pool._shapes["gate_up"][1]
        hiddens, indices = _expert_inputs(experts, k, rows, width)
        weights = [
            torch.rand(idx.shape, dtype=torch.float32) for idx in indices
        ]
        dy = [torch.randn(rows, width, dtype=hidden.dtype) for hidden in hiddens]

        ref_h = [h.clone().requires_grad_(True) for h in hiddens]
        ref_w = [w.clone().requires_grad_(True) for w in weights]
        outputs = [experts(h, idx, w) for h, idx, w in zip(ref_h, indices, ref_w, strict=True)]
        torch.autograd.backward(outputs, grad_tensors=dy)

        got_h, got_w = experts.backward_from_inputs(
            torch.cat(hiddens),
            torch.cat(indices),
            torch.cat(weights),
            torch.cat(dy),
            autocast_cfg={"enabled": False, "dtype": None, "cache_enabled": True},
            segment_counts=[rows] * k,
        )
        want_h = torch.cat([h.grad for h in ref_h])
        want_w = torch.cat([w.grad for w in ref_w])
        assert torch.equal(got_h, want_h), (got_h.float() - want_h.float()).abs().max()
        assert torch.equal(got_w, want_w), (got_w.float() - want_w.float()).abs().max()

        # The adapter's segments are the calls A0 makes: one per (expert, micro-batch) with that
        # micro-batch's row count, not one joined call over the window. Four matmuls per expert
        # (gate_up and down in the recompute, then the two input-gradient matmuls).
        seen: List[Tuple[int, ...]] = []

        def spy(x: Any, w: Any, transposed: bool, counts: Any) -> Any:
            seen.append(tuple(int(c) for c in counts))
            op = (lambda p: torch.mm(p, w.t())) if transposed else (lambda p: torch.mm(p, w))
            return torch.cat([op(p) for p in x.split(list(counts))])

        spy_h, spy_w = experts.backward_from_inputs(
            torch.cat(hiddens),
            torch.cat(indices),
            torch.cat(weights),
            torch.cat(dy),
            autocast_cfg={"enabled": False, "dtype": None, "cache_enabled": True},
            segment_counts=[rows] * k,
            mm=spy,
        )
        assert torch.equal(spy_h, want_h) and torch.equal(spy_w, want_w)
        want_calls = Counter()
        for expert in range(experts.num_experts):
            per_mb = [int((idx == expert).sum()) for idx in indices]
            rows_of = tuple(count for count in per_mb if count)  # micro-batch order, no zeros
            if rows_of:
                want_calls[rows_of] += 4
        assert Counter(seen) == want_calls, f"{Counter(seen)} != {want_calls}"
    finally:
        rig.close()
