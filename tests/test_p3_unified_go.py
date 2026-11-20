"""End-to-end integration tests for the DeepSeek P3 Unified GO stack:
M1 (decoupled sub-window) + M2+ (zero-materialization streaming backward) +
H2 (ABA-safe window NF4 dequant) + M3 (LargeLayerBufferPool release) +
Track 2 (C++17 PackedStreamAsync QD=4, 2-slot async, cross-layer backward pre-staging) +
Track 3 (Chunked CE head).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _has_native_async_ext() -> bool:
    try:
        import _native_expert_stream_async  # noqa: F401

        return True
    except ImportError:
        return False


def _build_nf4_fixture(tmp_path: Path, *, hidden_size: int = 128, dtype: str = "bfloat16"):
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    cfg = moe_config(
        vocab_size=256,
        tie_word_embeddings=False,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        moe_intermediate_size=hidden_size // 2,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=hidden_size // 2,
        kv_lora_rank=hidden_size // 4,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    weights = moe_weights_dir(tmp_path, cfg)
    shards = tmp_path / "shards"
    skeleton = build_meta_skeleton(str(weights), dtype=dtype, quant="nf4", double_quant=False)
    layout = shard_layout_of(skeleton)
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    index = shard_checkpoint(
        str(weights),
        str(shards),
        dtype=dtype,
        quant="nf4",
        double_quant=False,
        quant_device="cpu",
        arch="deepseek_v3",
        expert_layout="per_expert",
        layout=layout,
        quant_suffixes=suffixes,
    )
    return weights, shards, index


@pytest.mark.skipif(not _has_native_async_ext(), reason="_native_expert_stream_async not built")
def test_unified_go_stack_bitwise_parity_two_adamw_steps(tmp_path: Path):
    """Full 4-pillar GO stack matches plain Two-Read L2L across 2 AdamW steps."""
    import torch

    from benchmarks.harness.b13_moe_fixture import moe_lora_config, snapshot
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    weights, shards, index = _build_nf4_fixture(tmp_path, hidden_size=128, dtype="bfloat16")
    k, seq_len = 4, 24
    gen = torch.Generator().manual_seed(2026)
    step_batches = [
        [torch.randint(0, 64, (1, seq_len), generator=gen) for _ in range(k)],
        [torch.randint(0, 64, (1, seq_len), generator=gen) for _ in range(k)],
    ]

    def _run_arm(*, unified: bool, head_impl: str):
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=moe_lora_config(),
            device="cpu",
            dtype="bfloat16",
            quant="nf4",
            double_quant=False,
            tier="disk",
            pin=False,
            seed=11,
            two_read_experts=True,
            grouped_expert_routes=True,
        )
        controller = None
        if unified:
            cpu_packed_control(runtime.expert_pool, runtime.expert_readers[0], 0)
            controller = AsyncExpertController(
                runtime,
                mode="cpp_async_2slot",
                io_workers=4,
                ring_slots=8,
                cross_layer_prefetch=True,
            )
        store = ActivationStore(
            n_layers=3,
            k=k,
            chunk_shape=(1, seq_len, 128),
            dtype=torch.bfloat16,
            device="cpu",
            pin=False,
        )
        try:
            model.train()
            call = capture_layer_call(model, step_batches[0][0], draws=False)
            stepper = L2LStep(model, runtime, call)
            trainables = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainables, lr=1e-3)
            out_losses = []
            out_grads = []
            for batches in step_batches:
                opt.zero_grad(set_to_none=True)
                n_items = label_tokens(batches)
                head = l2l_sft_head(batches, n_items, impl=head_impl, chunk=8)
                losses = stepper.run(
                    batches,
                    store,
                    head=head,
                    route_preserving_mbr=unified,
                    mbr_group_size=k if unified else None,
                    mbr_decoupled_subwindow=unified,
                    mbr_recompute_subwindow=1 if unified else None,
                    mbr_window_dequant=unified,
                    release_large_pool_during_layers=unified,
                    expert_controller=controller,
                    collect_phase_stats=True,
                )
                out_losses.append([loss.detach().clone() for loss in losses])
                out_grads.append(snapshot(model, gradients=True))
                opt.step()
            adapter = snapshot(model, gradients=False)
            telemetry = controller.native_telemetry() if controller is not None else None
            return (
                out_losses,
                out_grads,
                adapter,
                stepper.expert_phase_stats,
                stepper.window_dequant_stats,
                stepper.large_pool_release_stats,
                telemetry,
            )
        finally:
            if controller is not None:
                controller.close()
            store.close()
            runtime.close()

    ref_losses, ref_grads, ref_adapter, _, _, _, _ = _run_arm(unified=False, head_impl="full")
    (
        uni_full_losses,
        uni_full_grads,
        uni_full_adapter,
        phase_stats,
        dequant_stats,
        pool_stats,
        native_stats,
    ) = _run_arm(unified=True, head_impl="full")
    (
        uni_chunk_losses,
        uni_chunk_grads,
        uni_chunk_adapter,
        chunk_phase_stats,
        chunk_dequant_stats,
        chunk_pool_stats,
        chunk_native_stats,
    ) = _run_arm(unified=True, head_impl="chunked")

    # 1. Unified GO (full head) must be BIT-EXACT against plain Two-Read L2L across both steps.
    for step_idx in range(2):
        for mb in range(k):
            assert torch.equal(uni_full_losses[step_idx][mb], ref_losses[step_idx][mb])
        for name, ref_g in ref_grads[step_idx].items():
            assert torch.equal(uni_full_grads[step_idx][name], ref_g), f"drift on {name}"
    for name, ref_w in ref_adapter.items():
        assert torch.equal(uni_full_adapter[name], ref_w), f"adapter drift on {name}"

    # 2. Unified GO + Chunked CE must match plain L2L tightly (exact BF16 gradients, ~1e-6 loss).
    for step_idx in range(2):
        for mb in range(k):
            torch.testing.assert_close(
                uni_chunk_losses[step_idx][mb], ref_losses[step_idx][mb], atol=1e-5, rtol=1e-5
            )
        for name, ref_g in ref_grads[step_idx].items():
            torch.testing.assert_close(
                uni_chunk_grads[step_idx][name], ref_g, atol=1e-5, rtol=1e-5
            )
    for name, ref_w in ref_adapter.items():
        torch.testing.assert_close(uni_chunk_adapter[name], ref_w, atol=1e-5, rtol=1e-5)

    # 3. Verify all 4 GO pillars actually executed!
    for stats_bundle in (
        (phase_stats, dequant_stats, pool_stats, native_stats),
        (chunk_phase_stats, chunk_dequant_stats, chunk_pool_stats, chunk_native_stats),
    ):
        p_stats, d_stats, lp_stats, n_stats = stats_bundle
        # M1 + M2+: 0 recompute expert reads, symmetric forward == backward expert reads
        for layer_idx in (1, 2):
            row = p_stats["layers"][layer_idx]
            assert row["recompute_forward"]["expert_fetches"] == 0
            assert row["init_forward"]["expert_fetches"] > 0
            assert row["init_forward"]["expert_fetches"] == row["backward_dX"]["expert_fetches"]
        # H2: window NF4 dequant reuse fired
        assert d_stats is not None and d_stats["cache_reuses"] > 0
        # M3: LargeLayerBufferPool released to 0 B during both forward & backward bodies
        assert lp_stats["enabled"] is True
        assert lp_stats["forward_body_allocated_bytes"] == 0
        assert lp_stats["backward_body_allocated_bytes"] == 0
        # Track 2: C++17 PackedStreamAsync cross-layer pre-staging fired between MoE layers 2 -> 1
        assert n_stats is not None
        assert n_stats["cross_layer_prefetched"] > 0
        assert n_stats["cross_layer_promotions"] > 0


@pytest.mark.skipif(not _has_native_async_ext(), reason="_native_expert_stream_async not built")
def test_unified_go_saved_storage_o1_with_chunked_ce_and_async_controller(tmp_path: Path):
    """Unified GO stack keeps saved tensor storage O(1) vs O(k) standard MBR."""
    import torch

    from benchmarks.harness.b13_moe_fixture import moe_lora_config
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head
    from tests.test_p3_decoupled_subwindow import SavedStorageLedger

    weights, shards, index = _build_nf4_fixture(tmp_path, hidden_size=128, dtype="bfloat16")
    k, seq_len = 8, 32
    gen = torch.Generator().manual_seed(77)
    batches = [torch.randint(0, 64, (1, seq_len), generator=gen) for _ in range(k)]

    def _measure(*, unified: bool) -> int:
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=moe_lora_config(),
            device="cpu",
            dtype="bfloat16",
            quant="nf4",
            double_quant=False,
            tier="disk",
            pin=False,
            seed=5,
            two_read_experts=True,
            grouped_expert_routes=True,
        )
        controller = None
        if unified:
            cpu_packed_control(runtime.expert_pool, runtime.expert_readers[0], 0)
            controller = AsyncExpertController(
                runtime,
                mode="cpp_async_2slot",
                io_workers=4,
                ring_slots=8,
                cross_layer_prefetch=True,
            )
        store = ActivationStore(
            n_layers=3,
            k=k,
            chunk_shape=(1, seq_len, 128),
            dtype=torch.bfloat16,
            device="cpu",
            pin=False,
        )
        try:
            model.train()
            call = capture_layer_call(model, batches[0], draws=False)
            stepper = L2LStep(model, runtime, call)
            ledger = SavedStorageLedger()
            n_items = label_tokens(batches)
            head = l2l_sft_head(batches, n_items, impl="chunked" if unified else "full", chunk=8)
            with ledger.track():
                stepper.run(
                    batches,
                    store,
                    head=head,
                    route_preserving_mbr=True,
                    mbr_group_size=k,
                    mbr_decoupled_subwindow=unified,
                    mbr_recompute_subwindow=1 if unified else None,
                    mbr_window_dequant=unified,
                    release_large_pool_during_layers=unified,
                    expert_controller=controller,
                )
            return ledger.peak_bytes
        finally:
            if controller is not None:
                controller.close()
            store.close()
            runtime.close()

    std_peak = _measure(unified=False)
    unified_peak = _measure(unified=True)
    assert unified_peak * 4 < std_peak, f"expected >=4x drop: {unified_peak} vs {std_peak}"


@pytest.mark.skipif(not _has_native_async_ext(), reason="_native_expert_stream_async not built")
def test_unified_go_train_embedding_and_error_recovery(tmp_path: Path):
    """M3 + Chunked CE + AsyncExpertController handles trainable embeds and error unwind."""
    import torch

    from benchmarks.harness.b13_moe_fixture import (
        EMBEDDING_TARGETS,
        moe_lora_config,
        randomize_adapters,
        trainables,
    )
    from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens, model_parts
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    weights, shards, index = _build_nf4_fixture(tmp_path, hidden_size=128, dtype="bfloat16")
    k, seq_len = 4, 24
    gen = torch.Generator().manual_seed(91)
    batches = [torch.randint(0, 64, (1, seq_len), generator=gen) for _ in range(k)]

    model, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=str(shards),
        index=index,
        lora_config=moe_lora_config(targets=EMBEDDING_TARGETS),
        device="cpu",
        dtype="bfloat16",
        quant="nf4",
        double_quant=False,
        tier="disk",
        pin=False,
        seed=3,
        two_read_experts=True,
        grouped_expert_routes=True,
    )
    cpu_packed_control(runtime.expert_pool, runtime.expert_readers[0], 0)
    controller = AsyncExpertController(
        runtime,
        mode="cpp_async_2slot",
        io_workers=4,
        ring_slots=8,
        cross_layer_prefetch=True,
    )
    store = ActivationStore(
        n_layers=3,
        k=k,
        chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16,
        device="cpu",
        pin=False,
    )
    try:
        model.train()
        randomize_adapters(model, seed=17)
        _, owner = model_parts(model)
        call = capture_layer_call(model, batches[0], draws=False)
        stepper = L2LStep(model, runtime, call)

        # 1. Inject a failure in custom head and verify clean state unwind.
        def _boom_head(_causal, _owner, _hidden, _mb):
            raise RuntimeError("injected head failure")

        with pytest.raises(RuntimeError, match="injected head failure"):
            stepper.run(
                batches,
                store,
                head=_boom_head,
                route_preserving_mbr=True,
                mbr_decoupled_subwindow=True,
                mbr_recompute_subwindow=1,
                mbr_window_dequant=True,
                release_large_pool_during_layers=True,
                expert_controller=controller,
            )
        # Controller unbound cleanly and large pool prefetcher restored
        for layer in owner.layers:
            moe = getattr(getattr(layer, "inner", layer).mlp, "experts", None)
            if moe is not None:
                assert getattr(moe, "_expert_controller", None) is None
        assert runtime.prefetcher.tail_prefetch is not None
        assert runtime.prefetcher.backward_tail_prefetch is not None

        # 2. Clean step right after failure succeeds with trainable embed_tokens + Chunked CE!
        model.zero_grad(set_to_none=True)
        n_items = label_tokens(batches)
        losses = stepper.run(
            batches,
            store,
            head=l2l_sft_head(batches, n_items, impl="chunked", chunk=8),
            route_preserving_mbr=True,
            mbr_decoupled_subwindow=True,
            mbr_recompute_subwindow=1,
            mbr_window_dequant=True,
            release_large_pool_during_layers=True,
            expert_controller=controller,
        )
        assert len(losses) == k
        assert stepper.large_pool_release_stats["forward_body_allocated_bytes"] == 0
        assert stepper.large_pool_release_stats["backward_body_allocated_bytes"] == 0
        embed_grads = [
            p.grad for name, p in trainables(model).items() if "embed_tokens" in name
        ]
        assert embed_grads and all(g is not None and g.count_nonzero() > 0 for g in embed_grads)
    finally:
        controller.close()
        store.close()
        runtime.close()
