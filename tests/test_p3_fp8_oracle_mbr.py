"""Tests for block-64 FP8 boundary activations + DeepseekRoutingOracle + decoupled MBR."""

from __future__ import annotations

import pytest
import torch

from soup_cli.utils import activation_codec, l2l_step
from soup_cli.utils.async_expert_controller import AsyncExpertController, cpu_packed_control
from soup_cli.utils.fp8_activations import FP8ActivationStore, fp8_activation_allocation
from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.l2l_activations import ActivationStore
from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint
from soup_cli.utils.layer_stream_runtime import (
    build_meta_skeleton,
    build_streamed_model,
    quantised_layer_suffixes,
    shard_layout_of,
)
from soup_cli.utils.routing_oracle import DeepseekRoutingOracle
from soup_cli.utils.stream_chunked_ce import l2l_sft_head


def _grad_agreement(ref: dict, cand: dict) -> tuple[float, float]:
    import math

    err_total = 0.0
    ref_total = 0.0
    min_cos = 1.0
    for name in sorted(ref):
        a = ref[name].detach().double().flatten()
        b = cand[name].detach().double().flatten()
        ref_sq = float(a.dot(a))
        cand_sq = float(b.dot(b))
        err_sq = float((b - a).pow(2).sum())
        err_total += err_sq
        ref_total += ref_sq
        if ref_sq > 0 and cand_sq > 0:
            cos = float(a.dot(b)) / math.sqrt(ref_sq * cand_sq)
        else:
            cos = 1.0 if ref_sq == cand_sq == 0 else 0.0
        if cos < min_cos:
            min_cos = cos
    rel_l2 = math.sqrt(err_total / max(ref_total, 1e-300))
    return rel_l2, min_cos


def _build_tiny_shards(tmp_path):
    from benchmarks.harness.b5_real_moe import _meta_construction
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir

    work = tmp_path / "tiny_fp8_oracle"
    work.mkdir(parents=True, exist_ok=True)
    cfg = moe_config(
        vocab_size=256,
        tie_word_embeddings=False,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    weights = moe_weights_dir(work, cfg)
    shards = work / "shards"
    with _meta_construction():
        skeleton = build_meta_skeleton(
            str(weights), dtype="bfloat16", quant="nf4", double_quant=False
        )
    layout = shard_layout_of(skeleton)
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    shard_checkpoint(
        str(weights),
        str(shards),
        dtype="bfloat16",
        arch="deepseek_v3",
        quant="nf4",
        double_quant=False,
        quant_device="cpu",
        expert_layout="per_expert",
        layout=layout,
        quant_suffixes=suffixes,
    )
    return weights, shards


def _open_streamed(weights, shards, lora_state=None, *, two_read_experts: bool = True):
    from peft import LoraConfig, TaskType

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness.b5_real_moe import _meta_construction

    torch.manual_seed(71)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_b_proj", "kv_b_proj", "o_proj"],
    )
    index = read_shard_index(str(shards))
    with _meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=lora_cfg,
            device="cpu",
            dtype="bfloat16",
            quant="nf4",
            double_quant=False,
            tier="ram",
            pin=False,
            two_read_experts=two_read_experts,
            grouped_expert_routes=True,
            expert_transfer="sync",
            expert_gemm="native",
        )
    model.train()
    trainables = fixture.trainables(model)
    if lora_state is not None:
        with torch.no_grad():
            for name, param in trainables.items():
                param.copy_(lora_state[name].to(param.device, dtype=param.dtype))
    else:
        fixture.randomize_adapters(model, seed=67)
    return model, runtime, trainables, fixture.snapshot(model, gradients=False)


def test_block64_fp8_codec_bitwise_matches_p3_realdata_codec():
    from benchmarks.harness import p3_realdata_codec as ref_codec

    gen = torch.Generator().manual_seed(123)
    for hidden in (128, 130):
        x = torch.randn((1, 17, hidden), generator=gen, dtype=torch.float32).to(torch.bfloat16)
        x[0, 3].zero_()  # exact zero row
        expected = ref_codec.round_trip(x, format="e4m3", group_size=64)
        payload, scales = activation_codec.encode(x, activation_codec.FP8, group_size=64)
        actual = activation_codec.decode(
            payload, scales, activation_codec.FP8, torch.bfloat16, hidden, group_size=64
        )
        assert torch.equal(actual, expected), f"block64 mismatch for hidden={hidden}"

    # Verify FP8ActivationStore(group_size=64) bitwise matches p3_realdata_codec.make_store
    x = torch.randn((1, 32, 128), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    alloc = fp8_activation_allocation(
        n_layers=2, mb_shapes=[(1, 32, 128)], pin=False, compute_dtype="bfloat16",
        device_type="cpu", group_size=64,
    )
    budget = HostMemoryBudget(alloc.host_bytes)
    store = FP8ActivationStore(
        host_budget=budget, n_layers=2, k=1, chunk_shape=(1, 32, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    policy = ref_codec.CodecPolicy("block64_e4m3", "e4m3", group_size=64)
    ref_alloc = ref_codec.allocation(
        policy, n_layers=2, mb_shapes=[(1, 32, 128)], dtype=torch.bfloat16,
        pin=False, device_type="cpu",
    )
    ref_store = ref_codec.make_store(
        policy, host_budget=HostMemoryBudget(ref_alloc["charged_host_bytes"]),
        n_layers=2, mb_shapes=[(1, 32, 128)], dtype=torch.bfloat16, device="cpu", pin=False,
    )
    try:
        store.put(0, 0, x)
        ref_store.put(0, 0, x)
        store.begin_backward()
        ref_store.begin_backward()
        got = store.get(0, 0)
        want = ref_store.get(0, 0)
        assert torch.equal(got, want)
        # Second get (decoupled MBR R1 + R2) returns identical decoded values
        assert torch.equal(store.get(0, 0), want)
        store.end_step()
        ref_store.end_step()
    finally:
        store.close()
        ref_store.close()
    assert budget.used_bytes == 0


def test_block64_fp8_store_refuses_nonfinite_on_end_step():
    x = torch.randn((1, 8, 128), dtype=torch.bfloat16)
    x[0, 2, 70] = float("nan")
    alloc = fp8_activation_allocation(
        n_layers=1, mb_shapes=[(1, 8, 128)], pin=False, compute_dtype="bfloat16",
        device_type="cpu", group_size=64,
    )
    budget = HostMemoryBudget(alloc.host_bytes)
    store = FP8ActivationStore(
        host_budget=budget, n_layers=1, k=1, chunk_shape=(1, 8, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    try:
        store.put(0, 0, x)
        with pytest.raises(activation_codec.NonFiniteActivationError):
            store.end_step()
    finally:
        store.close()
    assert budget.used_bytes == 0


def test_unified_go_fp8_oracle_bitwise_matches_batyr_probe_and_bf16_e3(tmp_path):
    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness import p3_deepseek_oracle as ref_oracle
    from benchmarks.harness import p3_realdata_codec as ref_codec

    torch.use_deterministic_algorithms(True)
    weights, shards = _build_tiny_shards(tmp_path)
    k, seq_len = 4, 32
    gen = torch.Generator().manual_seed(2026)
    batches = [torch.randint(0, 256, (1, seq_len), generator=gen) for _ in range(k)]
    mb_shapes = [(1, seq_len, 128)] * k

    # Arm 0: Exact BF16 unified_go reference
    model0, rt0, tr0, init_lora = _open_streamed(weights, shards)
    cpu_packed_control(rt0.expert_pool, rt0.expert_readers[0], 0)
    ctrl0 = AsyncExpertController(
        rt0, mode="cpp_async_2slot", cuda_nf4_decode=False,
        io_workers=2, ring_slots=4, cross_layer_prefetch=True,
    )
    store0 = ActivationStore(
        n_layers=3, k=k, chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16, device="cpu", pin=False,
    )
    try:
        stepper0 = l2l_step.L2LStep(model0, rt0, l2l_step.capture_layer_call(model0, batches[0]))
        opt0 = torch.optim.AdamW(tr0.values(), lr=1e-4)
        opt0.zero_grad(set_to_none=True)
        head0 = l2l_sft_head(batches, l2l_step.label_tokens(batches), impl="chunked", chunk=16)
        losses0 = stepper0.run(
            batches, store0, head=head0,
            route_preserving_mbr=True, mbr_group_size=k,
            mbr_decoupled_subwindow=True, mbr_recompute_subwindow=1,
            mbr_window_dequant=True, release_large_pool_during_layers=True,
            expert_controller=ctrl0, collect_phase_stats=True,
        )
        grads0 = fixture.snapshot(model0, gradients=True)
        opt0.step()
        bf16_store_bytes = 3 * k * seq_len * 128 * 2
    finally:
        store0.close()
        ctrl0.close()
        rt0.close()

    # Arm 1: Batyr's reference p3_realdata_codec (block64_e4m3) + DeepseekOracleProbe (plain L2L)
    model1, rt1, tr1, _ = _open_streamed(
        weights, shards, lora_state=init_lora, two_read_experts=False
    )
    policy = ref_codec.CodecPolicy("block64_e4m3", "e4m3", group_size=64)
    budget1 = HostMemoryBudget(256 * 1024**2)
    store1 = ref_codec.make_store(
        policy, host_budget=budget1, n_layers=3, mb_shapes=mb_shapes,
        dtype=torch.bfloat16, device="cpu", pin=False,
    )
    store1.k = k
    try:
        call1 = l2l_step.capture_layer_call(model1, batches[0])
        stepper1 = l2l_step.L2LStep(model1, rt1, call1)
        with ref_oracle.DeepseekOracleProbe(model1, mb_shapes, budget1) as probe1:
            opt1 = torch.optim.AdamW(tr1.values(), lr=1e-4)
            opt1.zero_grad(set_to_none=True)
            head1 = l2l_sft_head(batches, l2l_step.label_tokens(batches), impl="chunked", chunk=16)
            losses1 = stepper1.run(batches, store1, head=head1, routing_oracle=probe1)
            grads1 = fixture.snapshot(model1, gradients=True)
            opt1.step()
            adapter1 = fixture.snapshot(model1, gradients=False)
            assert probe1.report()["complete"] is True
    finally:
        store1.close()
        rt1.close()
    assert budget1.used_bytes == 0

    # Arm 2: Production unified_go + FP8ActivationStore(group_size=64) + DeepseekRoutingOracle
    model2, rt2, tr2, _ = _open_streamed(weights, shards, lora_state=init_lora)
    cpu_packed_control(rt2.expert_pool, rt2.expert_readers[0], 0)
    ctrl2 = AsyncExpertController(
        rt2, mode="cpp_async_2slot", cuda_nf4_decode=False,
        io_workers=2, ring_slots=4, cross_layer_prefetch=True,
    )
    budget2 = HostMemoryBudget(256 * 1024**2)
    store2 = FP8ActivationStore(
        host_budget=budget2, n_layers=3, k=k, chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    try:
        call2 = l2l_step.capture_layer_call(model2, batches[0])
        stepper2 = l2l_step.L2LStep(model2, rt2, call2)
        with DeepseekRoutingOracle(model2, host_budget=budget2, pin=False) as oracle2:
            opt2 = torch.optim.AdamW(tr2.values(), lr=1e-4)
            opt2.zero_grad(set_to_none=True)
            head2 = l2l_sft_head(batches, l2l_step.label_tokens(batches), impl="chunked", chunk=16)
            losses2 = stepper2.run(
                batches, store2, head=head2,
                route_preserving_mbr=True, mbr_group_size=k,
                mbr_decoupled_subwindow=True, mbr_recompute_subwindow=1,
                mbr_window_dequant=True, release_large_pool_during_layers=True,
                expert_controller=ctrl2, routing_oracle=oracle2,
                collect_phase_stats=True,
            )
            grads2 = fixture.snapshot(model2, gradients=True)
            opt2.step()
            adapter2 = fixture.snapshot(model2, gradients=False)
            oracle_report = stepper2.routing_oracle_stats
            fp8_store_bytes = store2.host_bytes()["owned_host_storage_bytes"]
            oracle_host_bytes = oracle2.host_bytes()["owned_host_storage_bytes"]
    finally:
        store2.close()
        ctrl2.close()
        rt2.close()
    assert budget2.used_bytes == 0

    # 1. Exact BIT-FOR-BIT match between Batyr's DeepseekOracleProbe (plain L2L) and unified_go!
    for l1, l2 in zip(losses1, losses2):
        assert torch.equal(l1, l2)
    for name in sorted(grads1):
        assert torch.equal(grads1[name], grads2[name]), f"grad mismatch vs Batyr Oracle: {name}"
    for name in sorted(adapter1):
        assert torch.equal(adapter1[name], adapter2[name]), f"adapter mismatch: {name}"

    # 2. Exact forward loss match and tight E3 gradient agreement vs BF16
    for l0, l2 in zip(losses0, losses2):
        assert torch.equal(l0, l2)
    rel_l2, min_cos = _grad_agreement(grads0, grads2)
    ref_rel_l2, ref_min_cos = _grad_agreement(grads0, grads1)
    assert rel_l2 == ref_rel_l2 and min_cos == ref_min_cos
    assert rel_l2 < 0.08, f"E3 global_rel_l2={rel_l2} unexpectedly high"
    assert min_cos >= 0.99, f"E3 min_cosine={min_cos} fell below 0.99"

    # 3. Telemetry & storage reduction checks
    assert fp8_store_bytes < bf16_store_bytes * 0.55
    assert oracle_host_bytes > 0
    assert oracle_report["complete"] is True
    assert oracle_report["step_counts"] == {
        "native_forward": 2 * k,
        "oracle_peek": 2 * k,
        "oracle_recompute": 2 * k,
        "oracle_vjp": 2 * k,
    }
    recomp_fetches = sum(
        row.get("recompute_forward", {}).get("expert_fetches", 0)
        for row in stepper2.expert_phase_stats["layers"].values()
    )
    assert recomp_fetches == 0
    assert stepper2.large_pool_release_stats["forward_body_allocated_bytes"] == 0
    assert stepper2.large_pool_release_stats["backward_body_allocated_bytes"] == 0


def test_multistep_fp8_oracle_and_fail_closed_without_oracle(tmp_path):
    weights, shards = _build_tiny_shards(tmp_path)
    k, seq_len = 2, 16
    gen = torch.Generator().manual_seed(99)
    step_batches = [
        [torch.randint(0, 256, (1, seq_len), generator=gen) for _ in range(k)]
        for _ in range(2)
    ]

    # Multi-step reuse (2 AdamW updates) with DeepseekRoutingOracle
    model, rt, tr, _ = _open_streamed(weights, shards)
    budget = HostMemoryBudget(128 * 1024**2)
    store = FP8ActivationStore(
        host_budget=budget, n_layers=3, k=k, chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    try:
        call = l2l_step.capture_layer_call(model, step_batches[0][0])
        stepper = l2l_step.L2LStep(model, rt, call)
        opt = torch.optim.AdamW(tr.values(), lr=1e-4)
        with DeepseekRoutingOracle(model, host_budget=budget, pin=False) as oracle:
            for batches in step_batches:
                opt.zero_grad(set_to_none=True)
                head = l2l_sft_head(
                    batches, l2l_step.label_tokens(batches), impl="chunked", chunk=8
                )
                stepper.run(
                    batches, store, head=head,
                    route_preserving_mbr=True, mbr_group_size=k,
                    mbr_decoupled_subwindow=True, mbr_recompute_subwindow=1,
                    mbr_window_dequant=True, release_large_pool_during_layers=True,
                    routing_oracle=oracle,
                )
                opt.step()
            rep = oracle.report()
            assert rep["steps_completed"] == 2
            assert rep["total_counts"]["native_forward"] == 2 * 2 * k
            assert rep["total_counts"]["oracle_peek"] == 2 * 2 * k
            assert rep["total_counts"]["oracle_recompute"] == 2 * 2 * k
            assert rep["total_counts"]["oracle_vjp"] == 2 * 2 * k
        # Gate method restored cleanly to two-read _routing_forward after oracle.close()
        from soup_cli.utils.expert_stream import _routing_forward

        for idx in (1, 2):
            gate = stepper.layers[idx].inner.mlp.gate
            assert getattr(gate.forward, "__func__", None) is _routing_forward
    finally:
        store.close()
        rt.close()
    assert budget.used_bytes == 0

    # Negative test: ReferencePass + routing_oracle is refused
    model_r, rt_r, _, _ = _open_streamed(weights, shards)
    store_r = ActivationStore(
        n_layers=3, k=k, chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16, device="cpu", pin=False,
    )
    try:
        stepper_r = l2l_step.L2LStep(
            model_r, rt_r, l2l_step.capture_layer_call(model_r, step_batches[0][0])
        )
        ref_pass = l2l_step.ReferencePass(head=lambda _c, _o, h, _m: h.sum())
        with DeepseekRoutingOracle(model_r, pin=False) as oracle_r:
            with pytest.raises(ValueError, match="ReferencePass"):
                stepper_r.run(
                    step_batches[0], store_r, reference=ref_pass, routing_oracle=oracle_r
                )
    finally:
        store_r.close()
        rt_r.close()

    # Negative test 1: plain L2L (two_read_experts=False) + perturbed FP8 WITHOUT Oracle
    # fails closed in RoutingGuard on recompute route flip.
    model_f, rt_f, _, _ = _open_streamed(weights, shards, two_read_experts=False)
    budget_f = HostMemoryBudget(128 * 1024**2)
    store_f = FP8ActivationStore(
        host_budget=budget_f, n_layers=3, k=4, chunk_shape=(1, 32, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    try:
        gen_f = torch.Generator().manual_seed(777)
        batches_f = [torch.randint(0, 256, (1, 32), generator=gen_f) for _ in range(4)]
        stepper_f = l2l_step.L2LStep(
            model_f, rt_f, l2l_step.capture_layer_call(model_f, batches_f[0])
        )
        orig_get = store_f.get

        def noisy_get(layer_i: int, mb_i: int):
            val = orig_get(layer_i, mb_i)
            return val + 0.5 * torch.ones_like(val)

        store_f.get = noisy_get  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="routing"):
            stepper_f.run(batches_f, store_f, routing_oracle=None)
    finally:
        store_f.close()
        rt_f.close()
    assert budget_f.used_bytes == 0

    # Negative test 2: decoupled MBR (two_read_experts=True) without Oracle has non-zero
    # gate_weight drift on FP8, whereas DeepseekRoutingOracle keeps gate_weight_rel_l2 == 0.0.
    model_m, rt_m, _, _ = _open_streamed(weights, shards, two_read_experts=True)
    budget_m = HostMemoryBudget(128 * 1024**2)
    store_m = FP8ActivationStore(
        host_budget=budget_m, n_layers=3, k=2, chunk_shape=(1, seq_len, 128),
        dtype=torch.bfloat16, device="cpu", pin=False, group_size=64,
    )
    try:
        stepper_m = l2l_step.L2LStep(
            model_m, rt_m, l2l_step.capture_layer_call(model_m, step_batches[0][0])
        )
        stepper_m.run(
            step_batches[0], store_m,
            route_preserving_mbr=True, mbr_group_size=2,
            mbr_decoupled_subwindow=True, mbr_recompute_subwindow=1,
            collect_phase_stats=True, routing_oracle=None,
        )
        drift_no_oracle = max(
            row["gate_weight_rel_l2"]
            for row in stepper_m.route_preserving_mbr_stats["layers"].values()
            if "gate_weight_rel_l2" in row
        )
        assert drift_no_oracle > 0.0
        with DeepseekRoutingOracle(model_m, host_budget=budget_m, pin=False) as oracle_m:
            stepper_m.run(
                step_batches[0], store_m,
                route_preserving_mbr=True, mbr_group_size=2,
                mbr_decoupled_subwindow=True, mbr_recompute_subwindow=1,
                collect_phase_stats=True, routing_oracle=oracle_m,
            )
            drift_with_oracle = max(
                row["gate_weight_rel_l2"]
                for row in stepper_m.route_preserving_mbr_stats["layers"].values()
                if "gate_weight_rel_l2" in row
            )
            assert drift_with_oracle == 0.0
    finally:
        store_m.close()
        rt_m.close()
    assert budget_m.used_bytes == 0
