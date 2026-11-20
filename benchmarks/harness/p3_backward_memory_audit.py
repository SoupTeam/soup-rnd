"""Empirical live-tensor backward memory audit for DeepSeek P3 (k=1..64, 30B & 671B).

Measures exact live unique storage bytes on a real streamed NF4 DeepSeek-V3 model via
``torch.autograd.graph.saved_tensors_hooks``, separating every contributor:
1. ``LargeLayerBufferPool`` (1.726 GiB persistent BF16 on DeepSeek-V3 671B; freed by M3),
2. ``k``-times dense NF4 ``Linear4bit`` fallback weight copies (eliminated by H2),
3. ``top_k``-fold gathered ``WeightedFrozenExpert`` saved inputs & ``dh_chunks`` (M2+),
4. ``O(k)`` live non-expert autograd graphs in recompute (collapsed to O(1) by M1),
5. PR #72/#73 group-by-group leak when ``mbr_group_size < k`` (fixed).

Also computes calibrated Kanana 30B-A3B and DeepSeek-V3 671B VRAM/RAM projections for
S=64 and S=512 on the target 8 GB GPU (7.45 GiB usable) / 32 GiB RAM (swap=0) stand.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _measure_tiny_deepseek(k_values: List[int], seq_len: int) -> Dict[str, Any]:
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils import l2l_step
    from soup_cli.utils.l2l_activations import ActivationStore
    from tests.test_b13_moe_runtime import build_moe_streamed
    from tests.test_p3_decoupled_subwindow import SavedStorageLedger

    cfg = fixture.moe_config(
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
    rows: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="p3_bwd_audit_") as tmp:
        root = Path(tmp)
        weights = fixture.moe_weights_dir(root, cfg)
        moe = build_moe_streamed(
            weights,
            root / "shards",
            storage="nf4",
            dtype="bfloat16",
            layout="per_expert",
            lora_config=fixture.moe_lora_config(targets=fixture.HEAD_SHARED_TARGETS),
            two_read_experts=True,
            grouped_expert_routes=True,
        )
        try:
            moe.model.train()
            max_k = max(k_values)
            all_batches = [
                fixture.moe_batch(cfg, seed=500 + mb, rows=1, seq=seq_len)["input_ids"]
                for mb in range(max_k)
            ]
            stepper = l2l_step.L2LStep(
                moe.model, moe.runtime, l2l_step.capture_layer_call(moe.model, all_batches[0])
            )
            store = ActivationStore(
                n_layers=cfg.num_hidden_layers,
                k=max_k,
                chunk_shape=(1, seq_len, cfg.hidden_size),
                dtype=torch.bfloat16,
                device="cpu",
                pin=False,
            )
            try:
                large_pool_bytes = (
                    int(moe.runtime.large_pool.nbytes)
                    if moe.runtime.large_pool is not None
                    else 0
                )
                for k in k_values:
                    batches = all_batches[:k]
                    arms = [
                        ("l2l_two_read", dict(route_preserving_mbr=False)),
                        ("mbr_std_full", dict(route_preserving_mbr=True, mbr_group_size=k)),
                        (
                            "mbr_h2_only",
                            dict(
                                route_preserving_mbr=True,
                                mbr_group_size=k,
                                mbr_window_dequant=True,
                            ),
                        ),
                        (
                            "mbr_m1_m2_h2_m3",
                            dict(
                                route_preserving_mbr=True,
                                mbr_group_size=k,
                                mbr_decoupled_subwindow=True,
                                mbr_recompute_subwindow=1,
                                mbr_window_dequant=True,
                                release_large_pool_during_layers=True,
                            ),
                        ),
                    ]
                    k_res: Dict[str, Any] = {}
                    for arm_name, kwargs in arms:
                        for p in fixture.trainables(moe.model).values():
                            p.grad = None
                        ledger = SavedStorageLedger()
                        torch.manual_seed(11)
                        with ledger.track():
                            stepper.run(batches, store, collect_phase_stats=True, **kwargs)
                        lp = stepper.large_pool_release_stats or {}
                        k_res[arm_name] = {
                            "peak_saved_bytes": ledger.peak_bytes,
                            "peak_saved_mib": round(ledger.peak_bytes / (1024**2), 3),
                            "pack_events": ledger.pack_events,
                            "large_pool_slot_bytes": large_pool_bytes,
                            "large_pool_bwd_bytes": int(
                                lp.get("backward_body_allocated_bytes", large_pool_bytes)
                            ),
                            "window_dequant": dict(stepper.window_dequant_stats or {}),
                        }
                    rows[f"k={k}"] = k_res
            finally:
                store.close()
        finally:
            moe.close()
    return {"config": {"hidden_size": cfg.hidden_size, "seq_len": seq_len}, "measured": rows}


def _project_model(
    *,
    name: str,
    vocab: int,
    hidden: int,
    moe_inter: int,
    shared_inter: int,
    n_layers: int,
    n_moe: int,
    n_experts: int,
    top_k: int,
    q_lora: int,
    kv_lora: int,
    n_heads: int,
    qk_nope: int,
    qk_rope: int,
    v_dim: int,
    k_values: List[int],
    seq_lens: List[int],
    gpu_usable_gib: float = 7.45,
    ram_usable_gib: float = 30.0,
) -> Dict[str, Any]:
    bf16 = 2
    fp32 = 4
    gib = 1024**3

    # Exact persistent GPU slots
    large_pool_bytes = vocab * hidden * bf16
    # Non-expert weights of one MoE layer (MLA + router + shared FFN)
    mla_params = (
        hidden * q_lora
        + q_lora * n_heads * (qk_nope + qk_rope)
        + hidden * (kv_lora + qk_rope)
        + kv_lora * n_heads * (qk_nope + v_dim)
        + (n_heads * v_dim) * hidden
    )
    shared_params = 3 * hidden * shared_inter
    router_params = hidden * n_experts
    layer_non_expert_params = mla_params + shared_params + router_params
    # Double-buffered LayerBufferPool stores NF4 (~0.56 B/param with absmax) for linears
    layer_pool_nf4_bytes = int(2 * layer_non_expert_params * 0.5625)
    # 1 dense BF16 dequant copy of the layer (shared across window under H2; k copies without H2!)
    one_layer_dense_bf16_bytes = layer_non_expert_params * bf16
    # 2-slot ExpertBufferPool (NF4 packed + BF16 unpacked per slot)
    expert_params = 3 * hidden * moe_inter
    expert_slot_bytes = int(expert_params * (bf16 + 0.5625))
    expert_pool_2slot_bytes = 2 * expert_slot_bytes
    # Resident LoRA (r=4 on q_b, kv_b, o_proj) + AdamW + CUDA context
    lora_params = n_layers * (
        (q_lora + n_heads * (qk_nope + qk_rope))
        + (kv_lora + n_heads * (qk_nope + v_dim))
        + (n_heads * v_dim + hidden)
    ) * 4
    lora_vram_bytes = int(lora_params * (bf16 + 2 * fp32))
    cuda_ctx_bytes = int(0.35 * gib)

    static_with_large = (
        large_pool_bytes
        + layer_pool_nf4_bytes
        + expert_pool_2slot_bytes
        + lora_vram_bytes
        + cuda_ctx_bytes
    )
    static_with_m3 = static_with_large - large_pool_bytes

    def non_expert_act_bytes_per_mb(s: int) -> int:
        # Exact MLA + Shared FFN saved tensors for 1 micro-batch of length s (BF16 + FP32 softmax):
        # input_norm + q_a + q_b (n_heads*(qk_nope+qk_rope)) + kv_a + kv_b (n_heads*(qk_nope+v_dim))
        # + attn_probs (n_heads * s * s * bf16) + attn_out + post_norm + shared gate/up/silu
        q_dim = n_heads * (qk_nope + qk_rope)
        kv_dim = n_heads * (qk_nope + v_dim)
        attn_tensors = s * (
            3 * hidden + 2 * q_lora + 2 * q_dim + 2 * kv_lora + 2 * kv_dim + n_heads * v_dim
        ) * bf16
        attn_scores = n_heads * s * s * bf16 * 2  # scores + softmax
        shared_tensors = s * (2 * hidden + 3 * shared_inter) * bf16
        return attn_tensors + attn_scores + shared_tensors

    scenarios: Dict[str, Any] = {}
    for s in seq_lens:
        act_mb = non_expert_act_bytes_per_mb(s)
        s_rows: Dict[str, Any] = {}
        for k in k_values:
            tok = k * s
            # Legacy PR #72/#73 MBR (full k autograd + top_k gathered WFE inputs + no H2)
            wfe_saved_inputs = tok * top_k * hidden * bf16  # 8x gathered current_state!
            dh_chunks_plus_sorted = 3 * tok * top_k * hidden * bf16  # h_sorted + dy_sorted + dh
            legacy_no_h2_peak = (
                static_with_large
                + k * one_layer_dense_bf16_bytes
                + k * act_mb
                + wfe_saved_inputs
            )
            # PR #72/#73 + H2 (1 dense copy, still k live graphs + top_k WFE saved inputs)
            pr72_h2_peak = (
                static_with_large
                + one_layer_dense_bf16_bytes
                + k * act_mb
                + wfe_saved_inputs
            )
            # separate_fast (p3-nf4-k64-4060ti): H2 + k live graphs + h_sorted/dy_sorted/dh_chunks
            sep_fast_peak = (
                static_with_large
                + one_layer_dense_bf16_bytes
                + k * act_mb
                + dh_chunks_plus_sorted
            )
            # Our M1 + M2+ + H2 (keeps LargeLayerBufferPool resident)
            # Peak is max(Phase E-Bwd, Phase R2 sub_w=1)
            exp_tok_peak = min(tok, max(64, (tok * top_k * 2) // n_experts))
            e_bwd_leaves = (
                tok * hidden * bf16 * 3  # norm_leaf + dy + d_norm (NOT top_k-fold!)
                + tok * top_k * fp32 * 2  # coeff_leaf + d_coeff
                + int(exp_tok_peak * (hidden + 3 * moe_inter) * bf16)
            )
            r2_sub1 = (
                one_layer_dense_bf16_bytes
                + act_mb
                + tok * hidden * bf16  # remaining d_norm parts
                + tok * top_k * fp32
            )
            m1_m2_h2_peak = static_with_large + max(e_bwd_leaves, r2_sub1)
            # Full M1 + M2+ + H2 + M3 (LargeLayerBufferPool freed across 61 layers)
            m1_m2_h2_m3_peak = static_with_m3 + max(e_bwd_leaves, r2_sub1)

            # Host RAM check (32 GiB, swap=0)
            ram_base = 3.5 * gib
            ram_act_bf16 = tok * hidden * bf16 * n_layers
            ram_act_fp8 = tok * hidden * 1 * n_layers
            ram_routes = tok * n_moe * top_k * 8
            ram_fixed = ram_base + ram_routes + lora_vram_bytes
            ram_avail_bf16 = max(0.0, ram_usable_gib * gib - (ram_fixed + ram_act_bf16))
            ram_avail_fp8 = max(0.0, ram_usable_gib * gib - (ram_fixed + ram_act_fp8))
            exp_nf4_bytes = int(expert_params * 0.5625)

            s_rows[f"k={k}"] = {
                "tokens": tok,
                "legacy_pr72_no_h2_gib": round(legacy_no_h2_peak / gib, 3),
                "pr72_with_h2_gib": round(pr72_h2_peak / gib, 3),
                "separate_fast_h2_dhchunks_gib": round(sep_fast_peak / gib, 3),
                "m1_m2_h2_gib": round(m1_m2_h2_peak / gib, 3),
                "m1_m2_h2_m3_gib": round(m1_m2_h2_m3_peak / gib, 3),
                "wfe_topk_saved_gib": round(wfe_saved_inputs / gib, 3),
                "dh_chunks_and_sorted_gib": round(dh_chunks_plus_sorted / gib, 3),
                "k_non_expert_acts_gib": round((k * act_mb) / gib, 3),
                "sub1_non_expert_act_gib": round(act_mb / gib, 4),
                "fits_8gb_pr72_h2": pr72_h2_peak / gib <= gpu_usable_gib,
                "fits_8gb_m1_m2_h2": m1_m2_h2_peak / gib <= gpu_usable_gib,
                "fits_8gb_m1_m2_h2_m3": m1_m2_h2_m3_peak / gib <= gpu_usable_gib,
                "headroom_8gb_m1_m2_h2_m3_gib": round(
                    gpu_usable_gib - m1_m2_h2_m3_peak / gib, 3
                ),
                "ram_act_bf16_gib": round(ram_act_bf16 / gib, 3),
                "ram_act_fp8_gib": round(ram_act_fp8 / gib, 3),
                "ram_cache_experts_bf16_act": int(ram_avail_bf16 // exp_nf4_bytes),
                "ram_cache_experts_fp8_act": int(ram_avail_fp8 // exp_nf4_bytes),
            }
        scenarios[f"S={s}"] = s_rows

    return {
        "model": name,
        "static_gib": {
            "large_layer_buffer_pool": round(large_pool_bytes / gib, 3),
            "layer_buffer_pool_nf4_2slot": round(layer_pool_nf4_bytes / gib, 3),
            "one_layer_dense_bf16_h2": round(one_layer_dense_bf16_bytes / gib, 3),
            "expert_buffer_pool_2slot": round(expert_pool_2slot_bytes / gib, 3),
            "lora_adamw_r4": round(lora_vram_bytes / gib, 3),
            "cuda_context": round(cuda_ctx_bytes / gib, 3),
            "total_static_with_large_pool": round(static_with_large / gib, 3),
            "total_static_with_m3": round(static_with_m3 / gib, 3),
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek P3 backward memory empirical audit.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    tiny = _measure_tiny_deepseek(args.k, args.seq_len)
    kanana = _project_model(
        name="Kanana-30B-A3B",
        vocab=128256,
        hidden=2048,
        moe_inter=1408,
        shared_inter=1408 * 2,
        n_layers=48,
        n_moe=32,
        n_experts=128,
        top_k=8,
        q_lora=1536,
        kv_lora=512,
        n_heads=32,
        qk_nope=128,
        qk_rope=64,
        v_dim=128,
        k_values=[8, 32, 64, 128],
        seq_lens=[64, 512],
    )
    deepseek_671b = _project_model(
        name="DeepSeek-V3-671B",
        vocab=129280,
        hidden=7168,
        moe_inter=2048,
        shared_inter=2048,
        n_layers=61,
        n_moe=58,
        n_experts=256,
        top_k=8,
        q_lora=1536,
        kv_lora=512,
        n_heads=128,
        qk_nope=128,
        qk_rope=64,
        v_dim=128,
        k_values=[8, 32, 64, 128],
        seq_lens=[64, 512],
    )
    report = {
        "tiny_deepseek_nf4_measured": tiny,
        "projections": {"kanana_30b": kanana, "deepseek_v3_671b": deepseek_671b},
    }
    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    from rich.console import Console

    Console().print(text)


if __name__ == "__main__":
    main()
