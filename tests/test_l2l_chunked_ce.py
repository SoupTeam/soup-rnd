"""Loss parity at the L2L SFT head seam (PR #63 Chunked CE & CCE bridge)."""

from __future__ import annotations


def test_chunked_ce_matches_full_with_masked_rows_and_window_divisor():
    import torch
    from torch.nn import functional as F  # noqa: N812

    from soup_cli.utils.stream_chunked_ce import chunked_causal_lm_loss

    generator = torch.Generator().manual_seed(78)
    hidden = torch.randn(2, 7, 13, generator=generator).requires_grad_()
    labels = torch.randint(0, 37, (2, 7), generator=generator)
    labels[0] = -100
    labels[1, 2:4] = -100
    head = torch.nn.Linear(13, 37)
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    full = F.cross_entropy(head(hidden).view(-1, 37), shifted, reduction="sum") / 11
    chunked = chunked_causal_lm_loss(hidden, head, labels, 11, chunk=4)
    torch.testing.assert_close(chunked, full)
    leaves = (hidden, head.weight, head.bias)
    grads = torch.autograd.grad(chunked, leaves)
    for got, expected in zip(grads, torch.autograd.grad(full, leaves)):
        torch.testing.assert_close(got, expected)


def test_fully_masked_microbatch_has_zero_connected_vjp():
    import torch

    from soup_cli.utils.stream_chunked_ce import chunked_causal_lm_loss

    hidden = torch.ones(1, 6, 8, requires_grad=True)
    labels = torch.full((1, 6), -100)
    loss = chunked_causal_lm_loss(hidden, torch.nn.Linear(8, 16), labels, 12, chunk=3)
    loss.backward()
    assert loss.item() == 0
    assert hidden.grad.count_nonzero() == 0


def test_cce_passes_the_live_slot_and_window_scale_to_the_kernel(monkeypatch):
    import sys
    from types import SimpleNamespace

    import torch
    from torch.nn import functional as F  # noqa: N812

    from soup_cli.utils.layer_stream_runtime import (
        LargeLayerBufferPool,
        _streamed_large_layer_class,
    )
    from soup_cli.utils.stream_chunked_ce import cce_causal_lm_loss

    weight = torch.arange(40, dtype=torch.bfloat16).view(10, 4) / 20
    pool = LargeLayerBufferPool({"head": ((10, 4), "bfloat16")}, {"head": 0}, device="cpu")
    pool.load_async("head", SimpleNamespace(get=lambda _idx, _key: weight))
    inner = torch.nn.Linear(4, 10, bias=False, device="meta").requires_grad_(False)
    head = _streamed_large_layer_class()(inner, "head", pool)
    hidden = torch.randn(1, 4, 4, dtype=torch.bfloat16).requires_grad_()
    labels = torch.tensor([[2, -100, 4, 5]])

    def kernel(x, w, y, *, ignore_index, reduction):
        assert w.data_ptr() == pool.buffer.data_ptr()
        assert y.tolist() == [[-100, 4, 5, -100]]
        return F.cross_entropy(
            F.linear(x, w).float().flatten(0, 1),
            y.flatten(),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    monkeypatch.setitem(
        sys.modules, "cut_cross_entropy", SimpleNamespace(linear_cross_entropy=kernel)
    )
    actual = cce_causal_lm_loss(hidden, head, labels, 9)
    expected = (
        F.cross_entropy(
            head(hidden).float().flatten(0, 1),
            torch.tensor([-100, 4, 5, -100]),
            reduction="sum",
        )
        / 9
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, hidden)[0], torch.autograd.grad(expected, hidden)[0]
    )


def test_chunked_head_preserves_two_read_l2l_lora_gradients(tmp_path):
    import torch

    from benchmarks.harness.b13_moe_fixture import moe_lora_config, moe_weights_dir, snapshot
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.stream_chunked_ce import l2l_sft_head

    weights = moe_weights_dir(tmp_path)
    shards = tmp_path / "shards"
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="deepseek_v3", expert_layout="per_expert"
    )
    model, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=str(shards),
        index=index,
        lora_config=moe_lora_config(),
        device="cpu",
        dtype="float32",
        tier="disk",
        pin=False,
        two_read_experts=True,
        grouped_expert_routes=True,
    )
    store = ActivationStore(
        n_layers=3, k=2, chunk_shape=(1, 11, 32), dtype=torch.float32, device="cpu", pin=False
    )
    try:
        model.train()
        generator = torch.Generator().manual_seed(63)
        ids = [torch.randint(0, 64, (1, 11), generator=generator) for _ in range(2)]
        labels = [torch.full_like(ids[0], -100), ids[1].clone()]
        labels[1][:, 3:6] = -100
        n_items = int((labels[1][:, 1:] != -100).sum())
        call = capture_layer_call(model, ids[0], draws=False)
        step = L2LStep(model, runtime, call)
        # Warm up layer buffer slots so cold-vs-warm layer_loads counters match between arms.
        step.run(
            ids,
            store,
            head=l2l_sft_head(labels, n_items, impl="full"),
            route_preserving_mbr=True,
            mbr_decoupled_subwindow=True,
            mbr_recompute_subwindow=1,
            release_large_pool_during_layers=True,
        )
        results = []
        for impl in ("full", "chunked"):
            model.zero_grad(set_to_none=True)
            losses = step.run(
                ids,
                store,
                head=l2l_sft_head(labels, n_items, impl=impl, chunk=4),
                route_preserving_mbr=True,
                mbr_decoupled_subwindow=True,
                mbr_recompute_subwindow=1,
                release_large_pool_during_layers=True,
            )
            results.append((torch.stack(losses), snapshot(model, gradients=True), step.counters))
        assert results[0][2].expert_fetches > 0
        assert results[0][2] == results[1][2]
        torch.testing.assert_close(results[1][0], results[0][0])
        for name in results[0][1]:
            torch.testing.assert_close(results[1][1][name], results[0][1][name])
    finally:
        store.close()
        runtime.close()
