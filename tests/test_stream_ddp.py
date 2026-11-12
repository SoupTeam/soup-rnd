"""Layer streaming under a distributed data-parallel launch (``soup train --gpus N``).

A streamed decoder keeps the frozen base on ``meta``; DDP refused such a module ("input
module parameters locate in {'cpu', 'meta'}"). With the frozen parameters and buffers on
DDP's ignore list, two ranks must train the adapters exactly as one process accumulating
both ranks' micro-batches.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

from soup_cli.utils import stream_ddp


def _tiny_qwen35_dir(tmp_path):
    """A tiny hybrid qwen3_5 checkpoint (3 DeltaNet + 1 attention layer), as in
    tests/test_l2l_qwen35.py; built here because the spawned ranks import this file alone."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(11)
    config = AutoConfig.for_model(
        "qwen3_5_text", vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, full_attention_interval=4, tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _qwen35_streamed(tmp):
    import pathlib

    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_qwen35_dir(pathlib.Path(tmp))
    shards = os.path.join(tmp, "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="qwen3")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora,
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    import torch

    gen = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.02)
    return model, runtime


def _batch(n):
    import torch

    gen = torch.Generator().manual_seed(29)
    return [torch.randint(0, 64, (1, 8), generator=gen) for _ in range(n)]


def _rank(rank, world, tmp, port, ignore):
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    out = os.path.join(tmp, f"rank{rank}.pt")
    try:
        model, runtime = _qwen35_streamed(os.path.join(tmp, f"rank{rank}"))
        if ignore:
            stream_ddp.ignore_frozen_for_ddp(model)
        try:
            wrapped = torch.nn.parallel.DistributedDataParallel(model)
        except ValueError as exc:
            torch.save(("refused", str(exc)), out)
            runtime.close()
            return
        batch = _batch(2 * world)
        total = sum(ids.shape[1] - 1 for ids in batch)
        for ids in batch[rank::world]:
            # DDP averages over ranks; the accumulated sum over the global window is wanted.
            (wrapped(input_ids=ids, labels=ids, num_items_in_batch=total).loss * world).backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(("ok", grads), out)
        runtime.close()
    finally:
        dist.destroy_process_group()


def _run_ddp(tmp_path, ignore):
    import socket
    import sys

    import torch
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    mp.spawn(_rank, args=(2, str(tmp_path), port, ignore), nprocs=2, join=True)
    return {r: torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(2)}


def test_ddp_refuses_a_streamed_model_without_the_ignore_list(tmp_path):
    pytest.importorskip("peft")
    results = _run_ddp(tmp_path, ignore=False)
    assert all(status == "refused" and "meta" in payload for status, payload in results.values())


def test_two_ranks_train_the_adapters_like_one_process_accumulating(tmp_path):
    pytest.importorskip("peft")
    import torch

    results = _run_ddp(tmp_path, ignore=True)
    assert all(status == "ok" for status, _ in results.values()), results
    rank0, rank1 = results[0][1], results[1][1]
    assert all(torch.equal(rank0[n], rank1[n]) for n in rank0)  # synchronised

    model, runtime = _qwen35_streamed(str(tmp_path / "single"))
    try:
        batch = _batch(4)
        total = sum(ids.shape[1] - 1 for ids in batch)
        for ids in batch:
            model(input_ids=ids, labels=ids, num_items_in_batch=total).loss.backward()
        single = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    finally:
        runtime.close()
    assert single.keys() == rank0.keys()
    assert any(float(g.abs().sum()) > 0 for g in single.values())
    for name in single:
        torch.testing.assert_close(rank0[name], single[name], rtol=1e-5, atol=1e-7)


def test_ignore_list_names_the_frozen_base_and_buffers_not_the_adapters():
    torch = pytest.importorskip("torch")

    model = torch.nn.Module()
    model.base = torch.nn.Linear(2, 2)
    model.base.weight.requires_grad_(False)
    model.base.bias.requires_grad_(False)
    model.lora = torch.nn.Linear(2, 2)
    model.register_buffer("rope", torch.zeros(2))
    assert stream_ddp.ignore_frozen_for_ddp(model) == 3
    assert sorted(getattr(model, stream_ddp.DDP_IGNORE_ATTR)) == [
        "base.bias", "base.weight", "rope"]


@pytest.mark.parametrize("env,expected", [
    ({}, None), ({"WORLD_SIZE": "1", "LOCAL_RANK": "0"}, None),
    ({"WORLD_SIZE": "2", "LOCAL_RANK": "1"}, 1), ({"WORLD_SIZE": "x", "LOCAL_RANK": "1"}, None),
])
def test_local_rank_only_in_a_distributed_launch(monkeypatch, env, expected):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert stream_ddp.local_rank() == expected


def test_pin_rank_device(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert stream_ddp.pin_rank_device("cuda") == "cuda:1"
    assert calls == [1]
    assert stream_ddp.pin_rank_device("cuda:0") == "cuda:0"
    assert stream_ddp.pin_rank_device("cpu") == "cpu"
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert stream_ddp.pin_rank_device("cuda") == "cuda"


def test_shard_cache_lock_serialises_ranks(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLD_SIZE", "2")
    shard_dir = str(tmp_path / "cache" / "model")
    inside, overlap = [0], []

    def writer():
        with stream_ddp.shard_cache_lock(shard_dir):
            inside[0] += 1
            overlap.append(inside[0])
            time.sleep(0.2)
            inside[0] -= 1

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlap == [1, 1]


def test_l2l_is_refused_in_a_distributed_launch(monkeypatch):
    from types import SimpleNamespace

    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="distributed launch"):
        refuse_unsupported_l2l(SimpleNamespace(model_type="llama"), n_gpu=1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    refuse_unsupported_l2l(SimpleNamespace(model_type="llama"), n_gpu=1)
