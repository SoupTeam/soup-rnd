"""The real Torch 2.13 failure requires joint placement, not two shard counts."""

from types import SimpleNamespace

import pytest


def test_frozen_tied_embedding_and_head_are_placed_in_one_fsdp2_group(monkeypatch):
    import torch.nn as nn
    from torch.distributed import fsdp

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    embed = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    head.weight = embed.weight
    embed.weight.requires_grad_(False)
    model = SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)
    policy = object()
    plugin = SimpleNamespace(fsdp_version=2, mixed_precision_policy=policy, cpu_offload=None)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    calls = []
    monkeypatch.setattr(fsdp, "fully_shard",
                        lambda modules, **kwargs: calls.append((modules, kwargs)))
    assert prepare_fsdp2_tied_embeddings(SimpleNamespace(model=model, accelerator=accelerator))
    assert calls[0][0] == [embed, head]  # embedding alone creates the hardware failure
    assert calls[0][1]["reshard_after_forward"] is False
    assert calls[0][1]["mp_policy"] is policy
    assert embed.weight is head.weight


@pytest.mark.parametrize("variant", ["untied", "trainable", "fsdp1"])
def test_other_routes_are_not_rewritten(monkeypatch, variant):
    import torch.nn as nn
    from torch.distributed import fsdp

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    embed = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    if variant != "untied":
        head.weight = embed.weight
    for module in (embed, head):
        module.requires_grad_(variant == "trainable")
    model = SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)
    plugin = SimpleNamespace(fsdp_version=1 if variant == "fsdp1" else 2)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    monkeypatch.setattr(fsdp, "fully_shard", lambda *a, **k: pytest.fail("unexpected placement"))
    assert not prepare_fsdp2_tied_embeddings(SimpleNamespace(model=model, accelerator=accelerator))


def test_non_distributed_trainer_is_a_noop():
    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    assert not prepare_fsdp2_tied_embeddings(SimpleNamespace())
