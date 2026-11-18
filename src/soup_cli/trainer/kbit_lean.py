"""ZERO-2: plain (non-streamed) QLoRA in a small VRAM budget.

Stage 2 of ZERO-2 (Qwen3.5-9B, NF4, 8 GiB cap) failed before step 1: PEFT's
``prepare_model_for_kbit_training`` casts every non-quantized weight to fp32, and one of the two
untied 248 320 x 4096 vocabulary matrices alone asked for 3.79 GiB. Two opt-in levers:

- ``training.kbit_keep_16bit``: run PEFT's preparation (freezing, input grads, gradient
  checkpointing) without its fp32 cast, so the model stays in the 16-bit dtype it was loaded in,
  as the streamed path already runs it.
- ``training.embed_on_host``: keep the frozen input-embedding matrix in pinned host memory and
  copy only the looked-up rows to the GPU each step (the idea of the Soup Kernel Challenge entry
  of team AG TEAM, github.com/BAITC-Hacks/hack-2227fe74-ag-team, ``solution/vocab.py``,
  Apache-2.0; rewritten here for an untied embedding).
"""

from __future__ import annotations

from typing import Any

HOST_EMBED_ATTR = "_soup_embed_on_host"


def prepare_kbit(model: Any, use_gradient_checkpointing: bool, keep_16bit: bool = False) -> Any:
    """PEFT's ``prepare_model_for_kbit_training``; with ``keep_16bit`` its fp32 cast is skipped.

    PEFT casts ``param.data`` of every fp16/bf16 parameter that is not a ``Params4bit``. With
    ``keep_16bit`` each such parameter's data is swapped for an empty tensor of its own dtype for
    the duration of the call (the cast then touches zero elements) and put back afterwards, so
    everything else PEFT does (freezing, enable_input_require_grads, gradient checkpointing) is
    unchanged."""
    import torch
    from peft import prepare_model_for_kbit_training

    if not keep_16bit:
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    hidden = []
    for param in model.parameters():
        if (param.dtype in (torch.float16, torch.bfloat16)
                and param.__class__.__name__ != "Params4bit"):
            hidden.append((param, param.data))
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    try:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    finally:
        for param, data in hidden:
            param.data = data
    return model


def embedding_on_host(model: Any) -> int:
    """Move the frozen input embedding to pinned host memory; return the bytes freed on the GPU.

    The module keeps its identity (hooks registered on it, such as PEFT's
    ``enable_input_require_grads``, still fire on its output): only its weight moves and its
    ``forward`` looks the rows up on the CPU and returns them on the device of ``input_ids``."""
    import torch
    from torch.nn import functional

    emb = model.get_input_embeddings()
    if not isinstance(emb, torch.nn.Embedding):
        raise ValueError(
            f"training.embed_on_host needs an nn.Embedding input embedding, got {type(emb).__name__}"
        )
    head = model.get_output_embeddings()
    if head is not None and getattr(head, "weight", None) is emb.weight:
        raise ValueError(
            "training.embed_on_host needs an untied embedding: this model's lm_head shares "
            "the input-embedding matrix, which the head needs on the GPU"
        )
    if emb.weight.requires_grad:
        raise ValueError("training.embed_on_host keeps the embedding frozen; it is trainable here")
    if getattr(emb, HOST_EMBED_ATTR, False):
        return 0
    weight = emb.weight
    freed = weight.numel() * weight.element_size() if weight.is_cuda else 0
    host = weight.data.to("cpu")
    if torch.cuda.is_available():
        host = host.pin_memory()
    weight.data = host

    def forward(input_ids: Any) -> Any:
        rows = functional.embedding(
            input_ids.to("cpu"), emb.weight, emb.padding_idx, emb.max_norm,
            emb.norm_type, emb.scale_grad_by_freq, emb.sparse,
        )
        if input_ids.device.type != "cpu":
            if torch.cuda.is_available() and input_ids.is_cuda:
                rows = rows.pin_memory()
            rows = rows.to(input_ids.device, non_blocking=True)
        return rows

    emb.forward = forward
    setattr(emb, HOST_EMBED_ATTR, True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return freed
