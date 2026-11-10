"""B1.4 ticket 09: write a native DeepSeek-V3 checkpoint of any size, one tensor at a time.

The tiny fixture of ``b13_moe_fixture`` instantiates the whole model, so it cannot make a layer
bigger than host memory. This writer builds the model on the ``meta`` device, draws each tensor
on CPU, and writes the per-expert keys
``model.layers.<n>.mlp.experts.<id>.{gate,up,down}_proj.weight`` into safetensors shards. Peak
memory is one shard (at most ``SHARD_BYTES``), not the model.

The weights are random: they carry no language and no training. They exist to give the streamed
path tensors of the right shapes and sizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SHARD_BYTES = 2_000_000_000
#: Draw scales. Experts and dense weights match a normal init; the router is wide enough that
#: the top-k picks differ between tokens; the router bias is non-zero so its path is used.
DENSE_STD = 0.02
ROUTER_STD = 0.05
BIAS_STD = 0.3


def expert_layer_bytes(config: Any, *, dtype: str) -> int:
    """Bytes of the routed experts of one MoE layer: gate, up and down for every expert."""
    import torch

    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    return (
        config.n_routed_experts * 3 * config.hidden_size * config.moe_intermediate_size
        * itemsize
    )


def _draw(shape: Any, std: float, generator: Any, dtype: Any) -> Any:
    import torch

    return (torch.randn(shape, generator=generator, dtype=torch.float32) * std).to(dtype)


class _ShardWriter:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.pending: Dict[str, Any] = {}
        self.pending_bytes = 0
        self.weight_map: Dict[str, str] = {}
        self.total = 0
        self.count = 0

    def add(self, name: str, tensor: Any) -> None:
        self.pending[name] = tensor.contiguous()
        size = tensor.numel() * tensor.element_size()
        self.pending_bytes += size
        self.total += size
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        from safetensors.torch import save_file

        file_name = f"model-{self.count:05d}.safetensors"
        save_file(self.pending, str(self.folder / file_name), metadata={"format": "pt"})
        for name in self.pending:
            self.weight_map[name] = file_name
        self.count += 1
        self.pending, self.pending_bytes = {}, 0

    def close(self) -> None:
        self.flush()
        index = {"metadata": {"total_size": self.total}, "weight_map": self.weight_map}
        (self.folder / "model.safetensors.index.json").write_text(json.dumps(index))


def write_checkpoint(root: Path, config: Any, *, dtype: str = "bfloat16", seed: int = 0) -> Path:
    """Write ``config`` and random weights of ``dtype`` under ``root / "weights"``; return it."""
    import torch
    from transformers import AutoModelForCausalLM

    weights = Path(root) / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype)
    config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
    config.save_pretrained(weights)
    generator = torch.Generator().manual_seed(seed)
    writer = _ShardWriter(weights)
    for name, tensor in model.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            experts, double, hidden = tensor.shape
            inner = double // 2
            for expert in range(experts):
                gate_up = _draw((double, hidden), DENSE_STD, generator, torch_dtype)
                writer.add(f"{prefix}{expert}.gate_proj.weight", gate_up[:inner].clone())
                writer.add(f"{prefix}{expert}.up_proj.weight", gate_up[inner:].clone())
        elif name.endswith("mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for expert in range(tensor.shape[0]):
                writer.add(
                    f"{prefix}{expert}.down_proj.weight",
                    _draw(tensor.shape[1:], DENSE_STD, generator, torch_dtype),
                )
        elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            writer.add(name, torch.ones(tensor.shape, dtype=torch_dtype))
        elif name.endswith("e_score_correction_bias"):
            writer.add(name, _draw(tensor.shape, BIAS_STD, generator, torch_dtype))
        elif name.endswith("mlp.gate.weight"):
            writer.add(name, _draw(tensor.shape, ROUTER_STD, generator, torch_dtype))
        else:
            writer.add(name, _draw(tensor.shape, DENSE_STD, generator, torch_dtype))
    writer.close()
    return weights
