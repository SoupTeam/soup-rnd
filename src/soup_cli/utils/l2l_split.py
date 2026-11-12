"""One L2L step split over several GPUs in one process (``training.l2l_split_gpus``).

ZERO (Kaggle T4 x2, Qwen3.8-27B NF4): a 27B step on one T4 is bound by compute about as much
as by the weight reads, and two data-parallel processes read every layer twice and do not
fit the host's 32 GB. Here one process reads each layer once into the main GPU's pool, copies
it to the other GPUs, and each GPU runs its share of the window's micro-batches through it:
the compute splits, the reads do not double.

* The micro-batches are dealt round robin (:func:`assign`); the main device keeps the
  embedding, the head and the activation store, so activations cross between cards at the
  layer boundary only.
* An extra device runs the layer's own module through ``torch.func.functional_call`` with
  the streamed weights copied out of the pool, rebuilt as ``Params4bit`` views there when
  they are NF4, and with replicas of the layer's resident tensors: the adapters (leaves
  that collect their own gradients) and anything else not streamed.
* The replicas are refreshed from the adapters at every step and their gradients are added
  into the main adapters' ``.grad`` at its end, before the optimizer (and the GradScaler's
  unscale) sees them.
* A cross-device copy in torch waits on both devices' streams, so every copy of a layer is
  issued before either device's compute for it; the two devices then compute in parallel.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Sequence, Tuple

#: Upper bound of ``training.l2l_split_gpus``.
MAX_SPLIT_GPUS = 8


def assign(k: int, n_devices: int) -> List[int]:
    """The device index of each of ``k`` micro-batches, dealt round robin."""
    return [mb % max(1, int(n_devices)) for mb in range(int(k))]


def to_device(value: Any, device: Any) -> Any:
    """``value`` with every tensor in nested tuples, lists and dicts moved to ``device``."""
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def split_devices(device: Any, split_gpus: int) -> Tuple[str, ...]:
    """The extra devices of a split run on main ``device``: the next cards by index."""
    if int(split_gpus) <= 1:
        return ()
    text = str(device)
    first = int(text.split(":", 1)[1]) if ":" in text else 0
    return tuple(f"cuda:{first + i}" for i in range(1, int(split_gpus)))


def check_split(device: Any, split_gpus: int) -> None:
    """Refuse a split the launch cannot give: no CUDA, too few cards, a distributed launch."""
    if int(split_gpus) <= 1:
        return
    from soup_cli.utils.stream_ddp import distributed_world_size

    if not str(device).startswith("cuda"):
        raise ValueError(
            f"training.l2l_split_gpus={split_gpus} needs CUDA devices; this run is on "
            f"{device!r}. Set l2l_split_gpus to 1."
        )
    import torch

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible < int(split_gpus):
        raise ValueError(
            f"training.l2l_split_gpus={split_gpus} needs {split_gpus} GPUs in this process; "
            f"{visible} are visible."
        )
    if distributed_world_size() > 1:
        raise ValueError(
            "training.l2l_split_gpus splits one process's window over its GPUs; it does not "
            "combine with a data-parallel launch (`soup train --gpus N`). Use one or the "
            "other."
        )


def keep_hf_on_one_gpu(training_args: Any, split_gpus: int) -> None:
    """A split run is one process that owns several cards; HF must not wrap the model in
    ``nn.DataParallel`` or multiply the batch by the card count, so it sees one GPU."""
    if int(split_gpus) > 1:
        training_args._n_gpu = 1


def _device_context(device: Any) -> Any:
    """``torch.cuda.device(device)`` for a CUDA device, so kernels that launch on the
    current device (Triton, bitsandbytes) land on it; nothing for a CPU device."""
    import torch

    dev = torch.device(device)
    return torch.cuda.device(dev) if dev.type == "cuda" else nullcontext()


class DeviceReplica:
    """What an extra device needs to run the decoder layers: replicas of their resident
    tensors (the adapters and anything not streamed) and the NF4 code tables."""

    def __init__(self, layers: Sequence[Any], device: Any) -> None:
        import torch

        self.device = torch.device(device)
        #: Per layer: name (relative to ``layer.inner``) -> (main parameter, replica leaf).
        self.trainable: List[Dict[str, Tuple[Any, Any]]] = []
        #: Per layer: name -> copy of a frozen resident parameter or buffer.
        self.frozen: List[Dict[str, Any]] = []
        self._codes: Dict[int, Dict[str, Any]] = {}
        for layer in layers:
            inner = layer.inner
            trainable: Dict[str, Tuple[Any, Any]] = {}
            frozen: Dict[str, Any] = {}
            named = list(inner.named_parameters(remove_duplicate=False))
            named += list(inner.named_buffers(remove_duplicate=False))
            for name, tensor in named:
                if tensor.is_meta or name in layer.name_map:
                    continue
                if getattr(tensor, "requires_grad", False):
                    replica = tensor.detach().to(self.device, copy=True).requires_grad_(True)
                    trainable[name] = (tensor, replica)
                else:
                    frozen[name] = tensor.detach().to(self.device, copy=True)
            self.trainable.append(trainable)
            self.frozen.append(frozen)

    def refresh(self) -> None:
        """Copy the adapters' current values into the replicas and clear their gradients."""
        import torch

        with torch.no_grad():
            for layer in self.trainable:
                for param, replica in layer.values():
                    replica.copy_(param.detach())
                    replica.grad = None

    def merge_grads(self) -> None:
        """Add each replica's gradient into its main parameter's ``.grad``."""
        import torch

        with torch.no_grad():
            for layer in self.trainable:
                for param, replica in layer.values():
                    if replica.grad is None:
                        continue
                    grad = replica.grad.to(param.device, dtype=param.dtype)
                    if param.grad is None:
                        param.grad = grad.clone() if grad is replica.grad else grad
                    else:
                        param.grad.add_(grad)
                    replica.grad = None

    def _codes_for(self, layer: Any) -> Dict[str, Any]:
        key = id(layer.codes)
        codes = self._codes.get(key)
        if codes is None:
            codes = {name: value.to(self.device) for name, value in layer.codes.items()}
            self._codes[key] = codes
        return codes

    def weights(self, idx: int, layer: Any, buffers: Dict[str, Any]) -> Dict[str, Any]:
        """Layer ``idx``'s weights on this device: the pool's buffers copied over (NF4 ones
        rebuilt as ``Params4bit`` views of the copies) plus the resident replicas."""
        from soup_cli.utils.layer_stream_runtime import rebuild_params4bit

        moved = {name: tensor.to(self.device, non_blocking=True)
                 for name, tensor in buffers.items()}
        out: Dict[str, Any] = {}
        codes = self._codes_for(layer) if layer.quant_specs else {}
        for meta, ckpt in layer.name_map.items():
            spec = layer.quant_specs.get(ckpt)
            out[meta] = moved[ckpt] if spec is None else rebuild_params4bit(
                ckpt, moved, spec, codes
            )
        out.update(self.frozen[idx])
        out.update({name: replica for name, (_p, replica) in self.trainable[idx].items()})
        return out

    def forward(self, layer: Any, weights: Dict[str, Any], hidden: Any,
                args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """The layer's own module on this device, with ``weights`` substituted."""
        from torch.func import functional_call

        with _device_context(self.device):
            out = functional_call(layer.inner, weights, (hidden, *args), kwargs)
        return out[0] if isinstance(out, tuple) else out
