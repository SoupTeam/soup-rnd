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
* A cross-device copy in torch is a two-way barrier between the two devices' current streams.
  :class:`Transfers` runs every copy on side streams instead and hands each result over with
  an event, so a copy waits only for what it reads and blocks only what reads it: layer i+1
  crosses to the extra devices while layer i computes, and the activations cross without
  stalling the other device. ``SOUP_L2L_SPLIT_OVERLAP=0`` turns that off (copies on the
  compute streams, issued when the layer starts), for an A/B against the overlap.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Upper bound of ``training.l2l_split_gpus``.
MAX_SPLIT_GPUS = 8
#: ``0`` keeps every cross-device copy on the compute streams (no overlap); for experiments.
OVERLAP_ENV = "SOUP_L2L_SPLIT_OVERLAP"


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

    def weights(self, idx: int, layer: Any, moved: Dict[str, Any]) -> Dict[str, Any]:
        """Layer ``idx``'s weights on this device: the pool's buffers already copied here
        (:func:`copy_weights`; NF4 ones rebuilt as ``Params4bit`` views of the copies) plus
        the resident replicas."""
        from soup_cli.utils.layer_stream_runtime import rebuild_params4bit

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


def overlap_enabled(main: Any) -> bool:
    """Side-stream copies: on CUDA, unless ``SOUP_L2L_SPLIT_OVERLAP=0``."""
    import torch

    return torch.device(main).type == "cuda" and os.environ.get(OVERLAP_ENV, "1") != "0"


class Transfers:
    """Cross-device copies on side streams, handed over with events.

    Two side streams per device, one for weights and one for activations, so a weight
    prefetch never queues behind an activation that waits for the other device. Without
    overlap (CPU, or ``SOUP_L2L_SPLIT_OVERLAP=0``) every method is the plain ``.to``.
    """

    def __init__(self, main: Any, devices: Sequence[Any]) -> None:
        import torch

        self.main = torch.device(main)
        self.enabled = overlap_enabled(self.main)
        self._side: Dict[Tuple[str, Any], Any] = {}
        if self.enabled:
            for dev in (self.main, *[torch.device(d) for d in devices]):
                for purpose in ("weights", "acts"):
                    self._side[(purpose, dev)] = torch.cuda.Stream(dev)

    def side(self, purpose: str, device: Any) -> Any:
        """The side stream of ``purpose`` on ``device`` (None without overlap)."""
        import torch

        return self._side.get((purpose, torch.device(device)))

    def _copy(self, tensor: Any, dst: Any, purpose: str, ready: Any) -> Tuple[Any, Any, Any]:
        """Copy ``tensor`` to ``dst`` on the side streams after ``ready`` (an event on the
        source device). Returns (copy, done-on-source event, done-on-destination event)."""
        import torch

        s_src, s_dst = self.side(purpose, tensor.device), self.side(purpose, dst)
        if ready is not None:
            s_src.wait_event(ready)
        with torch.cuda.stream(s_src), torch.cuda.stream(s_dst):
            out = tensor.to(dst, non_blocking=True)
            src_done = torch.cuda.Event()
            src_done.record(s_src)
            dst_done = torch.cuda.Event()
            dst_done.record(s_dst)
        tensor.record_stream(s_src)
        return out, src_done, dst_done

    def act(self, tensor: Any, dst: Any, consumer: Optional[Any] = None) -> Any:
        """``tensor`` on ``dst``, ready for ``consumer`` (default: ``dst``'s current stream).

        The copy waits for the work queued so far on the source's current stream, which is
        what produced ``tensor``; nothing else on either device waits for it."""
        import torch

        dst = torch.device(dst)
        if not self.enabled or tensor.device == dst:
            return tensor.to(dst, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
        out, _src_done, dst_done = self._copy(tensor, dst, "acts", ready)
        consumer = consumer if consumer is not None else torch.cuda.current_stream(dst)
        consumer.wait_event(dst_done)
        out.record_stream(consumer)
        return out


class WeightCopy:
    """One layer's pool buffers on one extra device, and when the copy is done."""

    def __init__(self, moved: Dict[str, Any], src_done: Any = None, dst_done: Any = None) -> None:
        self.moved = moved
        self.src_done = src_done
        self.dst_done = dst_done


def copy_weights(
    transfers: Transfers, buffers: Dict[str, Any], device: Any, ready: Any = None
) -> WeightCopy:
    """The pool buffers of a layer copied to ``device``.

    With overlap the copies run on the weight side streams after ``ready`` (the pool slot's
    load event); the caller makes the main compute stream wait ``src_done`` before the slot
    can be reloaded and the device's compute stream wait ``dst_done`` before using it."""
    import torch

    if not transfers.enabled:
        return WeightCopy({n: t.to(device, non_blocking=True) for n, t in buffers.items()})
    moved: Dict[str, Any] = {}
    src_done = dst_done = None
    s_src = transfers.side("weights", transfers.main)
    if ready is not None:
        s_src.wait_event(ready)
    for name, tensor in buffers.items():
        moved[name], src_done, dst_done = transfers._copy(tensor, torch.device(device),
                                                          "weights", None)
    return WeightCopy(moved, src_done, dst_done)
