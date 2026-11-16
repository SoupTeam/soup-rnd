"""Layer streaming under a distributed data-parallel launch (``soup train --gpus N``).

Each rank streams the frozen base layer by layer on its own GPU and trains a full copy of
the LoRA adapters on its share of the batch; DDP averages the adapter gradients. Three
things stand between a streamed model and that:

* DDP synchronises every parameter and buffer of the module at wrap time and refuses a
  module whose parameters sit on two device types. A streamed decoder keeps the frozen base
  on ``meta`` by design. Only the trainable adapters need synchronising (the base is the
  same checkpoint on every rank), so everything else goes on the module's
  ``_ddp_params_and_buffers_to_ignore`` list, which ``DistributedDataParallel`` honours.
* A plain ``"cuda"`` device would put every rank on GPU 0 before accelerate pins them.
* Every rank prepares the same layer-shard cache; two writers on one directory race. The
  write is serialised by a file lock, and the second rank finds the finished cache.

ZERO (Kaggle T4 x2): a 27B step on one T4 is bound by GPU compute, not by the weight reads,
so a second card is the lever that remains.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

#: The attribute ``torch.nn.parallel.DistributedDataParallel`` reads on the wrapped module.
DDP_IGNORE_ATTR = "_ddp_params_and_buffers_to_ignore"


def distributed_world_size() -> int:
    """``WORLD_SIZE`` of a torchrun / accelerate launch; 1 when absent or malformed."""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1") or "1"))
    except (TypeError, ValueError):
        return 1


def local_rank() -> Optional[int]:
    """This process's ``LOCAL_RANK`` in a distributed launch, else None."""
    if distributed_world_size() <= 1:
        return None
    try:
        return int(os.environ.get("LOCAL_RANK", ""))
    except (TypeError, ValueError):
        return None


def pin_rank_device(device: Any) -> Any:
    """``cuda:<LOCAL_RANK>`` for an unindexed CUDA device in a distributed launch.

    The current CUDA device is set too, so code that allocates on ``"cuda"`` lands on this
    rank's card. Anything else (CPU, an explicit index, no distributed launch) is returned
    unchanged.
    """
    rank = local_rank()
    if rank is None or str(device) != "cuda":
        return device
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return f"cuda:{rank}"


@contextlib.contextmanager
def shard_cache_lock(shard_dir: str) -> Iterator[None]:
    """Serialise the layer-shard write across ranks; a no-op outside a distributed launch."""
    if distributed_world_size() <= 1:
        yield
        return
    from filelock import FileLock

    parent = os.path.dirname(os.path.abspath(shard_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    with FileLock(os.path.abspath(shard_dir).rstrip(os.sep) + ".lock"):
        yield


def ignore_frozen_for_ddp(model: Any) -> int:
    """Put every frozen parameter and every buffer of ``model`` on DDP's ignore list.

    Returns how many names were listed. Names are relative to ``model``, the module the
    trainer hands to DDP. Trainable parameters stay synchronised.
    """
    names = [name for name, param in model.named_parameters() if not param.requires_grad]
    names += [name for name, _ in model.named_buffers()]
    setattr(model, DDP_IGNORE_ATTR, names)
    return len(names)
