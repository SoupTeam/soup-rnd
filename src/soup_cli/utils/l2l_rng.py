"""The random state of one decoder-layer call in an L2L step.

``lora.dropout`` > 0 draws a mask from the global generator on every layer call. Gradient
accumulation visits the (micro-batch, layer) pairs micro-batch by micro-batch; the L2L step
visits them layer by layer, and calls each layer a second time in the backward to recompute
it. One shared stream cannot give both orders the same masks, and saving the state per pair
does not help while the forward order differs.

So the state is not a position in a stream. It is a function of the call:
``layer_seed(seed, step, mb, layer)``. The L2L forward, the L2L recompute in the backward
and the accumulation reference (a forward pre-hook on every layer) all set the generator
from it before the layer runs. Same call, same masks, in any visiting order, and nothing
has to be stored between the forward and the recompute.

Where ``seed`` comes from: ``resolve_training_seed`` (``utils/seeding.py``). ``step`` is
``trainer.state.global_step``, ``mb`` the position in the accumulation window and ``layer``
the index in the decoder container.

Not the same as a stock HF Trainer: that one draws every mask from a single stream, so a
run with ``lora.dropout`` > 0 matches the accumulation reference that installs this
function, not an unpatched Trainer.

torch is imported inside functions.
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from typing import Iterator


def layer_seed(seed: int, step: int, mb: int, layer: int) -> int:
    """A 63-bit generator seed for one (seed, step, micro-batch, layer) call.

    Hashed rather than summed, so neighbouring calls do not get neighbouring seeds, and
    stable across processes (Python's ``hash`` is salted).
    """
    packed = struct.pack("<qqqq", int(seed), int(step), int(mb), int(layer))
    digest = hashlib.blake2b(packed, digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def set_layer_state(seed: int, step: int, mb: int, layer: int) -> None:
    """Set the global generators to this call's state. Nothing is restored afterwards."""
    import torch

    torch.manual_seed(layer_seed(seed, step, mb, layer))


@contextmanager
def layer_rng(seed: int, step: int, mb: int, layer: int) -> Iterator[None]:
    """Run a layer call at its state, then put the surrounding stream back.

    ``torch.manual_seed`` seeds the CPU generator and every initialised CUDA card, so all
    of them are forked and restored, not only the card of the layer's input. A CPU call
    on a machine with an initialised GPU would otherwise leave that GPU's stream moved.
    """
    import warnings

    import torch

    cards = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with warnings.catch_warnings():
        # fork_rng warns when it forks more than one card; that is the intent here
        warnings.simplefilter("ignore")
        with torch.random.fork_rng(devices=cards):
            set_layer_state(seed, step, mb, layer)
            yield
