"""Bounded packed NF4 transfers, with explicit host and device reuse fences.

Experimental direct API only. Each ticket owns a slot until dequantization has
finished reading it. Dense decoded weights are separate allocations. No worker
thread, model arithmetic, cache admission or routing decision lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def packing_layout(tensors: Mapping[str, Any]) -> tuple[list[tuple], int]:
    """Preserve exact bytes and align typed views, including scalar sidecars."""
    layout, cursor = [], 0
    if not tensors:
        raise ValueError("an expert transfer must be nonempty")
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.numel() == 0:
            raise ValueError("expert transfer requires nonempty contiguous CPU tensors")
        width = tensor.element_size()
        # Match standalone CUDA allocator alignment, including vectorized NF4 loads.
        alignment = max(256, width)
        cursor = -(-cursor // alignment) * alignment
        count = tensor.numel() * width
        layout.append((name, cursor, count, tensor.dtype, tuple(tensor.shape)))
        cursor += count
    return layout, cursor


@dataclass
class TransferTicket:
    slot: Any
    views: dict[str, Any]
    released: bool = False


class PackedExpertTransfer:
    """Two pinned/GPU byte slots; one coalesced H2D copy per expert ticket.

    Host reuse waits for the previous H2D completion. GPU reuse waits on the
    consumer's post-dequantization event, not merely H2D completion. Close drains
    only these events. This object is sequential and is not thread-safe.
    """

    def __init__(self, device: str, maximum_bytes: int = 16 * 2**20) -> None:
        import torch

        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be a positive integer")
        if torch.device(device).type != "cuda":
            raise ValueError("packed expert transfer requires CUDA")
        self.device = device
        self.maximum_bytes = maximum_bytes
        with torch.cuda.device(device):
            self.stream = torch.cuda.Stream(device=device)
        self.slots: list[dict[str, Any]] = []
        self.cursor = 0
        self.copies = 0
        self.bytes_copied = 0
        self.closed = False

    def _drain(self) -> None:
        for slot in self.slots:
            for name in ("ready", "consumed"):
                event = slot.get(name)
                if event is not None:
                    event.synchronize()

    def _allocate(self, size: int) -> None:
        import torch

        if size > self.maximum_bytes:
            raise ValueError("expert packed payload exceeds the configured transfer bound")
        if self.slots and self.slots[0]["host"].numel() >= size:
            return
        if any(slot["leased"] for slot in self.slots):
            raise RuntimeError("cannot resize an outstanding expert transfer")
        self._drain()
        self.slots.clear()
        for _ in range(2):
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            # Allocator reuse / deterministic empty fills also have stream dependencies.
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                device = torch.empty(size, dtype=torch.uint8, device=self.device)
            device.record_stream(self.stream)
            self.slots.append({"host": host, "device": device, "ready": None,
                               "consumed": None, "leased": False})

    def stage(self, tensors: Mapping[str, Any]) -> TransferTicket:
        import torch

        if self.closed:
            raise RuntimeError("packed expert transfer is closed")
        layout, size = packing_layout(tensors)
        self._allocate(size)
        slot = self.slots[self.cursor % 2]
        self.cursor += 1
        if slot["leased"]:
            raise RuntimeError("expert transfer slot still has an outstanding consumer")
        if slot["ready"] is not None and not slot["ready"].query():
            slot["ready"].synchronize()
        for name, offset, count, _dtype, _shape in layout:
            slot["host"][offset:offset + count].copy_(tensors[name].reshape(-1).view(torch.uint8))
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if slot["consumed"] is not None:
                self.stream.wait_event(slot["consumed"])
            slot["device"][:size].copy_(slot["host"][:size], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.stream)
            slot["ready"] = ready
        slot["leased"] = True
        self.copies += 1
        self.bytes_copied += size
        views = {name: slot["device"][offset:offset + count].view(dtype).view(shape)
                 for name, offset, count, dtype, shape in layout}
        return TransferTicket(slot, views)

    def consume(self, ticket: TransferTicket) -> dict[str, Any]:
        import torch

        if self.closed or ticket.released:
            raise RuntimeError("expert transfer ticket was already released")
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(ticket.slot["ready"])
        ticket.slot["device"].record_stream(consumer)
        return ticket.views

    def release(self, ticket: TransferTicket, *, consumed: bool) -> None:
        import torch

        if ticket.released:
            return
        if consumed:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self.device))
        else:
            done = ticket.slot["ready"]
        ticket.slot["consumed"] = done
        ticket.slot["leased"] = False
        ticket.released = True

    def stats(self) -> dict[str, int]:
        size = sum(slot["host"].numel() for slot in self.slots)
        return {"pinned_host_bytes": size, "device_staging_bytes": size,
                "h2d_copies": self.copies, "h2d_bytes": self.bytes_copied}

    def close(self) -> None:
        if not self.closed:
            self._drain()
            self.slots.clear()
            self.closed = True
