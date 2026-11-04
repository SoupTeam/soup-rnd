#!/usr/bin/env python3
"""A4: the host-RAM budget for L2L's boundary activations, and the spill past it.

Layer-major micro-batching keeps ``n_layers x k`` boundary activations between
the forward walk and the backward walk (``l2l_activations.ActivationStore``).
This module decides, before anything is allocated, which of them stay in
page-locked RAM and which go to an unbuffered spill file, and what that costs in
REAL bytes: torch's caching host allocator rounds every pinned request up to the
next power of two (the ``torch.cuda.host_memory_stats`` docstring; measured in
#901, ``benchmarks/gate-901-14b-on-8gb.md``), and ``aligned_buffer`` asks for one
sector more than a chunk (#1531), so a staging buffer for an 8 MiB chunk
page-locks 16 MiB.

Who gets the RAM, in this order (the priority rule of
``benchmarks/probe-a4-l2l-ram-budget-spill.md`` §2):

1. the reserve: the OS and the margin the box needs, never lent out;
2. everything the process already holds that is not activations: the weight
   source's page-locked bytes as the shipped runtime accounts them
   (``StreamRuntime.stats()["pinned_bytes"]``, layer_stream_runtime.py:2966:
   the RAM store's power-of-two arenas, or the disk tier's staging), and the
   pageable rest (CUDA context, skeleton, adapters, optimizer);
3. the activation minimum: both staging rings, plus any per-chunk floor that
   never spills (G4's scales), so that a step at this k can run at all;
4. activation residency, layer by layer from the top, up to all-pinned;
5. what is left is spare: the term the C3 pinned-residency planner reads
   ("RAM spare after staging, arenas and (later) L2L activations").

Activations rank above C3 weight residency even though both save the same
traffic per byte: a resident activation byte saves one write and one read of
that byte per step, and a resident weight byte on the disk tier saves its
forward and its backward read. Only the activation traffic writes to the
drive. Writes cost drive endurance, and they can slow the weight reads that
share the drive, which the record measures (§2, G4).

Spill granularity is the LAYER, the lowest ones first, which is
``ActivationStore(spill_layers=)`` unchanged. The forward stores layer l's
input at visit l and the backward reads it at visit ``L - 1 - l`` of its
reverse walk, so the lowest layers are written first and read last: they have
the most compute between their write and their read. A whole layer at a time
wastes at most ``k - 1`` chunks of the budget against a per-chunk planner;
``ActivationPlan.granularity_waste_bytes`` states how much it is in each plan.

``chunk_bytes`` is an input, never derived from a dtype. A bf16 store passes
``seq x hidden x 2``. G4's compressed store passes its payload bytes as
``chunk_bytes`` (what a resident chunk pins and what the spill file gets per
spilled chunk) and its per-row scales as ``pinned_floor_bytes`` (one chunk's
scales; G4 keeps all ``n_layers x k`` of them in one pinned slab, spilled or
not). Both are UNROUNDED per-chunk request sizes. This module applies the
rounding, once per real allocation.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: ``l2l_activations.SECTOR_BYTES``: unbuffered I/O needs offsets, lengths and
#: addresses on this boundary, and ``aligned_buffer`` over-allocates one of it.
SECTOR_BYTES = 4096
#: ``l2l_activations.DEFAULT_RING``: staging buffers per direction.
DEFAULT_RING = 8

BUDGET_BELOW_MINIMUM = "BUDGET_BELOW_MINIMUM"
RAM_OVER_BUDGET = "RAM_OVER_BUDGET"
SPILL_DISK_SHORT = "SPILL_DISK_SHORT"
UNALIGNED_SPILL = "UNALIGNED_SPILL"

#: Arm budgets the record's rule names, as fractions of the all-pinned cost.
ARM_BUDGETS = ("1", "0.5", "0.25", "min")


class BudgetRefusalError(ValueError):
    """A plan that cannot run. ``code`` names the reason; the message carries the arithmetic."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _b(value: int) -> str:
    return f"{value:,} B ({value / 2**20:,.1f} MiB)"


def pinned_block_bytes(request: int) -> int:
    """Bytes one ``pin_memory=True`` allocation of ``request`` bytes page-locks.

    The caching host allocator rounds each block up to the next power of two.
    This mirrors ``soup_cli.utils.layer_stream_runtime._next_power_of_two``
    (layer_stream_runtime.py:552), which the shipped arena planner uses for the
    same allocator. A zero-byte request allocates nothing.
    """
    if request <= 0:
        return 0
    return 1 if request == 1 else 1 << (request - 1).bit_length()


def allocation_cost(request: int, *, pin: bool) -> int:
    """Host bytes an allocation really holds: rounded when pinned. The CPU
    allocator does not round (layer_stream_runtime.py:631)."""
    return pinned_block_bytes(request) if pin else max(0, int(request))


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _non_negative(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return int(value)


@dataclass(frozen=True)
class StoreLayout:
    """The host allocations one activation store makes, as ``ActivationStore`` makes them.

    l2l_activations.py:304-306 allocates one tensor per resident chunk;
    :320-321 allocates two rings of ``ring`` buffers, each
    ``aligned_buffer(chunk_bytes)`` = ``chunk_bytes + SECTOR_BYTES`` (:49),
    and only when some layer spills. ``pinned_floor_bytes`` is the per-chunk
    floor that never spills (G4's per-row scales). G4's ``CodecActivationStore``
    holds every chunk's floor in ONE slab of ``n_layers x k x pinned_floor_bytes``
    (g4_activation_codec.py:461), so the page-lock rounding applies once, to the
    slab, and the slab exists whether the payload spills or not.
    """

    n_layers: int
    k: int
    chunk_bytes: int
    pinned_floor_bytes: int = 0
    ring: int = DEFAULT_RING
    pin: bool = True

    def __post_init__(self) -> None:
        _positive("n_layers", self.n_layers)
        _positive("k", self.k)
        _positive("chunk_bytes", self.chunk_bytes)
        _non_negative("pinned_floor_bytes", self.pinned_floor_bytes)
        _positive("ring", self.ring)

    @property
    def floor_slab_bytes(self) -> int:
        """The request size of the one never-spilled floor slab (0: no slab)."""
        return self.n_layers * self.k * self.pinned_floor_bytes

    def allocations(self, spill_layers: int) -> List[Tuple[str, int, int]]:
        """``(kind, request bytes, count)`` for each kind of allocation the store makes."""
        if not 0 <= spill_layers <= self.n_layers:
            raise ValueError(f"spill_layers must be in [0, {self.n_layers}]; got {spill_layers}")
        out: List[Tuple[str, int, int]] = []
        resident = (self.n_layers - spill_layers) * self.k
        if resident:
            out.append(("resident_chunk", self.chunk_bytes, resident))
        if self.pinned_floor_bytes:
            out.append(("floor_slab", self.floor_slab_bytes, 1))
        if spill_layers:
            out.append(("ring_buffer", self.chunk_bytes + SECTOR_BYTES, 2 * self.ring))
        return out

    def host_bytes(self, spill_layers: int) -> int:
        return sum(allocation_cost(req, pin=self.pin) * n for _, req, n in self.allocations(
            spill_layers))

    def requested_bytes(self, spill_layers: int) -> int:
        return sum(req * n for _, req, n in self.allocations(spill_layers))

    @property
    def all_pinned_bytes(self) -> int:
        return self.host_bytes(0)

    @property
    def minimum_bytes(self) -> int:
        """Every layer spilled: the rings and the floor only."""
        return self.host_bytes(self.n_layers)

    def spill_file_bytes(self, spill_layers: int) -> int:
        return spill_layers * self.k * self.chunk_bytes


@dataclass(frozen=True)
class ActivationPlan:
    """Which chunks stay pinned, which spill, and what each side costs."""

    layout: StoreLayout
    budget_bytes: int
    spill_layers: int

    @property
    def n_layers(self) -> int:
        return self.layout.n_layers

    @property
    def k(self) -> int:
        return self.layout.k

    @property
    def chunk_bytes(self) -> int:
        return self.layout.chunk_bytes

    @property
    def ring(self) -> int:
        return self.layout.ring

    @property
    def pin(self) -> bool:
        return self.layout.pin

    @property
    def host_bytes(self) -> int:
        """Real host bytes of the store: page-locked rounding, rings and floor included."""
        return self.layout.host_bytes(self.spill_layers)

    @property
    def requested_bytes(self) -> int:
        return self.layout.requested_bytes(self.spill_layers)

    @property
    def spill_file_bytes(self) -> int:
        return self.layout.spill_file_bytes(self.spill_layers)

    @property
    def resident_layers(self) -> int:
        return self.n_layers - self.spill_layers

    @property
    def unused_bytes(self) -> int:
        return self.budget_bytes - self.host_bytes

    @property
    def granularity_waste_bytes(self) -> int:
        """Budget a per-chunk planner could still have filled with resident chunks
        (at most ``k - 1`` of them; with that many more it would hold a whole layer)."""
        if not self.spill_layers:
            return 0
        chunk = allocation_cost(self.chunk_bytes, pin=self.pin)
        return min(self.unused_bytes // chunk, self.k - 1) * chunk

    def placement(self, layer: int, mb: int) -> str:
        if not (0 <= layer < self.n_layers and 0 <= mb < self.k):
            raise IndexError(f"chunk ({layer}, {mb}) is outside {self.n_layers} x {self.k}")
        return "spill" if layer < self.spill_layers else "pinned"

    def spilled_chunks(self) -> List[Tuple[int, int]]:
        return [(layer, mb) for layer in range(self.spill_layers) for mb in range(self.k)]

    def pinned_chunks(self) -> List[Tuple[int, int]]:
        return [(layer, mb) for layer in range(self.spill_layers, self.n_layers)
                for mb in range(self.k)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers, "k": self.k, "chunk_bytes": self.chunk_bytes,
            "pinned_floor_bytes": self.layout.pinned_floor_bytes, "ring": self.ring,
            "pin": self.pin, "budget_bytes": self.budget_bytes,
            "spill_layers": self.spill_layers, "resident_layers": self.resident_layers,
            "host_bytes": self.host_bytes, "requested_bytes": self.requested_bytes,
            "all_pinned_bytes": self.layout.all_pinned_bytes,
            "minimum_bytes": self.layout.minimum_bytes,
            "spill_file_bytes": self.spill_file_bytes, "unused_bytes": self.unused_bytes,
            "granularity_waste_bytes": self.granularity_waste_bytes,
            "allocations": [
                {"kind": kind, "request_bytes": req, "count": n,
                 "host_bytes_each": allocation_cost(req, pin=self.pin)}
                for kind, req, n in self.layout.allocations(self.spill_layers)
            ],
        }


def _minimum_breakdown(layout: StoreLayout) -> str:
    ring = layout.chunk_bytes + SECTOR_BYTES
    each = allocation_cost(ring, pin=layout.pin)
    text = (f"2 staging rings x {layout.ring} buffers x {_b(each)} (each a {ring:,} B request: "
            f"{layout.chunk_bytes:,} B chunk + {SECTOR_BYTES} B alignment slack"
            + (", page-locked as the next power of two)" if layout.pin else ")"))
    if layout.pinned_floor_bytes:
        slab = layout.floor_slab_bytes
        text += (f" + one never-spilled floor slab of {layout.n_layers} x {layout.k} x "
                 f"{layout.pinned_floor_bytes:,} B = {slab:,} B, holding "
                 f"{_b(allocation_cost(slab, pin=layout.pin))}")
    return text


def plan_activations(
    budget_bytes: int,
    *,
    n_layers: int,
    k: int,
    chunk_bytes: int,
    pinned_floor_bytes: int = 0,
    ring: int = DEFAULT_RING,
    pin: bool = True,
    spill_free_bytes: Optional[int] = None,
    spill_reserve_bytes: int = 0,
    require_aligned: bool = True,
) -> ActivationPlan:
    """The plan for an activation budget of ``budget_bytes`` real host bytes.

    Keeps as many top layers pinned as fit and spills the rest. Raises
    ``BudgetRefusalError`` (``BUDGET_BELOW_MINIMUM``, ``UNALIGNED_SPILL``,
    ``SPILL_DISK_SHORT``) with the arithmetic in the message. ``spill_free_bytes``
    is the spill volume's free space (None: not checked); the file must leave
    ``spill_reserve_bytes`` of it free.
    """
    layout = StoreLayout(n_layers=n_layers, k=k, chunk_bytes=chunk_bytes,
                         pinned_floor_bytes=pinned_floor_bytes, ring=ring, pin=pin)
    budget = _non_negative("budget_bytes", budget_bytes)
    spill_reserve = _non_negative("spill_reserve_bytes", spill_reserve_bytes)
    if layout.all_pinned_bytes <= budget:
        return ActivationPlan(layout=layout, budget_bytes=budget, spill_layers=0)
    minimum = layout.minimum_bytes
    if budget < minimum:
        raise BudgetRefusalError(
            BUDGET_BELOW_MINIMUM,
            f"the activation budget is {_b(budget)}, but the smallest store for k={k} over "
            f"{n_layers} layers, every layer spilled, needs {_b(minimum)}: "
            f"{_minimum_breakdown(layout)}; {_b(minimum - budget)} short",
        )
    per_layer = k * allocation_cost(chunk_bytes, pin=pin)
    resident = min(n_layers - 1, (budget - minimum) // per_layer)
    plan = ActivationPlan(layout=layout, budget_bytes=budget, spill_layers=n_layers - resident)
    if require_aligned and chunk_bytes % SECTOR_BYTES:
        raise BudgetRefusalError(
            UNALIGNED_SPILL,
            f"{plan.spill_layers} layers must spill, but a chunk is {chunk_bytes:,} B, "
            f"{chunk_bytes % SECTOR_BYTES:,} B past a multiple of the {SECTOR_BYTES} B sector, "
            f"so an unbuffered spill file cannot hold it at aligned offsets",
        )
    if spill_free_bytes is not None:
        free = _non_negative("spill_free_bytes", spill_free_bytes)
        need = plan.spill_file_bytes + spill_reserve
        if need > free:
            raise BudgetRefusalError(
                SPILL_DISK_SHORT,
                f"the spill file needs {plan.spill_layers} layers x k={k} x {chunk_bytes:,} B "
                f"= {_b(plan.spill_file_bytes)}, plus a {_b(spill_reserve)} reserve, "
                f"= {_b(need)}; the spill volume has {_b(free)} free, {_b(need - free)} short",
            )
    return plan


def arm_budget_bytes(layout: StoreLayout, spec: str) -> int:
    """The activation budget a rule arm names: a fraction of the all-pinned
    cost (``"1"``, ``"0.5"``, ``"0.25"``) or ``"min"``, the smallest that runs."""
    if spec == "min":
        return layout.minimum_bytes
    fraction = float(spec)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"an arm budget is a fraction in (0, 1] or 'min'; got {spec!r}")
    return int(layout.all_pinned_bytes * fraction)


@dataclass(frozen=True)
class HostSplit:
    """A declared host-RAM total divided by the priority rule."""

    total_bytes: int
    reserve_bytes: int
    process_bytes: int
    weight_pinned_bytes: int
    plan: ActivationPlan

    @property
    def activation_budget_bytes(self) -> int:
        return self.plan.budget_bytes

    @property
    def peak_pinned_bytes(self) -> int:
        """Page-locked bytes at the step's peak: the weights' and the activations'."""
        return self.weight_pinned_bytes + (self.plan.host_bytes if self.plan.pin else 0)

    @property
    def planned_host_bytes(self) -> int:
        return (self.reserve_bytes + self.process_bytes + self.weight_pinned_bytes
                + self.plan.host_bytes)

    @property
    def spare_bytes(self) -> int:
        """What is left for C3 pinned weight residency after the activations."""
        return self.total_bytes - self.planned_host_bytes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes, "reserve_bytes": self.reserve_bytes,
            "process_bytes": self.process_bytes,
            "weight_pinned_bytes": self.weight_pinned_bytes,
            "activation_budget_bytes": self.activation_budget_bytes,
            "peak_pinned_bytes": self.peak_pinned_bytes,
            "planned_host_bytes": self.planned_host_bytes, "spare_bytes": self.spare_bytes,
            "plan": self.plan.as_dict(),
        }


def split_host_ram(
    total_bytes: int,
    *,
    reserve_bytes: int,
    process_bytes: int,
    weight_pinned_bytes: int,
    layout: StoreLayout,
    activation_cap_bytes: Optional[int] = None,
    spill_free_bytes: Optional[int] = None,
    spill_reserve_bytes: int = 0,
) -> HostSplit:
    """Divide ``total_bytes`` by the priority rule and plan the activations in their share.

    The activations get what the reserve, the process and the weight source
    leave, capped at ``activation_cap_bytes`` (a rule arm's budget) and at
    all-pinned; the rest is ``spare_bytes``. ``weight_pinned_bytes`` is the
    runtime's own figure, ``StreamRuntime.stats()["pinned_bytes"]``.
    """
    total = _non_negative("total_bytes", total_bytes)
    reserve = _non_negative("reserve_bytes", reserve_bytes)
    process = _non_negative("process_bytes", process_bytes)
    weights = _non_negative("weight_pinned_bytes", weight_pinned_bytes)
    left = total - reserve - process - weights
    # The smallest store that runs: every layer spilled, or all-pinned when that
    # is cheaper (a short stack at k=1 can cost less than the two staging rings).
    smallest = min(layout.minimum_bytes, layout.all_pinned_bytes)
    if left < smallest:
        raise BudgetRefusalError(
            RAM_OVER_BUDGET,
            f"{_b(total)} declared = {_b(reserve)} reserve + {_b(process)} process + "
            f"{_b(weights)} page-locked weights + {_b(max(left, 0))} for activations"
            + (f" ({_b(-left)} over before any activation)" if left < 0 else "")
            + f", but the smallest activation store needs {_b(smallest)}: "
            f"{_minimum_breakdown(layout)}",
        )
    share = min(left, layout.all_pinned_bytes)
    if activation_cap_bytes is not None:
        share = min(share, _non_negative("activation_cap_bytes", activation_cap_bytes))
    plan = plan_activations(
        share, n_layers=layout.n_layers, k=layout.k, chunk_bytes=layout.chunk_bytes,
        pinned_floor_bytes=layout.pinned_floor_bytes, ring=layout.ring, pin=layout.pin,
        spill_free_bytes=spill_free_bytes, spill_reserve_bytes=spill_reserve_bytes,
    )
    return HostSplit(total_bytes=total, reserve_bytes=reserve, process_bytes=process,
                     weight_pinned_bytes=weights, plan=plan)


def _refusal_at(budget: int, k: int, *, spill: bool, n_layers: int, chunk_bytes: int,
                pinned_floor_bytes: int, ring: int, pin: bool,
                spill_free_bytes: Optional[int]) -> Optional[str]:
    """None when ``k`` micro-batches fit ``budget``, else the code of the refusal."""
    layout = StoreLayout(n_layers=n_layers, k=k, chunk_bytes=chunk_bytes,
                         pinned_floor_bytes=pinned_floor_bytes, ring=ring, pin=pin)
    if not spill:
        return None if layout.all_pinned_bytes <= budget else RAM_OVER_BUDGET
    try:
        plan_activations(budget, n_layers=n_layers, k=k, chunk_bytes=chunk_bytes,
                         pinned_floor_bytes=pinned_floor_bytes, ring=ring, pin=pin,
                         spill_free_bytes=spill_free_bytes)
    except BudgetRefusalError as exc:
        return exc.code
    return None


_BOUNDS = {
    RAM_OVER_BUDGET: "all-pinned host bytes",
    BUDGET_BELOW_MINIMUM: "minimum store (rings and floors)",
    SPILL_DISK_SHORT: "spill volume free space",
    UNALIGNED_SPILL: "chunk not sector-aligned",
}


def max_k(
    activation_budget_bytes: int,
    *,
    n_layers: int,
    chunk_bytes: int,
    pinned_floor_bytes: int = 0,
    ring: int = DEFAULT_RING,
    pin: bool = True,
    spill: bool,
    spill_free_bytes: Optional[int] = None,
    k_limit: int = 1 << 20,
) -> Tuple[int, str]:
    """The largest k an activation budget admits, and what bounds it.

    Without spill, all-pinned must fit. With spill, ``plan_activations`` must
    accept the budget at that k, spill volume included. Every term grows with
    k (all-pinned, the floors, and the spill file, whose resident layers can
    only shrink as a layer gets wider), so a doubling then bisection search is
    exact. ``k_limit`` stands in for a bound this module does not see (VRAM,
    the dataset); reaching it is reported as such.
    """
    def refusal(k: int) -> Optional[str]:
        return _refusal_at(activation_budget_bytes, k, spill=spill, n_layers=n_layers,
                           chunk_bytes=chunk_bytes, pinned_floor_bytes=pinned_floor_bytes,
                           ring=ring, pin=pin, spill_free_bytes=spill_free_bytes)

    first = refusal(1)
    if first is not None:
        return 0, _BOUNDS[first]
    low, high = 1, 2
    while high <= k_limit and refusal(high) is None:
        low, high = high, high * 2
    high = min(high, k_limit + 1)
    while high - low > 1:
        mid = (low + high) // 2
        if refusal(mid) is None:
            low = mid
        else:
            high = mid
    if low >= k_limit:
        return k_limit, "k_limit"
    return low, _BOUNDS[refusal(low + 1) or RAM_OVER_BUDGET]


def tokens_per_step(
    activation_budget_bytes: int,
    *,
    n_layers: int,
    seq: int,
    chunk_bytes: int,
    pinned_floor_bytes: int = 0,
    ring: int = DEFAULT_RING,
    pin: bool = True,
    spill_free_bytes: Optional[int] = None,
    k_limit: int = 1 << 20,
) -> Dict[str, Any]:
    """Tokens per step a fixed activation budget reaches with and without spill.

    Arithmetic, not a measurement: k micro-batches of ``seq`` tokens each.
    """
    common = dict(n_layers=n_layers, chunk_bytes=chunk_bytes, pinned_floor_bytes=pinned_floor_bytes,
                  ring=ring, pin=pin, k_limit=k_limit)
    k_pinned, bound_pinned = max_k(activation_budget_bytes, spill=False, **common)
    k_spill, bound_spill = max_k(activation_budget_bytes, spill=True,
                                 spill_free_bytes=spill_free_bytes, **common)
    spill_file = 0
    if k_spill:
        spill_file = plan_activations(
            activation_budget_bytes, n_layers=n_layers, k=k_spill, chunk_bytes=chunk_bytes,
            pinned_floor_bytes=pinned_floor_bytes, ring=ring, pin=pin,
            spill_free_bytes=spill_free_bytes).spill_file_bytes
    return {
        "activation_budget_bytes": int(activation_budget_bytes), "seq": int(seq),
        "without_spill": {"k": k_pinned, "tokens": k_pinned * seq, "bound": bound_pinned},
        "with_spill": {"k": k_spill, "tokens": k_spill * seq, "bound": bound_spill,
                       "spill_file_bytes": spill_file},
    }


def budget_from_box(avail_bytes: Optional[int], commit_avail_bytes: Optional[int],
                    margin_bytes: int) -> int:
    """An activation budget read from the box after the model is built: free
    physical memory and (on Windows) free commit, the smaller of the two, minus
    a margin, as ``l2l_probe.fits_in_ram`` (V3) reads them. Unknown free memory
    is no budget at all: pinning blind can take the box and its peers down."""
    if avail_bytes is None:
        raise BudgetRefusalError(RAM_OVER_BUDGET, "free physical memory is unknown on this box")
    room = int(avail_bytes) if commit_avail_bytes is None else min(int(avail_bytes),
                                                                   int(commit_avail_bytes))
    return max(0, room - int(margin_bytes))
