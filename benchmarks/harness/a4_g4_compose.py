#!/usr/bin/env python3
"""A4 x G4 composition smoke: G4's compressed boundary store, sized by A4's planner, spilled
through A4's real unbuffered spill file, and compared with the plan row by row.

For every row (a chunk shape, a G4 codec, a source dtype) and every arm budget of the
A4 rule (``1``, ``0.5``, ``0.25``, ``min``):

1. G4's byte functions give one chunk's payload and scales (``payload_chunk_bytes``,
   ``scale_chunk_bytes``, ``compressed_chunk_bytes``), UNROUNDED.
2. A4's ``plan_activations`` takes the payload as ``chunk_bytes`` and the scales as
   ``pinned_floor_bytes`` and decides ``spill_layers``, or refuses with a code. When it
   refuses ``UNALIGNED_SPILL``, G4's store must refuse the same spill (ValueError), before
   it opens any file.
3. A G4 ``CodecActivationStore`` is built with the plan's ``spill_layers`` and ring, its
   spill file opened by ``a4_spill_store.open_spill`` (``DirectSpillFile`` by default: the
   page cache bypassed) inside ``TimedSpill``.
4. The allocations are read from the tensors' storages (resident payload chunks, the two
   staging rings, the scale slab) and compared kind by kind with the plan's allocation
   list, in total with ``plan.host_bytes`` and the budget, and with G4's own
   ``store.host_bytes()`` report; the spill file's size with ``plan.spill_file_bytes``.
5. ``put`` for every (layer, micro-batch) in forward order, ``get`` in backward order,
   ``steps`` times: the payload each ``get`` hands back is bit-identical to what G4's
   ``encode`` made of the same input, the restored activation bit-identical to
   ``decode`` of it, and each step writes and reads exactly ``plan.spill_file_bytes``.
6. After ``close``: no spill file, no spill thread.

SYNTHETIC / CPU PILOT. CPU-only torch cannot page-lock, so the stores are pageable and
their measured bytes are compared with a ``pin=False`` plan. The ``pinned`` columns apply
A4's power-of-two rule (#901) to the MEASURED storage sizes and compare that with the
``pin=True`` arithmetic at the same placement: it checks that A4 rounds the right
requests, not what torch's host allocator does. Usage::

    PYTHONPATH=src python benchmarks/harness/a4_g4_compose.py \\
        --g4-harness ../soup-rnd-g4/benchmarks/harness --spill-dir .soup/compose-spill \\
        --out compose.json

Exit 0 when every row agrees, 1 when one does not, 3 when the spill volume is refused.
Benchmark harness code, not shipped. Standard library at the top level; torch inside
functions.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_activation_budget as budget  # noqa: E402
import a4_box  # noqa: E402
import a4_spill_store  # noqa: E402
import l2l_activations  # noqa: E402

HARNESS = Path(__file__).resolve().parent
LABEL = "SYNTHETIC/CPU PILOT: G4+A4 composition smoke, no verdict"
SPECS = budget.ARM_BUDGETS
DTYPE_BYTES = {"bfloat16": 2, "float32": 4}
_COUNTER = itertools.count()


class Row(collections.namedtuple("Row", "name shape codec dtype k")):
    """One composition row: a (1, seq, hidden) chunk, a G4 codec, the source dtype, k."""


#: Mistral-7B at seq 512 (the G4/A4 fixture), the SmolLM2-135M CPU-pilot shape, and int4
#: at odd hidden sizes: two whose packed payload is sector-aligned, one whose is not.
ROWS: Tuple[Row, ...] = (
    Row("mistral7b_int8", (1, 512, 4096), "int8", "bfloat16", 2),
    Row("mistral7b_fp8_e4m3", (1, 512, 4096), "fp8_e4m3", "bfloat16", 2),
    Row("mistral7b_identity_bf16", (1, 512, 4096), "bf16", "bfloat16", 2),
    Row("smollm2_identity_float32", (1, 64, 576), "bf16", "float32", 3),
    Row("smollm2_int8", (1, 64, 576), "int8", "float32", 3),
    Row("int4_odd_hidden_63", (1, 256, 63), "int4", "float32", 3),
    Row("int4_odd_hidden_4095", (1, 256, 4095), "int4", "float32", 2),
    Row("int4_odd_hidden_577_unaligned", (1, 64, 577), "int4", "float32", 3),
)


# ==========================================================================
# G4, loaded from its own harness directory
# ==========================================================================
def find_g4_harness(explicit: Optional[str] = None) -> Optional[Path]:
    """``explicit``, else this harness directory (after the tracks merge), else
    ``A4_G4_HARNESS``, else the sibling G4 worktree; None when none holds the codec."""
    candidates = [explicit] if explicit else [
        str(HARNESS), os.environ.get("A4_G4_HARNESS"),
        str(HARNESS.parents[2] / "soup-rnd-g4" / "benchmarks" / "harness")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "g4_activation_codec.py").is_file():
            return Path(os.path.realpath(candidate))
    return None


def load_g4(harness: Path) -> Any:
    """Import ``g4_activation_codec`` from ``harness``. Its ``import l2l_activations``
    resolves to the module already imported here, so both sides must carry the same file."""
    mine = Path(l2l_activations.__file__).resolve()
    theirs = harness / "l2l_activations.py"
    if theirs.is_file() and sha256(theirs) != sha256(mine):
        raise RuntimeError(f"{theirs} differs from {mine}: the composition would run G4's "
                           f"store over another ActivationStore than G4 tested")
    spec = importlib.util.spec_from_file_location("g4_activation_codec",
                                                  harness / "g4_activation_codec.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g4_activation_codec"] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==========================================================================
# pure arithmetic
# ==========================================================================
def chunk_sizes(g4: Any, shape: Tuple[int, ...], codec: str, source_element_bytes: int
                ) -> Dict[str, int]:
    """G4's unrounded bytes of one chunk."""
    return {
        "payload": int(g4.payload_chunk_bytes(shape, codec,
                                              source_element_bytes=source_element_bytes)),
        "scale": int(g4.scale_chunk_bytes(shape, codec)),
        "compressed": int(g4.compressed_chunk_bytes(shape, codec,
                                                    source_element_bytes=source_element_bytes)),
    }


def layout_of(sizes: Dict[str, int], *, n_layers: int, k: int, ring: int, pin: bool
              ) -> budget.StoreLayout:
    return budget.StoreLayout(n_layers=n_layers, k=k, chunk_bytes=sizes["payload"],
                              pinned_floor_bytes=sizes["scale"], ring=ring, pin=pin)


def plan_for(sizes: Dict[str, int], spec: str, *, n_layers: int, k: int, ring: int, pin: bool,
             spill_free_bytes: Optional[int] = None
             ) -> Tuple[int, Optional[budget.ActivationPlan], Optional[Dict[str, str]]]:
    """``(budget bytes, plan, refusal)``: payload as ``chunk_bytes``, scales as
    ``pinned_floor_bytes``, neither rounded first."""
    layout = layout_of(sizes, n_layers=n_layers, k=k, ring=ring, pin=pin)
    budget_bytes = budget.arm_budget_bytes(layout, spec)
    try:
        plan = budget.plan_activations(
            budget_bytes, n_layers=n_layers, k=k, chunk_bytes=sizes["payload"],
            pinned_floor_bytes=sizes["scale"], ring=ring, pin=pin,
            spill_free_bytes=spill_free_bytes)
    except budget.BudgetRefusalError as exc:
        return budget_bytes, None, {"code": exc.code, "message": str(exc)}
    return budget_bytes, plan, None


def expected_allocations(plan: budget.ActivationPlan) -> Dict[str, List[int]]:
    """The request size of every allocation the plan says the store makes, by kind."""
    out: Dict[str, List[int]] = {}
    for kind, request, count in plan.layout.allocations(plan.spill_layers):
        out.setdefault(kind, []).extend([request] * count)
    return out


def pinned_equivalent(sizes: Sequence[int]) -> int:
    """What the measured requests would page-lock under A4's power-of-two rule."""
    return sum(budget.pinned_block_bytes(int(size)) for size in sizes)


# ==========================================================================
# the store
# ==========================================================================
def measured_allocations(store: Any) -> Dict[str, List[int]]:
    """Storage bytes of every host allocation a G4 store holds, read from the tensors."""
    inner = store._inner
    out: Dict[str, List[int]] = {}
    resident = [int(t.untyped_storage().nbytes()) for t in inner._resident.values()]
    if resident:
        out["resident_chunk"] = resident
    if store._scale_slab is not None:
        out["floor_slab"] = [int(store._scale_slab.untyped_storage().nbytes())]
    if inner.spill_layers:
        out["ring_buffer"] = [int(buf.untyped_storage().nbytes())
                              for ring in (inner._write_ring, inner._read_ring)
                              for buf, _event in list(ring._free.queue)]
    return out


def _bits(tensor: Any) -> Any:
    import torch

    return tensor.detach().contiguous().view(torch.uint8)


def _source(shape: Tuple[int, ...], dtype: Any, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).mul_(3.0).to(dtype)


def _spill_threads() -> List[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("l2l-spill-")]


def g4_refuses_spill(g4: Any, row: Row, dtype: Any, spill_layers: int, ring: int,
                     n_layers: int) -> Dict[str, Any]:
    """Whether G4's store refuses ``spill_layers`` for this row, and opened no file."""
    opened: List[int] = []

    def factory(size: int) -> Any:
        opened.append(size)
        raise AssertionError("G4 opened a spill file for a payload it must refuse")

    try:
        store = g4.CodecActivationStore(
            codec=row.codec, n_layers=n_layers, k=row.k, chunk_shape=row.shape, dtype=dtype,
            device="cpu", pin=False, spill_layers=spill_layers, spill_factory=factory, ring=ring)
    except ValueError as exc:
        return {"refused": True, "message": str(exc), "opened_file": bool(opened)}
    store.close()
    return {"refused": False, "message": None, "opened_file": bool(opened)}


def run_spec(g4: Any, row: Row, spec: str, *, n_layers: int, ring: int, steps: int,
             spill_dir: str, backend: str) -> Dict[str, Any]:
    import torch

    dtype = getattr(torch, row.dtype)
    sizes = chunk_sizes(g4, row.shape, row.codec, DTYPE_BYTES[row.dtype])
    os.makedirs(spill_dir, exist_ok=True)
    free = a4_spill_store.free_bytes(spill_dir)
    budget_bytes, plan, refusal = plan_for(sizes, spec, n_layers=n_layers, k=row.k, ring=ring,
                                           pin=False, spill_free_bytes=free)
    cuda_budget, cuda_plan, cuda_refusal = plan_for(sizes, spec, n_layers=n_layers, k=row.k,
                                                    ring=ring, pin=True)
    out: Dict[str, Any] = {
        "spec": spec, "budget_bytes": budget_bytes,
        "plan": plan.as_dict() if plan else None, "refusal": refusal,
        "cuda_arithmetic": {"budget_bytes": cuda_budget,
                            "plan": cuda_plan.as_dict() if cuda_plan else None,
                            "refusal": cuda_refusal},
        "checks": {},
    }
    checks = out["checks"]
    if plan is None:
        if refusal["code"] == budget.UNALIGNED_SPILL:
            # What the planner would spill if alignment did not matter: G4 must refuse it.
            layout = layout_of(sizes, n_layers=n_layers, k=row.k, ring=ring, pin=False)
            loose = budget.plan_activations(
                budget_bytes, n_layers=n_layers, k=row.k, chunk_bytes=sizes["payload"],
                pinned_floor_bytes=sizes["scale"], ring=ring, pin=False, require_aligned=False)
            g4_side = g4_refuses_spill(g4, row, dtype, loose.spill_layers, ring, n_layers)
            out["g4_refusal"] = {**g4_side, "spill_layers": loose.spill_layers,
                                 "minimum_bytes": layout.minimum_bytes}
            checks["g4_refuses_the_same_unaligned_spill"] = (
                g4_side["refused"] and "4096" in (g4_side["message"] or "")
                and not g4_side["opened_file"])
        else:
            checks["refused_" + refusal["code"]] = True
        out["ok"] = all(checks.values())
        return out

    opened: Dict[str, Any] = {}

    def factory(size: int) -> Any:
        opened["requested_bytes"] = int(size)
        path = os.path.join(spill_dir, f"a4-spill-compose-{os.getpid()}-{next(_COUNTER)}.bin")
        opened["path"] = path
        opened["timed"] = a4_spill_store.TimedSpill(
            a4_spill_store.open_spill(backend, path, int(size)))
        return opened["timed"]

    store = g4.CodecActivationStore(
        codec=row.codec, n_layers=n_layers, k=row.k, chunk_shape=row.shape, dtype=dtype,
        device="cpu", pin=False, spill_layers=plan.spill_layers,
        spill_factory=factory if plan.spill_layers else None, ring=plan.ring)
    seen: Dict[Tuple[int, int], Any] = {}
    try:
        inner_get = store._inner.get

        def recording_get(layer: int, mb: int) -> Any:
            payload = inner_get(layer, mb)
            seen[(layer, mb)] = payload.clone()
            return payload

        store._inner.get = recording_get
        checks["g4_sizes_are_the_store_sizes"] = (
            store.payload_chunk_bytes == sizes["payload"]
            and store.scale_chunk_bytes == sizes["scale"]
            and store.chunk_bytes == sizes["compressed"] == sizes["payload"] + sizes["scale"])
        checks["planner_got_unrounded_sizes"] = (
            plan.chunk_bytes == sizes["payload"]
            and plan.layout.pinned_floor_bytes == sizes["scale"])
        measured = measured_allocations(store)
        expected = expected_allocations(plan)
        held = sum(sum(sizes_) for sizes_ in measured.values())
        cuda_layout = layout_of(sizes, n_layers=n_layers, k=row.k, ring=ring, pin=True)
        report = store.host_bytes()
        out["measured"] = {
            "allocations": {kind: {"count": len(v), "bytes_each": sorted(set(v)),
                                   "bytes": sum(v)} for kind, v in measured.items()},
            "expected": {kind: {"count": len(v), "bytes_each": sorted(set(v)), "bytes": sum(v)}
                         for kind, v in expected.items()},
            "pageable_bytes": held, "plan_host_bytes": plan.host_bytes,
            "pinned_equivalent_bytes": pinned_equivalent(
                [s for v in measured.values() for s in v]),
            "pinned_plan_arithmetic_bytes": cuda_layout.host_bytes(plan.spill_layers),
            "g4_report": report,
        }
        checks["allocations_match_kind_by_kind"] = (
            {kind: sorted(v) for kind, v in measured.items()}
            == {kind: sorted(v) for kind, v in expected.items()})
        checks["held_equals_plan"] = held == plan.host_bytes
        checks["held_within_budget"] = held <= budget_bytes
        checks["pinned_rounding_of_measured_equals_pinned_plan"] = (
            out["measured"]["pinned_equivalent_bytes"]
            == out["measured"]["pinned_plan_arithmetic_bytes"])
        checks["g4_report_matches_plan"] = (
            report["host_bytes"] == plan.requested_bytes
            and report["scale_bytes"] == plan.layout.floor_slab_bytes
            and report["spilled_payload_bytes"] == plan.spill_file_bytes
            and report["resident_chunks"] == plan.resident_layers * row.k)
        timed = opened.get("timed")
        if plan.spill_layers:
            writer = timed.inner._writer
            out["spill"] = {"path": opened["path"], "backend": backend,
                            "direct": bool(timed.direct),
                            "requested_bytes": opened["requested_bytes"],
                            "file_bytes": int(os.fstat(writer.fileno()).st_size),
                            "name_visible_while_open": os.path.exists(opened["path"])}
            checks["spill_request_equals_plan"] = opened["requested_bytes"] == plan.spill_file_bytes
            checks["spill_file_size_equals_plan"] = (
                out["spill"]["file_bytes"] == plan.spill_file_bytes)
            checks["spill_file_is_sector_aligned"] = plan.spill_file_bytes % 4096 == 0
            checks["spill_backend_as_requested"] = timed.direct == (backend == "direct")
        else:
            checks["no_spill_file_opened"] = "timed" not in opened
        payload_ok = decoded_ok = True
        steps_io = []
        hidden = int(row.shape[-1])
        for step in range(steps):
            before = timed.io() if timed is not None else None
            started = time.perf_counter()
            for layer in range(n_layers):
                for mb in range(row.k):
                    store.put(layer, mb, _source(row.shape, dtype, step * 10_000 + layer * 10 + mb))
            store.begin_backward()
            for layer in range(n_layers - 1, -1, -1):
                for mb in range(row.k):
                    restored = store.get(layer, mb)
                    x = _source(row.shape, dtype, step * 10_000 + layer * 10 + mb)
                    payload, scale = g4.encode(x, row.codec)
                    got = seen.pop((layer, mb))
                    payload_ok &= torch.equal(_bits(got), _bits(payload))
                    want = g4.decode(payload, scale, row.codec, dtype, hidden)
                    decoded_ok &= torch.equal(_bits(restored), _bits(want))
            store.end_step()
            elapsed = time.perf_counter() - started
            if timed is not None:
                after = timed.io()
                steps_io.append({"written": after["bytes_written"] - before["bytes_written"],
                                 "read": after["bytes_read"] - before["bytes_read"],
                                 "seconds": elapsed})
            else:
                steps_io.append({"written": 0, "read": 0, "seconds": elapsed})
        out["steps"] = steps_io
        out["stats"] = {"inner_bytes_written": store.stats.bytes_written,
                        "inner_bytes_read": store.stats.bytes_read,
                        "resident_chunks": store.stats.resident_chunks,
                        "spilled_chunks": store.stats.spilled_chunks}
        checks["payload_round_trip_bit_exact"] = bool(payload_ok)
        checks["restored_equals_g4_decode_bit_exact"] = bool(decoded_ok)
        checks["every_step_moves_exactly_the_spill_file"] = all(
            s["written"] == s["read"] == plan.spill_file_bytes for s in steps_io)
        checks["store_counters_agree"] = (
            store.stats.bytes_written == store.stats.bytes_read == steps * plan.spill_file_bytes)
    finally:
        store.close()
    path = opened.get("path")
    out["after_close"] = {
        "file_exists": bool(path) and os.path.exists(path),
        "left_behind": opened["timed"].left_behind if "timed" in opened else None,
        "spill_threads": _spill_threads(),
        "leftover_spill_files": a4_spill_store.leftover_spill_files(spill_dir),
    }
    checks["cleanup_leaves_nothing"] = (not out["after_close"]["file_exists"]
                                        and out["after_close"]["left_behind"] is None
                                        and not out["after_close"]["spill_threads"]
                                        and not out["after_close"]["leftover_spill_files"])
    out["ok"] = all(checks.values())
    return out


def run_row(g4: Any, row: Row, *, n_layers: int, ring: int, steps: int, spill_dir: str,
            backend: str, specs: Sequence[str] = SPECS,
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    sizes = chunk_sizes(g4, row.shape, row.codec, DTYPE_BYTES[row.dtype])
    result: Dict[str, Any] = {"row": row.name, "shape": list(row.shape), "codec": row.codec,
                              "dtype": row.dtype, "k": row.k, "n_layers": n_layers,
                              "ring": ring, "source_element_bytes": DTYPE_BYTES[row.dtype],
                              "sizes": sizes, "specs": []}
    for spec in specs:
        try:
            entry = run_spec(g4, row, spec, n_layers=n_layers, ring=ring, steps=steps,
                             spill_dir=spill_dir, backend=backend)
        except Exception as exc:  # noqa: BLE001 — kept in the record, the row fails
            entry = {"spec": spec, "error": f"{type(exc).__name__}: {exc}", "ok": False,
                     "checks": {}}
        result["specs"].append(entry)
        failed = [name for name, ok in entry["checks"].items() if not ok]
        plan = entry.get("plan")
        what = (f"spill {plan['spill_layers']}/{n_layers} layers, host "
                f"{entry['measured']['pageable_bytes']:,} B = plan {plan['host_bytes']:,} B "
                f"<= budget {entry['budget_bytes']:,} B" if plan and "measured" in entry
                else entry.get("error") or f"refused {(entry.get('refusal') or {}).get('code')}")
        log(f"{row.name:<31} {spec:<4} {what}  -> {'OK' if entry['ok'] else 'MISMATCH'}"
            + (f" {failed}" if failed else ""))
    result["ok"] = all(entry["ok"] for entry in result["specs"])
    return result


# ==========================================================================
# main
# ==========================================================================
def _git_head(path: Path) -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--g4-harness", required=True,
                        help="the G4 benchmarks/harness directory (holds g4_activation_codec.py)")
    parser.add_argument("--spill-dir", required=True, help="scratch directory for spill files")
    parser.add_argument("--out", required=True, help="the JSON record (never overwritten)")
    parser.add_argument("--backend", choices=a4_spill_store.BACKENDS, default="direct")
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--ring", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rows", default=",".join(row.name for row in ROWS))
    parser.add_argument("--min-free-ram-bytes", type=int, default=1 << 30,
                        help="refuse to start with less free physical RAM than this")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 2
    by_name = {row.name: row for row in ROWS}
    names = [name.strip() for name in args.rows.split(",") if name.strip()]
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        print(f"unknown rows {unknown}; known {sorted(by_name)}")
        return 2
    harness = find_g4_harness(args.g4_harness)
    if harness is None:
        print(f"no g4_activation_codec.py under {args.g4_harness}")
        return 2
    import psutil
    import torch

    free_ram = int(psutil.virtual_memory().available)
    if free_ram < args.min_free_ram_bytes:
        print(f"refusing: {free_ram:,} B of RAM free, under --min-free-ram-bytes "
              f"{args.min_free_ram_bytes:,}")
        return 2
    os.makedirs(args.spill_dir, exist_ok=True)
    volume = a4_box.spill_volume(args.spill_dir)
    if args.backend == "direct" and not volume["ok"]:
        print("spill volume refused: " + "; ".join(volume["refusals"]))
        return 3
    g4 = load_g4(harness)
    meta = {
        "label": LABEL, "argv": list(argv) if argv is not None else sys.argv[1:],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "free_ram_bytes_at_start": free_ram,
        "a4_head": _git_head(HARNESS), "g4_head": _git_head(harness),
        "a4_harness": str(HARNESS), "g4_harness": str(harness),
        "sha256": {"g4_activation_codec.py": sha256(harness / "g4_activation_codec.py"),
                   "l2l_activations.py (used)": sha256(Path(l2l_activations.__file__)),
                   "a4_activation_budget.py": sha256(HARNESS / "a4_activation_budget.py"),
                   "a4_spill_store.py": sha256(HARNESS / "a4_spill_store.py"),
                   "a4_g4_compose.py": sha256(Path(__file__))},
        "spill_volume": volume, "backend": args.backend,
        "pinned_note": "CPU-only torch: stores are pageable; pinned columns are A4's "
                       "power-of-two rule applied to measured request sizes, not a reading "
                       "of torch's host allocator",
    }
    print(f"{LABEL}\nG4 {harness} ({meta['g4_head']}), A4 {HARNESS} ({meta['a4_head']}), "
          f"backend {args.backend}, spill volume {volume['fstype']} {volume['scope']} "
          f"direct_io={volume['direct_io']}")
    rows = [run_row(g4, by_name[name], n_layers=args.n_layers, ring=args.ring,
                    steps=args.steps, spill_dir=args.spill_dir, backend=args.backend)
            for name in names]
    record = {"meta": meta, "rows": rows, "ok": all(row["ok"] for row in rows),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mismatched = [row["row"] for row in rows if not row["ok"]]
    print(f"{len(rows)} rows, {sum(len(r['specs']) for r in rows)} budgets: "
          + ("every row agrees with the plan" if not mismatched else f"MISMATCH in {mismatched}"))
    print(f"wrote {out}")
    return 0 if not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
