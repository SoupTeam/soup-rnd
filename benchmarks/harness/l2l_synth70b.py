#!/usr/bin/env python3
"""A5 ticket 05: L2L on a synthetic 70B (random weights of the 70B shape), signed as such.

The checkpoint written by ``synth_checkpoint.py --shape llama-70b --vocab 32768`` has the
bytes per layer, the layer count and the GEMM shapes of Llama-3.1-70B and random values. A run
on it checks mechanics and memory use. It is not evidence about the quality or the speed of a
real 70B or of a 1T model, and nothing here scales a number to another model.

This module holds the parts a CPU can check: the layer shapes, the size of the activation
store, the resource plan with the refusal when RAM or disk is short, the price of a rental,
the F1 exclusion of spoiled runs and the report. Every heading and every table of the report
carries ``LABEL``. A record of another model (the 7B runs of tickets 03 and 04) is refused.
The run itself is ``l2l_ticket04_arm.py`` with ``l2l_synth70b/l2l.yaml``.

Benchmark harness code, not shipped. torch is not imported here.

``plan`` prints the plan and exits 3 when the machine is short; ``build`` joins an arm record
to the report payload; ``report`` renders it::

    python benchmarks/harness/l2l_synth70b.py plan --k 4 --seq 256 --tier disk --path /root/work
    python benchmarks/harness/l2l_synth70b.py report --record record.json --out report.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth_checkpoint  # noqa: E402

LABEL = "синтетический 70B"
MODEL_KIND = "synthetic-70b"

#: The A4 F2 shape: Llama-3.1-70B with a 32768-row head (the Mistral tokenizer fits in it).
SHAPE: Dict[str, int] = {**synth_checkpoint.SHAPES["llama-70b"], "vocab": 32768}

#: Disk for the source, the NF4 cache and the spill is multiplied by this.
DISK_MARGIN = 1.1
#: CUDA context, staging and Python, in bytes of host RAM.
WORKSPACE_BYTES = 1_000_000_000
#: NF4 with double quant, bytes per parameter (``layer_stream.py``).
NF4_BYTES_PER_PARAM = 0.5 + 1 / 64 + 4 / (64 * 256)
GIB = 2**30


class SynthRefusedError(RuntimeError):
    """The machine is too small for the plan: a missing precondition, not a result."""


class MixedRecordsError(ValueError):
    """The record is not a pure synthetic 70B record (another model, or an extrapolation)."""


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def layer_shapes(index: int) -> Dict[str, Tuple[int, ...]]:
    """Tensor name to shape for decoder layer ``index``."""
    return synth_checkpoint.layer_tensors(SHAPE, index)


def _numel(dims: Tuple[int, ...]) -> int:
    count = 1
    for dim in dims:
        count *= dim
    return count


def layer_param_count() -> int:
    return sum(_numel(dims) for dims in layer_shapes(0).values())


def _linear_param_count() -> int:
    return SHAPE["layers"] * sum(
        _numel(dims) for name, dims in layer_shapes(0).items() if "layernorm" not in name
    )


def _embedding_param_count() -> int:
    return 2 * SHAPE["vocab"] * SHAPE["hidden"]


def total_params() -> int:
    return SHAPE["layers"] * layer_param_count() + _embedding_param_count() + SHAPE["hidden"]


def weight_bytes() -> Dict[str, int]:
    """The bf16 checkpoint on disk and the NF4 cache the loader builds from it."""
    return {
        "source_bf16": 2 * total_params(),
        "nf4_cache": int(_linear_param_count() * NF4_BYTES_PER_PARAM)
        + 2 * _embedding_param_count(),
    }


# --------------------------------------------------------------------------
# activation store and the resource plan
# --------------------------------------------------------------------------
def activation_store(*, k: int, batch: int, seq: int, itemsize: int = 2) -> Dict[str, int]:
    """Size of the L2L store: ``layers x k`` chunks of ``batch x seq x hidden``.

    A pinned chunk takes a power-of-two block (#901); ``logical_bytes`` is the data alone.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    chunk = batch * seq * SHAPE["hidden"] * itemsize
    block = _next_power_of_two(chunk)
    return {
        "chunk_bytes": chunk,
        "block_bytes": block,
        "logical_bytes": SHAPE["layers"] * k * chunk,
        "pinned_bytes": SHAPE["layers"] * k * block,
    }


def resource_plan(
    *, k: int, batch: int, seq: int, tier: str, spill_layers: int = 0
) -> Dict[str, Any]:
    """Disk and host RAM the run needs, by the trainer's own budget formula.

    ``tier`` is where the weights are read from: ``disk`` keeps them on the drive, ``ram``
    pins the NF4 weights too. The first ``spill_layers`` layers of activations go to a spill
    file, the rest stay pinned and must stay under ``RAM_TIER_HEADROOM`` of free RAM.
    """
    if tier not in ("disk", "ram"):
        raise ValueError(f"tier must be 'disk' or 'ram', got {tier!r}")
    if not 0 <= spill_layers <= SHAPE["layers"]:
        raise ValueError(f"spill_layers must be in [0, {SHAPE['layers']}], got {spill_layers}")
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM

    store = activation_store(k=k, batch=batch, seq=seq)
    weights = weight_bytes()
    resident_layers = SHAPE["layers"] - spill_layers
    activation_need = resident_layers * k * store["block_bytes"] / RAM_TIER_HEADROOM
    ram_weights = weights["nf4_cache"] if tier == "ram" else 0
    spill = spill_layers * k * store["chunk_bytes"]
    disk_sum = weights["source_bf16"] + weights["nf4_cache"] + spill
    return {
        "k": k, "batch": batch, "seq": seq, "tier": tier, "spill_layers": spill_layers,
        "store": store,
        "ram": {
            "weights": ram_weights,
            "activation_need": activation_need,
            "workspace": WORKSPACE_BYTES,
            "total": ram_weights + activation_need + WORKSPACE_BYTES,
        },
        "disk": {
            "source_bf16": weights["source_bf16"],
            "nf4_cache": weights["nf4_cache"],
            "spill": spill,
            "margin": DISK_MARGIN,
            "total": disk_sum * DISK_MARGIN,
        },
    }


def check_resources(
    plan: Mapping[str, Any],
    *,
    free_ram: Optional[int],
    free_disk: Optional[int],
    total_ram: Optional[int] = None,
) -> None:
    """Refuse when RAM or disk is below the plan. An unknown amount is a refusal too.

    On the ``ram`` tier the pinned NF4 weights must also stay under
    ``PHYSICAL_RAM_TIER_HEADROOM`` of total RAM, as the trainer requires (#622).
    """
    from soup_cli.utils.layer_stream import (
        PHYSICAL_RAM_TIER_HEADROOM,
        PHYSICAL_RAM_TIER_HEADROOM_PERCENT,
    )

    problems: List[str] = []
    if plan["tier"] == "ram":
        if total_ram is None:
            problems.append("RAM: total physical RAM is unknown, the harness will not guess")
        elif plan["ram"]["weights"] > total_ram * PHYSICAL_RAM_TIER_HEADROOM:
            problems.append(
                f"RAM: pinned weights of {plan['ram']['weights'] / GIB:.1f} GiB exceed "
                f"{PHYSICAL_RAM_TIER_HEADROOM_PERCENT}% of {total_ram / GIB:.1f} GiB physical RAM"
            )
    need_ram, need_disk = plan["ram"]["total"], plan["disk"]["total"]
    if free_ram is None:
        problems.append("RAM: the free amount is unknown, the harness will not guess")
    elif free_ram < need_ram:
        problems.append(
            f"RAM: the plan needs {need_ram / GIB:.1f} GiB free, {free_ram / GIB:.1f} GiB is"
            " available"
        )
    if free_disk is None:
        problems.append("disk: the free amount is unknown, the harness will not guess")
    elif free_disk < need_disk:
        problems.append(
            f"disk: the plan needs {need_disk / GIB:.1f} GiB free, {free_disk / GIB:.1f} GiB is"
            " available"
        )
    if problems:
        raise SynthRefusedError("; ".join(problems))


def rental_cost(*, rate_per_hour: float, hours: float) -> float:
    if rate_per_hour < 0 or not hours > 0:
        raise ValueError(
            f"need a rate of 0 or more and hours above 0, got {rate_per_hour}, {hours}"
        )
    return rate_per_hour * hours


# --------------------------------------------------------------------------
# F1
# --------------------------------------------------------------------------
def split_runs(
    entries: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, str]]]:
    """Runs kept for the table and the spoiled ones with the reason F1 gave."""
    kept: List[Mapping[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    for entry in entries:
        if entry.get("spoiled"):
            reasons = entry.get("void_reasons") or {}
            why = "; ".join(f"{name}: {text}" for name, text in reasons.items())
            excluded.append({"run": str(entry.get("name")), "reason": why or "void"})
        else:
            kept.append(entry)
    return kept, excluded


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------
SPILL_NOT_WIRED = "spill is not wired into the trainer (A4 integration is outside A5)"


def build_record(
    arm: Mapping[str, Any],
    *,
    plan: Optional[Mapping[str, Any]],
    rental: Optional[Mapping[str, Any]],
    logs: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The report payload from one ``l2l_ticket04_arm.py`` record.

    Weight reads are the process read counter of the run (the spill is not wired into the
    trainer, so none of it is spill). The F1 ``entries`` name the runs to exclude.
    """
    if arm.get("failure"):
        raise ValueError(f"the run failed ({arm['failure']}); a failed run has no record")
    run = arm.get("run")
    if not run:
        raise ValueError("the arm has no run record (needs at least one step after step 0)")
    _, excluded = split_runs(entries)
    cost = None
    if rental:
        cost = rental_cost(rate_per_hour=rental["rate_per_hour"], hours=rental["hours"])
    unavailable: Dict[str, str] = {
        "spill_written_bytes": SPILL_NOT_WIRED,
        "spill_read_bytes": SPILL_NOT_WIRED,
    }
    if run.get("os_read_bytes") is None:
        unavailable["weight_read_bytes"] = "the process read counter is missing"
    return {
        "model": MODEL_KIND,
        "run": dict(run),
        "gpu": arm.get("gpu"),
        "weight_read_bytes": run.get("os_read_bytes"),
        "spill_written_bytes": None,
        "spill_read_bytes": None,
        "unavailable": unavailable,
        "plan": plan,
        "rental": None if rental is None else {**rental, "cost_usd": cost},
        "excluded": excluded,
        "logs": list(logs),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _gib(value: Optional[int]) -> str:
    return "н/д" if value is None else f"{value / GIB:.2f} GiB"


def _heading(level: int, text: str) -> str:
    return f"{'#' * level} {text} — {LABEL}"


def _table(caption: str, rows: Sequence[Tuple[str, str]]) -> List[str]:
    lines = [f"**{caption} — {LABEL}**", "", "| Показатель | Значение |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines + [""]


def _value(
    payload: Mapping[str, Any], key: str, text: Any, unavailable: Mapping[str, str]
) -> str:
    if text is None:
        reason = unavailable.get(key)
        if not reason:
            raise ValueError(f"{key} is null without a reason in 'unavailable'")
        return f"не измерено ({reason})"
    return str(text)


def _check_record(payload: Mapping[str, Any]) -> None:
    if payload.get("model") != MODEL_KIND:
        raise MixedRecordsError(
            f"the record is for model {payload.get('model')!r}, not {MODEL_KIND!r}; results of"
            " other models do not go into this table"
        )
    extra = [key for key in payload if key.startswith("extrapol")]
    if extra:
        raise MixedRecordsError(f"no extrapolation to other models: {', '.join(extra)}")


def render_report(payload: Mapping[str, Any]) -> str:
    """The report as Markdown. Raises for a record that is not a pure synthetic 70B record."""
    _check_record(payload)
    run = payload["run"]
    unavailable = payload.get("unavailable") or {}
    lines: List[str] = [_heading(1, "L2L на случайных весах формы 70B"), ""]
    lines += [
        f"Это {LABEL}: случайные веса формы Llama-3.1-70B (80 слоёв, hidden 8192,"
        f" head {SHAPE['vocab']} строк), настоящая модель не загружалась. Прогон проверяет"
        " механику L2L и расход памяти. Это не доказательство качества или скорости"
        " настоящей 70B и тем более 1T. Числа на другие модели не переносятся.",
        "",
    ]
    if payload.get("gpu"):
        lines += [f"Карта: {payload['gpu'].get('name')}.", ""]

    lines += [_heading(2, "Замеры одного прогона"), ""]

    def mem(key: str) -> str:
        return _value(payload, key, None if run.get(key) is None else _gib(run[key]), unavailable)

    rows = [
        ("время шага", _value(payload, "step_s_mean", None if run.get("step_s_mean") is None
                              else f"{run['step_s_mean']:.2f} с", unavailable)),
        ("токены без паддинга в секунду", _value(
            payload, "tokens_per_s", None if run.get("tokens_per_s") is None
            else f"{run['tokens_per_s']:.2f}", unavailable)),
        ("пик VRAM allocated", mem("peak_vram_allocated_bytes")),
        ("пик VRAM reserved", mem("peak_vram_reserved_bytes")),
        ("пик RSS процесса", mem("peak_rss_bytes")),
        ("закреплённая память хранилища активаций", mem("pinned_store_bytes")),
        ("чтение весов с диска", _value(
            payload, "weight_read_bytes",
            None if payload.get("weight_read_bytes") is None
            else _gib(payload["weight_read_bytes"]), unavailable)),
        ("спилл записано", _value(
            payload, "spill_written_bytes",
            None if payload.get("spill_written_bytes") is None
            else _gib(payload["spill_written_bytes"]), unavailable)),
        ("спилл прочитано", _value(
            payload, "spill_read_bytes",
            None if payload.get("spill_read_bytes") is None
            else _gib(payload["spill_read_bytes"]), unavailable)),
        ("число загрузок слоя за шаг", _value(
            payload, "layer_loads_per_step", run.get("layer_loads_per_step"), unavailable)),
        ("восстановления NF4", _value(payload, "nf4_restores", run.get("nf4_restores"),
                                       unavailable)),
    ]
    lines += _table("Скорость, GPU, RAM и ввод-вывод", rows)
    if payload.get("plan"):
        plan = payload["plan"]
        lines += [_heading(2, "Расчёт ресурсов"), ""]
        lines += _table("Диск и RAM по расчёту", [
            ("хранилище активаций закреплено", _gib(plan["store"]["pinned_bytes"])),
            ("RAM нужно свободно", _gib(int(plan["ram"]["total"]))),
            ("диск под веса bf16", _gib(plan["disk"]["source_bf16"])),
            ("диск под кэш NF4", _gib(plan["disk"]["nf4_cache"])),
            ("диск нужно свободно (запас x" + f"{plan['disk']['margin']})",
             _gib(int(plan["disk"]["total"]))),
        ])
    rental = payload.get("rental")
    if rental:
        lines += [_heading(2, "Аренда"), ""]
        lines += _table("Цена и бюджет", [
            ("инстанс", str(rental.get("instance"))),
            ("оффер", str(rental.get("offer"))),
            ("цена в час, USD", str(rental.get("rate_per_hour"))),
            ("часов", str(rental.get("hours"))),
            ("стоимость, USD", str(rental.get("cost_usd"))),
            ("бюджет, подтверждённый пользователем, USD", str(rental.get("budget_usd"))),
        ])
        if rental.get("destroyed"):
            lines += ["Инстанс уничтожен, `vastai show instances` подтвердил.", ""]
        else:
            lines += ["Уничтожение инстанса НЕ подтверждено.", ""]
    lines += [_heading(2, "Исключённые замеры и журналы"), ""]
    excluded = payload.get("excluded") or []
    if excluded:
        lines += [f"- {item['run']}: {item['reason']}" for item in excluded]
    else:
        lines += ["Испорченных по F1 замеров нет."]
    lines += [""] + [f"- журнал: {path}" for path in payload.get("logs") or []]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _free_ram() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def _total_ram() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().total)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan", help="print the plan and check this machine against it")
    plan_cmd.add_argument("--k", type=int, required=True)
    plan_cmd.add_argument("--batch", type=int, default=1)
    plan_cmd.add_argument("--seq", type=int, required=True)
    plan_cmd.add_argument("--tier", choices=("disk", "ram"), default="disk")
    plan_cmd.add_argument("--spill-layers", type=int, default=0)
    plan_cmd.add_argument("--path", default=".", help="directory whose free disk is checked")
    build_cmd = sub.add_parser("build", help="turn an arm run.json into a report record")
    build_cmd.add_argument("--arm", required=True, help="run.json of l2l_ticket04_arm.py")
    build_cmd.add_argument("--plan", default=None, help="JSON printed by the plan command")
    build_cmd.add_argument("--rental", default=None, help="JSON: instance, offer, rate_per_hour,"
                           " hours, budget_usd, destroyed")
    build_cmd.add_argument("--entries", default=None, help="JSON list of F1 run entries")
    build_cmd.add_argument("--log", action="append", default=[])
    build_cmd.add_argument("--out", required=True)
    report_cmd = sub.add_parser("report", help="render a record as the labelled report")
    report_cmd.add_argument("--record", required=True)
    report_cmd.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = resource_plan(k=args.k, batch=args.batch, seq=args.seq, tier=args.tier,
                             spill_layers=args.spill_layers)
        print(json.dumps(plan, indent=1))
        try:
            check_resources(plan, free_ram=_free_ram(),
                            free_disk=shutil.disk_usage(args.path).free,
                            total_ram=_total_ram())
        except SynthRefusedError as exc:
            print(f"REFUSED: {exc}")
            return 3
        print("the machine fits the plan")
        return 0
    if args.command == "build":
        def load(path: Optional[str]) -> Any:
            return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))

        record = build_record(load(args.arm), plan=load(args.plan), rental=load(args.rental),
                              logs=args.log, entries=load(args.entries) or [])
        target = Path(args.out)
        if target.exists():
            print(f"{target} exists; a record is never overwritten")
            return 2
        target.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n",
                          encoding="utf-8")
        print(f"wrote {target}")
        return 0
    payload = json.loads(Path(args.record).read_text(encoding="utf-8"))
    target = Path(args.out)
    if target.exists():
        print(f"{target} exists; a report is never overwritten")
        return 2
    target.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
