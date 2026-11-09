"""B5 ticket 02: judge of adapter gradients and updates, with a real/synthetic signature.

The judge reads run journals (JSON Lines, the event names of the B6 journal) and answers
pass or fail on two questions: the adapter gradients are finite and nonzero, and the adapters
after the steps differ from the initial ones. A journal with no ``real`` or ``synthetic``
signature fails, and so does one report that mixes both kinds.

The rule, with the reasons for each number, is
``benchmarks/b5-adapter-judge-rule-20261109.md``. It was fixed before the first GPU run.
The numbers below must not be tuned after a run.

Benchmark harness code, not shipped. torch is not imported here; ``b5_recorder.py`` writes
the journal.

    python benchmarks/harness/b5_judge.py judge --journal events.jsonl [--journal more.jsonl]
    python benchmarks/harness/b5_judge.py template --kind real > record.json
    python benchmarks/harness/b5_judge.py report --record record.json --out report.md

The report template is ``render_report``. Every heading and every table caption ends with the
signature, as in the synthetic 70B report of A5, and the report cannot be rendered without the
command, the versions, the hardware and the links to the source logs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Global L2 norm of the adapter gradients, before clipping. About 8 float32 epsilons (2**-23).
GRADIENT_NORM_MIN = 1e-6
#: ``distance_from_initial / max(initial_norm, 1)``. The same 8 epsilons.
RELATIVE_MOVE_MIN = 1e-6
#: One step cannot show that the second gradient is computed on already updated weights.
MIN_STEPS = 2

#: Signature kind to the words the report must carry.
KINDS: Dict[str, str] = {"real": "реальная модель", "synthetic": "синтетические веса"}
#: What each kind must name about its weights.
SOURCE_FIELD: Dict[str, str] = {"real": "revision", "synthetic": "generator"}

#: Environment fields every report states, or marks unavailable with a reason.
ENVIRONMENT_FIELDS = ("torch", "transformers", "peft", "bitsandbytes", "gpu", "ram")

_UNPARSABLE = "_unparsable"


class MixedRecordsError(ValueError):
    """The record mixes real and synthetic results, or its verdict is for the other kind."""


def read_journal(path: Any) -> List[Dict[str, Any]]:
    """Events of a JSONL file. A line that is not a JSON object becomes an ``_unparsable`` row."""
    rows: List[Dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if not isinstance(row, dict):
            rows.append({"event": _UNPARSABLE, "line": number})
        else:
            rows.append(row)
    return rows


def _number(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _nonfinite_fields(row: Mapping[str, Any]) -> List[str]:
    return [key for key, value in row.items()
            if isinstance(value, float) and not math.isfinite(value)]


class _Failures:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.items: List[str] = []

    def add(self, code: str, message: str) -> None:
        self.items.append(f"{code}: {self.prefix}{message}")


def _check_signature(journal: Sequence[Mapping[str, Any]], out: _Failures) -> Optional[str]:
    """The kind of the journal, or ``None`` when the signature is missing or unusable."""
    rows = [row for row in journal if row.get("event") == "signature"]
    if not rows:
        out.add("signature", "no signature event; the journal must say real or synthetic")
        return None
    kinds = {row.get("kind") for row in rows}
    if len(kinds) > 1:
        out.add("mixed_signatures", f"one journal carries several kinds: {sorted(map(str, kinds))}")
        return None
    if len(rows) > 1:
        out.add("signature", f"{len(rows)} signature events, exactly one is required")
    row = rows[0]
    kind = row.get("kind")
    if kind not in KINDS:
        out.add("signature", f"kind {kind!r} is not 'real' or 'synthetic'")
        return None
    if not isinstance(row.get("model"), str) or not row["model"]:
        out.add("signature", "the signature names no model")
    field = SOURCE_FIELD[kind]
    if not isinstance(row.get(field), str) or not row[field]:
        out.add("signature", f"a {kind} signature needs a non-empty {field!r}")
    return str(kind)


def _check_gradients(steps: Sequence[Mapping[str, Any]], out: _Failures) -> None:
    for row in steps:
        label = f"step {row.get('step')}"
        norm = _number(row, "gradient_norm")
        if norm is None:
            out.add("malformed", f"{label} has no numeric gradient_norm")
        elif math.isfinite(norm) and norm <= GRADIENT_NORM_MIN:
            out.add("zero_gradient",
                    f"{label} gradient_norm {norm:g} is not above {GRADIENT_NORM_MIN:g}")
        bad = _number(row, "gradient_nonfinite_tensors")
        if bad is None:
            out.add("malformed", f"{label} has no gradient_nonfinite_tensors")
        elif bad != 0:
            out.add("nonfinite", f"{label} has {bad:g} gradient tensors with NaN or inf")
        got, want = _number(row, "gradient_tensors"), _number(row, "trainable_tensors")
        if got is None or want is None:
            out.add("malformed", f"{label} has no gradient_tensors or trainable_tensors")
        elif got != want:
            out.add("gradient_missing",
                    f"{label} {got:g} of {want:g} trainable tensors got a gradient")
        if _number(row, "loss") is None:
            out.add("malformed", f"{label} has no numeric loss")


def _check_update(initial: Mapping[str, Any], steps: Sequence[Mapping[str, Any]],
                  out: _Failures) -> None:
    norm = _number(initial, "l2_norm")
    if not _is_digest(initial.get("sha256")) or norm is None or norm < 0:
        out.add("no_initial", "adapter_initial has no valid sha256 and l2_norm")
        return
    start = initial["sha256"]
    if steps and steps[0].get("adapter_sha256_before") != start:
        out.add("digest_chain", "step 0 starts from an adapter that is not the recorded initial")
    for previous, row in zip(steps, steps[1:]):
        if row.get("adapter_sha256_before") != previous.get("adapter_sha256_after"):
            out.add("digest_chain", f"step {row.get('step')} does not start where "
                    f"step {previous.get('step')} ended")
    for row in steps:
        distance = _number(row, "distance_from_initial")
        if distance is None or not _is_digest(row.get("adapter_sha256_after")):
            out.add("malformed", f"step {row.get('step')} has no distance_from_initial "
                    "or adapter_sha256_after")
            continue
        if math.isfinite(distance) and (row["adapter_sha256_after"] == start) != (distance == 0):
            out.add("digest_distance_clash", f"step {row.get('step')}: the digest and the "
                    "distance disagree about whether the adapter moved")
    for label, row in (("first", steps[0]), ("last", steps[-1])) if steps else ():
        distance = _number(row, "distance_from_initial")
        if distance is None or not math.isfinite(distance):
            continue
        relative = distance / max(norm, 1.0)
        if relative <= RELATIVE_MOVE_MIN:
            out.add("no_update", f"after the {label} step the adapter is {relative:g} "
                    f"(relative) from the initial one, not above {RELATIVE_MOVE_MIN:g}")


def _check_reload(journal: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]],
                  out: _Failures) -> None:
    reloads = [row for row in journal if row.get("event") == "reload"]
    if not reloads:
        out.add("reload", "no reload event: the saved adapter was not read back")
    elif steps and reloads[-1].get("parameter_sha256") != steps[-1].get("adapter_sha256_after"):
        out.add("reload", "the adapter read back after saving differs from the trained one")


def _judge_one(journal: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    out = _Failures(prefix)
    for row in journal:
        if row.get("event") == _UNPARSABLE:
            out.add("unparsable", f"line {row.get('line')} is not a JSON object")
        for key in _nonfinite_fields(row):
            out.add("nonfinite", f"event {row.get('event')} step {row.get('step')}: "
                    f"{key} is {row[key]}")
    kind = _check_signature(journal, out)
    steps = [row for row in journal if row.get("event") == "step"]
    if len(steps) < MIN_STEPS:
        out.add("too_few_steps", f"{len(steps)} steps recorded, at least {MIN_STEPS} are required")
    _check_gradients(steps, out)
    initials = [row for row in journal if row.get("event") == "adapter_initial"]
    if not initials:
        out.add("no_initial", "no adapter_initial event: without the starting state "
                "'updated' cannot be shown")
    else:
        _check_update(initials[0], steps, out)
    _check_reload(journal, steps, out)
    return {"kind": kind, "failures": out.items, "steps": len(steps)}


def judge(journals: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Pass or fail for the journals of one report, with every reason for a fail."""
    failures: List[str] = []
    kinds = set()
    steps = 0
    if not journals:
        failures.append("signature: no journal given")
    for number, journal in enumerate(journals, start=1):
        prefix = f"journal {number}: " if len(journals) > 1 else ""
        one = _judge_one(journal, prefix)
        failures += one["failures"]
        steps += one["steps"]
        if one["kind"]:
            kinds.add(one["kind"])
    if len(kinds) > 1:
        failures.append("mixed_signatures: real and synthetic results are in one report; "
                        "make two reports")
    return {
        "passed": not failures, "failures": failures,
        "kind": next(iter(kinds)) if len(kinds) == 1 else None,
        "journals": len(journals), "steps": steps,
        "thresholds": {"gradient_norm_min": GRADIENT_NORM_MIN,
                       "relative_move_min": RELATIVE_MOVE_MIN, "min_steps": MIN_STEPS},
    }


# --------------------------------------------------------------------------
# report template
# --------------------------------------------------------------------------
def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(caption: str, label: str, rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [f"**{caption} — {label}**", "", "| Показатель | Значение |", "| --- | --- |"]
    lines += [f"| {_cell(row[0])} | {_cell(row[1])} |" for row in rows]
    return lines + [""]


def _check_report_kind(record: Mapping[str, Any]) -> str:
    kind = record.get("kind")
    if kind not in KINDS:
        raise ValueError(f"record kind {kind!r} is not 'real' or 'synthetic'")
    verdict = record.get("verdict")
    if not isinstance(verdict, Mapping):
        raise ValueError("record has no verdict: run the judge first")
    if verdict.get("kind") != kind:
        raise MixedRecordsError(
            f"the record is {kind!r} but the judge saw {verdict.get('kind')!r} in its journals; "
            "real and synthetic results do not share a report")
    for section in record.get("sections") or []:
        if section.get("kind", kind) != kind:
            raise MixedRecordsError(
                f"section {section.get('heading')!r} is {section.get('kind')!r} in a "
                f"{kind!r} report")
    return str(kind)


def render_report(record: Mapping[str, Any]) -> str:
    """The report as Markdown. Raises for a missing field or a mix of the two kinds."""
    kind = _check_report_kind(record)
    label = KINDS[kind]
    if not record.get("command"):
        raise ValueError("record has no command: the report must say how the run was started")
    if not record.get("logs"):
        raise ValueError("record has no logs: the report must link the source journals")
    if not record.get("title"):
        raise ValueError("record has no title")
    unavailable = record.get("unavailable") or {}
    environment = record.get("environment") or {}
    rows: List[Sequence[Any]] = [("команда запуска", f"`{record['command']}`")]
    for field in ENVIRONMENT_FIELDS:
        value = environment.get(field)
        if value in (None, ""):
            if not unavailable.get(field):
                raise ValueError(f"environment.{field} is empty without a reason in 'unavailable'")
            value = f"не измерено ({unavailable[field]})"
        rows.append((field, value))

    def heading(level: int, text: str) -> str:
        return f"{'#' * level} {text} — {label}"

    verdict = record["verdict"]
    lines = [heading(1, str(record["title"])), ""]
    lines += [f"Подпись отчёта: {label}. Реальные и синтетические результаты в одном отчёте "
              "не смешиваются.", ""]
    lines += [heading(2, "Запуск и окружение"), ""]
    lines += _table("Команда, версии и железо", label, rows)
    lines += [heading(2, "Вердикт судьи градиентов и обновлений"), ""]
    lines += _table("Градиенты и обновления адаптеров", label, [
        ("вердикт", "PASS" if verdict.get("passed") else "FAIL"),
        ("журналов", verdict.get("journals")),
        ("шагов", verdict.get("steps")),
        ("порог нормы градиента", (verdict.get("thresholds") or {}).get("gradient_norm_min")),
        ("порог относительного сдвига адаптера",
         (verdict.get("thresholds") or {}).get("relative_move_min")),
        ("причины fail", "; ".join(verdict.get("failures") or []) or "нет"),
    ])
    for section in record.get("sections") or []:
        lines += [heading(2, str(section["heading"])), ""]
        for table in section.get("tables") or []:
            lines += _table(str(table["caption"]), label, table["rows"])
    lines += [heading(2, "Исходные журналы"), ""]
    lines += [f"- {path}" for path in record["logs"]]
    notes = record.get("notes") or []
    if notes:
        lines += ["", heading(2, "Оговорки"), ""] + [f"- {note}" for note in notes]
    return "\n".join(lines).rstrip() + "\n"


def template_record(kind: str) -> Dict[str, Any]:
    """An empty record. ``render_report`` refuses it until every field is filled."""
    return {
        "kind": kind, "title": "", "command": "",
        "environment": {field: "" for field in ENVIRONMENT_FIELDS},
        "unavailable": {}, "logs": [], "verdict": None,
        "sections": [{"heading": "", "tables": [{"caption": "", "rows": [["", ""]]}]}],
        "notes": [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    judge_cmd = sub.add_parser("judge", help="judge the journals of one report")
    judge_cmd.add_argument("--journal", action="append", required=True)
    judge_cmd.add_argument("--output", default=None, help="write the verdict as JSON here")
    template_cmd = sub.add_parser("template", help="print an empty report record as JSON")
    template_cmd.add_argument("--kind", choices=sorted(KINDS), required=True)
    report_cmd = sub.add_parser("report", help="render a record as the signed report")
    report_cmd.add_argument("--record", required=True)
    report_cmd.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "template":
        print(json.dumps(template_record(args.kind), indent=1, ensure_ascii=False))
        return 0
    if args.command == "report":
        target = Path(args.out)
        if target.exists():
            print(f"{target} exists; a report is never overwritten")
            return 2
        record = json.loads(Path(args.record).read_text(encoding="utf-8"))
        try:
            text = render_report(record)
        except ValueError as exc:
            print(f"REFUSED: {exc}")
            return 3
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
        return 0
    result = judge([read_journal(path) for path in args.journal])
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1, allow_nan=False) + "\n",
                                     encoding="utf-8")
    console = Console()
    console.print("PASS" if result["passed"] else "FAIL", result.get("kind") or "")
    for failure in result["failures"]:
        console.print(f"  {failure}", markup=False)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
