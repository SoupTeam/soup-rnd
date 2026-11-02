#!/usr/bin/env python3
# ruff: noqa: E501  (the report text is prose; its lines are not wrapped)
"""Ticket A6-06: write the four per-method reports from the records of l2l_pref_check.py.

Reads ``records-<method>.json`` and ``environment.json`` from a results directory, plus
``rental.json`` (offer, instance, rate, times, cost, destroy check) written by hand after the
rental, and prints one Markdown report per method into ``benchmarks/``. The numbers come from the
records only; the prose is fixed text. Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
METHOD_NAMES = {"dpo": "DPO", "orpo": "ORPO", "simpo": "SimPO", "kto": "KTO"}
VERDICT_TEXT = {
    "pass": "ПРОШЁЛ",
    "fail": "НЕ ПРОШЁЛ",
    "no_verdict": "НЕТ ВЕРДИКТА",
}
DATA_FILES = {
    "dpo": "examples/data/dpo_pairs_synthetic.jsonl",
    "orpo": "examples/data/orpo_pairs_synthetic.jsonl",
    "simpo": "examples/data/simpo_pairs_synthetic.jsonl",
    "kto": "examples/data/kto_synthetic.jsonl",
}


def _used(cell: Dict[str, Any], judged: Dict[str, Any]) -> Dict[str, Any]:
    index = judged["used_attempt"]
    return cell["attempts"][index] if index is not None else cell["attempts"][-1]


def _fmt(value: float) -> str:
    return f"{value:.3e}" if value else "0"


def results_section(records: Dict[str, Any]) -> List[str]:
    lines = [
        "| k | окон | попытка | ступень | A против A побитово | max разница градиента | max разница потери | исход |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        attempt = _used(cell, judged)
        eq = attempt["equality"]
        steps = len(attempt.get("losses", {}).get("l2l", []))
        diff = attempt.get("max_abs", {}).get("l2l_vs_ga1", {})
        lines.append(
            f"| {cell['k']} | {steps} | {judged['used_attempt']} | {eq.get('tier')} | "
            f"{eq.get('a_vs_a_equal')} | {_fmt(diff.get('grad', 0.0))} | "
            f"{_fmt(diff.get('loss', 0.0))} | {VERDICT_TEXT[judged['verdict']]} |"
        )
    problems = [
        f"- k={cell['k']}: {problem}"
        for cell, judged in zip(records["cells"], records["verdict"]["cells"])
        for problem in _used(cell, judged)["equality"].get("problems", [])
    ]
    return lines + ([""] + problems if problems else [])


def losses_section(records: Dict[str, Any]) -> List[str]:
    lines = ["| k | шаг | потеря накопления (ga1) | потеря накопления (ga2) | потеря L2L |",
             "| --- | --- | --- | --- | --- |"]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        losses = _used(cell, judged).get("losses", {})
        for step, (a, b, c) in enumerate(zip(*(losses.get(x, []) for x in ("ga1", "ga2", "l2l")))):
            lines.append(f"| {cell['k']} | {step} | {a!r} | {b!r} | {c!r} |")
    return lines


def loads_section(records: Dict[str, Any], method: str) -> List[str]:
    name = METHOD_NAMES[method]
    lines = [
        f"| k | загрузки слоя за шаг, {name} | загрузки слоя за шаг, L2L SFT | "
        "большие загрузки (метод / SFT, сумма) | восстановления весов (метод / SFT, сумма) | "
        "загрузки в захватах формы (метод / SFT) | исход |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        loads = _used(cell, judged)["loads"]
        if "layer_loads" not in loads:
            lines.append(f"| {cell['k']} | нет записи | нет записи | | | | не оценено |")
            continue
        large = loads.get("l2l_large_loads", {"method": [], "sft": []})
        restores = loads.get("l2l_weight_restores", {"method": [], "sft": []})
        captures = loads["capture_layer_loads"]
        lines.append(
            f"| {cell['k']} | {loads['layer_loads']['method']} | {loads['layer_loads']['sft']} | "
            f"{sum(large['method'])} / {sum(large['sft'])} | "
            f"{sum(restores['method'])} / {sum(restores['sft'])} | "
            f"{captures['method']} / {captures['sft']} | {'совпадает' if loads['pass'] else 'нет'} |"
        )
    return lines


def validity_section(records: Dict[str, Any]) -> List[str]:
    lines = [
        "| k | попытка | ga1 | ga2 | l2l | sft | отброшена | причина |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        excluded = {item["attempt"]: item["reason"] for item in judged["excluded"]}
        for index, attempt in enumerate(cell["attempts"]):
            arms = attempt["arms"]
            row = " | ".join(arms.get(arm, "не запускалось") for arm in ("ga1", "ga2", "l2l", "sft"))
            note = excluded.get(index, attempt.get("crashed") or "")
            lines.append(
                f"| {cell['k']} | {index} | {row} | {'да' if index in excluded else 'нет'} | {note} |"
            )
    lines += ["", "Проверки `foreign_reads` (не обязательная, записана) по плечам использованных попыток:", ""]
    lines += ["| k | ga1 | ga2 | l2l | sft |", "| --- | --- | --- | --- | --- |"]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        watch = _used(cell, judged).get("watch", {})
        row = " | ".join(watch.get(arm, {}).get("foreign_reads", "-") for arm in ("ga1", "ga2", "l2l", "sft"))
        lines.append(f"| {cell['k']} | {row} |")
    return lines


def render(method: str, records: Dict[str, Any], env: Dict[str, Any], rental: Dict[str, Any],
           results_rel: str) -> str:
    name = METHOD_NAMES[method]
    verdict = records["verdict"]
    head = f"**Вердикт: {VERDICT_TEXT[verdict['verdict']]}** (правило `benchmarks/l2l-preference-check-rule-a6.md`)."
    if verdict["reason"]:
        head += f" Причина: {verdict['reason']}."
    out = [
        f"# A6 тикет 06: {name} на L2L против накопления градиентов, SmolLM2-135M, {rental['card']}, "
        f"{rental['date']}",
        "",
        head,
        "",
        "Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** "
        f"(`{DATA_FILES[method]}`, 12 строк, написаны вручную). Качество модели не измерялось. "
        "Реальных наборов в этом отчёте нет, результаты на них не заявляются. "
        "Результат относится к этой модели и этой карте и не переносится на другие.",
        "",
        "## Стенд",
        "",
        "| Параметр | Значение |",
        "| --- | --- |",
        f"| Сервер | Vast.ai, оффер {rental['offer']}, инстанс {rental['instance']}, "
        f"${rental['rate']}/ч, хост {rental['host']} |",
        f"| GPU | {env['nvidia_smi']} |",
        f"| Образ | `{rental['image']}`, Python {env['python']} |",
        f"| Пакеты | torch {env['torch']}, transformers {env['transformers']}, trl {env['trl']}, "
        f"peft {env['peft']} (полный список в `packages.txt`) |",
        f"| RAM | {env['ram'] or 'см. free.txt'}; нагрузка хоста на старте: {rental['uptime']} |",
        f"| Коммит | `{rental['commit']}` (дерево загружено по `git ls-files`) |",
        f"| Стоимость | {rental['cost']} (аренда {rental['start']} to {rental['end']}, один инстанс на все "
        "четыре метода) |",
        f"| Уничтожение | {rental['destroy']} |",
        "",
        "## Команды",
        "",
        "```bash",
        "bash benchmarks/harness/l2l_pref_check/remote.sh   # bootstrap, затем:",
        "export PYTHONPATH=$PWD/src HF_HOME=/root/hf",
        "python benchmarks/harness/l2l_pref_check.py run --device cuda --out /root/results/a6 \\",
        f"    --methods {method}   # на сервере шли все четыре метода подряд одной командой без --methods",
        "```",
        "",
        "k равен 1, 2, 4. Плечи: `ga1`, `ga2` (накопление дважды, A против A), `l2l`, `sft` (L2L SFT "
        "с теми же k и числом микропорций, для счёта загрузок). Каждое плечо идёт в своём процессе "
        "под `ArmWatch`, обязательные проверки `suspend` и `power`.",
        "",
        "## Равенство потерь и градиентов LoRA",
        "",
    ]
    out += results_section(records)
    out += ["", "Потери по шагам (одинаковые числа значат побитово равные `float`):", ""]
    out += losses_section(records)
    out += ["", "## Загрузки слоя" if method != "dpo" else
            "## Опорный проход не добавил загрузок слоя", ""]
    if method == "dpo":
        out += [
            "Счётчики берутся из лога обучения, это счётчики рантайма. В DPO каждая микропорция "
            "проходит через загруженный слой дважды: политика с градиентом и опорный проход без "
            "градиента с адаптером `ref`. Условие правила: загрузок слоя за шаг столько же, сколько у "
            "L2L SFT с тем же k, больших загрузок и восстановлений весов не больше.",
            "",
        ]
    out += loads_section(records, method)
    out += ["", "## Валидность замеров и попытки", ""]
    out += validity_section(records)
    out += [
        "",
        "Испорченными считаются плечи с проверкой `void`; их попытка отброшена по `benchmarks/METHOD.md`, "
        "запись осталась в каталоге результатов. Неудачные попытки приложены ниже.",
        "",
        "## Что не проверялось",
        "",
        "- Скорость и память не измерялись.",
        "- Другие модели, другие карты, NF4, реальные наборы данных.",
        "- Размер набора: 12 строк, шесть микропорций по две строки; неполное последнее окно есть при k=4.",
        "- `foreign_reads` не входит в обязательные проверки (пояснение в правиле).",
        "",
        "## Файлы",
        "",
        f"- [records-{method}.json]({results_rel}/records-{method}.json), "
        f"[environment.json]({results_rel}/environment.json)",
        f"- журналы и записи `ArmWatch` каждой попытки: `{results_rel}/{method}/k*/attempt*/`",
        f"- [packages.txt]({results_rel}/packages.txt), [nvidia-smi.txt]({results_rel}/nvidia-smi.txt), "
        f"[bootstrap.log]({results_rel}/bootstrap.log), [a6.log]({results_rel}/a6.log)",
        f"- [rental.json]({results_rel}/rental.json), [create.log]({results_rel}/create.log)",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="directory with records-*.json")
    parser.add_argument("--slug", default="rtx3060")
    args = parser.parse_args()
    results = Path(args.results).resolve()
    env = json.loads((results / "environment.json").read_text())
    rental = json.loads((results / "rental.json").read_text())
    rel = results.relative_to(ROOT / "benchmarks").as_posix()
    for method in METHOD_NAMES:
        path = results / f"records-{method}.json"
        if not path.exists():
            continue
        text = render(method, json.loads(path.read_text()), env, rental, rel)
        target = ROOT / "benchmarks" / f"probe-vast-l2l-{method}-a6-ticket06.md"
        target.write_text(text)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
