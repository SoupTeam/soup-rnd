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
            f"| {cell['k']} | {steps} | {judged['used_attempt']} | {eq.get('tier') or 'нет'} | "
            f"{eq.get('a_vs_a_equal')} | {_fmt(diff.get('grad', 0.0))} | "
            f"{_fmt(diff.get('loss', 0.0))} | {VERDICT_TEXT[judged['verdict']]} |"
        )
    problems = []
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        found = _used(cell, judged)["equality"].get("problems", [])
        tensors = [p for p in found if "differs" in p and "loss" not in p]
        steps = sorted({int(p.split(":")[0].split()[1]) for p in tensors if p.startswith("step ")})
        losses = [p for p in found if "loss" in p]
        if found:
            problems.append(
                f"- k={cell['k']}: различаются {len(tensors)} записей градиента (на шагах {steps}) "
                f"и {len(losses)} потерь; первое: {found[0]}"
            )
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
    lines += ["", "Проверки `foreign_reads` (не обязательная, записана) по плечам использованных попыток. "
              "Пока здесь `unknown`, довод правила «веса скачаны заранее, чтения тренера не шли с нуля» "
              "этой проверкой не подтверждён: путь модели не отображается на блочное устройство.", ""]
    lines += ["| k | ga1 | ga2 | l2l | sft |", "| --- | --- | --- | --- | --- |"]
    for cell, judged in zip(records["cells"], records["verdict"]["cells"]):
        watch = _used(cell, judged).get("watch", {})
        row = " | ".join(watch.get(arm, {}).get("foreign_reads", "-") for arm in ("ga1", "ga2", "l2l", "sft"))
        lines.append(f"| {cell['k']} | {row} |")
    return lines


def excluded_text(records: Dict[str, Any]) -> str:
    dropped = [(c["k"], e) for c in records["verdict"]["cells"] for e in c["excluded"]]
    if not dropped:
        return "В этом наборе испорченных замеров и неудачных запусков нет: по одной попытке на ячейку, все использованы."
    return "Исключены попытки: " + "; ".join(f"k={k}, попытка {e['attempt']}: {e['reason']}" for k, e in dropped) + "."


def cause_lines(records: Dict[str, Any]) -> List[str]:
    """What is and is not known about a failure; nothing is said when every cell passed."""
    used = [_used(c, j) for c, j in zip(records["cells"], records["verdict"]["cells"])]
    if all(a["equality"]["pass"] for a in used):
        return []
    return [
        "- Причина расхождения L2L и накопления на CUDA не диагностирована. Тип весов при загрузке "
        "в записях не сохранён: код берёт `torch_dtype=\"auto\"`, то есть тип чекпойнта "
        "(`resolve_frozen_base_load_dtype` в `src/soup_cli/utils/gpu.py`); градиенты LoRA записаны в float32. "
        "В тестах тикетов 02 to 05 (CPU, float32, крошечная Llama) то же равенство побитовое. "
        "Это не доказывает, что причина в точности вычислений.",
        *(
            ["- Допуск A3-07 с полом шума от второго бэкенда SDPA не измерялся: правило этого тикета берёт "
             "пол шума из `ga2` против `ga1`, а он равен 0, поэтому ступень 2 не включалась. Прогон с таким "
             "полом требует поправки правила до измерения и не входит в этот отчёт."]
            if all(a["equality"]["a_vs_a_equal"] for a in used)
            else []
        ),
    ]


def what_showed(records: Dict[str, Any]) -> List[str]:
    """One line per criterion of the rule, from the records."""
    cells = list(zip(records["cells"], records["verdict"]["cells"]))
    used = [(_used(c, j), j) for c, j in cells]
    equal = [j["verdict"] for _a, j in used]
    return [
        "- Равенство L2L и накопления (ступень 1, побитово): "
        + ("выполнено во всех ячейках." if all(a["equality"]["pass"] for a, _ in used)
           else "**не выполнено** (таблица ниже)."),
        "- Накопление против самого себя (`ga1` и `ga2`): "
        + ("побитово равно во всех ячейках, значит стенд повторяем и ступень 2 (допуск A3-07 по полу шума) "
           "не применяется." if all(a["equality"]["a_vs_a_equal"] for a, _ in used)
           else "различается хотя бы в одной ячейке (ступень 2 применялась там)."),
        "- «Лишнего чтения нет» (загрузки слоя за шаг равны L2L SFT): "
        + ("выполнено во всех ячейках." if all(a["loads"]["pass"] for a, _ in used) else "**не выполнено**."),
        "- Валидность плеч (`suspend`, `power`): "
        + ("`ok` на всех плечах всех использованных попыток." if all(
            all(v == "ok" for v in a["arms"].values()) for a, _ in used) else "есть плечи не `ok`, см. ниже."),
        "- Метод **не отмечается готовым**." if "fail" in equal or any(j["verdict"] != "pass" for _a, j in used)
        else "- Все три ячейки прошли.",
    ]


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
        "vastai create instance 58363712 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 60 --ssh --direct --label a6-06-pref-check --cancel-unavail",
        'vastai attach ssh 58864206 "$(cat ~/.ssh/id_ed25519.pub)"',
        "git ls-files -z | tar --null -T - -czf - | ssh ... 'mkdir -p /root/soup && tar xzf - -C /root/soup'",
        'ssh -f ... "cd /root/soup && CARD=\'RTX 3060 12GB\' nohup bash benchmarks/harness/l2l_pref_check/remote.sh > /root/remote.out 2>&1 < /dev/null &"',
        "vastai destroy instance 58864206 -y",
        "# на сервере, из каталога /root/soup, дерево коммита выше:",
        "CARD='RTX 3060 12GB' bash benchmarks/harness/l2l_pref_check/remote.sh",
        "# внутри: bootstrap (a0-bootstrap.sh), затем одна команда на все четыре метода:",
        "python benchmarks/harness/l2l_pref_check.py run --device cuda --out /root/results/a6 \\",
        '    --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf',
        "```",
        "",
        "k равен 1, 2, 4. Плечи: `ga1`, `ga2` (накопление дважды, A против A), `l2l`, `sft` (L2L SFT "
        "с теми же k и числом микропорций, для счёта загрузок). Каждое плечо идёт в своём процессе "
        "под `ArmWatch`, обязательные проверки `suspend` и `power`.",
        "",
        "## Что показали данные",
        "",
        *what_showed(records),
        "",
        "## Равенство потерь и градиентов LoRA",
        "",
    ]
    out += results_section(records)
    out += ["", "Потери по шагам (одинаковые числа значат побитово равные `float`):", ""]
    out += losses_section(records)
    loads_ok = all(_used(c, j)["loads"]["pass"] for c, j in zip(records["cells"], records["verdict"]["cells"]))
    heading = "## Загрузки слоя"
    if method == "dpo":
        heading += ": опорный проход не добавил загрузок" if loads_ok else ": опорный проход добавил загрузки"
    out += ["", heading, ""]
    if method == "dpo":
        out += [
            "Счётчики берутся из лога обучения, это счётчики рантайма. В DPO опорный проход идёт "
            "через тот же загруженный слой без градиента с адаптером `ref`. Условие правила: загрузок слоя за шаг столько же, сколько у "
            "L2L SFT с тем же k, больших загрузок и восстановлений весов не больше.",
            "",
        ]
    out += loads_section(records, method)
    out += ["", "## Валидность замеров и попытки", ""]
    out += validity_section(records)
    out += [
        "",
        "Испорченными считаются плечи с проверкой `void`; их попытка отбрасывается по `benchmarks/METHOD.md`, "
        "запись остаётся в каталоге результатов. "
        + excluded_text(records),
        "",
        "## Что не проверялось",
        "",
        *cause_lines(records),
        "- Шаги после нулевого начинаются с адаптеров, которые уже обновлены разными градиентами, поэтому их различия не независимы от шага 0. Чистое сравнение даёт только шаг 0.",
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
        f"- [rental.json]({results_rel}/rental.json) (написан вручную после аренды; число `16,50` нагрузки и credit 1,698 и 1,641 взяты из вывода `ssh` и `vastai show user` во время работы и не заархивированы), "
        f"[create.log]({results_rel}/create.log), [COMMIT]({results_rel}/COMMIT)",
        "- `pt-sha256.txt`: хеши градиентов и потерь каждого плеча. Сами `.pt` весят 184 МБ и в репозитории их нет.",
        "- Проверка ключей градиентов (только `.default.`, ни одного `.ref.`) и длины ряда потерь добавлена в судью после прогона. Она применена к сохранённым `.pt` на машине автора: 21120 ключей, нарушений нет, вердикты и числа различий те же.",
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
