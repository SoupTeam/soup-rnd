#!/usr/bin/env python3
# ruff: noqa: E501  (the report text is prose; its lines are not wrapped)
"""A6 readiness rerun: write the four per-method reports from ``ready-<method>.json``.

Reads ``ready-<method>.json`` and ``environment.json`` from a results directory, plus ``rental.json``
(written by hand after the rental), and writes one Markdown report per method into ``benchmarks/``.
The numbers come from the records only; the prose is fixed text. Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_pref_check  # noqa: E402

NAMES = {"dpo": "DPO", "orpo": "ORPO", "simpo": "SimPO", "kto": "KTO"}
VERDICT = {"pass": "ПРОШЁЛ", "fail": "НЕ ПРОШЁЛ", "no_verdict": "НЕТ ВЕРДИКТА"}
STATE = {"equal": "равно побитово", "not_equal": "не равно", "unavailable": "недоступен"}
ARM_ORDER = ["f32u_ga1", "f32u_ga2", "f32u_l2l", "native_ga1", "native_l2l", "native_floor", "sft"]
DATA_FILES = {
    "dpo": "dpo_pairs_synthetic.jsonl", "orpo": "orpo_pairs_synthetic.jsonl",
    "simpo": "simpo_pairs_synthetic.jsonl", "kto": "kto_synthetic.jsonl",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "нет"
    return f"{value:.3e}" if value else "0"


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _cells(records: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [records["cells"][key] for key in sorted(records["cells"], key=int)]


TENSORS_PER_STEP = 120  # SmolLM2-135M: 30 layers x (q_proj, v_proj) x (lora_A, lora_B)


def what_showed(records: Dict[str, Any]) -> List[str]:
    """The findings in plain sentences; every number comes from ``ready-<method>.json``."""
    out = []
    cells = _cells(records)
    equal = [c["k"] for c in cells if c["verdict"]["float32"]["state"] == "equal"]
    out.append(f"- Критерий 1 (float32 побитово): выполнен в ячейках k={equal} из {[c['k'] for c in cells]}.")
    for cell in cells:
        f32 = cell["verdict"]["float32"]
        if f32["state"] == "not_equal":
            overall = cell["pairs"]["f32u_l2l_vs_ga1"]["overall"]
            out.append(
                f"  - k={cell['k']}: не равно по правилу (потери сравниваются тоже). Наибольшая разница градиентов {_fmt(overall['grad'])}, потерь {_fmt(overall['loss'])}. "
                f"Судья записал: {f32['problems'][0]}. "
                + ("Градиенты равны побитово на всех шагах, различается только записанная потеря на последнем разряде float32; вероятная причина, не проверенная, в порядке суммирования потери окна." if overall["grad"] == 0 else "Различаются и градиенты.")
            )
    for cell in cells:
        bf16 = cell["verdict"]["bf16"]
        step0 = [p for p in bf16.get("problems", []) if p.startswith("step 0:")]
        if not step0:
            continue
        ratios = []
        for item in step0:
            found = re.search(r"differs by ([0-9.e+-]+), limit ([0-9.e+-]+)", item)
            if found and float(found.group(2)) > 0:
                ratios.append(float(found.group(1)) / float(found.group(2)))
        out.append(
            f"- Критерий 2 (bf16, шаг 0), k={cell['k']}: вне предела {len(step0)} записей из {TENSORS_PER_STEP} тензоров шага"
            + (f"; превышение предела от {min(ratios):.2f} до {max(ratios):.2f} раза." if ratios else ".")
        )
    if any("step 0:" in p for c in cells for p in c["verdict"]["bf16"].get("problems", [])):
        out.append("  Предел считается по каждому тензору отдельно из одного замера пола, и правило требует выполнения на всех тензорах шага 0. Провал по немногим тензорам при том, что наибольшая разница по всем тензорам шага 0 (таблица ниже) меньше наибольшего пола, не доказывает и не исключает ошибку: правило здесь строже, чем по общей величине, и ничего больше о причине провала эти данные не говорят.")
    other = [c["k"] for c in cells if c["verdict"]["state"] == "fail" and not c["verdict"]["reasons"]]
    if other:
        out.append(f"- Ячейки k={other} не прошли, причина не записана.")
    return out


def results_section(records: Dict[str, Any]) -> List[str]:
    rows = []
    for cell in _cells(records):
        verdict = cell["verdict"]
        f32, bf16 = verdict["float32"], verdict["bf16"]
        f32_pair = cell["pairs"]["f32u_l2l_vs_ga1"]
        step0 = cell["pairs"]["native_l2l_vs_ga1"]["steps"][0] if cell["pairs"]["native_l2l_vs_ga1"] else None
        floor0 = cell["pairs"]["native_floor_vs_ga1"]["steps"][0] if cell["pairs"]["native_floor_vs_ga1"] else None
        rows.append([
            cell["k"], cell["windows"],
            STATE[f32["state"]], _fmt(f32_pair["overall"]["grad"]) if f32_pair else "нет",
            "да" if bf16.get("step0_within") else ("нет" if bf16.get("step0_within") is False else "нет данных"),
            f"{_fmt(step0['grad'])} / {_fmt(floor0['grad'])}" if step0 and floor0 else "нет",
            f"{_fmt(step0['loss'])} / {_fmt(floor0['loss'])}" if step0 and floor0 else "нет",
            VERDICT[verdict["state"]],
        ])
    return _table(
        ["k", "окон", "float32 (критерий 1)", "float32 max разница градиента", "bf16 шаг 0 в 1,5 пола (критерий 2)",
         "шаг 0: градиент L2L / пол", "шаг 0: потеря L2L / пол", "исход ячейки"], rows)


def reasons_section(records: Dict[str, Any]) -> List[str]:
    out = []
    for cell in _cells(records):
        for reason in cell["verdict"]["reasons"][:6]:
            out.append(f"- k={cell['k']}: {reason}")
    return out or ["Причин провала и неопределённости нет."]


def later_steps_section(records: Dict[str, Any]) -> List[str]:
    rows = []
    for cell in _cells(records):
        l2l, floor = cell["pairs"]["native_l2l_vs_ga1"], cell["pairs"]["native_floor_vs_ga1"]
        if not l2l or not floor:
            continue
        for a, b in zip(l2l["steps"], floor["steps"]):
            rows.append([cell["k"], a["step"], _fmt(a["grad"]), _fmt(b["grad"]), _fmt(a["loss"]), _fmt(b["loss"])])
    return _table(["k", "шаг", "градиент L2L", "градиент пол", "потеря L2L", "потеря пол"], rows)


def loads_section(records: Dict[str, Any], method: str) -> List[str]:
    rows = []
    for cell in _cells(records):
        logs = cell["logs"]
        if not logs.get("native_l2l") or not logs.get("sft"):
            rows.append([cell["k"], "нет данных", "", "", "", "нет данных"])
            continue
        loads = l2l_pref_check.judge_loads(logs["native_l2l"], logs["sft"])
        rows.append([
            cell["k"], loads["layer_loads"]["method"], loads["layer_loads"]["sft"],
            f"{sum(loads['l2l_large_loads']['method'])} / {sum(loads['l2l_large_loads']['sft'])}",
            f"{sum(loads['l2l_weight_restores']['method'])} / {sum(loads['l2l_weight_restores']['sft'])}",
            "совпадает" if loads["pass"] else "НЕ совпадает: " + "; ".join(loads["problems"][:2]),
        ])
    return _table(["k", f"загрузки слоя за шаг, {NAMES[method]}", "загрузки слоя за шаг, L2L SFT",
                   "большие загрузки (метод / SFT)", "восстановления весов (метод / SFT)", "исход"], rows)


def validity_section(records: Dict[str, Any]) -> List[str]:
    rows = []
    for cell in _cells(records):
        for arm in ARM_ORDER:
            for item in cell["attempts"].get(arm, []):
                rows.append([cell["k"], f"`{arm}`", item["attempt"], item["outcome"], item["returncode"],
                             item.get("excluded", ""), "да" if item.get("excluded") else "нет"])
    return _table(["k", "плечо", "попытка", "исход ArmWatch", "код выхода", "причина исключения", "исключена"], rows)


def types_section(records: Dict[str, Any]) -> List[str]:
    cell = _cells(records)[0]
    fields = ("stream_dtype", "train_bf16", "mixed_precision", "base_weight", "layer_out", "grads", "sdpa", "allow_tf32")
    arms = ["f32u_ga1", "f32u_l2l", "native_ga1", "native_l2l", "native_floor"]
    rows = []
    for field in fields:
        rows.append([f"`{field}`"] + [("нет записи" if not cell["types"].get(a) else f"`{cell['types'][a].get(field)}`") for a in arms])
    return _table(["поле (ячейка k=" + str(cell["k"]) + ")"] + [f"`{a}`" for a in arms], rows)


def render(method: str, records: Dict[str, Any], env: Dict[str, Any], rental: Dict[str, Any], rel: str) -> str:
    verdict = records["method_verdict"]
    name = NAMES[method]
    out = [
        f"# A6 повторный прогон по правилу готовности: {name} на L2L против накопления, SmolLM2-135M, {rental['card']}, {rental['date']}",
        "",
        f"**Вердикт: {VERDICT[verdict['state']]}** (правило `benchmarks/l2l-preference-readiness-rule-a6.md`)."
        + (f" {verdict['reason']}." if verdict["reason"] else ""),
        "",
        "Правило записано и закоммичено до правок драйвера и до аренды. Тикет 06 не переписывается: его отчёты остаются «не прошёл» по его критерию. "
        f"Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** (`examples/data/{DATA_FILES[method]}`, 12 строк). "
        "Качество модели не измерялось, результат относится к этой модели и этой карте и не переносится на другие."
        + (" DPO k=1 уже измерен в тикете 08; критерий предложен после него, поэтому DPO здесь не независимое доказательство." if method == "dpo" else ""),
        "",
        "## Стенд",
        "",
        *_table(["Параметр", "Значение"], [
            ["Сервер", rental["server"]], ["GPU", env["nvidia_smi"]], ["Образ", rental["image"]],
            ["Пакеты", f"torch {env['torch']}, transformers {env['transformers']}, trl {env['trl']}, peft {env['peft']}, CUDA {env['cuda']}, Python {env['python']} (полный список в `packages.txt`)"],
            ["RAM", f"{env['ram']}; {rental['uptime']}"], ["Коммит", f"`{rental['commit']}`" + (" " + rental["tree_note"] if rental.get("tree_note") else "")],
            ["Стоимость", rental["cost"]], ["Аккаунт", rental["account"]], ["Уничтожение", rental["destroy"]],
        ]),
        "",
        "## Команды",
        "",
        "```bash",
        *rental["commands"],
        "```",
        "",
        "k равен 1, 2, 4. Плечи описаны в правиле: `f32u_*` (float32 подменой, ядра SDPA по умолчанию), `native_*` (bf16, `MATH` в обоих путях, пол на `EFFICIENT_ATTENTION`), `sft` (счёт загрузок). "
        "Каждое плечо идёт в своём процессе под `ArmWatch`, обязательные проверки `suspend` и `power`.",
        "",
        "## Результат по ячейкам",
        "",
        *results_section(records),
        "",
        "## Что показали данные",
        "",
        *what_showed(records),
        "",
        "Что не выполнено или не определено:",
        "",
        *reasons_section(records),
        "",
        "Что реально стояло в плечах (запись изнутри первого вызова слоя 0 и из `TrainingArguments`); float32 получен подменой двух функций драйвера, это обход, а не штатный режим:",
        "",
        *types_section(records),
        "",
        "## Все шаги bf16 (записаны, не судятся)",
        "",
        "Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами, поэтому правило их не судит. Наибольшая разница по градиенту LoRA и потере:",
        "",
        *later_steps_section(records),
        "",
        "## Загрузки слоя" + (": опорный проход " + ("не добавил загрузок" if all(c["verdict"]["state"] != "fail" or not any(r.startswith("loads:") for r in c["verdict"]["reasons"]) for c in _cells(records)) else "добавил загрузки") if method == "dpo" else ""),
        "",
        "Счётчики берутся из лога обучения, это счётчики рантайма. Условие правила: загрузок слоя за шаг столько же, сколько у L2L SFT с тем же k, больших загрузок и восстановлений весов не больше."
        + (" В DPO опорный проход идёт через тот же загруженный слой без градиента с адаптером `ref`." if method == "dpo" else ""),
        "",
        *loads_section(records, method),
        "",
        "## Валидность замеров и попытки",
        "",
        *validity_section(records),
        "",
        "Испорченные замеры (`void`) исключаются по `benchmarks/METHOD.md`, запись остаётся в каталоге результатов. Если колонка «исключена» пуста, исключать было нечего.",
        "",
        "## Что «прошёл» не значит и что не проверялось",
        "",
        "- Не значит побитового равенства в bf16 (в тикете 08 его нет даже на шаге 0) и допуска в bf16 при ядрах SDPA по умолчанию: критерий 2 закрепляет `MATH`, как в A3-07, и результат от этого зависит.",
        "- Пол шума это один замер второго бэкенда SDPA; GA в bf16 на повторяемость не проверялся.",
        "- Данные синтетические, 12 строк, неполное последнее окно есть при k=4. Качество модели не измерялось.",
        "- Другие модели, другие карты, NF4, реальные наборы, скорость и память не проверялись. `foreign_reads` не обязательна (пояснение в правиле).",
        "",
        "## Файлы",
        "",
        f"- [ready-{method}.json]({rel}/ready-{method}.json), [environment.json]({rel}/environment.json), [rental.json]({rel}/rental.json)",
        f"- журналы и записи `ArmWatch` каждой попытки: `{rel}/{method}/k*/<плечо>/attempt*/` (без каталогов `*-work`)",
        f"- [packages.txt]({rel}/packages.txt), [bootstrap.log]({rel}/bootstrap.log), [ready.log]({rel}/ready.log), [create.log]({rel}/create.log), [COMMIT]({rel}/COMMIT), [show-instances-after.txt]({rel}/show-instances-after.txt)",
        "- хеши `.pt` каждой попытки лежат в `ready-" + method + ".json` (`attempts`); сами `.pt` в репозиторий не входят.",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="directory with ready-*.json")
    args = parser.parse_args()
    results = Path(args.results).resolve()
    env = json.loads((results / "environment.json").read_text())
    rental = json.loads((results / "rental.json").read_text())
    rel = results.relative_to(ROOT / "benchmarks").as_posix()
    for method in NAMES:
        path = results / f"ready-{method}.json"
        if not path.exists():
            continue
        target = ROOT / "benchmarks" / f"probe-vast-l2l-{method}-a6-ready.md"
        target.write_text(render(method, json.loads(path.read_text()), env, rental, rel))
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
