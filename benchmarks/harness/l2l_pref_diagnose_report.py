#!/usr/bin/env python3
# ruff: noqa: E501  (the report text is prose; its lines are not wrapped)
"""Ticket A6-08: write the diagnosis report from the records of l2l_pref_diagnose.py.

Reads ``diagnosis.json`` and ``environment.json`` from a results directory, plus ``rental.json``
(offer, instance, rate, times, cost, destroy check) written by hand after the rental, and writes
one Markdown report into ``benchmarks/``. The numbers come from the records only; the prose is
fixed text. Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REPORT = "probe-vast-l2l-dpo-divergence-a6-ticket08.md"

ROW_TEXT = {
    1: "округление, ошибки нет. Критерий готовности методов предпочтений переписывается на допуск A3-07 новым правилом до нового прогона четырёх методов; в этом тикете он не переписывается.",
    2: "округление больше пола шума от второго бэкенда SDPA (накопление против себя даёт 0, пол взят от `EFFICIENT_ATTENTION` против `MATH`). Решение о допуске за пользователем.",
    3: "вероятна ошибка в общем пути L2L. Контроль 3 называет слой; чинится отдельным тикетом.",
    4: "расхождение того же порядка, что шум бэкенда. Тип точности не отделён, это предел результата.",
    5: "причина не установлена.",
}
STATE_TEXT = {
    "equal": "равно побитово",
    "not_equal": "не равно",
    "unavailable": "недоступен",
    "within": "в 1,5 пола",
    "outside": "вне 1,5 пола",
    "no_data": "нет данных",
    "differs": "расходится",
    "not_needed": "не нужен",
}
TYPE_FIELDS = (
    "dtype_mode", "path", "stream_dtype", "train_bf16", "train_fp16", "mixed_precision", "base_weight",
    "lora_weight", "layer_in", "layer_out", "grads", "l2l_state_dtype", "attn_impl", "sdpa",
    "matmul_precision", "allow_tf32",
)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "нет"
    return f"{value:.3e}" if value else "0"


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return out


def _pair_rows(summary: Dict[str, Any], names: List[str]) -> List[List[Any]]:
    rows = []
    for name in names:
        pair = summary["pairs"].get(name)
        if pair is None:
            rows.append([f"`{name}`", "нет данных", "", ""])
            continue
        for step in pair["steps"]:
            rows.append([f"`{name}`", step["step"], _fmt(step["grad"]), _fmt(step["loss"])])
    return rows


def types_section(summary: Dict[str, Any], arms: List[str]) -> List[str]:
    records = {arm: summary["types"].get(arm) for arm in arms}
    rows = []
    for field in TYPE_FIELDS:
        row: List[Any] = [f"`{field}`"]
        for arm in arms:
            record = records[arm]
            row.append("нет записи" if record is None else f"`{record.get(field)}`")
        rows.append(row)
    return _table(["поле"] + [f"`{arm}`" for arm in arms], rows)


def attempts_section(summary: Dict[str, Any]) -> List[str]:
    rows = []
    for arm, attempts in summary["attempts"].items():
        for item in attempts:
            rows.append([
                f"`{arm}`", item["attempt"], item["outcome"], item["returncode"],
                item.get("excluded", ""), "да" if item.get("excluded") else "нет",
            ])
    return _table(["плечо", "попытка", "исход ArmWatch", "код выхода", "причина исключения", "исключена"], rows)


def what_showed(summary: Dict[str, Any]) -> List[str]:
    """The findings in plain sentences; every number comes from ``diagnosis.json``."""
    c1, c2, pairs = summary["control1"], summary["control2"], summary["pairs"]
    out = []
    f32 = pairs.get("f32_l2l_vs_ga1")
    if c1["state"] == "equal" and f32:
        out.append(f"- Контроль 1: в float32 L2L побитово равен накоплению на всех {len(f32['steps'])} шагах (градиенты LoRA и потери); накопление в float32 повторяется побитово. Запись типов подтверждает float32 без автокаста в этих трёх плечах. Значит, при закреплённом `MATH` в общем пути L2L в этой ячейке на этой карте ошибки нет. Что разрыв условий тикета 06 (SDPA по умолчанию) вызван только точностью, это не доказано: float32 без закрепления не запускался.")
    elif c1["state"] == "not_equal":
        out.append("- Контроль 1: в float32 L2L не равен накоплению; смотрите контроль 3.")
    else:
        out.append("- Контроль 1 недоступен, причины в разделе контроля 1.")
    l2l, floor = pairs.get("native_l2l_vs_ga1"), pairs.get("native_floor_vs_ga1")
    if l2l and floor:
        step0_l2l, step0_floor = l2l["steps"][0], floor["steps"][0]
        out.append(f"- Контроль 2, шаг 0 (одинаковые адаптеры): разрыв L2L по градиенту {_fmt(step0_l2l['grad'])}, пол {_fmt(step0_floor['grad'])}; по потере {_fmt(step0_l2l['loss'])} и {_fmt(step0_floor['loss'])}. Шаг 0 в пределе: {'да' if c2.get('step0_within') else 'нет'}.")
        found = [re.search(r"differs by ([0-9.e+-]+), limit ([0-9.e+-]+)", item) for item in c2["problems"]]
        ratios = [float(m.group(1)) / float(m.group(2)) for m in found if m and float(m.group(2)) > 0]
        steps = sorted({int(re.match(r"step (\d+)", item).group(1)) for item in c2["problems"]})
        if c2["problems"]:
            out.append(f"- Контроль 2, все шаги: вне предела {len(c2['problems'])} записей на шагах {steps}; наибольшее превышение предела {max(ratios, default=0):.2f} раза (предел это 1,5 пола), меньше чем вдвое превышают {sum(r < 2 for r in ratios)} из {len(ratios)}. Пол здесь одно измерение второго бэкенда. Шаги после нулевого не независимы от шага 0 (адаптеры уже разошлись), поэтому по правилу исход читается как «вне», а размер превышения виден в таблице.")
    unpinned, pinned = pairs.get("unpinned_l2l_vs_ga"), pairs.get("native_l2l_vs_ga1")
    if unpinned and pinned:
        out.append(f"- Без закрепления SDPA (условия тикета 06) потеря шага 0 расходится на {_fmt(unpinned['steps'][0]['loss'])}, с `MATH` в обоих путях на {_fmt(pinned['steps'][0]['loss'])}. Это совпадает с тем, что L2L вызывает слои без маски внимания, а накопление с маской, и умолчание SDPA может выбирать для них разные ядра. Какое ядро выбрано по умолчанию, здесь не записано, поэтому это наблюдение, а не причина.")
    return out


def render(summary: Dict[str, Any], env: Dict[str, Any], rental: Dict[str, Any], rel: str) -> str:
    c1, c2, c3, outcome = summary["control1"], summary["control2"], summary["control3"], summary["outcome"]
    row = outcome["row"]
    verdict = f"исход {row}: {ROW_TEXT[row]}" if row else "исход не определён: у контроля 2 нет валидных данных."
    out = [
        f"# A6 тикет 08: откуда расхождение L2L и накопления градиентов, DPO k=1, SmolLM2-135M, {rental['card']}, {rental['date']}",
        "",
        f"**Вердикт: {verdict}**",
        "",
        "Правило: `benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md`, закоммичено до драйвера и до аренды; одна поправка (маска заполнения в контроле 3) внесена после смоука драйвера на CPU, тоже до аренды. "
        "Тикет 06 не переписывается: DPO, ORPO, SimPO и KTO там остаются «не прошёл», этот отчёт никого не отмечает готовым.",
        "",
        "Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** (`examples/data/dpo_pairs_synthetic.jsonl`, 12 строк, "
        "шесть окон). Результат относится к одной ячейке (DPO, k=1), этой модели и этой карте. ORPO, SimPO, KTO, другие k, другие карты и модели не заявляются.",
        "",
        "## Стенд",
        "",
        *_table(["Параметр", "Значение"], [
            ["Сервер", rental["server"]],
            ["GPU", f"{env['nvidia_smi']}"],
            ["Образ", rental["image"]],
            ["Пакеты", f"torch {env['torch']}, transformers {env['transformers']}, trl {env['trl']}, peft {env['peft']}, CUDA {env['cuda']}, Python {env['python']} (полный список в `packages.txt`)"],
            ["RAM", f"{env['ram']}; {rental['uptime']}"],
            ["Коммит", f"`{rental['commit']}`"],
            ["Стоимость", rental["cost"]],
            ["Аккаунт", rental["account"]],
            ["Уничтожение", rental["destroy"]],
        ]),
        "",
        "## Команды",
        "",
        "```bash",
        *rental["commands"],
        "```",
        "",
        "Плечи, пути, типы и бэкенды SDPA описаны в правиле. Каждое плечо идёт в своём процессе под `ArmWatch`, обязательные проверки `suspend` и `power`.",
        "",
        "## Что показали данные",
        "",
        *what_showed(summary),
        "",
        f"## Контроль 1: float32 на CUDA, {STATE_TEXT[c1['state']]}",
        "",
        "Поля конфига и переменной окружения для типа нет (`.scratch/a6-l2l-preference/research-float32-on-cuda.md`). Float32 получен подменой в процессе драйвера двух функций, "
        "`resolve_stream_dtype` и `bf16_fp16_flags` у обёртки DPO. Это обход, а не штатный режим. Что реально стояло в плечах (запись изнутри первого вызова слоя 0 и из `TrainingArguments`):",
        "",
        *types_section(summary, ["native_ga1", "native_l2l", "f32_ga1", "f32_ga2", "f32_l2l"]),
        "",
    ]
    if c1["problems"]:
        out += ["Что нашёл судья:", "", *[f"- {p}" for p in c1["problems"][:8]], ""]
        if len(c1["problems"]) > 8:
            out += [f"И ещё {len(c1['problems']) - 8} записей в `diagnosis.json`.", ""]
    out += [
        "Наибольшая разница по шагам (градиент LoRA, потеря):",
        "",
        *_table(["пара", "шаг", "градиент", "потеря"], _pair_rows(summary, ["f32_ga2_vs_ga1", "f32_l2l_vs_ga1"])),
        "",
        f"## Контроль 2: пол шума от второго бэкенда SDPA, {STATE_TEXT[c2['state']]}",
        "",
        f"Плечи `native_*`, `MATH` против `EFFICIENT_ATTENTION` для пола, множитель 1,5. На шаге 0: {STATE_TEXT['within'] if c2.get('step0_within') else STATE_TEXT['outside'] if c2.get('step0_within') is False else 'нет данных'}. "
        f"Наибольший пол {_fmt(c2.get('max_floor'))}, наибольший разрыв L2L {_fmt(c2.get('max_gap'))}.",
        "",
        *_table(["пара", "шаг", "градиент", "потеря"], _pair_rows(summary, ["native_floor_vs_ga1", "native_l2l_vs_ga1", "native_ga2_vs_ga1"])),
        "",
    ]
    if c2["problems"]:
        out += ["Что вне предела:", "", *[f"- {p}" for p in c2["problems"][:8]], ""]
    out += [
        "## Повтор условий тикета 06",
        "",
        "Плечи `unpinned_*`: бэкенд SDPA не закреплён, как в тикете 06. В вердикт не входят.",
        "",
        *_table(["пара", "шаг", "градиент", "потеря"], _pair_rows(summary, ["unpinned_l2l_vs_ga"])),
        "",
        f"## Контроль 3: первый слой, где выходы расходятся, {STATE_TEXT[c3['state']]}",
        "",
    ]
    if c3["state"] == "not_needed":
        out += ["Контроль 1 дал побитовое равенство, контроль 3 не нужен и не запускался.", ""]
    elif c3["state"] == "differs":
        out += [
            f"Тип плеч: {c3.get('dtype')}. Первое расхождение: {'вложение' if c3['first'] == 'embed' else 'слой ' + str(c3['first'])}, "
            f"максимум разницы {_fmt(c3.get('max_abs_at_first'))}. Расходятся слоёв: {len(c3['differing'])}. "
            f"Сравнение только на позициях без заполнения.",
            "",
        ]
    else:
        out += [f"Тип плеч: {c3.get('dtype')}. Состояние: {c3['state']}. {c3.get('reason', '')}", ""]
    out += [
        "## Валидность замеров и попытки",
        "",
        *attempts_section(summary),
        "",
        "Испорченные замеры (`void`) исключаются по `benchmarks/METHOD.md`, запись остаётся в каталоге результатов. Если колонка «исключена» пуста, исключать было нечего.",
        "",
        "## Неудачные попытки",
        "",
        *([f"- {item}" for item in rental.get("failed", [])] or ["Неудачных попыток не было."]),
        "",
        "## Что не проверялось",
        "",
        "- ORPO, SimPO, KTO, другие k, другие модели, другие карты, NF4, реальные наборы.",
        "- Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами; чистое сравнение даёт шаг 0.",
        "- Скорость и память не измерялись.",
        "- Float32 получен подменой функций драйвера, а не штатной настройкой продукта. Он запускался только с закреплённым `MATH`; float32 при SDPA по умолчанию (условия тикета 06) не запускался.",
        "- Исход 2 держится на шагах 1 to 5; на шаге 0 L2L в пределе. Пол шума это одна выборка второго бэкенда SDPA.",
        "- Начальные адаптеры (`start.pt`) не сохранены и хеш не записан; одинаковый старт виден только из равенства на шаге 0 в float32.",
        "- Ставка рабочего инстанса оценена по предложению, счёта vast.ai нет; стоимость в таблице стенда приблизительная.",
        "- Типы весов записаны изнутри первого вызова слоя 0, а не в момент загрузки хранилища.",
        "",
        "## Файлы",
        "",
        f"- [diagnosis.json]({rel}/diagnosis.json), [environment.json]({rel}/environment.json), [rental.json]({rel}/rental.json)",
        f"- журналы и записи `ArmWatch` каждой попытки: `{rel}/<плечо>/attempt*/` (без каталогов `*-work`)",
        f"- [create.log]({rel}/create.log), [create-attempt1-failed.log]({rel}/create-attempt1-failed.log), [COMMIT]({rel}/COMMIT)",
        f"- [packages.txt]({rel}/packages.txt), [bootstrap.log]({rel}/bootstrap.log), [a8.log]({rel}/a8.log), [show-instances-after.txt]({rel}/show-instances-after.txt)",
        "- хеши `.pt` каждой попытки лежат в `diagnosis.json` (`attempts`); сами `.pt` в репозиторий не входят.",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="directory with diagnosis.json")
    args = parser.parse_args()
    results = Path(args.results).resolve()
    summary = json.loads((results / "diagnosis.json").read_text())
    env = json.loads((results / "environment.json").read_text())
    rental = json.loads((results / "rental.json").read_text())
    rel = results.relative_to(ROOT / "benchmarks").as_posix()
    target = ROOT / "benchmarks" / REPORT
    target.write_text(render(summary, env, rental, rel))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
