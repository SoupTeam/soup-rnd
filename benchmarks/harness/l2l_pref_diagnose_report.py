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
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REPORT = "probe-vast-l2l-dpo-divergence-a6-ticket08.md"

ROW_TEXT = {
    1: "округление, ошибки нет. Критерий готовности методов предпочтений переписывается на допуск A3-07 новым правилом до нового прогона четырёх методов; в этом тикете он не переписывается.",
    2: "округление больше шума накопления. Решение о допуске за пользователем.",
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
        "## Что не проверялось",
        "",
        "- ORPO, SimPO, KTO, другие k, другие модели, другие карты, NF4, реальные наборы.",
        "- Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами; чистое сравнение даёт шаг 0.",
        "- Скорость и память не измерялись.",
        "- Float32 получен подменой функций драйвера, а не штатной настройкой продукта.",
        "",
        "## Файлы",
        "",
        f"- [diagnosis.json]({rel}/diagnosis.json), [environment.json]({rel}/environment.json), [rental.json]({rel}/rental.json)",
        f"- журналы и записи `ArmWatch` каждой попытки: `{rel}/<плечо>/attempt*/` (без каталогов `*-work`)",
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
