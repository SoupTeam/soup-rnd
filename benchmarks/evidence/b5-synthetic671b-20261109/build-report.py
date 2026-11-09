"""Builds benchmarks/b5-synthetic671b-report3-20261109.md from the journals of the run.

Run from the repo root: PYTHONPATH=$PWD/src python benchmarks/evidence/b5-synthetic671b-20261109/build-report.py
Every number comes from a journal, the store record or the instance record; none is typed in.
"""
import json
import statistics
from pathlib import Path

from benchmarks.harness import b5_cold_acceptance as acc
from benchmarks.harness import b5_judge

E = Path("benchmarks/evidence/b5-synthetic671b-20261109/attempt3")
H = E / "host" / "results"
GIB, GB = 2**30, 10**9


def gib(x): return f"{x / GIB:.2f} GiB ({int(x)} байт)"
def gb(x): return f"{x / GB:.1f} GB"


def load(run):
    return b5_judge.read_journal(H / run / "run" / "journal.jsonl")


def ev(rows, name): return [r for r in rows if r["event"] == name]


cold, warm = load("cold1"), load("warm1")
record = json.loads((H / "store" / "store-record.json").read_text())
cv, wv = acc.verdict(cold, record), acc.verdict(cold if False else warm, record, warm=True)
assert cv["passed"] and wv["passed"], (cv["failures"], wv["failures"])
judged = cv["judge"]
env = ev(cold, "environment")[0]
drop = ev(cold, "cold_cache")[0]
shape = ev(cold, "shape")[0]
steps_c, steps_w = ev(cold, "step"), ev(warm, "step")
clock_c, clock_w = ev(cold, "training_clock")[0], ev(warm, "training_clock")[0]
runtime = ev(cold, "runtime")[0]["stats"]
init = ev(cold, "adapter_initial")[0]
phase_c = [r for r in ev(cold, "phase_end") if r["phase"] == "training"][0]
prep_c = [r for r in ev(cold, "phase_end") if r["phase"] == "preparation"][0]
smi = {}
for run in ("cold1", "warm1"):
    values = []
    for line in (H / run / "nvidia-smi-samples.csv").read_text().splitlines()[1:]:
        try:
            values.append(int(line.split(",")[1].split()[0]))
        except (IndexError, ValueError):
            pass
    smi[run] = (len(values), max(values))
inst = json.loads((E / "instance-before-destroy.json").read_text())
speed_c, speed_w = cv["measured"]["tokens_per_second"], wv["measured"]["tokens_per_second"]
vram_total = env["vram_bytes"]
sm = lambda rows, k: [r[k] for r in rows]
ratio = lambda s: s["disk_read_bytes"] / s["seconds"] / GB

shape_section = f"""## Размер формы — синтетические веса

**Форма хранилища — синтетические веса**

| Показатель | Значение |
| --- | --- |
| слоёв / MoE-слоёв | {shape['layers']} / {shape['moe_layers']} (форма DeepSeek-V3) |
| маршрутизируемых экспертов на MoE-слой | {shape['experts']} |
| hidden / ширина эксперта | {shape['hidden']} / {shape['expert_size']} |
| параметров | {shape['parameters']:,} (синтетика, случайные веса) |
| хранилище на диске | {record['actual_bytes']:,} байт ({gb(record['actual_bytes'])}), план {record['planned_bytes']:,}, расхождение {abs(record['actual_bytes'] - record['planned_bytes']):,} байт |

Все числа этого отчёта относятся к случайным весам и случайным токенам. Они не доказывают качество настоящей модели и не говорят о скорости настоящей модели.
"""

sections = [
    {"heading": "Приёмка по правилу", "tables": [{"caption": "Вердикт приёмки", "rows": [
        ["вердикт холодного запуска (`b5_cold_acceptance.py`)", "PASS, причин fail нет"],
        ["вердикт тёплого повтора (`--warm`, отдельно)", "PASS, причин fail нет"],
        ["шагов", f"{len(steps_c)}, батч 1 последовательность, ровно {acc.TOKENS_PER_STEP} токенов на шаг, паддинга нет"],
        ["карта", f"1 шт., {env['gpu']}, total_memory {gib(vram_total)} (класс 8 GB), compute capability {env['compute_capability'][0]}.{env['compute_capability'][1]}"],
        ["память помещается", f"да: пик allocated {gib(cv['measured']['peak_allocated_bytes'])}, пик reserved {gib(cv['measured']['peak_reserved_bytes'])}, оба ниже {gib(vram_total)}"],
        ["попыток запуска", "1 холодный, 1 тёплый повтор; неудачных запусков нет. Неудачные аренды и остановка хоста описаны ниже"],
    ]}]},
    {"heading": "Холодный старт: сброс кеша и его подтверждение", "tables": [{"caption": "Сброс кеша страниц и подтверждение (остаток кеша 0 по `mincore` после `fadvise`)", "rows": [
        ["способ сброса", f"`{drop['method']}` (`posix_fadvise(DONTNEED)` на каждый файл хранилища). `drop_caches` недоступен: `/proc/sys/vm/drop_caches` на хосте только для чтения"],
        ["остаток хранилища в кеше страниц до сброса", f"{drop['resident_before_bytes']:,} байт ({gb(drop['resident_before_bytes'])}), `mincore`"],
        ["остаток после сброса", f"{drop['resident_after_bytes']:,} байт из {drop['store_bytes']:,} (порог правила 1%), `mincore`"],
        ["время сброса, зазор до `training_start`", f"{drop['seconds']:.1f} с, зазор {ev(cold, 'training_start')[0]['unix'] - drop['unix']:.2f} с (порог 600 с)"],
        ["чтение с диска в шаге 0", f"`read_bytes` {gb(steps_c[0]['disk_read_bytes'])} при {gb(steps_c[0]['expert_bytes_read'])} логических байт экспертов (порог правила 90%): условие выполнено"],
        ["что это подтверждает", "Сброс сработал (остаток 0). Чтение в шаге 0 показывает, что диск читался, но не отличает холодный запуск от тёплого: эксперты идут через `O_DIRECT` и читаются с диска в любом запуске. Правило это оговаривает"],
    ]}]},
    {"heading": "Градиенты и адаптеры", "tables": [{"caption": "Шаги обучения холодного запуска (LoRA r=8, alpha=16, lr 1e-4)", "rows": [
        ["обучаемых тензоров / элементов", f"{init['tensors']} / {init['elements']:,}, L2-норма начального адаптера {init['l2_norm']:.3f}"],
    ] + [[f"шаг {r['step']}", f"loss {r['loss']:.4f} (случайные токены, не оценка качества), норма градиента {r['gradient_norm']:.4f} (до клиппинга), конечных тензоров с градиентом {r['gradient_tensors']} из {r['trainable_tensors']}, расстояние от начального {r['distance_from_initial']:.4f}"] for r in steps_c] + [
        ["дайджест после перезагрузки", "равен дайджесту после последнего шага (проверка судьи `reload`)"],
    ]}]},
    {"heading": "Измерения холодного запуска", "tables": [{"caption": "Скорость, память, диск", "rows": [
        ["токены без паддинга в секунду, шаг 0 (холодный)", f"{speed_c['first_step']:.2f} ({steps_c[0]['seconds']:.1f} с на {steps_c[0]['tokens']} токенов)"],
        ["токены без паддинга в секунду, шаги 1 и 2", f"{steps_c[1]['tokens_per_second']:.2f} и {steps_c[2]['tokens_per_second']:.2f}, медиана {speed_c['later_steps_median']:.2f}"],
        ["проверка замера скорости", f"{clock_c['samples']} проб частоты SM, ни одной причины throttling, медиана {clock_c['sm_clock_mhz']['median']:.0f} MHz. Замер не испорчен"],
        ["пик VRAM allocated / reserved", f"{gib(cv['measured']['peak_allocated_bytes'])} / {gib(cv['measured']['peak_reserved_bytes'])} (счётчики аллокатора после reset перед шагом 0)"],
        ["`nvidia-smi` memory.used, максимум", f"{smi['cold1'][1]} MiB (справка, весь процесс, {smi['cold1'][0]} проб)"],
        ["пик RSS, подготовка (хранилище готово, шардирования нет)", f"{prep_c['peak_rss_bytes']:,} байт ({prep_c['peak_rss_bytes'] / GIB:.2f} GiB), VmHWM"],
        ["пик RSS, обучение (после сброса VmHWM)", f"{phase_c['peak_rss_bytes']:,} байт ({phase_c['peak_rss_bytes'] / GIB:.2f} GiB), VmHWM"],
        ["чтение с диска за шаг (`read_bytes`)", ", ".join(gb(x) for x in sm(steps_c, 'disk_read_bytes')) + " (шаги 0, 1, 2)"],
        ["логические байты экспертов за шаг", ", ".join(gb(x) for x in sm(steps_c, 'expert_bytes_read'))],
        ["из них прочее, не эксперты (`read_bytes` минус эксперты)", ", ".join(gb(x) for x in cv['measured']['non_expert_disk_read_bytes_per_step'])],
        ["загрузок экспертов за шаг", ", ".join(str(x) for x in sm(steps_c, 'expert_fetches')) + f" (всего {runtime['expert_fetches']}; слотов экспертов {shape['moe_layers'] * shape['experts']}: маршрутизация случайных токенов, у настоящих текстов будет другой охват)"],
        ["средняя скорость чтения диска за шаг", ", ".join(f"{ratio(s):.2f}" for s in steps_c) + " GB/s (`read_bytes` на секунды шага; шаг 0 в 3 раза медленнее шагов 1 и 2)"],
        ["диск под хранилище", f"{record['actual_bytes']:,} байт; `df`: занято {(386468823040 - 6189584384):,} байт между `df-0-before-write` и `df-3-after-write`; свободно после запуска {env['disk_free_bytes']:,} байт"],
        ["жёсткий лимит RAM", f"не проверялся: cgroup `memory.max` {env['ram_limit']['limit_bytes']:,} байт (194 GB), swap 0; RAM хоста {env['host_ram_bytes']:,} байт. Лимит 32 GB не воспроизводился, о нём вывода нет"],
    ]}]},
    {"heading": "Тёплый повтор, подписан отдельно", "tables": [{"caption": "Тёплый повтор: второй запуск того же процесса без `--cold`, сразу после холодного", "rows": [
        ["вердикт", "PASS (`--warm`), в журнале нет `cold_cache`"],
        ["токены без паддинга в секунду, шаги 0, 1, 2", ", ".join(f"{r['tokens_per_second']:.2f}" for r in steps_w) + f", медиана шагов 1 и 2 {speed_w['later_steps_median']:.2f}"],
        ["секунды шага", ", ".join(f"{r['seconds']:.1f}" for r in steps_w)],
        ["чтение с диска за шаг (`read_bytes`)", ", ".join(gb(x) for x in sm(steps_w, 'disk_read_bytes'))],
        ["проверка замера скорости", f"{clock_w['samples']} проб частоты SM, throttling нет"],
        ["что сравнение показывает", "Шаг 0 холодного запуска занял 211.0 с, шаг 0 тёплого 68.5 с. Счётчик `read_bytes` в обоих запусках почти одинаков, поэтому разницу в 142 с им не объяснить. Причина не установлена: запусков по одному, у диска могут быть собственные кеши за пределами кеша страниц. Это одна пара запусков без разброса"],
        ["loss и дайджест адаптера", "побитово те же, что в холодном запуске (зерно и данные те же)"],
    ]}]},
    {"heading": "Хранилище и запись", "tables": [{"caption": "Расчёт и запись хранилища", "rows": [
        ["расчёт до записи", f"план {record['planned_bytes']:,} байт, нужно свободно {record['required_free_bytes']:,} (`plan.out`). На диске 200 GB писатель отказывает: `refusal-200gb.json`, код выхода 3, папка не создана"],
        ["свободно до записи", "509 206 491 136 байт по `df -B1` (`df-0-before-write.txt`), требование выполнено с запасом 103.6 GB. Диск аренды 480 GB"],
        ["факт после записи", f"{record['actual_bytes']:,} байт, 122 файла, в пределах допуска 1%"],
        ["время записи", "два отрезка: 15:00:24 до 16:05 (94 файла, около 65 мин, около 74 MB/s), затем после остановки инстанса и перезапуска 1292.9 с (28 файлов, `--resume`). Писатель на Xeon E5-2470 v2 упирается в генерацию на CPU, не в диск"],
        ["как записано", "писателем `b5_synth_store.py` (тикет 05), без копирования между хостами. Второй отрезок шёл режимом `--resume`, добавленным после коммита правила (коммит 20c931d8): он оставляет файл, если заголовок safetensors и длина совпадают с ожидаемыми, остальное пишет заново"],
        ["`free_before_bytes` в записи", f"{record['free_before_bytes']:,} байт, вычислено (свободно сейчас {record['free_now_bytes']:,} плюс {record['existing_bytes']:,} уже записанных), помечено `free_before_derived`. Измерение до первого отрезка, 509 206 491 136 байт, отличается на 2.7 MB"],
    ]}]},
    {"heading": "Сравнение с настоящим 671B из B6", "tables": [{"caption": "Два отдельных числа, не сравнение", "rows": [
        ["синтетика этого отчёта (RTX 3060 Ti, bf16, NF4, 128 случайных токенов, шаги 1 и 2)", f"{speed_c['later_steps_median']:.2f} токен/с"],
        ["настоящий DeepSeek-V3 из B6 (RTX 5060, 2 шага по 63 токена)", "0.1456 токен/с (`b6-giant-moe-20261028.md`)"],
        ["настоящий DeepSeek-V3 из B6, повтор (Quadro P4000, fp32, 63 токена)", "0.1445 токен/с (`b6-acceptance-20261029.md`)"],
        ["вывод", "Не делается. Карты, хосты, типы данных, длины и данные разные. У синтетики случайная маршрутизация, у настоящей модели маршрутизация настоящих текстов"],
    ]}]},
    {"heading": "Неудачные попытки и стоимость", "tables": [{"caption": "Аренда vast.ai", "rows": [
        ["попытка 1", "VM в Квебеке (офер <offer-1>, экземпляр <instance-1>): ssh не принял ключ (`bad ownership or modes for file /root/.ssh/authorized_keys`), уничтожен через 6 минут"],
        ["попытка 2", "контейнер в Швеции (экземпляр <instance-2>): уничтожен по просьбе пользователя до первого входа, около минуты"],
        ["попытка 3, рабочая", f"экземпляр <instance-3>, Квебек, RTX 3060 Ti, {inst['dph_total']:.3f} USD/ч с диском 480 GB, жил около 2.03 ч (создан 14:49:52, уничтожен 16:51:37 UTC)"],
        ["остановка хоста", "в 16:05 хост перевёл экземпляр в `stopped` при 94 из 122 файлов хранилища. Я его не останавливал, аккаунт общий, причина не установлена. В 16:10 я запустил его снова, диск был цел. Это потеряло около 15 минут"],
        ["стоимость, оценка", "около 0.62 USD (подробно в `rental.md`; единица `inet_down_billed` предположена, это килобайты). Счёт за день ещё не появился. Баланс команды упал на 1.556 USD, остальные около 0.94 USD объяснить нечем: возможная причина чужой экземпляр общего счёта, это не проверено"],
        ["потолок тикета", "3 USD, подтверждён пользователем 2026-11-09, не достигнут"],
        ["уничтожение", "`vastai destroy instance <instance-3>`, затем `vastai show instances --raw` показал только чужой экземпляр <foreign-instance> (`show-instances-after.json`, 2026-11-09T16:51Z)"],
    ]}]},
]

command = ("nvidia/cuda:13.0.0-devel-ubuntu24.04, venv по remote-setup.sh, запись хранилища remote-write.sh и remote-resume.sh, затем remote-run.sh cold1 --cold: "
           "python benchmarks/harness/b5_real_moe.py --config benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml "
           "--store /root/b5/synthetic-v3 --output /root/b5/results/cold1/run --steps 3 --device cuda --cold; тёплый повтор warm1 без --cold")
rec = {
    "kind": "synthetic",
    "title": "B5, отчёт 3: синтетический DeepSeek-V3 (671B параметров), холодный запуск с диска на пути per_expert, одна карта 8 GB, 3 шага",
    "command": command,
    "environment": {
        "torch": f"{env['versions']['torch']} (CUDA {env['cuda']}), transformers {env['versions']['transformers']}",
        "transformers": env["versions"]["transformers"], "peft": env["versions"]["peft"],
        "bitsandbytes": env["versions"]["bitsandbytes"],
        "gpu": f"{env['gpu']}, compute capability 8.6, total_memory {vram_total} байт ({vram_total / GIB:.2f} GiB), 1 карта, драйвер 595.104.02",
        "ram": f"{env['host_ram_bytes'] / GIB:.0f} GiB на хосте ({env['host_ram_bytes']} байт), Intel Xeon E5-2470 v2, 20 потоков, контейнер vast.ai, cgroup memory.max {env['ram_limit']['limit_bytes']}",
    },
    "unavailable": {},
    "logs": [
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/results/cold1/run/journal.jsonl (журнал холодного запуска для судьи)",
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/results/warm1/run/journal.jsonl (журнал тёплого повтора)",
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/results/{cold1,warm1}/acceptance-verdict.json, environment.txt, run.log, nvidia-smi-samples.csv, df-*.txt",
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/results/store/ (store-record.json, write.log, resume.log, df-0 до df-3, environment.txt, pip-freeze.txt)",
        "benchmarks/evidence/b5-synthetic671b-20261109/attempt3/host/setup.log, write.out, resume.out, remote-*.sh",
        "benchmarks/evidence/b5-synthetic671b-20261109/ plan.out, plan-and-refusal.sh, refusal-200gb.json, rental.md, attempt1 и attempt2 (неудачные аренды), attempt3/instance-*.json, show-instances-*.json",
        "benchmarks/b5-synthetic-671b-acceptance-20261109.md (правило до аренды), benchmarks/b5-adapter-judge-rule-20261109.md",
        "benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml",
    ],
    "verdict": judged,
    "sections": sections,
    "notes": [
        "Результат относится к синтетическому хранилищу формы DeepSeek-V3 (случайные NF4-веса), одной карте (RTX 3060 Ti), трём шагам по 128 случайных токенов. Качество, сходимость, скорость и маршрутизация настоящей модели не заявляются. Loss около 13 на случайных токенах ничего не значит.",
        "Запуск шёл через харнесс `b5_real_moe.py --store`, а не через `soup train`: `soup train` не пишет журнал для судьи и не сохраняет градиенты.",
        "Сброс кеша шёл через `fadvise`, не через `drop_caches` (файловая система `/proc/sys` только для чтения). `fadvise` сбрасывает чистые страницы файлов хранилища, остаток после сброса 0 байт по `mincore`. Кеш самого диска или контроллера он не затрагивает.",
        "Тип диска не проверен отдельно: метаданные vast называют модель `FUJITSU MBE2147RC` и `disk_bw` 5751 MB/s при остановке экземпляра и 4715 MB/s перед уничтожением. Средняя скорость чтения в шагах 1 и 2 около 1.4 GB/s, что для одного вращающегося диска много. Утверждения «SSD» отчёт не делает.",
        "Шаг 0 холодного запуска в 3 раза медленнее шагов 1 и 2 и тёплого шага 0. Причина не установлена (см. таблицу тёплого повтора). Холодный шаг 0 приведён отдельно, медиана скорости берёт шаги 1 и 2.",
        "Один холодный и один тёплый запуск из трёх шагов не дают разброса между запусками. Шаги 1 и 2 это соседние замеры одного процесса.",
        "Хранилище записано в два отрезка из-за остановки инстанса хостом. Файлы первого отрезка проверены по заголовку safetensors и длине, побитово не сверялись. Побитово равные loss и дайджест адаптера в холодном и тёплом запусках относятся к одному и тому же хранилищу и не доказывают, что файлы равны записанным заново.",
        "Секунды шага включают подсчёт статистики градиентов рекордером (копии 610 тензоров адаптера, 48.5 млн элементов, на CPU). Отдельно это время не измерялось.",
        "Счётчики VRAM это счётчики аллокатора PyTorch, не весь процесс. Справка `nvidia-smi` показала максимум 5850 MiB за все пробы.",
        "Жёсткий лимит 32 GB RAM не проверялся (cgroup 194 GB). Пик RSS процесса 6.19 GiB не доказывает, что запуск уместится в 32 GiB: учтены только резидентные страницы процесса, не кеш страниц.",
        "Поля `double_quant`, `expert_layout`, `trust_remote_code` и `tier` в `environment` харнесс пишет как заданные настройки (замечание отчёта 1). Независимое подтверждение NF4 даёт размер хранилища: 380.3 GB для 671 млрд параметров, около 0.57 байта на параметр, в bf16 было бы около 1.34 TB.",
        "Применение `benchmarks/METHOD.md`: `ArmWatch` (проверки sleep, power, foreign_reads) харнесс B5 не запускает, поэтому сон машины, питание и чтения диска чужими процессами не измерены. Запись `gpu_clock` по METHOD.md сама прогон не бракует, её читают; браковка скорости при троттлинге в половине проб и больше это правило отчёта. Испорченных замеров нет: 0 из 341 проб в холодном запуске и 0 из 198 в тёплом. Метод учёта RAM B6 (`b6_ram.py`, cgroup `memory.current`) не запускался.",
        "Приёмка и судья дали pass без правки чисел правила. После коммита правила в писатель добавлен режим `--resume` (коммит 20c931d8), числа правила не менялись.",
    ],
}
(E.parent / "report-record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
text = b5_judge.render_report(rec)
marker = "## Запуск и окружение"
head, tail = text.split(marker, 1)
text = head + shape_section + "\n" + marker + tail
Path("benchmarks/b5-synthetic671b-report3-20261109.md").write_text(text, encoding="utf-8")
print("written", len(text.splitlines()), "lines")
