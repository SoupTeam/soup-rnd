# A0: L2L step 0 на Vast.ai RTX 3070, 2026-10-19

**Измерения завершены. G1–G4 PASS; полный ordinary-batch контроль не закрыт из-за V4 на batch 1/2/3.**

L2L k16: **74.84 tok/s**, ускорение **3.457×** относительно L2L k1; peak allocated **4.095 GB**.
Обычный batch 4: **78.71 tok/s**, peak allocated **7.360 GB**. На этой конфигурации L2L
медленнее него на **4.91%**, но расходует на **44.36%** меньше выделенной GPU-памяти.
Максимальный batch для измеренной конфигурации — **4**: batch 5 дал CUDA allocator OOM
в двух отдельных процессах. Это граница текущих настроек allocator/runtime, не всех возможных настроек GPU.

## Решение по A1 и B1

- **A1:** механизм прошёл G1–G4, но включение L2L по умолчанию ради скорости dense-70B
  на этой карте не обосновано: валидный обычный batch 4 быстрее. Прототип имеет смысл
  отдельно оценивать ради экономии VRAM или задач, где обычный batch ограничен памятью.
- Полное подтверждение ordinary-batch 1/2/3 требует новой заранее зафиксированной попытки
  в quiet window. Невалидные строки этой попытки сохранены; переизмерений с выбором лучших строк нет.
- **B1:** отдельная проверка MoE остаётся следующим исследованием. Этот замер не доказывает
  качество обучения, сходимость или скорость гигантских MoE. A1/B1 и C4 здесь не реализовывались.

## G1–G4 и валидность

| Gate | Статус | Результат |
| --- | --- | --- |
| G1 | pass | F1: 256 tensors + losses equal (deterministic); F2: 640 tensors + losses equal (deterministic) |
| G2 | pass | T_A(16) 74.8 vs 0.8 x T_B(16) 60.0 tok/s |
| G3 | pass | T_A(16) 74.8 vs 1.3 x T_A(1) 28.1 tok/s |
| G4 | pass | peak A16 4.095 vs A1 3.968 + 0.5 GB |

G1 использует предусмотренный исходным правилом deterministic fallback: обычный GA
не был битово воспроизводим сам с собой на F1/F2. В fallback GA/GA и L2L/GA совпали
точно: F1 256 и F2 640 ненулевых gradient tensors, плюс per-microbatch losses.

V1: direct I/O подтверждён. V2: plain и L2L k1 имеют 158 загрузок и 69.937 GB на шаг.
V3: перед CUDA 4 MiB, peers нет; доступно 23.026 GB RAM для 10.737 GB pinned + 4 GB margin.
Проверены 44 arm records и 88 GPU snapshots: везде один собственный
GPU PID, нет видимых тяжёлых чужих читателей; V5 не обнаружил battery/unknown samples
или gap >30 s. Все token-row fingerprints совпадают с общей последовательностью.
V4 проверялся для каждой пары; не прошли только отдельные ordinary batch 1/2/3.

Linux-питание явно декларировано как server, физического AC-датчика нет.
Чужой I/O наблюдается только в пределах процессов, видимых контейнеру; отсутствие
конкурирующего I/O других контейнеров/хоста этим не доказано. Причина разброса batch 1/2/3 не установлена.

## Основной timing: два раунда

Таблица использует tokens / среднее время двух раундов. Peak — максимум allocated PyTorch,
а не весь расход из nvidia-smi. A читает и копирует веса; B убирает чтение/H2D и служит
только timing ablation, без утверждения корректности его вычислений.

| k | A, tok/s | B, tok/s | A round spread | A peak GB | A/B valid |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.646 | 75.316 | 0.13% | 3.968 | True/True |
| 2 | 43.056 | 74.996 | 0.04% | 3.978 | True/True |
| 3 | 60.856 | 74.902 | 0.24% | 3.986 | True/True |
| 4 | 68.802 | 74.983 | 0.33% | 3.994 | True/True |
| 8 | 74.784 | 74.905 | 0.50% | 4.028 | True/True |
| 16 | 74.839 | 75.007 | 0.47% | 4.095 | True/True |

Дополнительные C/D k4: dequant share **15.00%**; C4/A4 time **0.9397**.
Plain A1: **21.192 tok/s**, spread **4.41%**, valid; plain B1: **81.748 tok/s**,
spread **1.22%**, valid. L2L B16 / (16 × plain B1) time = **1.0899**.

## Обычный batch и верхняя граница

| Batch | Раунд 0 tok/s | Раунд 1 tok/s | Среднее tok/s | Spread | Peak GB | Статус |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11.461 | 18.148 | 14.050 | 45.17% | 4.459 | INVALID V4; среднее не использовать как доказательство |
| 2 | 42.600 | 40.120 | 41.323 | 6.00% | 5.426 | INVALID V4; среднее не использовать как доказательство |
| 3 | 61.158 | 64.672 | 62.866 | 5.59% | 6.393 | INVALID V4; среднее не использовать как доказательство |
| 4 | 78.671 | 78.744 | 78.708 | 0.09% | 7.360 | valid |

Исходный протокол тестировал только batch 1–4. После того как batch 4 поместился,
до запуска batch 5 отдельно зафиксированы условия, команды и два процесса проверки границы.
Оба завершились с кодом 3 и memory.num_ooms=1; GPU перед каждым был свободен (1 MiB).
Новые данные не заменяют исходные строки и не снимают их INVALID V4.

**Исходный verdict.json оставлен без изменений:** `BATCH CONTROL INCOMPLETE: BUILD STEP 1`.
Его maximum_established=false относится к исходному плану 1–4. Отдельная проверка batch 5
устанавливает границу 4; control_complete всё ещё false из-за V4 на batch 1/2/3.

## Pinned RAM и spill

| Вариант | tok/s | Round spread | Peak allocated GB | Valid |
| --- | --- | --- | --- | --- |
| pinned | 74.685 | 1.14% | 4.095 | True |
| spill | 74.642 | 0.82% | 4.095 | True |

I4: spill / pinned = **0.999429**, выше порога 0.95; direct I/O=true.
Для 40 из 80 слоёв используется SSD: resident store 5.369 GB вместо 10.737 GB,
spill-файл 5.369 GB. Это бюджет хранилища активаций; уменьшение общего RSS процесса
не заявляется, поскольку allocator может сохранять освобождённые pinned pages.

## Стенд, версии и воспроизведение

- Vast.ai instance 56719626, machine 140736, host 583908, Бразилия.
- RTX 3070 8192 MiB; CUDA allocator total 8.220901376 GB; driver 580.159.04.
- PCIe Gen 3 x16; Core i7-10700K, 16 logical CPU; cgroup RAM 32,107,397,120 bytes.
- Один overlay-том 240 GiB. Видны два физических NVMe: T-FORCE 512 GB и Kingston 1 TB,
  плюс WD HDD 1 TB. Отдельный mount второго SSD не открыт; striping не проверялся.
- Python 3.12.13, torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2,
  bitsandbytes 0.50.2, accelerate 1.15.0. Полный freeze сохранён.
- F1: unsloth/mistral-7b-instruct-v0.3, revision 60b99dcbc7f853b7cd18472a607e8d045b96f03e;
  RAM NF4 cache 4,137,916,016 bytes.
- F2: synthetic Llama shape, 80 layers, hidden 8192, FFN 28672, heads 64, KV heads 8,
  vocab 32768, untied head; 68.99B parameters, 137.98 GB bf16.
  NF4 cache 36,388,401,328 bytes больше доступной RAM.
- NF4 double quant; buffers=2, read-ahead=2, seq=512, LoRA rank=8,
  targets=q/k/v/o, dropout=0, seed=3, input seed=17.
- Один warmup + три measured steps, два раунда. Автоматических timing retries не было.

Базовый commit: `3da204fadbd278c01ad5a4b23906b443d15e9dd1`; новых коммитов **0**, push/PR нет.
Архив измеренных исходников SHA256:
`d6fe64496f51e711dfed0371f8554deef170596b3a9842e332704aa9bb0d75ff`.
Протокол основной сессии SHA256:
`fa3dcad4b337c9e0a6497ccd76915c2e221c1514bf624168fd126818b8c3ccc3`.
Протокол проверки границы SHA256:
`b576924f47625f849b75dead4d54e2c8a776d07169b3292f0a77720434c8460d`.

608 Python source hashes до/после измерений совпали; локальные исходники тоже совпадают.
104 focused CPU-теста passed локально и на сервере, Ruff изменённых файлов passed.
Полный локальный Ruff имел 13 исходных I001 в нетронутых тестах; они не исправлялись.
Изменения касаются benchmark harness, Linux telemetry и воспроизводимости; `src/` не менялся.

Первая подготовка отказала V3 из-за Linux cgroup/cache accounting и GPU host PID, невидимого
в контейнерном /proc. До основной сессии исправлен учёт reclaimable inactive file cache
и добавлено строгое сопоставление единственного GPU PID при пустом списке до локальной
инициализации CUDA. preparation-01 сохранена; preparation-02 прошла реальное выделение
pinned buffers и direct I/O. Во время измерений исходники не менялись.

Приватный handoff владельца недоступен; использовано опубликованное правило.
Исторический результат RTX 5070 от 7 октября (G3/G4 NO VERDICT) не переписан.

Команды всех блоков находятся в protocol.json. Инструкции подготовки: [a0-reproduction.md](a0-reproduction.md).

## Сырые данные и проверка

- [Основной протокол и команды](results/a0-vast-rtx3070/a0-attempt-01/protocol.json)
- [Исходный verdict](results/a0-vast-rtx3070/a0-attempt-01/verdict.json)
- [Timing, все 32 точки](results/a0-vast-rtx3070/a0-attempt-01/timing.json)
- [Spill, все четыре точки](results/a0-vast-rtx3070/a0-attempt-01/spill.json)
- [Протокол дополнительной проверки batch 5](results/a0-vast-rtx3070/a0-batch-bound-01/protocol.json)
- [Batch 5: CUDA OOM round 0](results/a0-vast-rtx3070/a0-batch-bound-01/batch_5_r0.json)
- [Batch 5: CUDA OOM round 1](results/a0-vast-rtx3070/a0-batch-bound-01/batch_5_r1.json)
- [Полный package freeze](results/a0-vast-rtx3070/packages.txt)
- [SHA256 42 файлов, сверенных с сервером](results/a0-vast-rtx3070/remote-sha256.json)
- [Независимый локальный анализ всех arms](results/a0-vast-rtx3070/analysis.json)
- [Вывод проверки физических дисков](results/a0-vast-rtx3070/server-disks.txt)

В каталоге данных также сохранены preflight, все correctness и ordinary-batch JSON,
логи, обе подготовки и manifest исходников. Локальный strict evaluator воспроизвёл
gates/info итогового verdict без расхождений.

**Аренда Vast.ai instance 56719626 не остановлена автоматически.** Измерительные процессы
и сборщик завершились; прекращение аренды остаётся отдельным действием.
