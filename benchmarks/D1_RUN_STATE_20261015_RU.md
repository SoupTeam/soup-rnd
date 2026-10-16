# D1: сбор данных завершён, аренда закрыта 16 октября 2026

Актуальный итог: **98/98 ячеек исследованы, 81/98 сравнений принято**.
Проекции: 80/96 принято, 16 NO_VERDICT; A100 cached-zero ablation тоже NO_VERDICT.
QA: пять допустимых ABBA-блоков, 20 полных запусков. Полная приёмка D1 остаётся
открытой из-за отклонённых таймингов, отсутствия отдельной телеметрии GEMM ceiling
и ожидающего независимого review. Критерии не ослаблены.

## Итоговые файлы

- [Исследовательский отчёт](d1-final-research-20261016.md).
- [Пакет проверенных результатов](evidence/d1-a100-k-20261016/README.md):
  русский отчёт, полный index/CSV, coverage figure, CPU audits, manifest каждого
  сырого файла и скрипт CPU-проверки для reviewer.
- Локальный ZIP для передачи: `C:\Users\esenb\Desktop\D1_TEAMLEAD_20261016.zip`.
  954,443,728 bytes, SHA-256
  `a4dc005e549e3c5e904caa7baea97977ad234c7ea21a508799553cf506967828`.
  Внутри пакет, полный K raw archive, отдельный исторический RTX 3090 archive,
  подтверждение локального consumer replay и краткие расходы без данных аккаунта.
- Канонические локальные файлы:
  `C:\Users\esenb\.codex\worktrees\soup-d1-nf4-study\cloud-a100-k-20261015`.
  Подпапки `teamlead-packet`, `offline-k`, `all-evidence`, `final-consumer-replay`.
  Большой raw archive передаётся в ZIP и **не добавлен в Git**.

Измеряемый source неизменен: `87b3bc1747ff78ff0fe16b8bdd2fa2eb0105d77c`.
Последующие reporting commits не являются новыми CUDA прогонами.
Ветка `koko/d1-nf4-study`, draft PR https://github.com/SoupTeam/soup-rnd/pull/16.
Оригинальный D2 checkout и его пакеты не изменялись этой работой.

## Проверенные результаты

- 2,256 попыток; 1,892 допустимых arm в принятых сравнениях. Исходный report builder
  сохраняет INCOMPLETE. Coverage-controller не повторял clock-invalid ячейки.
- FP16/BF16 alias correctness PASS; 12 GPU regression tests passed, 17 deselected,
  0 failed/skipped/errors. CPU replay проверил 2,216 projection witnesses:
  2,054 exact-native и 162 заранее объявленных finite dense/fused differences.
- K QA: 20 × 276 обновлений и 64 predictions. Все 176 адаптерных матриц изменились;
  initial/final tensors, losses, predictions и gold совпали во всех 20 повторах.
  EM 50.0, F1 52.7815934; 9/29 ложных ответов среди unanswerable.
  Один фиксированный seed и маленький split не являются general quality validation.
- A100 QA native/forced: 325.0902 / 368.8206 ms; allocated 863,867,904 bytes,
  reserved 1,042,284,544 bytes с обеих сторон. Non-padding throughput
  781.244 / 685.781 tokens/s. Отношения описательные, без significance claim.
  RTX 3090 F показан отдельно; качество между cohorts различается, причина
  этого различия отдельным контролем не изолирована.
- Свежий локальный consumer replay завершился с exit 0 и PASS: все raw hashes,
  пересборка индекса, redraw figure, correctness/projection/mechanism/QA audits.
  Использован отдельный checkout source 87b3bc17; CUDA contexts/model downloads — 0.
  Это локальная проверка воспроизводимости, независимая приёмка ещё не выполнена.
- На CPU использовался Python 3.10.0, torch 2.13.0+cpu, rich 14.0.0,
  numpy 1.26.4, matplotlib 3.10.1. CUDA stack записан отдельно в неизменном raw.
  65 D1 CPU tests и выбранные проверки прежнего CI были зелёными;
  текущий reporting head проверяется отдельно в PR.

## Закрытие и расходы

Аренда **56108892 удалена**, официальный CLI подтвердил отсутствие
16 октября 2026 в **00:16:16 UTC / 05:16:16 Asia/Qyzylorda**.
Новых GPU jobs нет. Наблюдавшиеся начисления: $6.782 текущая аренда + $1.279
предыдущие = **$8.061 из разрешённых $10**. Это invoice snapshot с округлением
провайдера, не утверждение о финальном settlement аккаунта.

K archive: 909,977,469 bytes, SHA-256
`d90fb904ebab09cf7f94d2b460f7b3bae2697c8aea4ef0c1dba2420479a49eb3`.
14,285 файлов / 51,070,679,238 logical bytes, все SHA проверены; дубликаты —
hard links. Локальная проверка 9,861 уникального file hash не нашла мастер-ключ
Vast или приватный SSH-ключ. Сервер отдельно проверил своё instance credential.

При выгрузке шесть SFTP ranges завершились; метаданные двух медленных ranges
не показывали прогресс. Исходные partials сохранены после контролируемой остановки.
Два полных ranges восстановлены через SSH exec за 35.84 s, затем общий SHA
совпал. `finish_verified_archive.py` продолжил прежние materialize/scan/delete/
offline этапы. Научные прогоны и raw bytes не менялись; журнал — в пакете.
Старый transport error сохранён отдельно, текущей ошибки экспорта нет.

Для полной приёмки нужен отдельный заранее зафиксированный эксперимент с
доступным clock lock и telemetry window GEMM ceiling, затем независимый reviewer.
Старые NO_VERDICT не исправлять фильтрацией/нормализацией, physical-GPU cohorts
не объединять. sm120, 8B/70B full-model multiplier и новый fused backward не заявлены.
Сообщения тимлиду, merge/issue close и GitHub Release не выполнялись.

<details>
<summary>История возобновления 15 октября; ниже не текущее состояние</summary>

## Историческая операционная точка

Это операционная точка продолжения, **не итоговый отчёт и не утверждение о завершении D1**.
Историческое сохранение перед паузой: `D1_RESUME_20261015_RU.md`.
Пользователь возобновил работу и попросил закончить как можно быстрее.
Общий разрешённый бюджет остаётся $10; повторно запрашивать его без необходимости не нужно.

## Текущий эксперимент

- Рабочая ветка `koko/d1-nf4-study`, draft PR https://github.com/SoupTeam/soup-rnd/pull/16.
- Измеряемый K source: `87b3bc1747ff78ff0fe16b8bdd2fa2eb0105d77c`.
- Прежний K контроллер продолжен с контрольной точки после полного завершения
  текущего child, без повторения готовых сравнений. Операционная бюджетная
  ревизия: `d1-resource-budget-revision-20261015.md`; научные настройки прежние.
- Позже K остановился после 26 принятых сравнений: FP16 M=8, N=14336, K=4096
  dense/native нарушил прежний clock guard. Все пять ABBA-блоков этой ячейки
  отклонены; повторять её или ослаблять критерий нельзя. До следующих запусков
  зафиксирован `d1-coverage-continuation-20261015.md` (commit f5cdf85c).
  Внешний `launch-k-coverage.py` на прежнем чистом source 87b3bc17 собирает
  остальные независимые ячейки; coverage и acceptance учитываются раздельно.
  Исходный `d1_full_report.py` не менялся и оставляет INCOMPLETE при NO_VERDICT.
- Аренда **56108892**, machine 145857, NVIDIA A100-SXM4-40GB,
  UUID `GPU-b0245e84-f202-902b-9331-f732ab021e57`.
- Цена с диском 100 GB: $0.6944444444/ч. Предыдущие наблюдавшиеся расходы $1.279;
  резерв предыдущих расходов $1.50 и сети/неопределённости $0.30.
  Максимум 11.75 часа новой аренды; общий консервативный потолок $9.959722222.
- Измерительный deadline: 16 октября **05:47 Asia/Qyzylorda**;
  абсолютное удаление: 16 октября **07:17 Asia/Qyzylorda**.
  Эти пределы не продлевать. Значения Unix — в public `lease.json`.
- Изолированный D1 checkout:
  `C:\Users\esenb\.codex\worktrees\soup-d1-nf4-study\soup-rnd-d2`.
  Исходный D2 checkout и его пакеты не менять.

До запуска K выполнена диагностика J: FP16/BF16 correctness PASS,
GPU-регрессии PASS и два допустимых одноблочных clock-screen — 1095/1095 MHz
для M1 и 1410/1410 MHz для M2048. Они не входят в пять полных блоков K.
Locked clocks и application clocks провайдер запретил. Все решения используют
реальные частоты во время измерений и прежний guard ≤5%.

Первый полный J был остановлен: абляция дала 1185–1335 MHz и нарушила guard.
Сохранять J как неудачную попытку, не смешивать её тайминги с K.
K заново запускает correctness, GPU-регрессии и полный план: сначала
96 сравнений проекций, затем абляция с 200 warm-up шагами, затем QA с отдельным
200-step synthetic prelude и неизменными 276 основными обновлениями.
Правила: `d1-amendment-20261015-j.md`, `d1-amendment-20261015-k.md`.

## Управление и сохранение

Папки находятся в `C:\Users\esenb\.codex\worktrees\soup-d1-nf4-study`:

- `cloud-a100-k-20261015`: текущие `lease.json`, SSH transport,
  `progress-latest.json`, `finish_local.py`, `finish-detached.log`,
  `finish-detached-error.log`. Read-only `watch_k_v3.py` следит за прогрессом
  через `watch-detached.log`. Текущий exit marker — `setup-k-coverage.exit`.
  Прежний `setup-k-resume1.exit=1` относится к clock-остановке и не запускает
  текущую выгрузку. После сброса tool-host локальные helper processes были
  восстановлены вне его job: WMI-created hidden supervisor 6856 запустил
  finish launcher 20768 / worker 8204, watch 18892 / 19360 и budget guard
  11840 / 2028. Актуальные PID проверять по CIM, не по старому номеру.
- `cloud-a100-j-20261015`: исходная аренда того же инстанса, GPU context,
  установка Ubuntu/Python/torch, диагностические логи и **активный локальный
  бюджетный guard `api_budget_v2.py`** (`guard-v2-last.json`). Старый guard/finish
  J остановлены; текущий finish — K. Remote guard после замены имеет PID 54005.
- `cloud-a100-i-20261015`: неработавший первый хост 56106516. Он не дошёл до SSH
  или численных запусков; удалён и проверен отсутствующим. Логи загрузки сохранены.
- Старые `cloud-control`, `cloud-full-20261015`, `cloud-a100-g-20261015`
  содержат прежние архивы; их аренды удалены.

На GPU: `/workspace/d1/source`, `/workspace/d1/results/full-k`,
сохранённые `/workspace/d1/results/full-j` и `diagnostic-j`.
Текущий лог — `/workspace/d1/results/full-k-coverage.log`; `progress.json`
содержит `processed_cases`, `terminal_no_verdict_cases` и отдельно принятые
`completed_cases`. `coverage-complete.json` означает только обработку всех
98 ячеек, без глобального PASS. Validation перед запуском: 26 accepted reused,
1 terminal not retried, 71 pending, 0 новых CUDA contexts в проверке контроллера.

Снимок до продолжения сохранён локально как `d1-a100-k-before-coverage.tgz`:
177,094,108 bytes, SHA-256
`3ed056fd904cd44926a1a0cc369a33ccf6e7f53d4244c2d0c5a9cab85133caf8`.
4,929 файлов, 10,989,497,459 logical bytes; архив скачан за 343.33 s восемью
bounded range readers и проверен по общему SHA. Это ещё не materialization
каждого файла. Итоговый fetch также использует bounded parallel ranges.
Частичные прежние скачивания сохранены отдельно и не считаются доказательством.
Remote guard с привязанным per-instance credential работает независимо от
локального ноутбука и удаляет только свой инстанс по deadline. Приватный ключ SSH
и мастер API-ключ остаются локальными, не копировать их на сервер.
`supervise-d1-detached.ps1` в control parent запускает helpers через скрытый
WMI supervisor, вне дерева процессов инструмента. Он переиспользует уже
работающие роли и не создаёт дубликаты. `supervisor-detached.json` хранит их PID;
guard logs — `budget-detached*.log` в папке J. При сбое мониторинга сначала
проверить существующие helpers и удалённый процесс; CUDA job повторно не запускать.

Локальный K finish-controller после любого завершения, включая ошибку:

1. Архивирует все завершённые, невалидные и прерванные результаты J/K и диагностику.
2. Скачивает `d1-a100-k-results.tgz`, проверяет SHA-256 архива и каждого файла.
3. Проверяет отсутствие точных совпадений с локальным API-ключом/приватным SSH-ключом.
4. Удаляет ровно 56108892, проверяет отсутствие и сохраняет `released.json`.
5. Запускает `finalize_analysis.py`: полный индекс, CPU numerical replay,
   QA audit при допустимых повторах, CSV и график только при полной валидности.
   Результаты — `offline-k`; `status.json` явно сохраняет INCOMPLETE при нехватке данных.
   Затем `build_packet.py` собирает `teamlead-packet/D1_TEAMLEAD_RU.md`,
   таблицы, manifest, подтверждение закрытия аренды и DATA_NOTICE.

Если контроллер записал `finish-error.json`, устранить причину до deadline;
не запускать новый измерительный процесс и не переименовывать старые выводы.
Файл `started.json` и remote `setup-k-started` защищают от дублирующего запуска.

## Что уже добавлено в ветку

`benchmarks/evidence/d1-efg-replays-20261015`: шесть PASS correctness replays
E/F/G, replay всех 20 QA-запусков каждого E и F, F projection replay,
исторический неполный F index и G diagnostic summary. Они относятся к своим
старым CUDA source, а не к K. F имеет пять допустимых ABBA-блоков QA;
native 241.1833 ms, forced 255.9017 ms, одинаковые allocator peaks и качество.
Маленькая фиксированная выборка и один seed не доказывают общее качество модели.

После выгрузки K нужны итоговый честный отчёт, интерпретация полной сетки,
пакет файлов тимлиду, проверка CI и актуализация draft PR. Сообщения тимлиду
и публикация GitHub Release не поручались. Не объявлять глобальное завершение
по части сетки, по одним correctness-тестам или по сгенерированному отчёту.

Локально все D1 CPU tests: **65 passed**, полный Ruff чистый. Повторный CI run
38754323943 для измеряемого source 87b3bc17 завершён успешно, включая Ubuntu
Python 3.12; остальные платформы/smoke были пропущены планом quick CI.
Не выдавать это за полный cross-platform или финальный-head CUDA прогон.
Для doc-head 9037f689 CI run 38765064152 также завершён успешно, включая Ubuntu
Python 3.12. После новых документов нужно проверить актуальный CI отдельно.

</details>
