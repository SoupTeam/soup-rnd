# D1: короткая проверка реальной Mistral-7B, 2026-10-19

**На полном шаге Mistral-7B в RAM-стриминге Soup кандидат сэкономил63.5 MiB
allocated (7.90%) и32 MiB reserved (3.87%).** Два запуска каждого пути дали
одинаковые соответствующие peaks. В коротком ABBA время практически совпало:
native1.726373 s, candidate1.730409 s (+0.234%). По одному блоку нельзя установить
достоверную разницу скорости. Это `CLEAN_DESCRIPTIVE_SCREEN`, **не новый formal F1**.

Это меньшая экономия, чем синтетические157/140 MiB и около40%. Синтетические
проценты нельзя выдавать за экономию реальной7B. Здесь также не повторилось
синтетическое замедление в1.62 раза. Это отдельные workload/resource cohorts;
их samples не объединялись и причинное объяснение различий не устанавливалось.

## Что измерено

| Полный forward/backward/SGD step | Native A | Candidate B | Разница |
|---|---:|---:|---:|
| Peak allocated | 843303424 B /804.2368 MiB | 776718848 B /740.7368 MiB | −63.5 MiB /−7.90% |
| Peak reserved | 866123776 B /826 MiB | 832569344 B /794 MiB | −32 MiB /−3.87% |
| Медиана двух arm-медиан | 1726.373499 ms | 1730.408540 ms | +0.234%, описательно |
| Диапазон arm-медиан | 1718.898158–1733.848841 ms | 1726.372656–1734.444425 ms | Диапазоны перекрываются |

Практический вывод для Soup: подтверждена небольшая экономия памяти на настоящих
весах и полном streaming-шаге. Ни один путь не получил OOM; переход от «не помещается»
к «помещается» не показан. По этому тесту нельзя обещать40% экономии, ускорение или
возможность обучать более крупную модель. Production dispatch не переключён.

## Модель и условия

- Полная pretrained `mistralai/Mistral-7B-v0.3`, revision
  `caa1feb0e54d415e2df31207e5f4e273e33509b1`: все32 слоя, hidden4096,
  intermediate14336, vocab32768, untied embeddings/head. SHA всех трёх исходных
  safetensors совпали с LFS oid официального checkpoint; это не случайные веса.
- Soup `build_streamed_model`: **RAM**, два **pageable** переиспользуемых GPU-буфера,
  nested NF4, BF16; LoRA q/v rank4 alpha8 dropout0, стандартный zero-B init,
  seed842, SGD lr0.001. Batch1, seq128; обычный английский текст из harness,
  causal cross-entropy. Это runtime Soup и полный optimizer step, не isolated GEMM.
- Candidate использует прежний `wmma_native64` только на покрытых descriptors;
  остальные проекции сохраняют штатный fallback. Другие длины последовательности,
  LoRA ranks и модели не проверялись. BF16 reduced-precision reduction включён,
  TF32 выключен. Численный kernel и эталон не менялись.
- RTX3090 sm_86, VM56671053, machine97012,
  UUID `GPU-259fccc1-4cda-8c67-d13d-e08932415283`, guest driver580.95.05,
  Torch2.11.0+cu126, bitsandbytes0.50.2, Transformers5.17.0, PEFT0.20.0.
  Во всех primary/reference samples SM1200–1200 MHz, неизменный power limit350 W.
- По [отдельному правилу, записанному до CUDA](d1-real7b-rule-20261019.md): один
  fresh-process ABBA,5 warmup и20 timed steps на arm; всего80 timed steps.
  Все четыре ArmWatch/peer/telemetry и same-session dense GEMM controls прошли.
  Peaks сняты до reference. Memory-controller utilization: A1–12%, B2–9%.

## Корректность и качество

Три последовательных обновления native RAM streaming и candidate совпали
**bit-exact и finite**: logits, loss, все128 trainable adapter tensors и128
градиентов на каждом шаге —774 величины. Ненулевых градиентов64/128/128:
первый zero-B шаг ожидаемо даёт нулевой градиент A. Candidate dispatch384 вызова;
штатный fallback тоже реально использован. Даже SHA сериализованных свидетелей
native/candidate совпали.

Train CE losses обоих путей:3.4398639202 →3.4379322529 →3.4317741394.
Это training-loss trace одного текста, **не оценка качества**. Длительного обучения,
held-out/task metrics и отдельного сравнения этой7B с unpatched resident NF4 нет.
Общий для обоих streaming-путей дефект этот A/B gate сам по себе исключить не может.

На том же GPU отдельно прошёл большой nested BF16 recycled-buffer gate:
packed192 MiB, exact forward/dX, два recompute и finite differing same-process
plain-ctx negative control. Он и малые checkpoint gates выполнены до real7B screen.

## Источники, аудит и ресурсы

Executed real7B source: `e04424acb273132e40e4774de7cee2e47f871a43`.
Первоначальная подготовка на `2a34687f` остановилась: Soup отвергает symlinked
shards кеша HF. До real7B training сделана только обычная локальная копия уже
скачанных файлов. Численное правило/kernel/reference не менялись; source/log
первой попытки сохранены. Native gates выполнены на исходном commit, их численный
код идентичен исправленному. Ремонт I/O не выдаётся за численное улучшение kernel.

[Неизменяемый архив и CPU replay](https://github.com/SoupTeam/soup-rnd/tree/d08f224aa245ebad5d565019cf3fbba4743074a4/followups/real7b-20261019)
содержит raw132706782 B в трёх частях, source archives, CUDA tensors, input IDs,
пошаговые timings, GEMM/telemetry, ошибки подготовки, manifest, audit и receipts.
Raw SHA-256: `2e86f769e21719165021befc9ff38f2b3103d2f43403d9696a535bf746809799`.
Независимый CPU replay проверил694 source files, upstream weight hashes,774
величины,80 timings/медианы, refs, lifecycle/validity и clean-screen classification.
Он не является дополнительным CUDA-прогоном. Все публичные blobs проверены
побайтно; все предыдущие archive blobs и исторические NO_VERDICT неизменны.

Старый VM-хост стал недоступен до аренды. Замена ограничена45 минутами; guest
network stop23 GB и резерв$1 на трафик удерживают эту проверку в$1.50, общий D1
лимит$15. VM удалена после экспорта, clocks сброшены, persistence выключен,
временная личная SSH-регистрация отозвана. Снимок начислений: новая VM$0.187,
D1$8.854/$15; **трафик и другие начисления могут обновиться**, резерв не расходуется
на дополнительные прогоны. Полная D1/#842 по-прежнему требует
[решения владельца о покрытии](D1_ACCEPTANCE.md).
