# Ticket 04: L2L против накопления градиентов, SmolLM2-135M, RTX 3060, 2026-10-21

**Вердикт: PASS в обоих режимах (`quant none` и `nf4`), k=1, 2, 4.**

Проверялся поставляемый шаг `soup_cli.utils.l2l_step.L2LStep` с `ActivationStore`
на коммите 2e274f05 (драйвер) поверх c70661d4. LoRA dropout равен 0.
Режим `--deterministic` не понадобился: A-против-A (два прогона накопления
градиентов) побитово равны во всех шести записях.

## Стенд

| Параметр | Значение |
| --- | --- |
| Сервер | Vast.ai, оффер 54148061, инстанс 57086397, $0.072/ч, удалён после скачивания |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| Драйвер | 595.91.07 (CUDA 13.2 по `nvidia-smi`) |
| Образ | `nvidia/cuda:13.0.0-devel-ubuntu24.04`, Python 3.12.3 |
| Пакеты | torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2 (полный список в `packages.txt`) |
| Модель | `HuggingFaceTB/SmolLM2-135M`, реальные веса, llama, 30 слоёв, tied embeddings, bf16 |
| Вход | `seq=128`, случайные токены (seed 17), метки равны входу |

Bootstrap: `bash benchmarks/a0-bootstrap.sh` (104 CPU-теста и ruff прошли, `bootstrap.log`).

## Команды

```bash
export PYTHONPATH=$PWD/src
.venv/bin/python benchmarks/harness/l2l_smollm2_check.py --quant none --out /root/runs/none-01
.venv/bin/python benchmarks/harness/l2l_smollm2_check.py --quant nf4  --out /root/runs/nf4-01
```

## Результаты

Реальная модель SmolLM2-135M, реальные веса, синтетический вход (случайные токены).
Качество модели не измерялось.

Равенство: для каждого k потери и все 240 LoRA-градиентов (30 слоёв x 4 проекции x A/B)
равны побитово, максимальная разница 0, нулевых градиентов нет. Это верно для `none` и `nf4`.

| k | Загрузки слоёв L2L | Загрузки слоёв накопления | Восстановления NF4 L2L | Восстановления NF4 накопления |
| --- | --- | --- | --- | --- |
| 1 | 58 | 58 | 0 (`none`) / 210 (`nf4`) | 0 / 210 |
| 2 | 58 | 115 | 0 / 210 | 0 / 210 |
| 4 | 58 | 229 | 0 / 210 | 0 / 210 |

Загрузки L2L не зависят от k (58 при k=1, 2, 4). У накопления градиентов они растут почти
линейно (58, 115, 229). Восстановления NF4 не зависят от k и равны 210 в обоих способах
(30 слоёв x 7 квантованных линейных).
`large_loads` равен 0 во всех записях: эмбеддинг и голова tied и живут в обычных слотах.

## Что не проверялось

- Синтетическая модель: в этом отчёте её нет. Синтетические результаты A0 (dense-70B)
  остаются в [A0](probe-vast-rtx3070-l2l-step0-a0.md) и не смешаны с этими.
- Скорость и память: не измерялись.
- Dropout больше 0, MoE, несколько GPU: вне тикета.
- Холодные NF4-представления сбрасываются перед каждым подсчитанным шагом, иначе
  восстановления были бы 0 (кэш #841). Это зафиксировано в драйвере (`_cold_views`).

## Попытки и испорченные замеры

Было по одному запуску на режим, оба валидны. Испорченных замеров нет, повторных
запусков не делали. Все исходные файлы сохранены без правок.

## Файлы

- [none-01/records.json](results/l2l-smollm2-vast-rtx3060/none-01/records.json), [none-01.log](results/l2l-smollm2-vast-rtx3060/none-01.log)
- [nf4-01/records.json](results/l2l-smollm2-vast-rtx3060/nf4-01/records.json), [nf4-01.log](results/l2l-smollm2-vast-rtx3060/nf4-01.log)
- [packages.txt](results/l2l-smollm2-vast-rtx3060/packages.txt), [nvidia-smi.txt](results/l2l-smollm2-vast-rtx3060/nvidia-smi.txt), [bootstrap.log](results/l2l-smollm2-vast-rtx3060/bootstrap.log)
- [source-sha256.txt](results/l2l-smollm2-vast-rtx3060/source-sha256.txt): хеши драйвера и `l2l_step.py` на сервере

## Checklist ticket 04

- [x] Отчёт содержит команды, версии пакетов, GPU и исходные журналы.
- [x] Потери и LoRA-градиенты равны для k=1, 2, 4 при dropout 0 (deterministic-режим не понадобился).
- [x] Число загрузок не растёт в k раз (58 при k=1, 2, 4).
- [x] Синтетические и реальные результаты подписаны отдельно.
- [x] Испорченных замеров нет, исходные попытки сохранены.
