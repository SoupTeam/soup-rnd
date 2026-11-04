# A2 / PR #31: L2L равен обычному накоплению при dropout 0.05 на CUDA, replay потока, RTX 3060, 2026-10-22

**Вердикт: PASS.** Шаг L2L с replay нативной последовательности генератора на карте даёт потери и все
240 LoRA-градиентов, равные обычному накоплению без хуков (`torch.equal`), для k=1, 2, 4 при dropout 0 и 0.05.
Состояние CPU- и CUDA-генератора после шага тоже равно состоянию после накопления. Оба контроля
(общий поток без replay, другой seed) показывают отличие. Повтор в отдельном процессе даёт тот же sha256
потерь и градиентов, режим `--deterministic` даёт те же digest. Этот отчёт заменяет CUDA-часть отчёта
`probe-vast-rtx3060-l2l-smollm2-a2-ticket03.md`, измеренного на варианте А (хуки с функцией состояния).

## Стенд

| Параметр | Значение |
| --- | --- |
| Сервер | Vast.ai, аккаунт `soup-rnd`, оффер 55562140, инстанс 57259826, $0.076/ч, удалён после скачивания результатов |
| GPU | NVIDIA GeForce RTX 3060, 12 GB, драйвер 580.65.06, CUDA 13.0 |
| Образ | `nvidia/cuda:13.0.0-devel-ubuntu24.04`, Python 3.12.3 |
| Пакеты | torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, safetensors 0.8.0 (`packages.txt`) |
| Код | коммит `0a054109` ветки `a2-l2l-dropout-rng`, кроме `tests/test_l2l_rng_cuda.py` (одна строка, см. ниже); хеши файлов в `source-sha256.txt` |
| Модель | `HuggingFaceTB/SmolLM2-135M`, реальные веса (sha256 `80521b40...`), llama, 30 слоёв, bf16, `--quant none` |
| Вход | `seq=128`, случайные токены, метки равны входу |
| Seed | `--rng-seed 1234`: `torch.manual_seed` перед накоплением и перед шагом L2L |

Bootstrap: `bash benchmarks/a0-bootstrap.sh` (104 CPU-теста и ruff прошли, `bootstrap.log`).

## Проверка памяти до запуска

`l2l_fit_check.py --records`: измеренный пик `torch.cuda.max_memory_allocated` равен 188 MB (k=4), свободно
12.37 GB, запас ×1.25, `fits: true` (`run3/fit-measured.json`). Пробный прогон `probe-k4` вернул код 1, потому что
запись одного k=4 без k=1 не проходит проверку плоских загрузок. Так было и в прошлом отчёте; записи из него
использованы только для оценки памяти.

## Тесты на карте

`pytest -m gpu tests/test_l2l_rng_cuda.py`: 12 passed, 1 skipped (второй картой не располагали).
Среди них `test_replay_moves_the_cuda_offset_like_the_native_dropout`: dropout на нулях с теми же формой и
шагами сдвигает CUDA-состояние (seed и смещение philox) так же, как нативный dropout. Это был главный
риск варианта Б из `research-rng-reference.md`; он не подтвердился.

Первый запуск (`gpu-tests.log`) дал 11 passed, 1 failed: тест
`test_cpu_and_cuda_state_are_set_from_the_snapshot_and_restored_apart` не инициализировал CUDA до снимка,
а `snapshot()` видит только инициализированные карты. Ошибка в тесте, не в коде: в тест добавлена одна строка
`torch.zeros(1, device=DEVICE)`. Второй запуск (`gpu-tests2.log`) с исправленным файлом прошёл.

## Результаты на реальной модели

| Прогон | k=1 | k=2 | k=4 | вердикт | digest k=1 / 2 / 4 |
| --- | --- | --- | --- | --- | --- |
| `d0-01`, dropout 0 | равно | равно | равно | pass | 4b268139 / b7ee0773 / 4f031f7c |
| `d005-01`, dropout 0.05 | равно | равно | равно | pass | 30b4844c / 98982288 / 1545f2da |
| `d005-02`, dropout 0.05, повтор | равно | равно | равно | pass | те же, что у `d005-01` |
| `d005-det`, dropout 0.05, `--deterministic` | равно | равно | равно | pass | те же, что у `d005-01` |

«Равно» значит: потери, все 240 градиентов и состояние генератора после шага совпали с накоплением
(`max |d|` = 0.0). У dropout 0.05 `control_shared_stream_equal` и `control_other_seed_equal` равны false для
каждого k. Загрузки слоёв: накопление 58 / 115 / 229, L2L 58 / 58 / 58 для k = 1 / 2 / 4. Пик памяти карты
184-188 MB (234-238 MB в deterministic-режиме).

## Что не проверено

- Скрытые состояния и логиты напрямую: сравнивались потери и градиенты.
- Полный путь `SFTTrainer` с реальным collator и DataLoader на карте. Тренер при `lora.dropout` 0.05 проверен на CPU.
- Две карты (тест пропущен), NF4 (`--quant nf4` не запускался), модели крупнее SmolLM2-135M, реальный текст.
- Отказанные случаи (NEFTune с dropout, `attention_dropout` при `sdpa`, окна разной длины с dropout) на карте не проверялись.

## Неудачные попытки

Две первые попытки лежат в `results/l2l-replay-a2-pr31-vast-rtx3060/failed-attempts/`:
1. `run3-failed`: сервер не достучался до huggingface.co (SSL EOF), веса не скачались. Веса скачаны локально и переданы на сервер, запуск шёл с `HF_HUB_OFFLINE=1` и `--weights /root/smollm2`.
2. `run3-failed2`: служебные файлы macOS `._model.safetensors` из tar-архива попали под маску `*.safetensors` («header too large»). Файлы `._*` удалены.

## Файлы

Каталог `results/l2l-replay-a2-pr31-vast-rtx3060/`: `run3/` (записи, журналы, `status`, `fit-*.json`), `gpu-tests.log`,
`gpu-tests2.log`, `bootstrap.log`, `packages.txt`, `nvidia-smi.txt`, `source-sha256.txt`, `runall3.sh`, `local-commit.txt`.
