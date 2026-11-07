# A6 повторный прогон по правилу готовности: ORPO на L2L против накопления, SmolLM2-135M, RTX 3060 12 GB, 2026-11-04

**Вердикт: НЕ ПРОШЁЛ** (правило `benchmarks/l2l-preference-readiness-rule-a6.md`). k=1: fail; k=2: fail; k=4: fail.

Правило записано и закоммичено до правок драйвера и до аренды. Тикет 06 не переписывается: его отчёты остаются «не прошёл» по его критерию. Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** (`examples/data/orpo_pairs_synthetic.jsonl`, 12 строк). Качество модели не измерялось, результат относится к этой модели и этой карте и не переносится на другие.

## Стенд

| Параметр | Значение |
| --- | --- |
| Сервер | Vast.ai, оффер 58817449, инстанс 59139680, машина 149975, Poland PL, AMD EPYC 7452 |
| GPU | NVIDIA GeForce RTX 3060, 580.173.02, 12288 MiB |
| Образ | nvidia/cuda:13.0.0-devel-ubuntu24.04, Python 3.12.3, драйвер 580.173.02 |
| Пакеты | torch 2.14.1+cu130, transformers 5.19.0, trl 0.29.1, peft 0.21.2, CUDA 13.0, Python 3.12.3 (полный список в `packages.txt`) |
| RAM | MemTotal:       131744572 kB; нагрузка хоста: load average 4,76 / 4,59 / 4,69 на 64 потоках (uptime.txt, 18:11 UTC) |
| Коммит | `7d671d1f960e734c4a4ec750cf4af497be1ac4c0` Дерево на инстансе без `benchmarks/results/`: канал с Mac на этот хост дал около 76 KB/s, полное дерево (46 MB) не доезжало; на прогон эти данные не влияют. |
| Стоимость | около 0,085 USD: инстанс 59139680 65 минут (17:42:35Z to 18:47:34Z) по фактической ставке 0,0782 USD/ч из `vastai show instance`; около 25 минут из них ушло на медленную загрузку дерева, около 33 на замеры. Потолок 0,3 USD, остановка на 0,25. Счёта vast.ai нет, число приблизительное. |
| Аккаунт | команда vast.ai `soup-rnd` (по указанию пользователя) |
| Уничтожение | `vastai destroy instance 59139680 -y` в 18:47:34Z; `vastai show instances` после этого: 0 instances (show-instances-after.txt) |

## Команды

```bash
vastai create instance 58817449 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 60 --ssh --direct --label a6-ready --cancel-unavail
vastai attach ssh 59139680 "$(cat ~/.ssh/id_ed25519.pub)"
git ls-files -z | grep -z -v '^benchmarks/results/' | tar --null -T - -czf tree2.tgz; scp tree2.tgz ... :/root/; ssh ... 'mkdir -p /root/soup && tar xzf /root/tree2.tgz -C /root/soup'
ssh -f ... "cd /root/soup && CARD='RTX 3060 12GB' nohup bash benchmarks/harness/l2l_pref_diagnose/remote_ready.sh > /root/remote.out 2>&1 < /dev/null &"
# внутри remote_ready.sh, из /root/soup: bootstrap (a0-bootstrap.sh), затем
python benchmarks/harness/l2l_pref_diagnose.py ready --device cuda --out /root/results/ready --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
vastai destroy instance 59139680 -y
```

k равен 1, 2, 4. Плечи описаны в правиле: `f32u_*` (float32 подменой, ядра SDPA по умолчанию), `native_*` (bf16, `MATH` в обоих путях, пол на `EFFICIENT_ATTENTION`), `sft` (счёт загрузок). Каждое плечо идёт в своём процессе под `ArmWatch`, обязательные проверки `suspend` и `power`.

## Результат по ячейкам

| k | окон | float32 (критерий 1) | float32 max разница градиента | bf16 шаг 0 в 1,5 пола (критерий 2) | шаг 0: градиент L2L / пол | шаг 0: потеря L2L / пол | исход ячейки |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | равно побитово | 0 | нет | 1.312e-03 / 4.395e-03 | 7.951e-04 / 8.606e-03 | НЕ ПРОШЁЛ |
| 2 | 3 | равно побитово | 0 | нет | 1.091e-03 / 2.686e-03 | 1.860e-04 / 8.179e-03 | НЕ ПРОШЁЛ |
| 4 | 2 | равно побитово | 0 | нет | 7.324e-04 / 1.331e-03 | 4.199e-04 / 7.302e-03 | НЕ ПРОШЁЛ |

## Что показали данные

- Критерий 1 (float32 побитово): выполнен в ячейках k=[1, 2, 4] из [1, 2, 4].
- Критерий 2 (bf16, шаг 0), k=1: вне предела 3 записей из 120 тензоров шага; превышение предела от 1.15 до 1.78 раза.
- Критерий 2 (bf16, шаг 0), k=2: вне предела 1 записей из 120 тензоров шага; превышение предела от 1.25 до 1.25 раза.
- Критерий 2 (bf16, шаг 0), k=4: вне предела 1 записей из 120 тензоров шага; превышение предела от 1.08 до 1.08 раза.
  Предел считается по каждому тензору отдельно из одного замера пола, и правило требует выполнения на всех тензорах шага 0. Провал по немногим тензорам при том, что наибольшая разница по всем тензорам шага 0 (таблица ниже) меньше наибольшего пола, не доказывает и не исключает ошибку: правило здесь строже, чем по общей величине, и ничего больше о причине провала эти данные не говорят.

Что не выполнено или не определено:

- k=1: bf16 step 0 is outside 1.5 floors: step 0: base_model.model.model.layers.23.self_attn.v_proj.lora_B.default.weight differs by 1.850e-04, limit 1.602e-04
- k=2: bf16 step 0 is outside 1.5 floors: step 0: base_model.model.model.layers.29.self_attn.v_proj.lora_B.default.weight differs by 1.106e-04, limit 8.869e-05
- k=4: bf16 step 0 is outside 1.5 floors: step 0: base_model.model.model.layers.1.self_attn.q_proj.lora_A.default.weight differs by 5.188e-04, limit 4.807e-04

Что реально стояло в плечах (запись изнутри первого вызова слоя 0 и из `TrainingArguments`); float32 получен подменой двух функций драйвера, это обход, а не штатный режим:

| поле (ячейка k=1) | `f32u_ga1` | `f32u_l2l` | `native_ga1` | `native_l2l` | `native_floor` |
| --- | --- | --- | --- | --- | --- |
| `stream_dtype` | `float32` | `float32` | `bfloat16` | `bfloat16` | `bfloat16` |
| `train_bf16` | `False` | `False` | `True` | `True` | `True` |
| `mixed_precision` | `no` | `no` | `bf16` | `bf16` | `bf16` |
| `base_weight` | `torch.float32` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
| `layer_out` | `torch.float32` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
| `grads` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` |
| `sdpa` | `None` | `None` | `MATH` | `MATH` | `EFFICIENT_ATTENTION` |
| `allow_tf32` | `False` | `False` | `False` | `False` | `False` |

## Все шаги bf16 (записаны, не судятся)

Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами, поэтому правило их не судит. Наибольшая разница по градиенту LoRA и потере:

| k | шаг | градиент L2L | градиент пол | потеря L2L | потеря пол |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1.312e-03 | 4.395e-03 | 7.951e-04 | 8.606e-03 |
| 1 | 1 | 6.439e-03 | 6.104e-03 | 1.113e-02 | 7.703e-03 |
| 1 | 2 | 1.080e-02 | 9.949e-03 | 7.381e-03 | 2.933e-04 |
| 1 | 3 | 9.094e-03 | 1.086e-02 | 4.189e-03 | 2.746e-03 |
| 1 | 4 | 7.782e-03 | 1.003e-02 | 7.707e-03 | 2.396e-02 |
| 1 | 5 | 2.462e-02 | 1.976e-02 | 1.681e-02 | 1.176e-02 |
| 2 | 0 | 1.091e-03 | 2.686e-03 | 1.860e-04 | 8.179e-03 |
| 2 | 1 | 8.606e-03 | 6.622e-03 | 6.186e-03 | 4.768e-03 |
| 2 | 2 | 1.468e-02 | 9.644e-03 | 2.818e-03 | 3.452e-04 |
| 4 | 0 | 7.324e-04 | 1.331e-03 | 4.199e-04 | 7.302e-03 |
| 4 | 1 | 7.385e-03 | 1.439e-02 | 8.717e-04 | 7.413e-03 |

## Загрузки слоя

Счётчики берутся из лога обучения, это счётчики рантайма. Условие правила: загрузок слоя за шаг столько же, сколько у L2L SFT с тем же k, больших загрузок и восстановлений весов не больше.

| k | загрузки слоя за шаг, ORPO | загрузки слоя за шаг, L2L SFT | большие загрузки (метод / SFT) | восстановления весов (метод / SFT) | исход |
| --- | --- | --- | --- | --- | --- |
| 1 | [58, 57, 57, 57, 57, 57] | [58, 57, 57, 57, 57, 57] | 0 / 0 | 0 / 0 | совпадает |
| 2 | [58, 57, 57] | [58, 57, 57] | 0 / 0 | 0 / 0 | совпадает |
| 4 | [58, 57] | [58, 57] | 0 / 0 | 0 / 0 | совпадает |

## Валидность замеров и попытки

| k | плечо | попытка | исход ArmWatch | код выхода | причина исключения | исключена |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `f32u_ga1` | 0 | ok | 0 |  | нет |
| 1 | `f32u_ga2` | 0 | ok | 0 |  | нет |
| 1 | `f32u_l2l` | 0 | ok | 0 |  | нет |
| 1 | `native_ga1` | 0 | ok | 0 |  | нет |
| 1 | `native_l2l` | 0 | ok | 0 |  | нет |
| 1 | `native_floor` | 0 | ok | 0 |  | нет |
| 1 | `sft` | 0 | ok | 0 |  | нет |
| 2 | `f32u_ga1` | 0 | ok | 0 |  | нет |
| 2 | `f32u_ga2` | 0 | ok | 0 |  | нет |
| 2 | `f32u_l2l` | 0 | ok | 0 |  | нет |
| 2 | `native_ga1` | 0 | ok | 0 |  | нет |
| 2 | `native_l2l` | 0 | ok | 0 |  | нет |
| 2 | `native_floor` | 0 | ok | 0 |  | нет |
| 2 | `sft` | 0 | ok | 0 |  | нет |
| 4 | `f32u_ga1` | 0 | ok | 0 |  | нет |
| 4 | `f32u_ga2` | 0 | ok | 0 |  | нет |
| 4 | `f32u_l2l` | 0 | ok | 0 |  | нет |
| 4 | `native_ga1` | 0 | ok | 0 |  | нет |
| 4 | `native_l2l` | 0 | ok | 0 |  | нет |
| 4 | `native_floor` | 0 | ok | 0 |  | нет |
| 4 | `sft` | 0 | ok | 0 |  | нет |

Испорченные замеры (`void`) исключаются по `benchmarks/METHOD.md`, запись остаётся в каталоге результатов. Если колонка «исключена» пуста, исключать было нечего.

## Что «прошёл» не значит и что не проверялось

- Не значит побитового равенства в bf16 (в тикете 08 его нет даже на шаге 0) и допуска в bf16 при ядрах SDPA по умолчанию: критерий 2 закрепляет `MATH`, как в A3-07, и результат от этого зависит.
- Пол шума это один замер второго бэкенда SDPA; GA в bf16 на повторяемость не проверялся.
- Данные синтетические, 12 строк, неполное последнее окно есть при k=4. Качество модели не измерялось.
- Другие модели, другие карты, NF4, реальные наборы, скорость и память не проверялись. `foreign_reads` не обязательна (пояснение в правиле).

## Файлы

- [ready-orpo.json](results/l2l-pref-ready-a6-vast-rtx3060/ready-orpo.json), [environment.json](results/l2l-pref-ready-a6-vast-rtx3060/environment.json), [rental.json](results/l2l-pref-ready-a6-vast-rtx3060/rental.json)
- журналы и записи `ArmWatch` каждой попытки: `results/l2l-pref-ready-a6-vast-rtx3060/orpo/k*/<плечо>/attempt*/` (без каталогов `*-work`)
- [packages.txt](results/l2l-pref-ready-a6-vast-rtx3060/packages.txt), [bootstrap.log](results/l2l-pref-ready-a6-vast-rtx3060/bootstrap.log), [ready.log](results/l2l-pref-ready-a6-vast-rtx3060/ready.log), [create.log](results/l2l-pref-ready-a6-vast-rtx3060/create.log), [COMMIT](results/l2l-pref-ready-a6-vast-rtx3060/COMMIT), [show-instances-after.txt](results/l2l-pref-ready-a6-vast-rtx3060/show-instances-after.txt)
- хеши `.pt` каждой попытки лежат в `ready-orpo.json` (`attempts`); сами `.pt` в репозиторий не входят.
