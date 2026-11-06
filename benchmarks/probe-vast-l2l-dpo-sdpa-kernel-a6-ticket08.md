# A6 тикет 08, второй прогон (дополнение): откуда расхождение L2L и накопления градиентов, DPO k=1, SmolLM2-135M, RTX 3060 12 GB, 2026-11-04

Второй прогон по дополнению к правилу: те же плечи, что в первом отчёте (`probe-vast-l2l-dpo-divergence-a6-ticket08.md`), плюс `f32u_*` и запись ядра SDPA. Исход ниже это повторный замер первых контролей в новой сессии на том же хосте (машина 43183), он ничего не заменяет.

**Вердикт: исход 2: округление больше пола шума от второго бэкенда SDPA (накопление против себя даёт 0, пол взят от `EFFICIENT_ATTENTION` против `MATH`). Решение о допуске за пользователем.**

Правило: `benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md`, закоммичено до драйвера и до аренды; одна поправка (маска заполнения в контроле 3) внесена после смоука драйвера на CPU, тоже до аренды. Тикет 06 не переписывается: DPO, ORPO, SimPO и KTO там остаются «не прошёл», этот отчёт никого не отмечает готовым.

Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** (`examples/data/dpo_pairs_synthetic.jsonl`, 12 строк, шесть окон). Результат относится к одной ячейке (DPO, k=1), этой модели и этой карте. ORPO, SimPO, KTO, другие k, другие карты и модели не заявляются.

## Стенд

| Параметр | Значение |
| --- | --- |
| Сервер | Vast.ai, оффер 47491621, инстанс 59133143, машина 43183 (тот же хост, что в первом прогоне), Wisconsin US, Xeon E5-2696 v3, 72 потока |
| GPU | NVIDIA GeForce RTX 3060, 580.173.02, 12288 MiB |
| Образ | nvidia/cuda:13.0.0-devel-ubuntu24.04, Python 3.12.3, драйвер 580.173.02 |
| Пакеты | torch 2.14.1+cu130, transformers 5.19.0, trl 0.29.1, peft 0.21.2, CUDA 13.0, Python 3.12.3 (полный список в `packages.txt`) |
| RAM | MemTotal:       263801688 kB; нагрузка хоста: load average 7,27 / 6,19 / 7,07 на 72 потоках (uptime.txt, 16:51 UTC) |
| Коммит | `26013409352a2caaeab88849e850682e32f09631` |
| Стоимость | около 0,02 USD: инстанс 59133143 12,5 минуты (16:47:45Z to 17:00:14Z) по фактической ставке 0,10 USD/ч из `vastai show instance`. С первым прогоном (около 0,03 USD, ставка оценена) вместе около 0,05 USD при потолке 0,5 USD, остановка на 0,4. Счёта vast.ai нет, число приблизительное. |
| Аккаунт | команда vast.ai `soup-rnd` (по указанию пользователя) |
| Уничтожение | `vastai destroy instance 59133143 -y` в 17:00:14Z; `vastai show instances` после этого: 0 instances (show-instances-after.txt; чужой инстанс 59126915 к этому времени уже завершился без моего участия) |

## Команды

```bash
vastai create instance 47491621 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 60 --ssh --direct --label a6-08-kernel --cancel-unavail
vastai attach ssh 59133143 "$(cat ~/.ssh/id_ed25519.pub)"
git ls-files -z | tar --null -T - -czf - | ssh ... 'mkdir -p /root/soup && tar xzf - -C /root/soup'
ssh -f ... "cd /root/soup && CARD='RTX 3060 12GB' nohup bash benchmarks/harness/l2l_pref_diagnose/remote.sh > /root/remote.out 2>&1 < /dev/null &"
# внутри remote.sh, из /root/soup: bootstrap (a0-bootstrap.sh), затем
python benchmarks/harness/l2l_pref_diagnose.py run --device cuda --out /root/results/a8 --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
vastai destroy instance 59133143 -y
```

Плечи, пути, типы и бэкенды SDPA описаны в правиле. Каждое плечо идёт в своём процессе под `ArmWatch`, обязательные проверки `suspend` и `power`.

## Что показали данные

- Контроль 1: в float32 L2L побитово равен накоплению на всех 6 шагах (градиенты LoRA и потери); накопление в float32 повторяется побитово. Запись типов подтверждает float32 без автокаста в этих трёх плечах. Значит, при закреплённом `MATH` в общем пути L2L в этой ячейке на этой карте ошибки нет. Без закрепления SDPA см. контроль 1u ниже.
- Контроль 2, шаг 0 (одинаковые адаптеры): разрыв L2L по градиенту 1.648e-03, пол 5.371e-03; по потере 2.980e-07 и 5.248e-03. Шаг 0 в пределе: да.
- Контроль 2, все шаги: вне предела 238 записей на шагах [1, 2, 3, 4, 5]; наибольшее превышение предела 12.67 раза (предел это 1,5 пола), меньше чем вдвое превышают 195 из 238. Пол здесь одно измерение второго бэкенда. Шаги после нулевого не независимы от шага 0 (адаптеры уже разошлись), поэтому по правилу исход читается как «вне», а размер превышения виден в таблице.
- Контроль 1u: в float32 при ядрах SDPA по умолчанию L2L побитово равен накоплению на всех 6 шагах. Ошибки в общем пути L2L нет и без закрепления `MATH`.
- Ядра по умолчанию в bf16: судья ядер вернул `differ`, но общий бэкенд, совпавший с умолчанием в каждой записанной сигнатуре обоих путей: EFFICIENT_ATTENTION. Данные согласуются с одним ядром у накопления и у L2L; `MATH` от умолчания отличается (около 2e-3 на выходе внимания), так что умолчание это не `MATH`. Разрыв тикета 06 выбором ядра по этим данным не объяснён: остаётся округление bf16 в операции, где пути различаются (в float32 они равны побитово).
- Без закрепления SDPA (условия тикета 06) потеря шага 0 расходится на 8.280e-03, с `MATH` в обоих путях на 2.980e-07. Какие ядра выбирает умолчание, записано в дополнении ниже.

## Дополнение: float32 без закрепления SDPA, равно побитово; ядра по умолчанию: differ

Плечи `f32u_*`: float32 (подмена, как в контроле 1), бэкенд SDPA не закреплён. Разница по шагам:

| пара | шаг | градиент | потеря |
| --- | --- | --- | --- |
| `f32u_ga2_vs_ga1` | 0 | 0 | 0 |
| `f32u_ga2_vs_ga1` | 1 | 0 | 0 |
| `f32u_ga2_vs_ga1` | 2 | 0 | 0 |
| `f32u_ga2_vs_ga1` | 3 | 0 | 0 |
| `f32u_ga2_vs_ga1` | 4 | 0 | 0 |
| `f32u_ga2_vs_ga1` | 5 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 0 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 1 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 2 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 3 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 4 | 0 | 0 |
| `f32u_l2l_vs_ga1` | 5 | 0 | 0 |

Ядро SDPA по умолчанию: вызов записан в плече, потом выполнен без закрепления и под каждым бэкендом отдельно; `equal` значит побитово равный результат.

| плечо | форма и тип запроса | маска | is_causal | равно умолчанию | остальные бэкенды |
| --- | --- | --- | --- | --- | --- |
| `unpinned_ga` | [4, 9, 31, 64] bfloat16 | да | нет | EFFICIENT_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `unpinned_ga` | [4, 9, 15, 64] bfloat16 | да | нет | EFFICIENT_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `unpinned_ga` | [4, 9, 25, 64] bfloat16 | да | нет | EFFICIENT_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `unpinned_ga` | [4, 9, 22, 64] bfloat16 | да | нет | EFFICIENT_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `unpinned_ga` | [4, 9, 19, 64] bfloat16 | да | нет | EFFICIENT_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `unpinned_l2l` | [4, 9, 31, 64] bfloat16 | нет | да | EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03 |
| `unpinned_l2l` | [4, 9, 15, 64] bfloat16 | нет | да | EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03 |
| `unpinned_l2l` | [4, 9, 25, 64] bfloat16 | нет | да | EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03 |
| `unpinned_l2l` | [4, 9, 22, 64] bfloat16 | нет | да | EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION | MATH: differs by 1.953e-03 |
| `unpinned_l2l` | [4, 9, 19, 64] bfloat16 | нет | да | EFFICIENT_ATTENTION, FLASH_ATTENTION | MATH: differs by 1.953e-03; CUDNN_ATTENTION: differs by 6.104e-05 |
| `f32u_ga1` | [4, 9, 31, 64] float32 | да | нет | EFFICIENT_ATTENTION | MATH: differs by 5.364e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_ga1` | [4, 9, 15, 64] float32 | да | нет | EFFICIENT_ATTENTION | MATH: differs by 5.513e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_ga1` | [4, 9, 25, 64] float32 | да | нет | EFFICIENT_ATTENTION | MATH: differs by 3.874e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_ga1` | [4, 9, 22, 64] float32 | да | нет | EFFICIENT_ATTENTION | MATH: differs by 4.172e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_ga1` | [4, 9, 19, 64] float32 | да | нет | EFFICIENT_ATTENTION | MATH: differs by 4.768e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_l2l` | [4, 9, 31, 64] float32 | нет | да | EFFICIENT_ATTENTION | MATH: differs by 5.364e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_l2l` | [4, 9, 15, 64] float32 | нет | да | EFFICIENT_ATTENTION | MATH: differs by 5.513e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_l2l` | [4, 9, 25, 64] float32 | нет | да | EFFICIENT_ATTENTION | MATH: differs by 3.874e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_l2l` | [4, 9, 22, 64] float32 | нет | да | EFFICIENT_ATTENTION | MATH: differs by 4.172e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |
| `f32u_l2l` | [4, 9, 19, 64] float32 | нет | да | EFFICIENT_ATTENTION | MATH: differs by 4.768e-07; FLASH_ATTENTION: unavailable: No available kernel. Aborting execution.; CUDNN_ATTENTION: unavailable: No available kernel. Aborting execution. |

Чтение по таблице правила (дополнение) не применено, причина ниже.

**Поправка после данных.** Судья ядер, как его определяет правило, сравнивает списки совпавших бэкендов и вернул `differ`, потому что в списках L2L есть лишний `FLASH_ATTENTION`. При этом `EFFICIENT_ATTENTION` совпадает с умолчанием в каждой записанной сигнатуре обоих путей, а результаты разных бэкендов могут совпасть случайно. Данные согласуются с одним и тем же ядром у накопления и у L2L, поэтому строка таблицы правила «разрыв тикета 06 это выбор ядра» здесь **не заявляется**. Судья дополнен полем `common` после этого прогона (тест `test_kernels_name_the_backends...`).

## Повтор первого прогона

Те же плечи в новой сессии на том же хосте (машина 43183). Совпадают ли разрывы по шагам (градиент и потеря) с первым отчётом:

| пара | совпало |
| --- | --- |
| `f32_ga2_vs_ga1` | да |
| `f32_l2l_vs_ga1` | да |
| `native_floor_vs_ga1` | да |
| `native_ga2_vs_ga1` | да |
| `native_l2l_vs_ga1` | да |
| `unpinned_l2l_vs_ga` | да |

| контроль | первый прогон | второй прогон |
| --- | --- | --- |
| control1 | equal | equal |
| control2 | outside | outside |

## Контроль 1: float32 на CUDA, равно побитово

Поля конфига и переменной окружения для типа нет (`.scratch/a6-l2l-preference/research-float32-on-cuda.md`). Float32 получен подменой в процессе драйвера двух функций, `resolve_stream_dtype` и `bf16_fp16_flags` у обёртки DPO. Это обход, а не штатный режим. Что реально стояло в плечах (запись изнутри первого вызова слоя 0 и из `TrainingArguments`):

| поле | `native_ga1` | `native_l2l` | `f32_ga1` | `f32_ga2` | `f32_l2l` |
| --- | --- | --- | --- | --- | --- |
| `dtype_mode` | `native` | `native` | `f32` | `f32` | `f32` |
| `path` | `ga` | `l2l` | `ga` | `ga` | `l2l` |
| `stream_dtype` | `bfloat16` | `bfloat16` | `float32` | `float32` | `float32` |
| `train_bf16` | `True` | `True` | `False` | `False` | `False` |
| `train_fp16` | `False` | `False` | `False` | `False` | `False` |
| `mixed_precision` | `bf16` | `bf16` | `no` | `no` | `no` |
| `base_weight` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` |
| `lora_weight` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` |
| `layer_in` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` |
| `layer_out` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` |
| `grads` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` | `['torch.float32']` |
| `l2l_state_dtype` | `None` | `torch.bfloat16` | `None` | `None` | `torch.float32` |
| `attn_impl` | `sdpa` | `sdpa` | `sdpa` | `sdpa` | `sdpa` |
| `sdpa` | `MATH` | `MATH` | `MATH` | `MATH` | `MATH` |
| `matmul_precision` | `highest` | `highest` | `highest` | `highest` | `highest` |
| `allow_tf32` | `False` | `False` | `False` | `False` | `False` |

Наибольшая разница по шагам (градиент LoRA, потеря):

| пара | шаг | градиент | потеря |
| --- | --- | --- | --- |
| `f32_ga2_vs_ga1` | 0 | 0 | 0 |
| `f32_ga2_vs_ga1` | 1 | 0 | 0 |
| `f32_ga2_vs_ga1` | 2 | 0 | 0 |
| `f32_ga2_vs_ga1` | 3 | 0 | 0 |
| `f32_ga2_vs_ga1` | 4 | 0 | 0 |
| `f32_ga2_vs_ga1` | 5 | 0 | 0 |
| `f32_l2l_vs_ga1` | 0 | 0 | 0 |
| `f32_l2l_vs_ga1` | 1 | 0 | 0 |
| `f32_l2l_vs_ga1` | 2 | 0 | 0 |
| `f32_l2l_vs_ga1` | 3 | 0 | 0 |
| `f32_l2l_vs_ga1` | 4 | 0 | 0 |
| `f32_l2l_vs_ga1` | 5 | 0 | 0 |

## Контроль 2: пол шума от второго бэкенда SDPA, вне 1,5 пола

Плечи `native_*`, `MATH` против `EFFICIENT_ATTENTION` для пола, множитель 1,5. На шаге 0: в 1,5 пола. Наибольший пол 5.798e-03, наибольший разрыв L2L 9.262e-03.

| пара | шаг | градиент | потеря |
| --- | --- | --- | --- |
| `native_floor_vs_ga1` | 0 | 5.371e-03 | 5.248e-03 |
| `native_floor_vs_ga1` | 1 | 5.798e-03 | 3.956e-03 |
| `native_floor_vs_ga1` | 2 | 4.333e-03 | 9.565e-03 |
| `native_floor_vs_ga1` | 3 | 5.371e-03 | 3.760e-03 |
| `native_floor_vs_ga1` | 4 | 2.197e-03 | 7.970e-03 |
| `native_floor_vs_ga1` | 5 | 4.395e-03 | 1.214e-03 |
| `native_l2l_vs_ga1` | 0 | 1.648e-03 | 2.980e-07 |
| `native_l2l_vs_ga1` | 1 | 6.592e-03 | 2.093e-03 |
| `native_l2l_vs_ga1` | 2 | 4.028e-03 | 2.682e-06 |
| `native_l2l_vs_ga1` | 3 | 9.262e-03 | 2.778e-02 |
| `native_l2l_vs_ga1` | 4 | 2.197e-03 | 1.027e-03 |
| `native_l2l_vs_ga1` | 5 | 7.202e-03 | 5.714e-03 |
| `native_ga2_vs_ga1` | 0 | 0 | 0 |
| `native_ga2_vs_ga1` | 1 | 0 | 0 |
| `native_ga2_vs_ga1` | 2 | 0 | 0 |
| `native_ga2_vs_ga1` | 3 | 0 | 0 |
| `native_ga2_vs_ga1` | 4 | 0 | 0 |
| `native_ga2_vs_ga1` | 5 | 0 | 0 |

Что вне предела:

- step 1: base_model.model.model.layers.14.self_attn.q_proj.lora_A.default.weight differs by 2.014e-03, limit 1.808e-03
- step 1: base_model.model.model.layers.14.self_attn.q_proj.lora_B.default.weight differs by 1.251e-03, limit 1.190e-03
- step 1: base_model.model.model.layers.17.self_attn.q_proj.lora_A.default.weight differs by 8.430e-04, limit 8.211e-04
- step 1: base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight differs by 1.099e-03, limit 6.609e-04
- step 1: base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight differs by 4.501e-04, limit 2.403e-04
- step 1: base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weight differs by 1.312e-03, limit 1.190e-03
- step 1: base_model.model.model.layers.22.self_attn.v_proj.lora_A.default.weight differs by 6.447e-04, limit 5.436e-04
- step 1: base_model.model.model.layers.23.self_attn.q_proj.lora_A.default.weight differs by 4.272e-04, limit 2.146e-04

## Повтор условий тикета 06

Плечи `unpinned_*`: бэкенд SDPA не закреплён, как в тикете 06. В вердикт не входят.

| пара | шаг | градиент | потеря |
| --- | --- | --- | --- |
| `unpinned_l2l_vs_ga` | 0 | 2.563e-03 | 8.280e-03 |
| `unpinned_l2l_vs_ga` | 1 | 3.082e-03 | 3.639e-03 |
| `unpinned_l2l_vs_ga` | 2 | 3.906e-03 | 1.791e-02 |
| `unpinned_l2l_vs_ga` | 3 | 9.033e-03 | 2.490e-04 |
| `unpinned_l2l_vs_ga` | 4 | 3.174e-03 | 1.495e-02 |
| `unpinned_l2l_vs_ga` | 5 | 5.249e-03 | 1.900e-02 |

## Контроль 3: первый слой, где выходы расходятся, не нужен

Контроль 1 дал побитовое равенство, контроль 3 не нужен и не запускался.

## Валидность замеров и попытки

| плечо | попытка | исход ArmWatch | код выхода | причина исключения | исключена |
| --- | --- | --- | --- | --- | --- |
| `native_ga1` | 0 | ok | 0 |  | нет |
| `native_ga2` | 0 | ok | 0 |  | нет |
| `native_l2l` | 0 | ok | 0 |  | нет |
| `native_floor` | 0 | ok | 0 |  | нет |
| `f32_ga1` | 0 | ok | 0 |  | нет |
| `f32_ga2` | 0 | ok | 0 |  | нет |
| `f32_l2l` | 0 | ok | 0 |  | нет |
| `f32u_ga1` | 0 | ok | 0 |  | нет |
| `f32u_ga2` | 0 | ok | 0 |  | нет |
| `f32u_l2l` | 0 | ok | 0 |  | нет |
| `unpinned_ga` | 0 | ok | 0 |  | нет |
| `unpinned_l2l` | 0 | ok | 0 |  | нет |

Испорченные замеры (`void`) исключаются по `benchmarks/METHOD.md`, запись остаётся в каталоге результатов. Если колонка «исключена» пуста, исключать было нечего.

## Неудачные попытки

Неудачных попыток не было.

## Что не проверялось

- ORPO, SimPO, KTO, другие k, другие модели, другие карты, NF4, реальные наборы.
- Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами; чистое сравнение даёт шаг 0.
- Скорость и память не измерялись.
- Float32 получен подменой функций драйвера, а не штатной настройкой продукта.
- Контроль 1 запускался с закреплённым `MATH`; float32 при SDPA по умолчанию проверен плечами `f32u_*` (дополнение), в этой сессии на этой карте.
- Ядро SDPA по умолчанию определено по совпадению результатов с отдельными бэкендами; совпадение двух бэкендов бывает случайным, поэтому названы все совпавшие.
- Исход 2 держится на шагах 1 to 5; на шаге 0 L2L в пределе. Пол шума это одна выборка второго бэкенда SDPA.
- Начальные адаптеры (`start.pt`) не сохранены и хеш не записан; одинаковый старт виден только из равенства на шаге 0 в float32.
- Счёта vast.ai нет; стоимость в таблице стенда приблизительная.
- Типы весов записаны изнутри первого вызова слоя 0, а не в момент загрузки хранилища.

## Файлы

- [diagnosis.json](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/diagnosis.json), [environment.json](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/environment.json), [rental.json](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/rental.json)
- журналы и записи `ArmWatch` каждой попытки: `results/l2l-pref-diagnose-a6-vast-rtx3060-run2/<плечо>/attempt*/` (без каталогов `*-work`)
- [create.log](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/create.log), [COMMIT](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/COMMIT)
- [packages.txt](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/packages.txt), [bootstrap.log](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/bootstrap.log), [a8.log](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/a8.log), [show-instances-after.txt](results/l2l-pref-diagnose-a6-vast-rtx3060-run2/show-instances-after.txt)
- хеши `.pt` каждой попытки лежат в `diagnosis.json` (`attempts`); сами `.pt` в репозиторий не входят.
