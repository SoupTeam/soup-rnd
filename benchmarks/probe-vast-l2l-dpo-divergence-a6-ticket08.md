# A6 тикет 08: откуда расхождение L2L и накопления градиентов, DPO k=1, SmolLM2-135M, RTX 3060 12 GB, 2026-11-04

**Вердикт: исход 2: округление больше пола шума от второго бэкенда SDPA (накопление против себя даёт 0, пол взят от `EFFICIENT_ATTENTION` против `MATH`). Решение о допуске за пользователем.**

Правило: `benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md`, закоммичено до драйвера и до аренды; одна поправка (маска заполнения в контроле 3) внесена после смоука драйвера на CPU, тоже до аренды. Тикет 06 не переписывается: DPO, ORPO, SimPO и KTO там остаются «не прошёл», этот отчёт никого не отмечает готовым.

Модель `HuggingFaceTB/SmolLM2-135M`, реальные веса. Данные **синтетические** (`examples/data/dpo_pairs_synthetic.jsonl`, 12 строк, шесть окон). Результат относится к одной ячейке (DPO, k=1), этой модели и этой карте. ORPO, SimPO, KTO, другие k, другие карты и модели не заявляются.

## Стенд

| Параметр | Значение |
| --- | --- |
| Сервер | Vast.ai, оффер 47491621, инстанс 59130581, машина 43183, Wisconsin US, Xeon E5-2696 v3, 72 потока |
| GPU | NVIDIA GeForce RTX 3060, 580.173.02, 12288 MiB |
| Образ | nvidia/cuda:13.0.0-devel-ubuntu24.04, Python 3.12.3, драйвер 580.173.02 |
| Пакеты | torch 2.14.1+cu130, transformers 5.19.0, trl 0.29.1, peft 0.21.2, CUDA 13.0, Python 3.12.3 (полный список в `packages.txt`) |
| RAM | MemTotal:       263801688 kB; нагрузка хоста на старте: load average 5,83 / 6,96 / 10,61 на 72 потоках (uptime.txt, 16:31 UTC) |
| Коммит | `14562346865fcd416a69d24d2a53bd4a07771713` |
| Стоимость | около 0,03 USD всего: рабочий инстанс 59130581 11 минут (16:24:28Z to 16:35:27Z), около 0,085 USD/ч (ставка из предложения 0,0694 USD/ч; у первого инстанса фактическая ставка была на 22% выше предложения), около 0,016 USD; неудавшийся инстанс 59129442 10 минут по 0,0833 USD/ч, около 0,014 USD. Потолок 0,5 USD, остановка на 0,4. Падение credit аккаунта (0,96 до 0,62 за время работы) не считается моим расходом: на общем аккаунте работает чужой инстанс. |
| Аккаунт | команда vast.ai `soup-rnd` (по указанию пользователя) |
| Уничтожение | `vastai destroy instance 59130581 -y` в 16:35:27Z; `vastai show instances` после этого: только чужой инстанс 59126915 (show-instances-after.txt) |

## Команды

```bash
vastai create instance 47491621 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 60 --ssh --direct --label a6-08-diagnose --cancel-unavail
vastai attach ssh 59130581 "$(cat ~/.ssh/id_ed25519.pub)"
git ls-files -z | tar --null -T - -czf - | ssh ... 'mkdir -p /root/soup && tar xzf - -C /root/soup'
ssh -f ... "cd /root/soup && CARD='RTX 3060 12GB' nohup bash benchmarks/harness/l2l_pref_diagnose/remote.sh > /root/remote.out 2>&1 < /dev/null &"
# внутри remote.sh, из /root/soup: bootstrap (a0-bootstrap.sh), затем
python benchmarks/harness/l2l_pref_diagnose.py run --device cuda --out /root/results/a8 --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
vastai destroy instance 59130581 -y
```

Плечи, пути, типы и бэкенды SDPA описаны в правиле. Каждое плечо идёт в своём процессе под `ArmWatch`, обязательные проверки `suspend` и `power`.

## Что показали данные

- Контроль 1: в float32 L2L побитово равен накоплению на всех 6 шагах (градиенты LoRA и потери); накопление в float32 повторяется побитово. Запись типов подтверждает float32 без автокаста в этих трёх плечах. Значит, при закреплённом `MATH` в общем пути L2L в этой ячейке на этой карте ошибки нет. Что разрыв условий тикета 06 (SDPA по умолчанию) вызван только точностью, это не доказано: float32 без закрепления не запускался.
- Контроль 2, шаг 0 (одинаковые адаптеры): разрыв L2L по градиенту 1.648e-03, пол 5.371e-03; по потере 2.980e-07 и 5.248e-03. Шаг 0 в пределе: да.
- Контроль 2, все шаги: вне предела 238 записей на шагах [1, 2, 3, 4, 5]; наибольшее превышение предела 12.67 раза (предел это 1,5 пола), меньше чем вдвое превышают 195 из 238. Пол здесь одно измерение второго бэкенда. Шаги после нулевого не независимы от шага 0 (адаптеры уже разошлись), поэтому по правилу исход читается как «вне», а размер превышения виден в таблице.
- Без закрепления SDPA (условия тикета 06) потеря шага 0 расходится на 8.280e-03, с `MATH` в обоих путях на 2.980e-07. Это совпадает с тем, что L2L вызывает слои без маски внимания, а накопление с маской, и умолчание SDPA может выбирать для них разные ядра. Какое ядро выбрано по умолчанию, здесь не записано, поэтому это наблюдение, а не причина.

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
| `unpinned_ga` | 0 | ok | 0 |  | нет |
| `unpinned_l2l` | 0 | ok | 0 |  | нет |

Испорченные замеры (`void`) исключаются по `benchmarks/METHOD.md`, запись остаётся в каталоге результатов. Если колонка «исключена» пуста, исключать было нечего.

## Неудачные попытки

- Инстанс 59129442 (оффер 54058503, RTX 3060, Корея) создан 16:14:13Z и 10 минут оставался в статусе `loading`; по просьбе пользователя уничтожен в 16:24Z (первая команда без `-y` отменилась сама, повторена с `-y`). Измерений на нём нет. Записи: `create-attempt1-failed.log`, `failed-instance-1-destroyed.txt`.

## Что не проверялось

- ORPO, SimPO, KTO, другие k, другие модели, другие карты, NF4, реальные наборы.
- Шаги после нулевого начинаются с адаптеров, обновлённых разными градиентами; чистое сравнение даёт шаг 0.
- Скорость и память не измерялись.
- Float32 получен подменой функций драйвера, а не штатной настройкой продукта.
- Контроль 1 запускался только с закреплённым `MATH`; float32 при SDPA по умолчанию (условия тикета 06) не запускался.
- Исход 2 держится на шагах 1 to 5; на шаге 0 L2L в пределе. Пол шума это одна выборка второго бэкенда SDPA.
- Начальные адаптеры (`start.pt`) не сохранены и хеш не записан; одинаковый старт виден только из равенства на шаге 0 в float32.
- Счёта vast.ai нет; стоимость в таблице стенда приблизительная, ставка рабочего инстанса оценена по предложению.
- Типы весов записаны изнутри первого вызова слоя 0, а не в момент загрузки хранилища.

## Файлы

- [diagnosis.json](results/l2l-pref-diagnose-a6-vast-rtx3060/diagnosis.json), [environment.json](results/l2l-pref-diagnose-a6-vast-rtx3060/environment.json), [rental.json](results/l2l-pref-diagnose-a6-vast-rtx3060/rental.json)
- журналы и записи `ArmWatch` каждой попытки: `results/l2l-pref-diagnose-a6-vast-rtx3060/<плечо>/attempt*/` (без каталогов `*-work`)
- [create.log](results/l2l-pref-diagnose-a6-vast-rtx3060/create.log), [create-attempt1-failed.log](results/l2l-pref-diagnose-a6-vast-rtx3060/create-attempt1-failed.log), [COMMIT](results/l2l-pref-diagnose-a6-vast-rtx3060/COMMIT)
- [packages.txt](results/l2l-pref-diagnose-a6-vast-rtx3060/packages.txt), [bootstrap.log](results/l2l-pref-diagnose-a6-vast-rtx3060/bootstrap.log), [a8.log](results/l2l-pref-diagnose-a6-vast-rtx3060/a8.log), [show-instances-after.txt](results/l2l-pref-diagnose-a6-vast-rtx3060/show-instances-after.txt)
- хеши `.pt` каждой попытки лежат в `diagnosis.json` (`attempts`); сами `.pt` в репозиторий не входят.
