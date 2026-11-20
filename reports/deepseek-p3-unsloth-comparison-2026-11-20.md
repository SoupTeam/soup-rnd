# DeepSeek P3 — Сравнение `feat/p3-unified-go` с Unsloth (`2026.11.7`) и HF+BnB на RTX 4060 Ti 8 ГБ

**Дата:** 20 ноября 2026  
**Ветка / PR:** `feat/p3-unified-go` ([Draft PR #77](https://github.com/SoupTeam/soup-rnd/pull/77))  
**Стенд:** та же физическая `NVIDIA GeForce RTX 4060 Ti 8 ГБ` (`offer_id=58316190`, `8188 MiB`, `7.651 GiB` CUDA, SM 8.9), 62 GiB RAM, NVMe, Linux, Python 3.11. Окружение Unsloth: `unsloth 2026.11.7`, `torch 2.14.1+cu130`, `transformers 5.17.0`, `bitsandbytes 0.50.2`, `triton 3.8.0` *(инстанс Vast.ai удалён сразу после выгрузки JSON)*.  
**Условия:** `k=16, S=256` (`4 096` токенов/шаг), 4-bit NF4, LoRA `r=16, alpha=32`, `use_gradient_checkpointing="unsloth"`, фоновый поллер NVML и RSS каждые **20 мс**.

---

## Главное за 30 секунд

По запросу `nurikw` прогнали `unsloth 2026.11.7` и стандартный `HF + BitsAndBytes NF4 CPU-offload` на той же 8 ГБ карте и тех же моделях (`Qwen/Qwen3.8-27B` и `Kanana-2-30B-A3B`), плюс добавили положительный контроль `Qwen/Qwen2.5-0.5B`:

- **Контроль стенда (`Qwen/Qwen2.5-0.5B`, 4-bit):** Unsloth `2026.11.7` отрабатывает шаг штатно: **`314.79 tok/s`** (`13.012 с`), `loss = 14.0394`, пик VRAM **`1.171 GiB`** NVML (`0.518 GiB` alloc), пик RAM **`2.736 GiB`** RSS.
- **`Qwen/Qwen3.8-27B` (27B dense/hybrid):**
  - **Soup `unified_go` (`FP8 block64`):** `159.44 tok/s` (`25.33 с/шаг`), `5.106 GiB` NVML, `13.241 GiB` RSS *(на `k=64, S=512`: `199.73 tok/s`, `4.948 GiB` NVML, `27.86 GiB` RSS)*.
  - **Unsloth `2026.11.7` (`unsloth_default`):** падает на загрузке (`6.87 с`) с `ValueError` в `raise_if_bnb_cpu_spill` (`cuda:0 has 7.46 GB free`, CPU-spill bitsandbytes не квантизует).
  - **HF + BitsAndBytes NF4 (`device_map="auto"`, `fp32_cpu_offload=True`):** за `82.42 с` загрузки забирает **`56.690 GiB` RSS** (в 4.28× больше Soup, не влезает в 32 ГБ ПК) и падает на первом forward (`NotImplementedError: Cannot copy out of meta tensor; no data!`).
- **`Kanana-2-30B-A3B` (30B MoE, 3B активных):**
  - **Soup `unified_go` (`FP8 block64 + Oracle`):** `317.4–322.7 tok/s` (`12.60–12.90 с/шаг`), `1.905–1.968 GiB` NVML, `7.85–19.27 GiB` RSS *(на `k=64, S=512` с 3.5 GiB кэшем: `630.68 tok/s`, `3.567 GiB` NVML, `10.88 GiB` RSS)*.
  - **Unsloth `2026.11.7` (`unsloth_default` и `device_map="auto"`):** падает на загрузке (`6.0–8.2 с`) с тем же `ValueError` в `raise_if_bnb_cpu_spill`.
  - **HF + BitsAndBytes NF4 (`device_map="auto"`, `fp32_cpu_offload=True`):** за `87.24 с` загрузки забирает **`40.370 GiB` RSS** + **`7.817 GiB` NVML** и падает в **`CUDA out of memory`** (`768 MiB`) на сборке `layers.17.mlp.experts.gate_up_proj`.

---

## 1. Сводная таблица замеров (`k=16, S=256`, 4 096 tok/step, `lora_r=16`, 4-bit NF4)

*Сырые JSON:* `benchmarks/results/vast-4060ti-unsloth-compare-20261120/{control_05b,qwen38_27b,kanana}_summary.json` (данные Soup — из Серии 5, `benchmarks/results/vast-4060ti-qwen38-agentic-20261120/`).

| Модель | Вариант | Статус | Загрузка / всего (с) | Шаг (с) | Скорость (tok/s) | `alloc` / `resv` / **`nvml_peak`** | `RSS peak` | Итог / ошибка |
|---|---|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-0.5B` *(контроль)* | `unsloth_default` (`2026.11.7`) | ok | 7.64 / 20.66 | 13.01 | `314.79` | `0.518 / 0.604 /` **`1.171 GiB`** | `2.736 GiB` | `loss=14.0394`; стенд исправен |
| `Qwen/Qwen3.8-27B` | **Soup `unified_go`** (`FP8 block64`) | **ok** | ~10 *(индекс)* | **25.33** | **`159.44`** | `4.630 / 4.922 /` **`5.106 GiB`** | **`13.241 GiB`** | 16 шагов SFT за `6.85 мин` (`$0.048`), Exact `50% → 100%` |
| `Qwen/Qwen3.8-27B` | `unsloth_default` | error (`ValueError`) | — / 6.87 | — | — | `0.008 / 0.022 / 0.534 GiB` | `1.891 GiB` | `raise_if_bnb_cpu_spill` (`loader_utils.py:520`) |
| `Qwen/Qwen3.8-27B` | `unsloth_cpu_offload` | error (`TypeError`) | — / 5.84 | — | — | `0.008 / 0.022 / 0.534 GiB` | `1.887 GiB` | Баг вызова в 1-й версии харнеса (см. разд. 4.1) |
| `Qwen/Qwen3.8-27B` | `hf_bnb_cpu_offload` | error (`NotImplementedError`) | 82.42 / 90.25 | — | — | `4.273 / 4.369 / 4.882 GiB` | **`56.690 GiB`** | >32 GiB RAM; meta tensor в `bitsandbytes/functional.py:582` |
| `Kanana-2-30B-A3B` | **Soup `unified_go`** (`FP8 + Oracle`) | **ok** | ~6 *(индекс)* | **12.60–12.90** | **`317.4–322.7`** | `1.214–1.252 / — /` **`1.905–1.968 GiB`** | `7.85–19.27 GiB` | 16 шагов SFT за `3.44 мин` (`$0.024`), JSON `0% → 81.25%` |
| `Kanana-2-30B-A3B` | `unsloth_default` | error (`ValueError`) | — / 8.17 | — | — | `0.008 / 0.022 / 0.534 GiB` | `1.883 GiB` | `raise_if_bnb_cpu_spill` (`loader_utils.py:520`) |
| `Kanana-2-30B-A3B` | `unsloth_cpu_offload` (`device_map="auto"`) | error (`ValueError`) | — / 5.99 | — | — | `0.008 / 0.022 / 0.534 GiB` | `1.886 GiB` | Тот же `raise_if_bnb_cpu_spill` при `max_memory={0:"7GiB","cpu":"50GiB"}` |
| `Kanana-2-30B-A3B` | `hf_bnb_cpu_offload` | error (`RuntimeError` / `CUDA OOM`) | — / 87.24 | — | — | `7.171 / 7.359 /` **`7.817 GiB`** | **`40.370 GiB`** | >32 GiB RAM; `CUDA OOM` (`768 MiB`) на `layers.17.mlp.experts.gate_up_proj` |

---

## 2. Точные ошибки из логов и JSON

### 2.1. `Unsloth 2026.11.7` (`unsloth_default` на обеих моделях и `unsloth_cpu_offload` на Kanana)
Из `qwen38_27b_summary.json` и `kanana_summary.json` (`unsloth/models/loader_utils.py`, строка 520):
```text
File "/opt/conda/lib/python3.11/site-packages/unsloth/models/loader_utils.py", line 520, in raise_if_bnb_cpu_spill
ValueError: Unsloth: /workspace/models/kanana does not fit in GPU memory (cuda:0 has 7.46 GB free),
so transformers placed part of it on the CPU, which bitsandbytes cannot quantize.
Pass `offload_layers = "auto"` to from_pretrained to keep the decoder layers the GPU cannot hold
in host RAM and stream them in during training (slower, and it needs that much free system RAM),
load a smaller model, or free GPU memory held by other programs.
```
В 4-bit NF4 веса 27B–30B моделей занимают ~14.5–16 ГБ, поэтому `transformers` при `device_map="auto"` сбрасывает часть модулей на CPU, а `bitsandbytes` не умеет квантизовать CPU-слои.

### 2.2. `HF + BitsAndBytes NF4` (`device_map="auto"`, `llm_int8_enable_fp32_cpu_offload=True`)
- **На `Qwen/Qwen3.8-27B`** (`qwen38_27b_summary.json`): модель загрузилась за `82.418 с`, заняв **`56.690 GiB` RSS** (оффлоад-слои остаются не квантизованными на CPU), и упала на первом же forward в `accelerate/hooks.py:382` → `bitsandbytes/functional.py:582`:
  ```text
  File "/opt/conda/lib/python3.11/site-packages/bitsandbytes/functional.py", line 582, in to
    self.code = self.code.to(device)
  NotImplementedError: Cannot copy out of meta tensor; no data!
  ```
- **На `Kanana-2-30B-A3B`** (`run_unsloth_kanana.log`, строки 68–85): в `transformers 5.17.0` конвертер `core_model_loading.py:182` объединяет 128 экспертов каждого слоя в единый тензор `mlp.experts.gate_up_proj`. На 17-м слое при **`40.370 GiB` RSS** и **`7.817 GiB` NVML** сработало:
  ```text
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB.
  GPU 0 has a total capacity of 7.65 GiB of which 667.06 MiB is free.
  Error: Concatenate on tensors destined for model.layers.17.mlp.experts.gate_up_proj.
  ```

---

## 3. Почему стеки решают разные задачи

| Механизм | `Unsloth 2026.11.7` | `HF + BitsAndBytes CPU-offload` | `Soup feat/p3-unified-go` |
|---|---|---|---|
| Где лежат 4-bit веса 27B–30B | Должны влезать в VRAM (или слои целиком в System RAM через `offload_layers`) | Часть в VRAM, остаток в System RAM (40–57 GiB RSS, >32 ГБ ПК) | Шарды `per_expert` / послойные NF4 на **NVMe**, потоковое чтение через C++17 `PackedStreamAsync` |
| По-экспертный стриминг MoE | Нет | Нет (склейка всех 128 экспертов слоя в один тензор → OOM) | Есть: читаются и деквантуются только активные эксперты окна (`M1/M2+` + `H2`) |
| Потолок головы (`V = 128k–248k`) | Fused CE (при условии, что веса влезли) | Полный слот + логиты | `M3` освобождение слота `LargeLayerBufferPool` (`−0.50..−2.37 GiB` VRAM) + `Chunked CE (256)` |
| Граничные активации в RAM | Offloaded gradient checkpointing в BF16 | Нет | `FP8 (block64_e4m3)` + `DeepseekRoutingOracle` (`−46.9%` RAM, `gate_w_drift=0.0`) |

---

## 4. Честно про ограничения этого прогона

1. **Баг в первом запуске `qwen38_27b / unsloth_cpu_offload`:** при прогоне Qwen в харнесе в `FastLanguageModel.from_pretrained` ошибочно передавался `llm_int8_enable_fp32_cpu_offload=True`, поэтому ветка упала с `TypeError` раньше проверки VRAM. Перед прогоном Kanana аргумент убрали из `p3_unsloth_4060ti_compare.py` и получили чистый `ValueError` из `raise_if_bnb_cpu_spill`. Повторно Qwen для этой одной ветки не перезапускали, т.к. `unsloth_default` уже показал тот же `raise_if_bnb_cpu_spill` (`cuda:0 has 7.46 GB free`).
2. **Режим `offload_layers="auto"` в Unsloth `2026.11.7` не запускался:** о существовании этого нового флага мы узнали прямо из сообщения об ошибке на стенде. Судя по описанию самого Unsloth (*«keep the decoder layers the GPU cannot hold in host RAM and stream them in during training (slower, and it needs that much free system RAM)»*), он держит вытесненные слои целиком в System RAM без NVMe-стриминга и по-экспертного окна MoE — для 671B (`~368 ГБ`) и для 32 ГБ ПК с 27B–30B + активациями это не закрывает ограничение P3, но для полной чистоты сравнения на 27B–30B в 62 GiB RAM этот флаг стоит отдельным коротким прогоном замерить.
3. **Сравнение `tok/s` с `Qwen2.5-0.5B` — только как контроль стенда:** в `control_05b` один первый шаг Unsloth включает компиляцию/прогрев (`13.01 с`), поэтому цифра `314.8 tok/s` на 0.5B не является steady-state пиком Unsloth — она лишь подтверждает, что `unsloth 2026.11.7` + `triton 3.8.0` на этой RTX 4060 Ti реально работают без проблем с окружением.
4. **Предупреждения окружения на образе Vast (`torch 2.14.1+cu130`):** `xformers` не загрузился (использовался нативный PyTorch SDPA) и `import unsloth` шёл после `import transformers` (выведено в `run_unsloth_kanana.log`). На `raise_if_bnb_cpu_spill` и физический объём 4-bit весов (14.5–16 ГБ > 7.46 ГБ VRAM) это не влияет.

---

## 5. Артефакты и проверка

- **Харнес:** `benchmarks/harness/p3_unsloth_4060ti_compare.py` (`ruff check` — чисто).
- **Сырые JSON и логи:** `benchmarks/results/vast-4060ti-unsloth-compare-20261120/`:
  - `control_05b_summary.json`, `control_05b_unsloth_default.json`
  - `qwen38_27b_summary.json`, `qwen38_27b_{unsloth_default,unsloth_cpu_offload,hf_bnb_cpu_offload}.json`
  - `kanana_summary.json`, `kanana_{unsloth_default,unsloth_cpu_offload,hf_bnb_cpu_offload}.json`
  - `run_unsloth2.log`, `run_unsloth_kanana.log`
- **Основной отчёт по ветке:** `reports/deepseek-p3-unified-go-2026-11-20.md` (раздел 4.4).
