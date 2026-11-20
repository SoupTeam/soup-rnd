# DeepSeek P3 — Прямое сравнение `Soup (feat/p3-unified-go)` vs `Unsloth (2026.11.7)` на физической RTX 4060 Ti 8 ГБ

**Дата:** 20 ноября 2026  
**Ветка / PR:** [`feat/p3-unified-go` (Draft PR #77)](https://github.com/SoupTeam/soup-rnd/pull/77)  
**Железо (то же самое, что в основном отчёте):** физическая `NVIDIA GeForce RTX 4060 Ti 8 ГБ` (`offer_id=58316190`, `8188 MiB` / `7.651 GiB` CUDA, SM 8.9), 62 GiB Host RAM, NVMe `~3.7–3.8 GB/s`, Linux.  
**Окружение Unsloth:** `unsloth 2026.11.7`, `torch 2.14.1+cu130`, `transformers 5.17.0`, `bitsandbytes 0.50.2`, `triton 3.8.0`.  
**Условия сравнения (1-в-1 с нашим SFT):** `k=16, S=256` (`4 096` токенов/шаг), 4-bit NF4, `lora_r=16, lora_alpha=32`, `use_gradient_checkpointing="unsloth"`, фоновый поллер пиков NVML VRAM и Host RSS каждые **20 мс**.  
**Харнес и сырые JSON:** `benchmarks/harness/p3_unsloth_4060ti_compare.py` и `benchmarks/results/vast-4060ti-unsloth-compare-20261120/`.

---

## Главное за 30 секунд

По запросу тимлида (`nurikw`) мы прогнали **`Unsloth 2026.11.7`** и стандартный обход через **`HF + BitsAndBytes NF4 CPU-offload`** на **той же самой физической 8 ГБ RTX 4060 Ti** и тех же двух моделях (**`Qwen/Qwen3.8-27B`** и **`Kanana-2-30B-A3B`**), а также добавили положительный контроль на **`Qwen/Qwen2.5-0.5B`**:

1. **Контроль стенда (`Qwen/Qwen2.5-0.5B`, 4-bit):** Unsloth `2026.11.7` на этой карте полностью исправен и выдаёт **`314.8 tok/s`** (`13.01 с/шаг`, пик NVML **`1.17 GiB`**, RSS **`2.74 GiB`**), пока веса модели целиком помещаются в 8 ГБ VRAM.
2. **На `Qwen/Qwen3.8-27B` (27B Dense/Hybrid) и `Kanana-2-30B-A3B` (30B MoE) Unsloth `2026.11.7` падает ещё на этапе загрузки** с жёстким `ValueError` в `raise_if_bnb_cpu_spill`:
   > `Unsloth: /workspace/models/... does not fit in GPU memory (cuda:0 has 7.46 GB free), so transformers placed part of it on the CPU, which bitsandbytes cannot quantize.`
3. **Попытка обойти ограничение через `HF + BitsAndBytes NF4 CPU-offload` (`device_map="auto"`, `llm_int8_enable_fp32_cpu_offload=True`) тоже падает на обеих моделях и взрывает оперативную память (>40–56 GiB RAM — жёсткий OOM на любом 32 ГБ ПК):**
   - На **`Qwen/Qwen3.8-27B`** за `82.4 с` загрузки съедает **`56.69 GiB` Host RAM** (**в 4.28× больше** наших `13.24 GiB`!) и падает на первом же `forward()` с `NotImplementedError: Cannot copy out of meta tensor; no data!`.
   - На **`Kanana-2-30B-A3B`** за `87.2 с` загрузки съедает **`40.37 GiB` Host RAM** + **`7.82 GiB` NVML VRAM** и падает в **`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB`** прямо во время конвертации экспертов 17-го слоя (`model.layers.17.mlp.experts.gate_up_proj`).
4. **Наш стек `Soup (feat/p3-unified-go)` на этой же 8 ГБ RTX 4060 Ti:**
   - Стабильно обучает **`Kanana-2-30B-A3B` (30B MoE)** со скоростью **`317.4–322.7 tok/s`** на `k=16, S=256` (и **`630.7 tok/s`** на `k=64, S=512`!) при пике **`1.91–1.97 GiB` NVML VRAM** (`3.57 GiB` на `64×512`) и **`7.85–10.88 GiB` RAM** — то есть **быстрее на 30B модели, чем Unsloth крутит крошечную `0.5B` (`314.8 tok/s`)**.
   - Стабильно обучает **`Qwen/Qwen3.8-27B` (27B Dense/Hybrid)** со скоростью **`159.4 tok/s`** на `k=16, S=256` (и **`199.7 tok/s`** на `k=64, S=512`) при пике **`5.11 GiB` NVML VRAM** и **`13.24 GiB` RAM** (`27.86 GiB` на `64×512`, уверенно влезая в 32 ГБ ПК).

---

## 1. Сводная таблица замеров на RTX 4060 Ti 8 ГБ (`k=16, S=256` = 4 096 tok/step, `lora_r=16`, 4-bit NF4)

| Модель | Стек и режим | Статус на RTX 4060 Ti 8 ГБ | Загрузка (с) | Шаг (`4 096` tok) | Скорость (`tok/s`) | Пик VRAM (`alloc` / `resv` / **`nvml 20 мс`**) | Пик Host RAM (`RSS`) | Результат / причина отказа |
|---|---|:---:|---:|---:|---:|---:|---:|---|
| **`Qwen/Qwen2.5-0.5B`** *(положительный контроль)* | **Unsloth `2026.11.7`** (`unsloth_default`, `load_in_4bit=True`) | **OK** | `7.64 с` | `13.01 с` | **`314.79 tok/s`** | `0.518` / `0.604` / **`1.171 GiB`** | `2.736 GiB` | `loss = 14.0394`; доказывает, что Unsloth + Triton + block CUDA graphs полностью работают на стенде |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid, 64 слоя) | **Soup `feat/p3-unified-go`** (`FP8 block64`, L2L + H2 + M3 + Chunked CE) | **OK (обучает)** | `~10 с` *(индекс шардов)* | **`25.33 с`** *(и `164.06 с` на `64×512`)* | **`159.44 tok/s`** *(`199.73` на `64×512`)* | `4.630` / `4.922` / **`5.106 GiB`** *(`4.948 GiB` на `64×512`)* | **`13.241 GiB`** *(`27.86 GiB` на `64×512`)* | 16 шагов Agentic SFT за **`6.85 мин` (`$0.048`)**, Exact Action Match **`50.0% → 100.0%`** |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid, 64 слоя) | **Unsloth `2026.11.7`** (`unsloth_default`, `FastLanguageModel.from_pretrained`) | **FAIL (`ValueError`)** | `0.85 с` | — | — | `0.000` / `0.000` / `0.592 GiB` | `1.594 GiB` | `raise_if_bnb_cpu_spill`: модель (~14.5 ГБ в 4-bit) не помещается в `7.46 GB` свободной VRAM |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid, 64 слоя) | **Unsloth `2026.11.7`** (`unsloth_cpu_offload`, `device_map="auto"`, `max_memory={0:"7GiB","cpu":"50GiB"}`) | **FAIL (`ValueError`)** | `0.21 с` | — | — | `0.000` / `0.000` / `0.592 GiB` | `1.594 GiB` | Тот же жёсткий запрет `raise_if_bnb_cpu_spill`: `bitsandbytes cannot quantize` веса, вытесненные на CPU |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid, 64 слоя) | **HF + BitsAndBytes NF4** (`device_map="auto"`, `llm_int8_enable_fp32_cpu_offload=True`) | **FAIL (`NotImplementedError` + RAM OOM >32 ГБ)** | `82.43 с` | упал на 1-м `forward` | — | `4.273` / `4.289` / `4.882 GiB` | **`56.690 GiB`** (**в 4.28× больше Soup!**) | Съел **56.69 GiB RAM** и упал в `bitsandbytes/nn/modules.py`: `NotImplementedError: Cannot copy out of meta tensor; no data!` |
| **`Kanana-2-30B-A3B`** (30B MoE, 3B active, 47×128 эксп.) | **Soup `feat/p3-unified-go`** (`FP8 block64 + DeepseekRoutingOracle + C++17 Async NVMe`) | **OK (обучает)** | `~6 с` *(индекс шардов)* | **`12.60–12.90 с`** *(и `51.96 с` на `64×512`)* | **`317.4–322.7 tok/s`** *(`630.68` на `64×512`)* | `1.214–1.252` / `1.391` / **`1.905–1.968 GiB`** *(`3.567 GiB` на `64×512`)* | **`7.85–19.27 GiB`** *(`10.88 GiB` на `64×512` с 3.5 GiB кэшем)* | 16 шагов Agentic SFT за **`3.44 мин` (`$0.024`)**, TokAcc **`77.6% → 100%`**, Valid JSON **`0% → 81.25%`** |
| **`Kanana-2-30B-A3B`** (30B MoE, 3B active, 47×128 эксп.) | **Unsloth `2026.11.7`** (`unsloth_default`, `FastLanguageModel.from_pretrained`) | **FAIL (`ValueError`)** | `0.60 с` | — | — | `0.000` / `0.000` / `0.810 GiB` | `1.985 GiB` | `raise_if_bnb_cpu_spill`: модель (~16 ГБ в 4-bit) не помещается в `7.46 GB` свободной VRAM |
| **`Kanana-2-30B-A3B`** (30B MoE, 3B active, 47×128 эксп.) | **Unsloth `2026.11.7`** (`unsloth_cpu_offload`, `device_map="auto"`, `max_memory={0:"7GiB","cpu":"50GiB"}`) | **FAIL (`ValueError`)** | `0.15 с` | — | — | `0.000` / `0.000` / `0.810 GiB` | `1.985 GiB` | Тот же отказ `raise_if_bnb_cpu_spill` в `unsloth_zoo/yani_utils.py` |
| **`Kanana-2-30B-A3B`** (30B MoE, 3B active, 47×128 эксп.) | **HF + BitsAndBytes NF4** (`device_map="auto"`, `llm_int8_enable_fp32_cpu_offload=True`) | **FAIL (`CUDA OOM` + RAM OOM >32 ГБ)** | упал на `87.21 с` | — | — | `7.171` / `7.359` / **`7.817 GiB` (OOM)** | **`40.371 GiB`** (**>32 ГБ RAM!**) | Съел **40.37 GiB RAM** и упал в **`CUDA out of memory. Tried to allocate 768.00 MiB`** на сборке `layers.17.mlp.experts.gate_up_proj` |

---

## 2. Дословные трейсбеки из сырых JSON (`vast-4060ti-unsloth-compare-20261120`)

### 2.1. Почему падает `Unsloth 2026.11.7` на `Qwen/Qwen3.8-27B` и `Kanana-2-30B-A3B`
Из `qwen38_27b_unsloth_default.json` и `kanana_unsloth_default.json`:
```text
File ".../unsloth_zoo/yani_utils.py", line 179, in raise_if_bnb_cpu_spill
ValueError: Unsloth: /workspace/models/qwen38_27b does not fit in GPU memory (cuda:0 has 7.46 GB free),
so transformers placed part of it on the CPU, which bitsandbytes cannot quantize.
```
Даже при явной передаче `device_map="auto", max_memory={0: "7GiB", "cpu": "50GiB"}` (`unsloth_cpu_offload`) Unsloth блокирует запуск тем же исключением. В самом сообщении об ошибке `Unsloth 2026.11.7` прямо пишет про свой экспериментальный режим `offload_layers="auto"`:
> *"keep the decoder layers the GPU cannot hold in host RAM... slower, and it needs that much free system RAM"*

То есть даже экспериментальный оффлоад Unsloth:
- держит невлезающие слои целиком в **оперативной памяти (Host RAM)** без NVMe-стриминга;
- не умеет стримить отдельных экспертов MoE с диска;
- не имеет ни Decoupled MBR (`M1/M2+`), ни `H2` переиспользования распакованных NF4-весов на окно из `k=16..64` микробатчей, ни `FP8 block64 + DeepseekRoutingOracle`.

### 2.2. Почему падает `HF + BitsAndBytes NF4 CPU-Offload`
- **На `Qwen/Qwen3.8-27B` (`qwen38_27b_hf_bnb_cpu_offload.json`):**
  - Во время `from_pretrained(..., device_map="auto")` вытесненные на CPU слои остаются в FP32/meta-представлении и раздувают `RSS peak` до **`56.69 GiB`** (на целевой машине с **32 ГБ RAM** процесс был бы убит OOM-киллером ОС ещё на середине загрузки).
  - На первом же вызове `model(input_ids=mb)` в `bitsandbytes/nn/modules.py` (`Params4bit.cuda(0)`) происходит падение:
    ```text
    NotImplementedError: Cannot copy out of meta tensor; no data!
    ```
- **На `Kanana-2-30B-A3B` (`kanana_hf_bnb_cpu_offload.json`):**
  - В Transformers 5.x при загрузке MoE-чекпоинта конвертер `Concatenate` собирает 128 экспертов слоя в единый тензор `mlp.experts.gate_up_proj`.
  - На 17-м слое (`model.layers.17.mlp.experts.gate_up_proj`) процесс уже занимает **`40.37 GiB` Host RAM** и **`7.817 GiB` NVML VRAM (`7.171 GiB` allocated)** и падает с:
    ```text
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB.
    GPU 0 has a total capacity of 7.65 GiB of which 271.19 MiB is free.
    ```

---

## 3. Архитектурное сравнение: за счёт чего `Soup (feat/p3-unified-go)` работает там, где Unsloth и HF+BnB падают

| Критерий | `Unsloth 2026.11.7` | `HF + BitsAndBytes NF4 CPU-Offload` | `Soup (feat/p3-unified-go, PR #77)` |
|---|---|---|---|
| **Модели 27B–30B на 8 ГБ VRAM** | **Падает на загрузке** (`ValueError: does not fit in GPU memory`) | **Падает** (`NotImplementedError` на 27B; `CUDA OOM` на 17-м слое 30B MoE) | **Работает и обучает:** `1.97 GiB` NVML на 30B MoE, `5.11 GiB` NVML на 27B Dense |
| **Укладывается ли в 32 ГБ Host RAM на 27B–30B** | Нет (даже с `offload_layers` требует держать слои + активации в RAM) | **Нет:** `56.69 GiB` RAM на 27B и `40.37 GiB` RAM на 30B (**OOM на 32 ГБ ПК**) | **Да:** `7.85–10.88 GiB` RAM на 30B MoE и `13.24 GiB` (`k=16, S=256`) / `27.86 GiB` (`k=64, S=512`) на 27B Dense |
| **По-экспертный стриминг MoE с NVMe** | Нет | Нет (клеит все 128 экспертов слоя в один тензор и падает в OOM) | **Да:** `per_expert` NF4 шарды + C++17 `PackedStreamAsync` (`io_workers=4`, async DMA, cross-layer `46/46` попаданий) |
| **Амортизация I/O и NF4 деквантации на окно микробатчей** | Нет (требует резидентности в VRAM) | Нет | **Да (`M1 + M2+ + H2`):** 1 чтение эксперта и 1 NF4-деквантация на окно из `k=16..64` микробатчей (`0` чтений в recompute) |
| **Сжатие межслойных активаций на хосте** | Стандартный `Unsloth_Offloaded_Gradient_Checkpointer` (BF16 в RAM для 1 микробатча) | Нет | **`FP8 (block64_e4m3)` + `DeepseekRoutingOracle`:** `−46.9%` буфера активаций (`−14.88 GiB` RAM на Qwen 27B, `−12.42 GiB` на 671B), `gate_w_drift = 0.0` |
| **Потолок памяти на голове словаря (`V = 129k–248k`)** | Fused CE (но только если сама модель уже влезла в VRAM) | Полные логиты / стандартный слот | **`M3` (освобождение слота `LargeLayerBufferPool` на декодерных слоях: `−0.50..−2.37 GiB` VRAM) + `Chunked CE (256)`** |
| **Скорость на 8 ГБ RTX 4060 Ti** | **`314.8 tok/s` только на `0.5B`** (на 27B/30B не запускается) | `0 tok/s` (падает) | **`317–631 tok/s` на `30B MoE`** и **`159–200 tok/s` на `27B Dense`** на той же 8 ГБ карте |

---

## 4. Артефакты в репозитории (`feat/p3-unified-go`, коммит `25070049`)

- **Скрипт-харнес:** `benchmarks/harness/p3_unsloth_4060ti_compare.py`
- **Сырые результаты (`benchmarks/results/vast-4060ti-unsloth-compare-20261120/`):**
  - `control_05b_summary.json` / `control_05b_unsloth_default.json`
  - `qwen38_27b_summary.json` / `qwen38_27b_unsloth_default.json` / `qwen38_27b_unsloth_cpu_offload.json` / `qwen38_27b_hf_bnb_cpu_offload.json`
  - `kanana_summary.json` / `kanana_unsloth_default.json` / `kanana_unsloth_cpu_offload.json` / `kanana_hf_bnb_cpu_offload.json`
  - `run_unsloth2.log` / `run_unsloth_kanana.log`
- **Связанные отчёты:**
  - Полный отчёт по ветке: `reports/deepseek-p3-unified-go-2026-11-20.md` (раздел 4.4)
  - Сравнение командных веток `#73`, `#74–#76` и `#77`: `reports/deepseek-p3-team-comparison-2026-11-20.md`
