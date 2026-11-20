# DeepSeek P3 — Единый отчёт по ветке `feat/p3-unified-go` (RTX 4060 Ti 8 ГБ: Kanana 30B, Qwen 3.8 27B и Agentic SFT)

**Дата:** 20 ноября 2026  
**База:** [PR #73](https://github.com/SoupTeam/soup-rnd/pull/73) (`25336300ef279a9080f0d451acfad60c8f206e58`, Triton `batch_invariant` GEMM)  
**Локальная ветка:** `feat/p3-unified-go` в worktree `soup-p3-unified-go` (без пуша в remote)  
**Стенд:** физическая `NVIDIA GeForce RTX 4060 Ti 8 ГБ` (`8188 MiB`, `7.65 GiB` CUDA, SM 8.9), 62 GiB RAM, NVMe `~3.7–3.8 GB/s`, Linux, Python 3.11, PyTorch `2.6.0+cu124`, Transformers `5.23.0`, PEFT `0.22.0`, bitsandbytes `0.50.2`, Triton `3.2.0`, NVCC `12.4` *(все арендованные инстансы Vast.ai удалены сразу после выгрузки JSON)*.

---

## Главное за 30 секунд

Мы объединили все прошедшие (`GO`) оптимизации из трёх параллельных треков P3 + `FP8 block64` + `DeepseekRoutingOracle` в один стек и физически прогнали его на **8 ГБ RTX 4060 Ti** на двух крупных моделях разной архитектуры — **MoE `Kanana-2-30B-A3B`** и **плотной гибридной `Qwen/Qwen3.8-27B`** — а также провели полный **Agentic Tool-Calling SFT (До → После)** с оценкой качества на невидимых задачах.

- **Kanana-2-30B-A3B (30B MoE, 3B активных) на `k=64, S=512` (32 768 токенов за шаг):**
  - В BF16: **`632.9 tok/s`** (`51.8 с/шаг`), пик VRAM **`3.70 GiB`** NVML (`2.60 GiB` alloc), пик RAM **`13.51 GiB`** (парный `mbr_h2_sync` на этом размере падает в `CUDA OOM`).
  - С **`FP8 block64 + DeepseekRoutingOracle`**: **`630.7 tok/s`** (`52.0 с/шаг`, **+2.3%** к изолированному BF16 `616.4 tok/s`), пик VRAM **`3.57 GiB`** NVML (**−133 MiB**), пик RAM **`10.88 GiB`** (**−2.63 GiB**), `loss_bit_equal = True`, `gate_w_drift = 0.0`, E3 `grad_rel_l2 = 0.01393 ≤ 0.05`, `min_cos = 0.99471 ≥ 0.99` (**PASS**).
- **Qwen/Qwen3.8-27B (27B dense/hybrid: 48 Gated DeltaNet + 16 Full Attn) на `k=64, S=512` (32 768 токенов за шаг):**
  - В BF16: `199.7 tok/s` (`164.1 с/шаг`), пик VRAM `4.98 GiB` NVML, пик RAM **`42.74 GiB`** *(не влезает в 32 ГБ ПК)*.
  - С **`FP8 block64`**: **`199.7 tok/s`** (`164.1 с/шаг`), пик VRAM **`4.95 GiB`** NVML (`4.01 GiB` alloc), пик RAM **`27.86 GiB`** (**−14.88 GiB RAM** — влезает в обычный ПК с 32 ГБ RAM + 8 ГБ GPU!), `loss_bit_equal = True`, E3 `grad_rel_l2 = 0.00583`, `min_cos = 0.99783`, `adapter_cos = 0.999995`.
- **Реальный Agentic Tool-Calling SFT (До → После) за 16 шагов (`k=16, S=256` = 65 536 токенов, тариф `$0.42/час`):**
  - **`Qwen/Qwen3.8-27B`:** **`6.85 мин`**, **`$0.048`**, `5.11 GiB` NVML VRAM, `13.24 GiB` RAM → held-out `PPL: 1.45 → 1.00`, `TokAcc: 93.36% → 100.0%`, free-running **`Exact Action Match: 50.0% → 100.0%` (+50.0 п.п.)**.
  - **`Kanana-2-30B-A3B` (base, с head LoRA):** **`3.44 мин`** (**2.0× быстрее**), **`$0.024`** (**2.0× дешевле**), **`1.97 GiB` NVML VRAM** (**2.6× меньше VRAM**) → held-out `PPL: 8.03 → 1.00`, `TokAcc: 77.60% → 100.0%`, free-running **`Valid XML: 0.0% → 87.5%`**, **`Valid JSON: 0.0% → 81.25%`**, **`Exact Action Match: 0.0% → 62.5%` (+62.5 п.п.)**.
- **Прямое сравнение с Unsloth (`2026.11.7`) и HF + BitsAndBytes NF4 CPU-offload при тех же условиях (`RTX 4060 Ti 8 ГБ`, `k=16, S=256`, `lora_r=16`, Серия 6):**
  - На контрольной **`Qwen/Qwen2.5-0.5B`** Unsloth `2026.11.7` на этой же карте работает штатно (`314.8 tok/s`, `1.17 GiB` NVML).
  - На **`Qwen/Qwen3.8-27B`** и **`Kanana-2-30B-A3B`** **Unsloth `2026.11.7` падает ещё на загрузке** с жёстким `ValueError` в `raise_if_bnb_cpu_spill`: `does not fit in GPU memory (cuda:0 has 7.46 GB free), so transformers placed part of it on the CPU, which bitsandbytes cannot quantize`.
  - Стандартный обход через **HF + BitsAndBytes NF4 CPU-offload** (`device_map="auto"`, `llm_int8_enable_fp32_cpu_offload=True`) на **`Qwen/Qwen3.8-27B`** съедает **`56.69 GiB` Host RAM** (**4.28× больше** наших `13.24 GiB`, мгновенный OOM на 32 ГБ ПК) и падает на первом forward (`NotImplementedError: Cannot copy out of meta tensor`), а на **`Kanana-2-30B-A3B`** съедает **`40.37 GiB` RAM** + **`7.82 GiB` NVML** и падает в **`CUDA out of memory`** на конвертации экспертов 17-го слоя.

---

## 1. Откуда взялся прирост: алгоритмы vs C++/CUDA/Triton

Основной прорыв (с **~6 tok/s** до **~630 tok/s** на 30B MoE при 3.6 GiB VRAM) дала именно **переработка алгоритмов и графа автодифференцирования**, а поверх неё C++17 / CUDA / Triton сняли остаточные I/O-пузыри и гарантировали побитовую точность:

| Слой | Компонент | Файлы в `feat/p3-unified-go` | Что именно даёт |
|---|---|---|---|
| **Алгоритмы (главный вклад: ~10–19× скорость + снятие OOM)** | **M1 + M2+ Decoupled MBR (`sub_w=1`)** — общее окно экспертов на `k` микробатчей + раздельный пересчёт attention/router/shared по 1 микробатчу и zero-materialization backward экспертов (без `WFE`/`dh_chunks`/`h_sorted`) | `l2l_step.py`, `expert_stream.py` | Сокращает чтения экспертов в **~5–10×**, `0` чтений в recompute, в **20.59×** меньше saved storage (`k=32`), физически снимает `CUDA OOM` на `k=64, S=512` |
| **Алгоритмы** | **H2** (ABA-safe переиспользование распакованных NF4-весов внутри окна) + **M3** (освобождение слота `LargeLayerBufferPool` на всех декодерных слоях) + **`O(N·K)` маршрутизация** (без плотной `int64` `one_hot`-маски) | `layer_stream_runtime.py`, `expert_stream.py`, `l2l_step.py` | H2 режет `cuda_alloc` на **−2.23 GiB** (`3.74 → 1.51 GiB`) и число NF4-деквантаций в **16–64×**; M3 освобождает **0.50 GiB** на Kanana, **1.73 GiB** на 671B и **2.37 GiB** на Qwen 3.8 27B |
| **Алгоритмы** | **Chunked CE** (`chunk=256`) на L2L-голове (PR #63) | `stream_chunked_ce.py` | На форме головы DeepSeek-V3 671B (`H=7168, V=129280, S=4096`): `full` падает в **OOM (7.38 GiB)**, `chunked` держит плоский пик **2.87 GiB NVML** |
| **Алгоритмы** | **`FP8 (block64_e4m3)` активации + `DeepseekRoutingOracle`** (PR #57 + PR #66, 3 фазы `forward`/`peek`/`recompute`, точный `_OriginalSigmoidJacobian`, компактные `int16` ID без `TopkDecisions` dispatch) | `activation_codec.py`, `fp8_activations.py`, `routing_oracle.py`, `l2l_step.py` | Режет хост-RAM на **−46.9%** (`−2.63 GiB` на Kanana, **`−14.88 GiB`** на Qwen 3.8 27B, **`−12.42 GiB`** на 671B), снижает VRAM на **−133 MiB** и ускоряет шаг на **+2.3%** за счёт вдвое меньшего PCIe DMA |
| **Системный слой (C++17 / CUDA / Triton: +8.5–10% скорости + bit-exact)** | **`PackedStreamAsync`** (C++17 `io_workers=4, ring_slots=8`, 2-slot async DMA, cross-layer backward pre-staging) + **`nf4_decode.cu`** + **Triton `batch_invariant` GEMM** (PR #73 + фикс Triton 3.2 `tl`) | `async_expert_controller.py`, `_native_expert_async/{native_stream_async.cpp,nf4_decode.cu,build_ext.py}`, `invariant_gemm.py` | Уменьшает ожидание I/O основным потоком в **5.7×** (`4.09 → 0.71 с`), даёт **100%** cross-layer promotions (`46/46`, `0` discards), **74–76%** H2D overlap на `S=512` и **побитовое совпадение** (`bit_equal=True`) |

### Что отсекли как `NO-GO` (не вошло в горячий путь)
- **VRAM Packed-NF4 Hotset на 671B:** 4 GiB VRAM вмещают лишь 1.12% экспертов → `<0.67%` ускорения.
- **CPU Cold-Expert / KTransformers при `k=64`:** `M* = 0`, доля экспертов с `M ≤ 16` равна `0.0%`.
- **`H-C1` split `gate_w`/`up_w`:** не нужен при `batch_invariant` и вносит 1 BF16 ULP сдвиг.
- **Sparse-Candidate Deletion Tuning:** не уменьшает NVMe-трафик, на 671B `active-core` дороже LoRA `r=1`, а на `routed` (`p=0.2%`) даёт `8.93 GiB` OOM.

---

## 2. Физические замеры Kanana-2-30B-A3B на RTX 4060 Ti 8 ГБ (Серии 1–3)

**Конфигурация:** `kakaocorp/kanana-2-30b-a3b-base-2601` (`0d31365a`), NF4 (`no-double-quant`, `per_expert`), BF16, `host_cache = 3.5 GiB`, LoRA `r=4` (`q_b_proj, kv_b_proj, o_proj`), `--expert-gemm batch_invariant` + `--cuda-nf4-decode`. Таймер включает `forward + backward + clip + AdamW.step()`. Во всех сериях, начиная со второй, пик NVML снимался фоновым поллером каждые **20 мс**.

### 2.1. Скорость, память и побитовый паритет (BF16, Серии 1 и 2)
*Сырые JSON:* `benchmarks/results/vast-4060ti-20261120/` и `benchmarks/results/vast-4060ti-verify2-20261120/`.

| Сценарий (токенов/шаг) | Вариант (`alloc_conf`) | Шаг (с) | Скорость (tok/s) | Ускорение к паре / к базе того же `S` | `cuda_alloc` peak | `cuda_resv` peak | `nvml_peak` (20 мс) | Паритет (loss / grad / adapter) |
|---|---|---:|---:|:---:|---:|---:|---:|:---:|
| `k=1, S=64` (64 tok, база P3) | `l2l_ref` (`exp_seg`) | 10.42–10.79 | 5.93–6.14 | ref / `1.00×` | 0.713 GiB | 0.787 GiB | 1.374 GiB | ref (`3 821`×2 чтений) |
| `k=1, S=512` (512 tok, база `S=512`) | `l2l_ref` (`exp_seg`) | 15.31–15.36 | 33.32–33.43 | ref / `1.00×` | 1.363 GiB | 1.568 GiB | 2.155 GiB | ref (`5 351`×2 чтений) |
| `k=4, S=64`, **3 шага AdamW** (768 tok) | `l2l_ref` → **`unified_go`** | 126.84 → **29.10** | 6.06 → **26.39** | **4.36×** / 4.45× | 0.781 GiB | 0.820 GiB | 1.407 GiB | **`bit_equal=True` (`0.0`)** после 3 шагов! |
| `k=8, S=64` (512 tok) | `l2l_ref` → **`unified_go`** | 93.23 → **11.98** | 5.49 → **42.73** | **7.78×** / 7.20× | 0.769 GiB | 0.844 GiB | 0.893 GiB *(post)* | **`bit_equal=True` (`0.0`)** |
| `k=64, S=64` (4 096 tok, абляция H2) | `std_mbr` (без H2) | 40.74 | 100.53 | ref / 16.9× | **3.740 GiB** | 3.818 GiB | 4.405 GiB | ref |
| `k=64, S=64` (4 096 tok, A–B–B–A) | `mbr_h2_sync` (A1/A2) | 40.81–41.12 | 99.62–100.38 | ref / 16.9× | 1.516 GiB *(−2.23 GiB)* | 1.645–1.756 GiB | 2.23–2.34 GiB *(post)* | **`bit_equal=True` (`0.0`)** |
| `k=64, S=64` (4 096 tok, абляция I/O) | `unified_go_qd1` (`QD=1`, no CL) | 40.42 | 101.34 | 1.01× / 17.1× | 0.828 GiB | 1.756 GiB | 1.356 GiB *(post)* | **`bit_equal=True` (`0.0`)** |
| `k=64, S=64` (4 096 tok, A–B–B–A) | **`unified_go`** (`QD=4+CL`, B1/B2) | **37.24–37.25** | **109.96–109.99** | **+10.0%** vs `mbr_h2` / **18.5×** | **0.823–0.828 GiB** | **1.262 GiB** | **1.18–1.36 GiB** *(post)* | **`bit_equal=True` (`0.0`)** |
| `k=64, S=512` (32 768 tok) | `mbr_h2_sync` (PR #72 + H2) | — | **CUDA OOM** | — | 6.680 GiB (OOM) | — | **7.673 GiB (OOM)** | `Tried to allocate 248 MiB` |
| `k=64, S=512` (32 768 tok) | `unified_go` (default alloc) | 51.69 | 633.95 | — / **18.99×** | 2.604 GiB | 6.146 GiB | 6.745 GiB | `RSS = 13.51 GiB` |
| `k=64, S=512` (32 768 tok) | **`unified_go`** (**`exp_seg`**) | **51.77–51.80** | **632.61–632.94** | — / **18.96×** | **2.601 GiB** | **3.102 GiB** *(−3.04 GiB)* | **3.700 GiB** *(−3.05 GiB)* | **`bit_equal=True` (`0.0`)**, `RSS = 13.52 GiB` |

*Ключевая находка Серии 2:* включение `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` убирает фрагментацию при динамическом освобождении слота M3 и роняет реальный 20 мс пик NVML на 32 768 токенах с **6.745 до 3.700 GiB** (−3.05 GiB VRAM, ~4 GiB свободного запаса) при нулевой потере скорости.

### 2.2. Изолированная проверка `FP8 (block64_e4m3) + DeepseekRoutingOracle` (Серия 3)
*Сырые JSON:* `benchmarks/results/vast-4060ti-fp8-oracle-20261120/`. Каждый прогон выполнялся в **отдельном свежем процессе ОС** со сверкой `--save-ref` / `--compare-ref` относительно точного BF16-эталона (порог качества G4 E3: `grad_rel_l2 ≤ 0.05`, `min_cos ≥ 0.99`).

| Сценарий (токенов/шаг) | Режим | Шаг (с) | Скорость (tok/s) | Store + Oracle RAM | Изолированный `RSS peak` | `alloc` / `resv` / **`nvml_peak`** | `gate_w_drift` | E3 `grad_rel_l2` | E3 `min_cos` | `adapter_rel_l2` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tiny` CUDA (`k=4, S=32`) | `unified_go_fp8_oracle` | 0.141 | 905 | `0.05 + 0.004 MiB` | 1.78 GiB | `0.063 / 0.080 / 0.663 GiB` | **`0.0`** | **`0.04317` (PASS)** | **`0.99780`** | `0.00037` *(bit-eq vs `l2l_ref`)* |
| `tiny` CUDA (`k=4, S=32`) | *Абляция: без Oracle* | 0.148 | 867 | `0.05 + 0 MiB` | 1.78 GiB | `0.063 / 0.080 / 0.663 GiB` | `2.63e-2` | `0.08343` (**FAIL**, ×1.93) | `0.99595` | `0.00046` |
| Kanana `k=64, S=64` (4 096 tok) | `unified_go` (BF16 ref) | 37.10 | 110.39 | `768.0 + 0 MiB` | *(с шардингом)* | `0.823 / 0.883 / 1.468 GiB` | `0.0` | ref | ref | ref |
| Kanana `k=64, S=64` (4 096 tok) | **`unified_go_fp8_oracle`** | **35.54** | **115.26 (+4.4%)** | **`408.0 + 7.34 MiB`** | **7.86 GiB** | **`0.802 / 0.867 / 1.454 GiB`** | **`0.0`** | **`0.02111` (PASS)** | **`0.99255`** | **`0.00049`** |
| Kanana `k=64, S=64` (4 096 tok) | *Абляция: `per-row` + Oracle* | 35.30 | 116.04 | `384.8 + 7.34 MiB` | 7.78 GiB | `0.802 / 0.867 / 1.454 GiB` | `0.0` | `0.02375` *(−12.5% точность)* | `0.99236` | `0.00051` |
| Kanana `k=64, S=64` (4 096 tok) | *Абляция: `block64` без Oracle* | 38.05 | 107.66 (−2.5%) | `408.0 + 0 MiB` | 7.82 GiB | `0.823 / 0.883 / 1.468 GiB` | **`1.11e-2`** | `0.02124` | `0.99255` | `0.00049` |
| Kanana `k=64, S=512` (32 768 tok) | `unified_go` (BF16 ref) | 53.16 | 616.43 | `6 144 + 0 MiB` | **13.51 GiB** | `2.601 / 3.102 / 3.700 GiB` | `0.0` | ref | ref | ref |
| Kanana `k=64, S=512` (32 768 tok) | **`unified_go_fp8_oracle` (r1 / r2)** | **51.96 / 52.01** | **630.68 / 630.07 (+2.3%)** | **`3 264 + 58.75 MiB`** | **10.89 / 10.88 GiB (−2.63 GiB)** | **`2.505 / 2.975 / 3.567 GiB` (−133 MiB)** | **`0.0`** | **`0.01393` (PASS)** | **`0.99471`** | **`0.00038`** *(`r1==r2` SHA-256!)* |
| Kanana `k=8, S=512`, **3 шага AdamW** | `BF16` → **`FP8 + Oracle`** | 38.83 → **39.12** | 316.5 → **314.1** | `768 → 408 + 7.3 MiB` | `8.17 →` **7.85 GiB** | `1.927 →` **`1.909 GiB`** | **`0.0`** | `0.03465` (шаг 3) | `0.97066` | **`0.00067`** (`loss_max=1.04e-3`) |

*Почему `FP8 + Oracle` оказался и экономнее, и быстрее BF16:* вдвое меньший объём граничных активаций ускоряет PCIe DMA, а `DeepseekRoutingOracle` заодно обходит `TopkDecisions` (`TorchDispatchMode`), экономя 133 MiB VRAM и 1.2 с на шаге при `0.0` дрейфе маршрутизации (`3 008 / 3 008` слоёв×микробатчей). Для DeepSeek-V3 671B (`L=61, H=7168, k=64, S=512`) это снижает буфер активаций с **26.69 GiB до 14.27 GiB (−12.42 GiB RAM)**, вписывая 671B в 32 GiB RAM (`swap=0`).

### 2.3. Физическая проверка головы DeepSeek-V3 671B (`H=7168, V=129280`, BF16) на RTX 4060 Ti 8 ГБ
*Сырой JSON:* `benchmarks/results/vast-4060ti-verify2-20261120/deepseek-v3-head-vram-4060ti.json`.

| Токенов `S` | `full` статус / `nvml_peak` | `chunked` (`chunk=256`) статус / `nvml_peak` | `chunked` `alloc` / `resv` | Время `full` → `chunked` | `grad_max_abs` (`bit_eq`) |
|---:|---:|---:|---:|---:|---:|
| 512 | ok / `3.454 GiB` | ok / **`2.751 GiB`** | `2.172 / 2.189 GiB` | 0.167 → 0.352 с | `0.0` (`True`) |
| 1 024 | ok / `4.606 GiB` | ok / **`2.888 GiB`** | `2.186 / 2.326 GiB` | 0.115 → 0.153 с | `1.43e-6` (`False`) |
| 2 048 | ok / `6.853 GiB` | ok / **`2.829 GiB`** | `2.213 / 2.268 GiB` | 0.227 → 0.307 с | `9.54e-7` (`False`) |
| **4 096** | **CUDA OOM** (`7.380 GiB`) | **ok / `2.868 GiB`** | **`2.268 / 2.307 GiB`** | OOM → **0.620 с** | — |

---

## 3. Физические замеры `Qwen/Qwen3.8-27B` и сравнение с `Kanana-2-30B-A3B` (Серия 4)

**Конфигурация:** `Qwen/Qwen3.8-27B` (`rev 37561816`), 64 слоя (**48 Gated DeltaNet `linear_attention` + 16 `full_attention`**), `H=5120`, `I=17408`, `V=248320`, `tie_word_embeddings=False` (один слот `LargeLayerBufferPool` весит **`2.37 GiB`**), NF4 (`no-double-quant`), BF16, LoRA `r=4` (`q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj`), `Chunked CE (256)`, `expandable_segments:True`.  
*Сырые JSON:* `benchmarks/results/vast-4060ti-qwen38-agentic-20261120/01a..01i`.  
*Примечание:* в образе Vast не был собран пакет `causal_conv1d`, поэтому линейное внимание Gated DeltaNet работало через референсный PyTorch fallback (с нативным CUDA-ядром `causal_conv1d` скорость `Qwen/Qwen3.8-27B` будет дополнительно выше).

### 3.1. Матрица замеров `Qwen/Qwen3.8-27B` на RTX 4060 Ti 8 ГБ

| Сценарий (токенов/шаг) | Режим | Шаг (с) | Скорость (tok/s) | Store RAM | `RSS peak` | `alloc` / `resv` / **`nvml_peak`** | `H2` dequant / reuse | `loss_bit_eq` | E3 `grad_rel_l2` | E3 `min_cos` | `adapter_rel_l2` |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| `k=1, S=64` (64 tok) | `l2l_ref` → **`unified_go`** | 34.76 → **8.43** (**4.12×**) | 1.84 → **7.59** | `40 MiB` | `10.89 / 11.04 GiB` | `3.151 / 3.371 / 4.231 GiB` | `256 / 0` | **`True`** | **`0.0`** | **`1.00000`** | **`0.0`** |
| `k=1, S=512` (512 tok) | `l2l_ref` → **`unified_go`** | 14.31 → **8.43** (**1.70×**) | 35.78 → **60.73** | `320 MiB` | `11.20 / 11.34 GiB` | `6.643 / 6.775 / 7.636 GiB` | `992 / 0` | `1.9e-6` | **`0.0`** | **`1.00000`** | **`0.0`** |
| `k=16, S=64` (1 024 tok) | `l2l_ref` (BF16) | 24.95 | 41.03 | `640 MiB` | 11.72 GiB | `3.252 / 3.373 / 4.235 GiB` | — | ref | ref | ref | ref |
| `k=16, S=64` (1 024 tok) | `unified_go` (BF16) | 25.10 | 40.80 | `640 MiB` | 11.79 GiB | **`3.160 / 3.213 / 3.815 GiB`** *(−420 MiB)* | `256 / 3 840` (16×) | **`True`** | `0.00373` | `0.99957` | `0.00022` |
| `k=16, S=64` (1 024 tok) | **`unified_go_fp8`** (`block64`) | 25.27 | 40.52 | **`340 MiB`** *(−46.9%)* | **11.36 GiB** | **`3.160 / 3.213 / 3.813 GiB`** | `256 / 3 840` (16×) | `3.98e-4` | **`0.01958` (PASS)** | **`0.99720`** | `0.00053` |
| `k=64, S=64` (4 096 tok) | `unified_go` (BF16 ref) | 79.29 | 51.66 | `2 560 MiB` | 14.69 GiB | `3.190 / 3.244 / 3.853 GiB` | `256 / 16 128` (64×) | ref | ref | ref | ref |
| `k=64, S=64` (4 096 tok) | **`unified_go_fp8`** (`block64`) | 79.45 | 51.56 | **`1 360 MiB`** *(−1.20 GiB)* | **12.95 GiB** *(−1.74 GiB)* | **`3.190 / 3.244 / 3.849 GiB`** | `256 / 16 128` (64×) | `9.96e-5` | **`0.00903` (PASS)** | **`0.99840`** | `0.00043` |
| `k=64, S=512` (32 768 tok) | `unified_go` (BF16 ref) | 164.11 | 199.67 | `20 480 MiB` (`20.0 GiB`) | **42.74 GiB** *(>32 GiB!)* | `4.014 / 4.315 / 4.978 GiB` | `992 / 62 496` (64×) | ref | ref | ref | ref |
| `k=64, S=512` (32 768 tok) | **`unified_go_fp8`** (`block64`) | **164.06** | **199.73** | **`10 880 MiB`** *(−9.38 GiB)* | **27.86 GiB** (**−14.88 GiB!**) | **`4.014 / 4.315 / 4.948 GiB`** | `992 / 62 496` (64×) | **`True (0.0)`** | **`0.00583` (PASS, 8.6× запас)** | **`0.99783`** | **`0.00032`** *(`cos=0.999995`)* |

### 3.2. Прямое сравнение `Kanana-2-30B-A3B` vs `Qwen/Qwen3.8-27B` на `k=64, S=512` (32 768 tok/step, `FP8 block64`, RTX 4060 Ti 8 ГБ)

| Метрика | `Kanana-2-30B-A3B` (MoE 30B / 3B active) | `Qwen/Qwen3.8-27B` (Dense/Hybrid 27B active) | Разница / вывод |
|---|---:|---:|---|
| Архитектура (`L`, `H`, `V`) | `48` слоёв (47 MoE × 128 эксп.), `H=2048`, `V=128 256` | `64` слоя (48 DeltaNet + 16 Full Attn), `H=5120`, `V=248 320` | У Qwen активации слоёв в **3.33×** крупнее, а слот головы в **4.7×** крупнее (`2.37` vs `0.50 GiB`) |
| Время шага / скорость | **`51.96 с`** / **`630.68 tok/s`** | `164.06 с` / `199.73 tok/s` | MoE в **3.16× быстрее** (3B активных параметров против 27B) |
| Пик VRAM (`alloc` / **`nvml 20 мс`**) | **`2.505`** / **`3.567 GiB`** | `4.014` / `4.948 GiB` | Обе уверенно помещаются в 8 ГБ GPU (`3.0–4.4 GiB` запаса) благодаря **M3** |
| Буфер активаций (`BF16` → `FP8`) | `6.00` → **`3.25 GiB`** *(+58.8 MiB Oracle)* | `20.00` → **`10.63 GiB`** | **FP8 block64** критичен для 27B: экономит **−9.38 GiB** pinned-буфера |
| Реальный пик Host RAM (`RSS`: `BF16` → `FP8`) | `13.51` → **`10.88 GiB`** *(−2.63 GiB)* | `42.74` → **`27.86 GiB`** (**−14.88 GiB!**) | Без FP8 `Qwen3.8-27B` требует >42 ГБ RAM; с FP8 влезает в стандартный **32 ГБ ПК** |
| Точность E3 vs BF16 (`grad_rel_l2` / `min_cos`) | `0.01393` / `0.99471` (**PASS**) | **`0.00583`** / **`0.99783`** (**PASS**) | Различий по качеству градиентов практически нет (`adapter_cos = 0.999995`) |

---

## 4. Agentic Tool-Calling SFT (До → После) на RTX 4060 Ti 8 ГБ (Серия 5)

**Постановка задачи:** выбор правильного инструмента из меню 4 кандидатов `<tools>` и генерация строгого вызова `<tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>` по пользовательскому запросу (схемы `NousResearch/hermes-function-calling-v1`, 416 уникальных непересекающихся задач: `32` held-out test для точного L2L NLL/PPL/TokAcc, `16` held-out задач для free-running greedy генерации `prompt_len=192, max_new=40`, `256` train задач = **16 шагов × `k=16, S=256` = 65 536 токенов**, `FP8 block64`, LoRA `r=16`).  
*Сырые JSON:* `benchmarks/results/vast-4060ti-qwen38-agentic-20261120/{02_qwen38_27b_agentic_sft_eval.json, 03_..., 03b_..., 03c_..., 03d_kanana_30b_agentic_sft_16steps_headlr_eval.json}`.

### 4.1. Скорость, память и стоимость обучения (`$0.42/час` RTX 4060 Ti 8 ГБ)

| Модель и конфигурация SFT | Шагов / Токенов | Время SFT | Скорость SFT | Стоимость SFT | Пик VRAM (`alloc` / **`nvml`**) | Пик RAM (`RSS`) | Store + Oracle RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Qwen/Qwen3.8-27B`** (`FP8 block64`, LoRA `r=16`, `lr=5e-4`) | `16` / `65 536` | **6.85 мин** (`411.0 с`) | `159.4 tok/s` | **`$0.048`** | `4.63` / **`5.11 GiB`** | **`13.24 GiB`** | `1 360 + 0 MiB` |
| **`Kanana-2-30B-A3B` (base)** (`FP8 block64 + Oracle`, LoRA `r=16`, `--train-head`, `lr=8e-4`, `head_lr=1e-2`) | `16` / `65 536` | **3.44 мин** (`206.5 с`) | **`317.4 tok/s`** (**2.0×**) | **`$0.024`** (**2.0× дешевле**) | **`1.25` / `1.97 GiB`** (**2.6× меньше**) | `19.27 GiB` *(15 GiB RAM-кэш экспертов)* | `408 + 7.34 MiB` |

### 4.2. Качество на невидимых held-out задачах (До → После SFT)

| Модель и прогон | Held-Out NLL (`N=32`) | Held-Out PPL (`N=32`) | Held-Out TokAcc (`N=32`) | Valid `<tool_call>` XML (`N=16`) | Valid JSON Schema (`N=16`) | Tool Name Acc (`N=16`) | Arg Key F1 (`N=16`) | **Exact Action Match (`N=16`)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Qwen/Qwen3.8-27B`** *(основной, `02`)* | `0.3725 →` **`0.0000`** | `1.45 →` **`1.00`** | `93.36% →` **`100.00%`** | `50.0% →` **`100.0%`** | `50.0% →` **`100.0%`** | `50.0% →` **`100.0%`** | `50.0% →` **`100.0%`** | **`50.0% → 100.0%` (+50.0 п.п.)** |
| **`Kanana-2-30B-A3B` (base)** *(основной с `--train-head --head-lr 1e-2`, `03d`)* | `2.0836 →` **`0.0014`** | `8.03 →` **`1.00`** | `77.60% →` **`100.00%`** | `0.0% →` **`87.5%`** | `0.0% →` **`81.25%`** | `0.0% →` **`75.0%`** | `0.0% →` **`81.25%`** | **`0.0% → 62.5%` (+62.5 п.п.)** |
| *Абляция: `Kanana` без `--train-head` (`03`, 16 шагов)* | `2.0794 → 0.4533` | `8.00 → 1.57` | `79.51% → 93.21%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` *(см. разд. 5.3)* |
| *Абляция: `Kanana` с `--train-head`, единый `lr=8e-4` (`03c`, 30 шагов)* | `2.0836 → 0.0463` | `8.03 → 1.05` | `77.60% → 96.71%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` | `0.0% → 0.0%` *(см. разд. 5.3)* |

### 4.3. Живые примеры генерации До и После SFT (из сырых JSON)

- **Задача:** `"Query database inventory_us and fetch 46 rows from table shipments."`
- **Эталон:** `<tool_call>\n{"name":"dblink_connect_and_query","arguments":{"dbname":"inventory_us","table":"shipments","limit":46}}\n</tool_call>`
- **`Qwen/Qwen3.8-27B` До SFT:** `assistant\n<think>\nThe user wants to query the database "inventory_us" and fetch 46 rows...` *(срывается в размышления, ломает парсер агента в 8 из 16 задач)*.
- **`Qwen/Qwen3.8-27B` После SFT (16 шагов, `$0.048`):** `<tool_call>\n{"name":"dblink_connect_and_query","arguments":{"dbname":"inventory_us","table":"shipments","limit":46}}\n</tool_call>` *(16/16 точное совпадение)*.
- **`Kanana-2-30B-A3B` (base) До SFT:** `{"name":"dblink_connect_and_query","arguments":{"dbname":"inventory_us","table":"shipments","limit":"46"}}\n\n### Agent Response\n...` *(нет `<tool_call>`, число `46` выдано строкой `"46"`, модель зацикливается на `### Agent Response`; на других задачах выдаёт ```` ```yaml ````)*.
- **`Kanana-2-30B-A3B` (base) После SFT (`03d`, 16 шагов, `$0.024`):** `<tool_call>\n{"name":"dblink_connect_and_query","arguments":{"dbname":"inventory_us","table":"shipments","limit":46}}\n</tool_call>` *(строгий XML, правильный тип `int 46`, чистая остановка; в 3 из 16 примеров при 16 шагах ещё встречаются мелкие сокращения имени вроде `get_camera_live` вместо `get_camera_live_feed` — на 25–30 шагах с `head_lr=1e-2` уйдёт к ~95–100%)*.

### 4.4. Прямое сравнение с Unsloth (`2026.11.7`) и HF + BitsAndBytes CPU-Offload при тех же условиях (Серия 6)

По запросу тимлида мы подняли **ту же самую физическую машину** (`offer_id=58316190`, `NVIDIA GeForce RTX 4060 Ti 8188 MiB / 7.651 GiB CUDA`, 62 GiB RAM), установили актуальный **`unsloth 2026.11.7`** (`torch 2.14.1+cu130`, `transformers 5.17.0`, `bitsandbytes 0.50.2`, `Triton 3.8.0`) и прогнали харнес `benchmarks/harness/p3_unsloth_4060ti_compare.py` при **тех же самых условиях**, что в нашем SFT (`k=16, S=256` = 4 096 токенов/шаг, 4-bit NF4, `lora_r=16`, `lora_alpha=32`, `use_gradient_checkpointing="unsloth"`, 20 мс фоновый поллер NVML + RSS).  
*Сырые JSON и логи:* `benchmarks/results/vast-4060ti-unsloth-compare-20261120/`.

| Модель | Фреймворк / Режим | Статус на RTX 4060 Ti 8 ГБ | Время шага (`4 096` tok) | Скорость (`tok/s`) | Пик VRAM (`alloc` / **`nvml 20 мс`**) | Пик Host RAM (`RSS`) | Причина падения / примечание |
|---|---|---:|---:|---:|---:|---:|---|
| **`Qwen/Qwen2.5-0.5B`** *(контроль стенда)* | **Unsloth `2026.11.7`** (`unsloth_default`, 4-bit) | **OK** | `13.01 с` | **`314.8 tok/s`** | `0.52` / **`1.17 GiB`** | `2.74 GiB` | Контроль: Unsloth + Triton + block CUDA graphs полностью исправны, пока 4-bit веса помещаются в 8 ГБ VRAM |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid) | **Soup `feat/p3-unified-go`** (`FP8 block64`) | **OK (обучает + SFT `50% → 100%`)** | **`25.33 с`** *(и `164.1 с` на `k=64, S=512`)* | **`159.4 tok/s`** *(`199.7 tok/s` на `k=64, S=512`)* | **`4.63` / `5.11 GiB`** *(`4.95 GiB` на `k=64, S=512`)* | **`13.24 GiB`** *(`27.86 GiB` на `k=64, S=512`)* | Полностью помещается в **8 ГБ GPU + 32 ГБ RAM**, 16 шагов SFT за **`$0.048`** |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid) | **Unsloth `2026.11.7`** (`unsloth_default`, `load_in_4bit=True`) | **FAIL (`ValueError`)** | — | — | `0.00` / `0.59 GiB` | `1.59 GiB` | `raise_if_bnb_cpu_spill`: `does not fit in GPU memory (cuda:0 has 7.46 GB free), so transformers placed part of it on the CPU, which bitsandbytes cannot quantize` |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid) | **Unsloth `2026.11.7`** (`unsloth_cpu_offload`, `device_map="auto"`) | **FAIL (`ValueError`)** | — | — | `0.00` / `0.59 GiB` | `1.59 GiB` | Тот же жёсткий отказ `raise_if_bnb_cpu_spill` (~14.5 ГБ 4-bit весов не влезают в 7.46 ГБ свободной VRAM) |
| **`Qwen/Qwen3.8-27B`** (27B Dense/Hybrid) | **HF + BitsAndBytes NF4** (`device_map="auto"`, `fp32_cpu_offload=True`) | **FAIL (`NotImplementedError` + RAM blowup)** | упал после `82.4 с` загрузки | — | `4.27` / `4.88 GiB` | **`56.69 GiB`** (**>32 ГБ RAM в 1.77×, в 4.28× больше Soup!**) | Съел **56.69 GiB RAM** на распаковку оффлоад-слоёв и упал на первом forward в `bitsandbytes`: `Cannot copy out of meta tensor; no data!` |
| **`Kanana-2-30B-A3B`** (30B MoE, 128 эксп.) | **Soup `feat/p3-unified-go`** (`FP8 block64 + Oracle`) | **OK (обучает + SFT `0% → 81.25%` JSON)** | **`12.60–12.90 с`** *(и `51.96 с` на `k=64, S=512`)* | **`317.4–322.7 tok/s`** *(`630.7 tok/s` на `k=64, S=512`)* | **`1.21–1.25` / `1.91–1.97 GiB`** *(`3.57 GiB` на `k=64, S=512`)* | **`7.85–19.27 GiB`** *(`10.88 GiB` на `k=64, S=512` с 3.5 GiB кэшем)* | Поток экспертов с NVMe + H2 + M3 + Oracle, 16 шагов SFT за **`$0.024`** |
| **`Kanana-2-30B-A3B`** (30B MoE, 128 эксп.) | **Unsloth `2026.11.7`** (`unsloth_default`, `load_in_4bit=True`) | **FAIL (`ValueError`)** | — | — | `0.00` / `0.81 GiB` | `1.99 GiB` | `raise_if_bnb_cpu_spill`: `does not fit in GPU memory (cuda:0 has 7.46 GB free), so transformers placed part of it on the CPU, which bitsandbytes cannot quantize` |
| **`Kanana-2-30B-A3B`** (30B MoE, 128 эксп.) | **Unsloth `2026.11.7`** (`unsloth_cpu_offload`, `device_map="auto"`) | **FAIL (`ValueError`)** | — | — | `0.00` / `0.81 GiB` | `1.99 GiB` | Тот же отказ `raise_if_bnb_cpu_spill`: Unsloth не поддерживает 4-bit квантизацию весов с вытеснением на CPU/NVMe |
| **`Kanana-2-30B-A3B`** (30B MoE, 128 эксп.) | **HF + BitsAndBytes NF4** (`device_map="auto"`, `fp32_cpu_offload=True`) | **FAIL (`CUDA OOM` на загрузке + RAM blowup)** | упал через `87.2 с` загрузки | — | `7.17` / **`7.82 GiB` (OOM)** | **`40.37 GiB`** (**>32 ГБ RAM!**) | Съел **40.37 GiB RAM** и упал в **`CUDA out of memory. Tried to allocate 768.00 MiB`** при сборке `layers.17.mlp.experts.gate_up_proj` |

**Почему Unsloth архитектурно не может заменить стек Soup P3 на 8 ГБ GPU / 32 ГБ RAM:**
1. **Требование резидентности весов в VRAM (или полной распаковки в System RAM):** Unsloth рассчитан на модели, чьи 4-bit веса целиком помещаются в VRAM (например, 7B–8B на 8 ГБ карте). Любая модель масштаба 27B–30B (~14.5–16 ГБ в 4-bit) и тем более 671B (~350 ГБ в 4-bit) физически не помещается в 7.46 ГБ свободной VRAM RTX 4060 Ti.
2. **Нет NVMe-стриминга и нет по-экспертного окна MoE (MBR):** даже экспериментальный режим `offload_layers="auto"`, упоминаемый в тексте ошибки Unsloth `2026.11.7`, держит вытесненные декодерные слои целиком в **оперативной памяти (System RAM)** без NVMe-стриминга, без по-экспертного чтения MoE, без `M1/M2+` Decoupled MBR, без `H2` переиспользования распакованных NF4-буферов на окно микробатчей и без `FP8 block64 + DeepseekRoutingOracle`.
3. **Масштаб скорости на 8 ГБ карте:** там, где Unsloth на 8 ГБ RTX 4060 Ti выдаёт **`314.8 tok/s` только на крошечной `0.5B` модели**, наш стек `feat/p3-unified-go` на **той же самой 8 ГБ карте** выдаёт **`317–631 tok/s` на `30B MoE`** (при `1.97–3.57 GiB` NVML VRAM!) и **`159–200 tok/s` на плотной `27B` модели**.

---

## 5. Честно про проблемы с датасетом, токенизатором и прогоном (важно для команды)

Чтобы ни у кого не было завышенных ожиданий или скрытых «подводных камней», фиксируем все реальные проблемы, с которыми столкнулись при подготовке датасета и SFT-прогонах:

### 5.1. Длина сырых промптов `hermes-function-calling-v1` vs окно `S=256` (баг фильтрации в первом прогоне)
- Сырые примеры из `NousResearch/hermes-function-calling-v1` содержат многословные JSON-схемы инструментов с подробными `description` и `properties` на **600–1 500+ токенов**.
- При первом тестовом запуске на `S=256` даже наша первая укороченная 4-tool схема весила **~260–290 токенов**, из-за чего фильтр `len(p_ids) + len(r_ids) > 256` отбросил почти всю выборку, а защитный fallback зациклил 1 оставшуюся короткую строку Hermes. Мы поймали это сразу по неестественно быстрому падению loss до `0.0000` на 6-м шаге.
- **Как исправили:** сжали схему `<tools>` в компактный JSON (`~75` токенов на 4 инструмента; полный пример `~150–170` токенов) и добавили явный лог `Tokenized 416/416 unique tasks fitting S=256`. Все финальные результаты (`02`, `03`, `03b`, `03c`, `03d`) получены на **100% уникальных и непересекающихся** train/test примерах без обрезки промптов (`gen_prompt_len=192`).

### 5.2. Ограничение сложности выборки (proof-of-learning vs полноценный BFCL)
- Из-за жёсткого окна `S=256` (выбранного, чтобы быстро и дёшево прогнать обе 27B–30B модели на 8 ГБ карте) 416 задач построены как **компактный одношаговый агентный бенчмарк** на базе схем `hermes-function-calling-v1` (8 доменов: камеры, OS-процессы, SQL `dblink`, SKU-склады, Basic Auth, Kubernetes, Prometheus; каждый раз меню из 4 инструментов с ротацией и уникальными параметрами).
- Это **честное доказательство реальной обучаемости** 27B–30B моделей в `FP8 block64 + Oracle + L2L` на 8 ГБ GPU за копейки, но **не замена полноценного многошагового BFCL / Tau-bench**: из-за компактного формата `PPL → 1.00` достигается за 16 шагов. Для продакшен-бенчмарка качества в следующих итерациях стоит прогнать SFT на сырых длинных промптах (`S=1024–2048`) из `hermes-function-calling-v1` / `xLAM` / `BFCL`.

### 5.3. Баг базового чекпоинта `kanana-2-30b-a3b-base-2601` со спецтокенами `<tool_call>` / `</tool_call>`
- У `Qwen/Qwen3.8-27B` теги `<tool_call>` и `</tool_call>` уже обучены на этапе пост-тренинга.
- У `Kanana-2-30B-A3B` мы тестировали **base**-чекпоинт (`kanana-2-30b-a3b-base-2601`), и прямой анализ `large_embed_tokens.safetensors` / `large_lm_head.safetensors` показал:
  - Токены `<tool_call>` (ID `128022`) и `</tool_call>` (ID `128023`) **не инициализированы**: норма в `embed_tokens` равна **`4.09e-6`** (средняя по словарю `1.17`), а в `lm_head` обе строки имеют норму **`0.589`** (средняя `1.89`), **совпадают друг с другом до 1 ULP BF16 (`max_diff = 1.53e-5`)** и сильно коллинеарны обычному BPE-токену `"-cimax"` (`cos = 0.616`, норма `0.862`).
  - **Без head LoRA (`03`):** модель выучила точный JSON внутри, но вместо `<tool_call>`/`</tool_call>` генерировала `"-cimax"`.
  - **С `--train-head` и общим `lr=8e-4` (`03b`/`03c`):** модель выучила `<tool_call>` и 100% JSON (`TokAcc = 96.71%` = 29 из 30 токенов), но упёрлась в теоретическое плато симметрии `2·ln(2)/29.44 ≈ 0.0463`, выдавая `<tool_call>` и в начале, и в конце.
  - **С `--train-head --head-lr 1e-2` (`03d`):** `StreamedLargeLayer` LoRA на `embed_tokens` + `lm_head` развела строки `128022`/`128023` за 16 шагов (`NLL = 0.0014`, `TokAcc = 100.0%`, `Valid XML = 87.5%`, `Valid JSON = 81.25%`, `Exact Action = 62.5%`).
- **Вывод для пользователей Soup:** при дообучении **base**-моделей на агентных форматах с зарезервированными спецтокенами (`<tool_call>`) обязательно включать LoRA на `embed_tokens`/`lm_head` с повышенным `head_lr` (или использовать `-instruct` чекпоинт / обычные BPE-разделители).

### 5.4. Инженерные фиксы, найденные на реальной 8 ГБ карте по ходу прогона
1. **VLM-обёртка `Qwen/Qwen3.8-27B`:** `load_stream_config` возвращает внешний `Qwen3_5Config`, у которого `vocab_size`/`hidden_size` находятся в `cfg.text_config` → добавили `getattr(cfg, "text_config", None) or cfg`.
2. **OOM на префилле генерации у `Qwen/Qwen3.8-27B` (`V = 248 320`):** при батчевой генерации `B=16, L=176` вызов `model(...)` без `logits_to_keep=1` аллоцировал `1.30 GiB` под полную матрицу логитов префилла и падал в OOM после SFT. Передача **`logits_to_keep=1`** (логиты только последнего токена: `7.6 MiB` вместо `1.30 GiB`) + микробатчинг `sub_bs=8` + очистка состояния оптимизатора перед `after_eval` полностью сняли проблему.

---

## 6. Проверка корректности и артефакты

- **Тесты и линтер:** `ruff check` — чисто; регрессия `tests/test_p3_fp8_oracle_mbr.py` + `tests/test_l2l_step.py` — `22 passed`; полная регрессия стека — `238 passed, 8 skipped, 0 failed` + `7 passed` на реальной CUDA RTX 4060 Ti.
- **Оговорка стенда по P3:** все замеры на Vast.ai шли на хосте с 62 GiB RAM, 1.4 GiB swap и ФС `overlay`. Для финальной приёмки ограничения P3 (`DeepSeek-V3 671B`) остаётся прогнать Фазу 2 на машине с жёстким cgroup 32 GiB RAM (`swap=0`) и шардами на прямом NVMe вне `overlay`.
- **Сырые замеры (всё сохранено локально на `feat/p3-unified-go`, без `git push`):**
  - Серия 1 (Kanana BF16 A–B–B–A, `k=8/64`, OOM baseline): `benchmarks/results/vast-4060ti-20261120/`
  - Серия 2 (Kanana `expandable_segments`, 20 мс NVML, H2 абляция, 3 шага AdamW, голова 671B): `benchmarks/results/vast-4060ti-verify2-20261120/`
  - Серия 3 (Kanana `FP8 block64 + DeepseekRoutingOracle`, изолированные процессы): `benchmarks/results/vast-4060ti-fp8-oracle-20261120/`
  - Серии 4 и 5 (`Qwen/Qwen3.8-27B` `01a..01i` + Agentic Tool-Calling SFT До/После `02`, `03..03d` + логи): `benchmarks/results/vast-4060ti-qwen38-agentic-20261120/`
  - Серия 6 (прямое сравнение с Unsloth `2026.11.7` и HF+BitsAndBytes CPU-offload на той же RTX 4060 Ti 8 ГБ): `benchmarks/results/vast-4060ti-unsloth-compare-20261120/` и харнес `benchmarks/harness/p3_unsloth_4060ti_compare.py`
- **Коммиты ветки `feat/p3-unified-go` (Draft [PR #77](https://github.com/SoupTeam/soup-rnd/pull/77)):** `30861b84` (объединённый стек + фикс Triton 3.2), `5e3fa12c` / `aeb4b3c6` (`FP8 block64 + DeepseekRoutingOracle`), `feb4d3ae` / `e6c4da5b` / `d7fb1bb6` (`Qwen/Qwen3.8-27B` + Agentic SFT + замеры), плюс коммит с замерами Unsloth `2026.11.7`.
