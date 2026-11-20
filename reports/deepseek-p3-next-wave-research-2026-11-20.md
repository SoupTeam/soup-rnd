# DeepSeek P3 — R&D-разбор следующей волны оптимизаций для Soup на Ada (8–16 ГБ GPU, 32 ГБ RAM, NVMe)

**Дата:** 20 ноября 2026  
**База:** `feat/p3-unified-go` ([PR #77](https://github.com/SoupTeam/soup-rnd/pull/77)), аудит открытых PR команды (`#28`, `#32`, `#51`, `#63–#69`, `#73`, `#77`), кода `src/soup_cli/utils/{l2l_step.py,invariant_gemm.py,expert_shard.py}`, сырых JSON `vast-4060ti-fp8-oracle-20261120`, исходников `Triton 3.2/3.5`, `PTX 8.7` и актуальных работ 2024–2026 (`SageAttention2++`, `DeepSeek-V3`, `CUTLASS Example 94`, `SGLang`, `TorchAO`, `ZipNN`, `DFloat11`, `KT-FT`, `Unsloth`, `LoRA+`).  
**Правило документа:** явно разделены **(а)** факты из кода/замеров/первоисточников, **(б)** расчёты с первых принципов и **(в)** гипотезы, требующие проверки на железе.

---

## Главное за 30 секунд

Аудит сырых JSON PR #77 и кода показал, что на `Kanana-2-30B-A3B` (`k=64, S=512`, `~52 с/шаг`) главный поток ждёт I/O-воркеры всего **`0.49–0.52 с` (~1%)**, а экономия времени от `FP8+Oracle` укладывается в разброс отдельных прогонов BF16 (`51.77–51.80 с` в серии 2 vs `51.96–52.01 с` в серии 3; реальный твёрдый выигрыш `FP8+Oracle` — это память: `6 144 → 3 323 MiB` store, `−2.63 GiB` RSS на Kanana и `−14.88 GiB` на Qwen 27B).

Поэтому правильный порядок действий перед крупными правками ядра такой:

1. **Прогон #1 (сначала замер, потом код):** снять пофазный профиль **`Fwd / R1 / E-Bwd / R2 / Head / AdamW`** (CUDA events) и гистограмму `M` на Kanana `k=64, S=512`. По закону Амдала для цели `+10% tok/s` участок с локальным ускорением `2×` обязан занимать **`≥18.2%` критического пути** (`≥27.3%` при `1.5×`). Параллельно закрыть пробел в сравнении с конкурентом: прогнать **`Unsloth 2026.11.7` с `offload_layers="auto"`** на Qwen 27B и Kanana 30B (сначала в 62 GiB, затем в `cgroup 32 GiB, swap=0`).
2. **Compute на Ada `sm_89` (`CONDITIONAL GO`, в два шага):**
   - *Шаг 1 (контроль):* **`E4M3` с FP32-аккумулятором (`88.28 TFLOPS` пик, 2× к BF16 `44.14`)** в фиксированном K-цикле `invariant_gemm.py` со шкалами строго **per-token × per-K-group** (`1×64`/`1×128`), чтобы не сломать `batch_invariant`. Работает уже на `CUDA 12.4 / Triton 3.2`.
   - *Шаг 2 (после контроля):* **`E4M3` с FP16-аккумулятором + FP32-промоция каждые `G=64` (`176.56 TFLOPS` пик, 4× к BF16)** по схеме SageAttention2++ (`arXiv:2505.21136`). **Важное ограничение тулчейна:** инструкция `.f16.e4m3.e4m3.f16` появилась только в **`PTX 8.7 / CUDA 12.8` и `Triton 3.5.0`** (в `Triton 3.2` из PR #77 её нет!).
3. **Устранение `Phase R1` в `l2l_step.py` (`CONDITIONAL GO`, зависит от профиля):** `Phase R1` повторно гоняет полный MLA + router без графа ради `expert_input` (`~0.82 PFLOPs` ≈ `18–27 с/шаг` на 671B). Если профиль подтвердит значимую долю `R1`, его можно убрать через асинхронный `O_DIRECT` scratch `expert_input` в `FP8 block64` (`231 MiB/слой`, `14.05 GB` на 58 слоёв V3, `0 MiB` лишней RAM сверх одного pinned-буфера).
4. **NVMe для 671B (`O_DIRECT` уже в коде):** `ExpertShardReader._open()` (`expert_shard.py:264-285`) уже открывает `O_DIRECT`, но тихо откатывается в buffered при `OSError` (как на `overlayfs` Vast). Для 671B (`367.82 GB/фазу` NF4) пороги, ниже которых диск становится узким местом при идеальном перекрытии: **`4.52 GB/s` Fwd / `1.93 GB/s` Bwd** при `30 TFLOPS` и **`6.66 / 2.84 GB/s`** при `44.2 TFLOPS`. Нужно сначала замерить фактический `direct=True` и sustained-полосу холодного чтения на целевом SSD.
5. **Ускорение сходимости (`LoRA+`) и `NO-GO` по lossless-сжатию NF4:** самый дешёвый тест по времени до качества — **`LoRA+` (`lr_B/lr_A = 2, 4`, нулевой оверхед на шаг)** против настроенного baseline на ненасыщенном held-out тесте. Lossless сжатие (`DFloat11` / `ZipNN`) поверх NF4 — **`NO-GO`** (квантили NF4 равновероятны, потолок сжатия одних FP32-шкал `11.1%`, а `ZipNN` на E4M3 даёт `4.4–5.6 bpw` до шкал и выравнивания).

---

## 1. Карта SOTA, исходников и открытых реализаций (2024–2026)

| Метод / Статья / Репо | Год | Суть алгоритма / ядра | Железо и модель в замерах | Заявленный эффект | Главный подвох / ограничение | Источник |
|---|---|---|---|---|---|---|
| **CUTLASS Example 94 & SGLang `fp8_kernel.py`** | 2025–2026 | Блочный `sm_89` FP8 GEMM с фиксированными scale-блоками и внешней FP32-суммой (без Hopper TMA) | RTX 4090 / L4 (`sm_89`); DeepSeek-V3/Qwen3 inference (`vLLM #61224`: L4 `1 929 → 2 739 tok/s` на Qwen3-8B-FP8) | Ближайшая рабочая основа для `sm_89` blockwise FP8 | Только inference (нет `dX`); на `sm_89` лимит shared memory `48–99 KB/block`; на batch 1 бывает регресс (`4.77 → 4.94 с`) | [CUTLASS Ex. 94](https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/examples/94_ada_fp8_blockwise/ada_fp8_blockwise.cu), [SGLang](https://github.com/sgl-project/sglang/blob/a4347f43c2cec02f5b89ab1c787c56c3c55ed8fb/python/sglang/kernels/ops/gemm/fp8_kernel.py#L95-L128), [vLLM #61224](https://github.com/vllm-project/vllm/pull/61224) |
| **DeepSeek-V3 FP8 Training** (`arXiv:2412.19437`, §3.3) | 2024–2025 | Блочный `E4M3` (`1×128` акт., `128×128` веса) **в Fwd и Bwd без `E5M2`** + FP32-промоция на CUDA-ядрах каждые 128 `K` (обход 14-битного усечения мантиссы Tensor Core) | H800 (`sm_90`); DeepSeek-V3 671B | `<0.25%` расхождение loss vs BF16 | `DeepGEMM` привязан к `sm_90+` (TMA + WGMMA) | [arXiv:2412.19437 §3.3.2](https://arxiv.org/html/2412.19437v2#S3.SS3.SSS2) |
| **SageAttention2++** (`arXiv:2505.21136`, §3) | 2025 | `FP8 × FP8` с **FP16-аккумулятором** (`mma.f16.f8.f8.f16`) + FP32-промоция каждые `G=64` `K` и сужение диапазона | RTX 4090 (`sm_89`), L40S; LLM / Video | `2×` к `FP8→FP32`, до `3.9×` к FlashAttention | Требует **PTX 8.7 / CUDA 12.8** и **Triton 3.5+**; доказано на `P @ V`, не на `dX`; нужна граница `G·amax(qA)·amax(qB) ≤ 65504` | [arXiv:2505.21136 §3](https://arxiv.org/html/2505.21136v3#S3), [Triton 3.5 MMAv2](https://github.com/triton-lang/triton/blob/v3.5.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L387-L423) |
| **TorchAO `Float8Linear`** & **COAT** (`arXiv:2410.19313`) | 2025–2026 | PyTorch-native FP8 (`torch._scaled_mm`); COAT добавляет mixed-granularity FP8 активации и range expansion | H100 / Ada; Llama / Torchtitan | Контроль цены `amax + cast + GEMM`; COAT: `1.54×` меньше памяти, `1.43×` быстрее | Опубликованный crossover TorchAO на H100; `tensorwise amax` по `M` ломает `batch_invariant` MBR | [TorchAO](https://docs.pytorch.org/ao/stable/workflows/training.html#performance), [arXiv:2410.19313](https://arxiv.org/abs/2410.19313) |
| **DFloat11** (`arXiv:2504.11651`) & **ZipNN** (`arXiv:2508.19263`) | 2025 | Lossless entropy-кодирование экспонент BF16/FP8 (Huffman / ANS) | Llama-405B (8×80GB), E4M3 модели | DFloat11: `−30%` на BF16 (~11 bpw); ZipNN E4M3: `55–70%` (`4.4–5.6 bpw`) | Хуже NF4 (`4.5 bpw`) до шкал и 4 KiB выравнивания; поверх NF4 энтропия квантилей близка к 4.0 бит | [arXiv:2504.11651](https://arxiv.org/abs/2504.11651), [ZipNN E4M3 §4.2](https://arxiv.org/html/2508.19263v1#S4.SS2) |
| **LoRA+** (`arXiv:2402.12354`) vs **LoRA-Pro** / **LoRA-GA** / **sMuon** | 2024–2026 | `LoRA+`: `lr_B = λ·lr_A`; `LoRA-Pro`: уравнение Сильвестра `r×r`; `sMuon` (`2608.14492`) / `LoRA-TSD` (`2609.02734`): Muon для адаптеров | Llama / Mistral / Qwen; GSM8K, MMLU, HumanEval | Ускорение сходимости по шагам без правки замороженных весов | В отдельной ячейке `LoRA-Pro` дал `101` мин против `90` у LoRA из-за доп. вычислений; начинать с бесплатного `LoRA+` | [LoRA+ §5.3](https://arxiv.org/html/2402.12354#S5.SS3), [LoRA-Pro](https://arxiv.org/html/2407.18242#S3.SS3), [sMuon](https://arxiv.org/abs/2608.14492) |
| **OOM** (`arXiv:2611.11364 §7`) | 2026 | Interleaved layer residency и политики prefetch для MoE offloading | Qwen3.6-35B-A3B на 4× RTX 4090; 122B на H100 | Снижение шага `7.53 → 6.69 с` при тех же 12 резидентных слоях | Другой масштаб памяти/GPU, но переносима методика сравнения политик prefetch на одном trace | [arXiv:2611.11364 §7](https://arxiv.org/html/2611.11364v1#S7) |

---

## 2. Разбор с первых принципов под RTX 4060 Ti 8 ГБ (`sm_89`), 32 ГБ RAM, NVMe

### 2.1. Аппаратная лестница Ada (`sm_89`) и требования тулчейна

*(а) Факты из Ada Whitepaper v2.02 (Table 2), PTX ISA 8.7 и исходников Triton 3.2 vs 3.5:*

| Режим MMA на RTX 4060 Ti (`AD106`, 34 SM) | PTX / CUDA минимум | Поддержка в Triton | Пик (dense TFLOPS) | Множитель к BF16 |
|---|---|---|---:|---:|
| `BF16 × BF16 → FP32` *(текущий `invariant_gemm.py`)* | PTX 7.0 / CUDA 11 | Triton 3.2+ | **`44.14`** | `1.00×` |
| `FP8 (e4m3) × FP8 → FP32` *(1-й контроль)* | PTX 8.4 / CUDA 12.4 | **Triton 3.2+** (`MMAv2.cpp:194-258`) | **`88.28`** | **`2.00×`** |
| **`FP8 (e4m3) × FP8 → FP16`** *(2-й шаг, с FP32 промоцией)* | **PTX 8.7 / CUDA 12.8** | **Triton 3.5.0+** (`ptxas 12.8.93`, `MMAv2.cpp:387-423`) | **`176.56`** | **`4.00×`** |

*(б) Условия точности и инвариантности для `invariant_gemm.py`:*
- **Защита от переполнения FP16-аккумулятора:** при нулевом начальном аккумуляторе на K-блоке длины $G$ требуется $G \cdot \text{amax}(q_A) \cdot \text{amax}(q_B) \le 65\,504$ с запасом на округление. Для $G=64$ стандартный диапазон `E4M3` (`448 × 448 = 200 704`) **переполняет FP16**; пробные caps `30 / 30` дают границу `64 × 30 × 30 = 57 600 < 65 504` (а `16 / 16` — `16 384`).
- **Сохранение `batch_invariant`:** нельзя использовать `tensorwise amax` по всему `M` (тогда строка в `M=1024` получит другой scale, чем в `M=16`). Квантование активаций и `dY` должно быть строго **per-token × per-K-group** (`1×64` или `1×128`), фиксированный tile и постоянный порядок K-цикла, включая хвосты.
- **Учёт всех накладных в микробенче:** измерять не только голый GEMM, но полный конвейер `NF4 decode → amax/pack → transpose для dX → GEMM`, а также проверить дамп PTX/SASS на отсутствие тихого сворачивания в Triton (`triton-lang/triton#12392`).

### 2.2. Пороги диска и память на DeepSeek-V3 671B (`k=64, S=512 = 32 768 tok/step`)

*(а) Факты из кода (`expert_shard.py:264-285, 348-374`, `async_expert_controller.py:43-76`, `native_stream_async.cpp:158-177`):*
- `O_DIRECT` **уже реализован** до C++: `ExpertShardReader._open()` открывает direct FD с тихим fallback на buffered при `OSError`, передаёт FD в C++, где проверяется `F_GETFL`. На Vast `overlayfs` мог срабатывать тихий fallback.
- Один routed-эксперт V3 (`3 × 7168 × 2048`) в NF4 `4.5 bpw` весит **`23.625 MiB` (`24.77 MB`)**, вся фаза (`58 × 256 = 14 848` экспертов) — **`367.82 GB`** до выравнивания (`735.65 GB` Two-Read).

*(б) Пороги sustained полосы SSD, ниже которых диск становится узким местом (из расчёта `8.16 PFLOPs` = `2.44 Fwd + 5.72 Bwd`):*
- При реалистичных **`30 TFLOPS` BF16** (`81.4 с` Fwd / `190.5 с` Bwd): диск ограничивает ниже **`4.52 GB/s` в Forward** и **`1.93 GB/s` в Backward**.
- При пике **`44.2 TFLOPS` BF16** (`55.2 с` Fwd / `129.3 с` Bwd): диск ограничивает ниже **`6.66 GB/s` в Forward** и **`2.84 GB/s` в Backward**.
- **Вывод про lossless-сжатие весов:** `ZipNN` для E4M3 оставляет `55–70%` (`4.4–5.6 bpw` до шкал и 4 KiB выравнивания — больше NF4 `4.5 bpw`), сжатие одних FP32-шкал NF4 ограничено сверху `11.1%` (проще и быстрее перевести шкалы в BF16/double-quant → `4.125 bpw`). При 20% экономии на `6 GB/s` одно чтение эксперта ускоряется на `~0.83 мс`, что окупает последовательную CPU-распаковку только при скорости декода `>30 GB/s`.

### 2.3. Стоимость `Phase R1` в `l2l_step.py:1130–1243`

*(а) Из кода `l2l_step.py`:* в Backward `R1` пересчитывает `MLA + router` без графа ради `expert_input`, затем `E-Bwd` считает `dX` и `dα`, затем `R2` снова считает `MLA + router + shared` по `sub_w=1` с графом.  
*(б) Память для кэша `expert_input` (если профиль покажет высокую долю `R1`):* на 671B полный RAM-кэш добавил бы **`25.38 GiB` BF16** или **`13.48 GiB` `FP8 block64`** поверх уже занятых `14.27 GiB` boundary store — в 32 GiB RAM это не влезает. Реалистичный путь — **асинхронный `O_DIRECT` scratch на NVMe** (`231 MiB/слой` через один переиспользуемый pinned-буфер, `0 MiB` лишней RAM), и начинать проверку нужно с 1–2 слоёв на Kanana (`66 MiB/слой`), только если доля `R1` в профиле окажется значимой.

### 2.4. Что уже закрыли коллеги в `SoupTeam/soup-rnd`
- **PR #72 (merged в `main`, включает `#74–#76`):** группировка MBR, `async_lease`, `fast_row_gather`, `mbr_ordered_backward`.
- **PR #28 (`D3`, `Artix6034`):** экспериментальный fused NF4 dequant-GEMM сэкономил `52 MiB` VRAM, но был **на `22.2–30.6%` медленнее** на RTX 3090 — подтверждает `NO-GO` для классического BF16 fusion.
- **PR #51 (`H2`):** Liger CE-only на RTX 4090 дал `1.013×` (`p=0.077`, ускорения >3% нет), `−33%` reserved VRAM.
- **PR #65 / #69 (`nurseit14`):** поддержка FP16 `GradScaler` в L2L (pre-Ampere) и `--gpus N` L2L DDP.

---

## 3. Сравнение с конкурентами на ноябрь 2026 (проверенные цифры из первоисточников)

| Стек | Архитектура оффлоада | Реальные требования / измеренные точки | Пробел / отличия от Soup |
|---|---|---|---|
| **Unsloth (`2026.11.7`, PR `#11832`, `#12705`, `#15159`)** | `offload_layers="auto"` (ранее `block_swap_layers`): держит вытесненные замороженные слои целиком в **pinned Host RAM** + gradient checkpointing | PR `#15159` починил fused MoE под `offload_layers` (проверено 3 шага tiny Qwen3-MoE на T4). Для DeepSeek-V3 671B одни сырые 4-bit routed-веса весят **`~304.5 GiB` RAM** — в 32 GiB RAM физически не влезает | **Нужен 1 догон:** в нашей Серии 6 мы не передавали `offload_layers="auto"`. Короткий прогон на Qwen 27B и Kanana 30B (в 62 GiB и в `cgroup 32 GiB, swap=0`) закроет вопрос для 27–30B |
| **KTransformers Fine-Tune (`KT-FT`, PDF Fig. 8(d) & Table 2)** | Attention/Shared на GPU, routed-эксперты в FP8/INT8 исполняются на CPU (Intel Xeon AMX) | Ближайшая опубликованная single-GPU точка 671B LoRA (`r=8`, 3.4B trainable): **`1× RTX 4090 + 1 TB RAM + 2× Intel Xeon AMX`**, batch 1. Цифра **`166.8 tok/s @2K`** снята на **`8× L40 + 2 TB RAM`**, BF16, batch 8 | Требует сервер с 1–2 TB RAM и Xeon AMX; не работает на consumer 32 GiB RAM + NVMe |
| **Axolotl / TorchTitan / MegaTrain** | Axolotl: выгрузка слоёв в pinned CPU RAM (+ обязательное сохранение router top-k); TorchTitan: `DP=4, EP=2`; MegaTrain (120B): H200 + `1.5 TB` RAM | Кластерные или `>1 TB` RAM стенды | Не решают задачу `1× 8 ГБ GPU + 32 GiB RAM + NVMe` |
| **DeepSpeed ZeRO-Infinity / LoHan** | Послойный offload параметров/оптимизатора в RAM/NVMe | На динамическом MoE-роутинге превращает чтение в random I/O | У Soup детерминированное окно MBR (`k=64`) даёт крупные последовательные чтения `24.8 MB` |

---

## 4. Red Flags (что может пойти не так)

1. **Накопление дрейфа FP8 на длинной траектории:** на Kanana `k64×S512` (1 шаг) `FP8+Oracle` дал `grad_rel_l2 = 0.01393, min_cos = 0.99471`, но на 3-м шаге `k8×S512` уже `0.03465` и `0.97066` (из-за расхождения траекторий адаптеров). Ошибку ядра нужно мерить на **одном и том же снимке адаптеров**, а качество обучения — на отдельной траектории и held-out метрике.
2. **Несоответствие версий CUDA/Triton на стенде:** попытка собрать `mma.f16.f8.f8.f16` на `Triton 3.2 / CUDA 12.4` либо упадёт, либо тихо откатится в медленный путь. Фиксировать `ptxas --version`, `triton.__version__`, `git_sha` (в JSON серии 3 было `unknown`) и проверять PTX/SASS.
3. **Тихий откат `O_DIRECT` на `overlayfs`:** `ExpertShardReader._open()` перехватывает `OSError` и молча ставит `self.direct = False`. Без явной проверки `direct == True` и сверки `io.stat rbytes` можно случайно тестировать Linux page cache вместо NVMe.
4. **Насыщенный eval для `LoRA+`:** на нашем компактном `S=256` Agentic SFT `PPL` падает до `1.00` за 16 шагов (`~7.5–7.7k` target-токенов) — на таком тесте невозможно отличить `LoRA+` от baseline. Нужен ненасыщенный held-out тест (например, `S=512–1024`, GSM8K / MMLU / длинные схемы инструментов).

---

## 5. План проверочных экспериментов (строгий порядок, бюджет `< $2.50` на 1× RTX 4060 Ti 8 ГБ)

### Шаг 1 — Пофазный профиль Kanana `k=64, S=512` + догон `Unsloth offload_layers="auto"` (~45 мин, `~$0.25`)
- **Профиль Kanana (BF16, 8 GB):** снять CUDA events / NVTX для `Fwd`, `R1`, `E-Bwd`, `R2`, `Head`, `AdamW` (не ломая перекрытие стримов), гистограмму `M`, реальный `direct` флаг и повторы wall-time после прогрева.
  - **Ветвление по результатам профиля (порог Амдала для `+10%` при локальном `2×`: `p ≥ 18.2%`):**
    - Большая доля `E-Bwd`/GEMM → запускаем Шаг 2 (FP8 GEMM).
    - Большая доля `R1` → проверяем сохранение `expert_input` на 1–2 слоях (`O_DIRECT` scratch vs `R1`).
    - Большая доля `R2` → профилируем отдельно `MLA / router / shared` и проверяем размер `sub_w` в бюджете 8 ГБ.
- **Догон `Unsloth 2026.11.7` (`offload_layers="auto"`):** те же Qwen3.8-27B и Kanana-30B-A3B (`k=16, S=256`, `r=16`, те же targets, закреплённые версии `unsloth`/`unsloth_zoo`), подтвердить фактическое число выгруженных слоёв, замерить startup, 2 warmup, 3–5 шагов, NVML, RSS и `cgroup memory.peak` (сначала 62 GiB, затем `32 GiB, swap=0`).

### Шаг 2 — Микробенч `sm_89` FP8 GEMM на 4 формах DeepSeek-V3 A2 (~30 мин, `~$0.20`)
- **Формы:** `gate_up.fwd`, `down.fwd`, `down.dx`, `gate_up.dx` (`7168×2048`), `M = 1, 2, 3, 128, 512, 1024` + взвешивание по гистограмме маршрутов.
- **Сравнение:** (0) текущий BF16 `invariant_gemm.py` и `cuBLAS`; (1) **`E4M3` с FP32-acc** (`per-token × groupK=64/128`); (2) на `Triton 3.5+ / CUDA 12.8+` — **`E4M3` с FP16-acc** (`G=64`, cap `30/30` и `16/16`) на тех же FP8-байтах; (3) полная цепочка `NF4 decode + amax/pack + transpose + GEMM`.
- **`PASS / FAIL`:**
  - `PASS`: подтверждённые PTX/SASS; побитовый паритет merged/split внутри варианта (`max_abs = 0.0`); `0` overflow/NaN/Inf; на одинаковом состоянии адаптеров `grad_rel_l2 ≤ 0.05` и `min_cos ≥ 0.99` (порог E3) для `dX` и `dα`; реальный выигрыш на полной цепочке покрывает долю критического пути (`≥10%` проекция на полный шаг).
  - `FAIL`: провал точности, поломка `batch_invariant` или выигрыш меньше порога окупаемости → остановка на микробенче.

### Шаг 3 — Целевой NVMe-контроль (`cgroup 32 GiB, swap=0`) и тест `LoRA+` (~2–3 ч, `~$0.80`)
- **Холодное чтение NVMe:** на `ext4/xfs` bind-mount (вне `overlay`) подтвердить `direct=True`, измерить sustained-полосу после прогрева и троттлинга (`p95`, `io.stat rbytes/wbytes`, `memory.peak`, `swap == 0`) и сравнить с порогами V3 (`4.52 / 1.93 GB/s` при 30 TFLOPS; `6.66 / 2.84 GB/s` при 44.2 TFLOPS). Вкладываться в `IQ3_XXS/Q3_K` или `double-quant` только если диск ниже порога.
- **`LoRA+` по времени до качества:** на ненасыщенном held-out наборе сравнить `lr_B/lr_A = 1` (настроенный baseline) vs `2` и `4` на 2–3 сидах (чекпоинты на шагах `0/4/8/16/32`). Критерий `PASS` — меньшее суммарное **wall-time до заранее заданного held-out NLL / GSM8K порога** при равном бюджете VRAM/RAM.

---

## 6. Источники

1. **NVIDIA Ada Lovelace Whitepaper v2.02 (Table 2 — `sm_89` TFLOPS):** https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf#page=30
2. **NVIDIA PTX ISA 8.7 (CUDA 12.8.1 — `mma.sync` FP8 `.f16` / `.f32`):** https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
3. **Triton 3.2 vs 3.5 MMAv2 FP8 lowering & toolchain:** [Triton 3.2 MMAv2.cpp](https://github.com/triton-lang/triton/blob/v3.2.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L194-L258), [Triton 3.5 MMAv2.cpp](https://github.com/triton-lang/triton/blob/v3.5.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L387-L423), [triton#12392](https://github.com/triton-lang/triton/issues/12392)
4. **CUTLASS Example 94 (Ada FP8 blockwise) & SGLang Triton blockwise FP8:** [ada_fp8_blockwise.cu](https://github.com/NVIDIA/cutlass/blob/af951046cd9cac9301d8e3e90863fd8d7edc9fa9/examples/94_ada_fp8_blockwise/ada_fp8_blockwise.cu), [SGLang fp8_kernel.py](https://github.com/sgl-project/sglang/blob/a4347f43c2cec02f5b89ab1c787c56c3c55ed8fb/python/sglang/kernels/ops/gemm/fp8_kernel.py#L95-L128), [vLLM #61224](https://github.com/vllm-project/vllm/pull/61224)
5. **DeepSeek-V3 Technical Report (§3.3.2 FP8):** https://arxiv.org/html/2412.19437v2#S3.SS3.SSS2
6. **SageAttention2++ (`arXiv:2505.21136 §3`):** https://arxiv.org/html/2505.21136v3#S3
7. **TorchAO FP8 Training & COAT (`arXiv:2410.19313`):** [TorchAO](https://docs.pytorch.org/ao/stable/workflows/training.html#performance), [arXiv:2410.19313](https://arxiv.org/abs/2410.19313)
8. **DFloat11 (`arXiv:2504.11651`) & ZipNN E4M3 (`arXiv:2508.19263 §4.2`):** https://arxiv.org/abs/2504.11651 и https://arxiv.org/html/2508.19263v1#S4.SS2
9. **Unsloth `offload_layers` / MoE PRs:** [#11832](https://github.com/unslothai/unsloth/pull/11832), [#12705](https://github.com/unslothai/unsloth/pull/12705), [#15159](https://github.com/unslothai/unsloth/pull/15159), [V3.1 recipe #3200](https://github.com/unslothai/unsloth/issues/3200)
10. **KTransformers Fine-Tune (`KT-FT`, Table 2 & Fig. 8) & OOM (`arXiv:2611.11364 §7`):** [KT-FT PDF](https://raw.githubusercontent.com/kvcache-ai/ktransformers/paper-ktft-v1.1/papers/kt-ft/ktransformers-finetune.pdf), [arXiv:2611.11364 §7](https://arxiv.org/html/2611.11364v1#S7)
11. **LoRA+ (`arXiv:2402.12354`), LoRA-Pro (`arXiv:2407.18242`), LoRA-GA (`arXiv:2407.05000`), sMuon (`arXiv:2608.14492`), LoRA-TSD (`arXiv:2609.02734`):** [LoRA+](https://arxiv.org/html/2402.12354#S5.SS3), [LoRA-Pro](https://arxiv.org/html/2407.18242#S3.SS3)
12. **Код и PR `SoupTeam/soup-rnd`:** PR `#28`, `#51`, `#65`, `#66`, `#69`, `#72`, `#73`, `#77`; `src/soup_cli/utils/{l2l_step.py,invariant_gemm.py,expert_shard.py,async_expert_controller.py}`.
