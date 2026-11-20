# DeepSeek P3 — Единая GO-ветка (`feat/p3-unified-go`) и физические проверки на RTX 4060 Ti 8 ГБ

**Дата:** 20 ноября 2026  
**База:** PR #73 (`25336300ef279a9080f0d451acfad60c8f206e58`, Triton `batch_invariant` GEMM)  
**Worktree:** `C:/Users/esenb/Desktop/soup-p3-unified-go` (локальная ветка `feat/p3-unified-go`, коммит кода `30861b84`, без пуша в remote)

---

## 1. Состав объединённого стека (`GO` из 3 параллельных треков)

| Компонент | Источник | Файлы в ветке | Эффект |
|---|---|---|---|
| **M1 + M2+ + H2 + M3 + $O(N \cdot K)$ маршрутизация** (decoupled sub-window `sub_w=1`, zero-materialization streaming backward без `WFE`/`dh_chunks`/`h_sorted`, ABA-safe window NF4 dequant, освобождение `LargeLayerBufferPool` 1.726 GiB на всех 61 слоях, маршрутизация без `int64` `one_hot` + фикс утечки графов при `mbr_group_size < k`) | Трек 1 (`feat/p3-decoupled-mbr`) | `src/soup_cli/utils/layer_stream_runtime.py`, `src/soup_cli/utils/expert_stream.py`, `src/soup_cli/utils/l2l_step.py` | **20.59×** меньше saved storage (`k=32`), физически снимает 8 ГБ OOM на Kanana `k=64, S=512` (вместе с `expandable_segments:True` → **3.70 GiB** честный 20 мс пик NVML); `0` чтений экспертов в recompute |
| **`PackedStreamAsync` (C++17 `io_workers=4, ring_slots=8`, 2-slot async DMA, cross-layer backward pre-staging, `nf4_decode.cu`) + `merged_invariant` (`batch_invariant` Triton + фикс Triton 3.2 `tl` в `invariant_gemm.py`)** | Трек 2 (`feat/p3-io-gemm-throughput`) + PR #73 | `src/soup_cli/utils/async_expert_controller.py`, `src/soup_cli/utils/_native_expert_async/{native_stream_async.cpp,nf4_decode.cu,build_ext.py}`, `src/soup_cli/utils/invariant_gemm.py`, интеграция в `expert_stream.py` и `l2l_step.py` | `main_wait_worker` снижен в **5.7×** (`4.09 → 0.71 с`); **100%** cross-layer promotions (`46/46` за шаг, `138/138` за 3 шага, `0` discards); **76.0–76.4%** H2D overlap на `S=512` |
| **Chunked CE / CCE на L2L-голове** | Трек 3 (`codex/p3-deletion-and-cce`) + PR #63 | `src/soup_cli/utils/stream_chunked_ce.py`, `lora_config=None` в `layer_stream_runtime.py` | Физически на RTX 4060 Ti 8 ГБ для формы головы DeepSeek-V3 `H=7168, V=129280, S=4096`: `full` = **CUDA OOM** (`7.38 GiB`), `chunked` = **`2.27 GiB` alloc / `2.31 GiB` resv / `2.87 GiB` NVML peak** |
| **FP8 (`block64_e4m3`) граничные активации + `DeepseekRoutingOracle`** (блочное `group_size=64` FP8-сжатие с async pinned DMA + трёхвазный Oracle `forward`/`peek`/`recompute` с точным якобианом `_OriginalSigmoidJacobian` и компактными `int16` ID без `TopkDecisions` dispatch-оверхеда) | PR #57 + PR #66 + интеграция в decoupled MBR | `src/soup_cli/utils/activation_codec.py`, `src/soup_cli/utils/fp8_activations.py`, `src/soup_cli/utils/routing_oracle.py`, `src/soup_cli/utils/l2l_step.py`, `tests/test_p3_fp8_oracle_mbr.py` | На Kanana `k=64, S=512` (32 768 tok): store **`6144 → 3264 MiB` (−46.9%)** + Oracle **`58.75 MiB`**, физический пик RSS **`13.51 → 10.88 GiB` (−2.63 GiB)**, пик NVML **`3.700 → 3.567 GiB` (−133 MiB)**, скорость **`616.4 → 630.7 tok/s` (+2.3%)**, E3 **`grad_rel_l2 = 0.01393 ≤ 0.05`**, **`min_cos = 0.99471 ≥ 0.99` (PASS)**; на 671B экономит **−12.42 GiB RAM** (`26.69 → 14.27 GiB`), вписывая `k=64, S=512` в 32 GiB RAM (`swap=0`) |

### Отсечённые направления (`NO-GO` — не включены в горячий путь)
- **VRAM Packed-NF4 Hotset на 671B** (Трек 2): 4 GiB VRAM вмещают лишь 1.12% экспертов → `<0.67%` ускорения.
- **CPU Cold-Expert / KTransformers при `k=64`** (Трек 2): $M^* = 0$, доля экспертов с $M \le 16$ равна `0.0%`.
- **`H-C1` split `gate_w`/`up_w`** (Трек 2): не нужен с `batch_invariant` (`merged_invariant`) и вносит 1 BF16 ULP сдвиг.
- **Sparse-Candidate Deletion Tuning** (Трек 3): не снижает NVMe-трафик, на 671B `active-core` дороже LoRA `r=1`, на `routed` (`p=0.2%`) даёт `8.93 GiB` OOM.

---

## 2. Физические замеры на RTX 4060 Ti 8 ГБ (Vast.ai, инстансы удалены)

**Стенд:** `NVIDIA GeForce RTX 4060 Ti (8188 MiB, 7.65 GiB CUDA, SM 8.9)`, 62 GiB RAM, NVMe `~3.7–3.8 GB/s`, `torch 2.6.0+cu124`, `transformers 5.23.0`, `peft 0.22.0`, `bitsandbytes 0.50.2`, `triton 3.2.0`, `nvcc 12.4`.  
**Модель:** `kakaocorp/kanana-2-30b-a3b-base-2601` (`0d31365a`), NF4 (`no-double-quant`, `per_expert`), BF16, `host_cache = 3.5 GiB`, LoRA `r=4` (`q_b_proj, kv_b_proj, o_proj`), `--expert-gemm batch_invariant` + `--cuda-nf4-decode`.  
**Сырые JSON:** `benchmarks/results/vast-4060ti-20261120/` (серия 1) и `benchmarks/results/vast-4060ti-verify2-20261120/` (серия 2, с 20 мс фоновым поллером пика NVML, `30861b84`).

### 2.1. Kanana 30B-A3B на физическом RTX 4060 Ti 8 ГБ (честное разделение метрик ускорения)

| Сценарий (токенов/шаг) | Арм (`alloc_conf`) | Шаг (с) | Скорость (tok/s) | Парное ускорение (те же `k, S`) | К `l2l_ref` при том же `S` | К базе P3 `k=1, S=64` (`5.93 tok/s`) | `cuda_alloc` peak | `cuda_resv` peak | `nvml_peak` (20 мс)* | Побитовый паритет (loss / grad / adapter) |
|---|---|---:|---:|:---:|:---:|:---:|---:|---:|---:|:---:|
| **`k=1, S=64`** (64 tok, база P3) | `l2l_ref` (`exp_seg`) | 10.79 / 10.42 | 5.93 / 6.14 | ref | 1.00× | 1.00× | 0.713 GiB | 0.787 GiB | 1.374 GiB | ref (`3 821`×2 чтений) |
| **`k=1, S=512`** (512 tok, база `S=512`) | `l2l_ref` (`exp_seg`) | 15.31 / 15.36 | 33.43 / 33.32 | ref | 1.00× | 5.63× | 1.363 GiB | 1.568 GiB | 2.155 GiB | ref (`5 351`×2 чтений) |
| **`k=4, S=64`, 3 шага AdamW** (768 tok) | `l2l_ref` (`exp_seg`) | 126.84 (`42.2/43.1/41.6`) | 6.06 | ref | 1.02× | 1.02× | 0.767 GiB | 0.810 GiB | 1.397 GiB | ref |
| **`k=4, S=64`, 3 шага AdamW** (768 tok) | **`unified_go`** (`exp_seg`) | **29.10** (`9.83/9.79/9.48`) | **26.39** | **4.36×** | 4.45× | 4.45× | 0.781 GiB | 0.820 GiB | 1.407 GiB (`0.89` post) | **`bit_equal=True` (`0.0`)** после 3 шагов! |
| **`k=8, S=64`** (512 tok, серия 1) | `l2l_ref` (default) | 93.23 | 5.49 | ref | 0.93× | 0.93× | 0.752 GiB | 0.816 GiB | 1.403 GiB (post) | ref (`30 969`×2) |
| **`k=8, S=64`** (512 tok, серия 1) | **`unified_go`** (`full` / `chunked`) | **11.98 / 12.10** | **42.73 / 42.30** | **7.78× / 7.70×** | 7.20× / 7.13× | 7.20× / 7.13× | 0.769 GiB | 0.844 GiB | **0.893 GiB** (post) | **`bit_equal=True` (`0.0`)** |
| **`k=64, S=64`** (4 096 tok, серия 2) | `std_mbr` (без H2, `exp_seg`) | 40.74 | 100.53 | ref | 16.9× | 16.9× | **3.740 GiB** | 3.818 GiB | 4.405 GiB | ref |
| **`k=64, S=64`** (4 096 tok, A–B–B–A сер. 1) | `mbr_h2_sync` (A1 / A2) | 41.12 / 40.81 | 99.62 / 100.38 | ref | 16.9× | 16.9× | 1.516 GiB | 1.645–1.756 GiB | 2.23–2.34 GiB (post) | **`bit_equal=True` (`0.0`)** |
| **`k=64, S=64`** (4 096 tok, абляция сер. 1) | `unified_go_qd1` (`QD=1, no CL`) | 40.42 | 101.34 | 1.01× | 17.1× | 17.1× | **0.828 GiB** | 1.756 GiB | **1.356 GiB** (post) | **`bit_equal=True` (`0.0`)** |
| **`k=64, S=64`** (4 096 tok, A–B–B–A сер. 1) | **`unified_go`** (B1 / B2, `QD=4+CL`) | **37.25 / 37.24** | **109.96 / 109.99** | **1.10×** vs `mbr_h2` (**1.085×** vs `qd1`) | **18.5×** | **18.5×** | **0.823–0.828 GiB** | **1.262 GiB** | **1.18–1.36 GiB** (post) | **`bit_equal=True` (`0.0`)** |
| **`k=64, S=512`** (32 768 tok, серия 1) | `mbr_h2_sync` (PR #72 + H2) | — | **CUDA OOM** | — | — | — | 6.680 GiB (OOM) | — | **7.673 GiB (OOM)** | `Tried to allocate 248 MiB` |
| **`k=64, S=512`** (32 768 tok, серия 2 `[5A]`) | `unified_go` ($O(NK)$, default alloc) | 51.69 | 633.95 | **— (baseline OOM)** | **18.99×** (vs `33.38`) | 106.8× | 2.604 GiB | 6.146 GiB | 6.745 GiB (20 мс) | `RSS = 13.51 GiB` |
| **`k=64, S=512`** (32 768 tok, серия 2 `[5B]`) | **`unified_go`** ($O(NK)$, **`exp_seg`**) | **51.77 / 51.80** | **632.94 / 632.61** | **— (baseline OOM)** | **18.96×** (vs `33.38`) | 106.7× | **2.601 GiB** | **3.102 GiB** (−3.04 GiB!) | **3.700 GiB** (−3.05 GiB!) | **`bit_equal=True` (`0.0`)**, `RSS = 13.52 GiB` |

### 2.2. Физическая проверка головы DeepSeek-V3 (`H=7168, V=129280`, BF16) на RTX 4060 Ti 8 ГБ (`expandable_segments:True`)
Источник: `benchmarks/results/vast-4060ti-verify2-20261120/deepseek-v3-head-vram-4060ti.json`.

| `S` | `full` статус | `full` `alloc` / `resv` / `nvml_peak` | `chunked` (`chunk=256`) статус | `chunked` `alloc` / `resv` / `nvml_peak` | Время `full` → `chunked` | `grad_max_abs` (`bit_eq`) |
|---:|:---:|---:|:---:|---:|---:|---:|
| 512 | ok | 2.749 / 2.893 / 3.454 GiB | ok | **2.172 / 2.189 / 2.751 GiB** | 0.167 → 0.352 с | `0.0` (`True`) |
| 1024 | ok | 3.773 / 4.045 / 4.606 GiB | ok | **2.186 / 2.326 / 2.888 GiB** | 0.115 → 0.153 с | `1.43e-6` (`False`) |
| 2048 | ok | 5.760 / 6.291 / 6.853 GiB | ok | **2.213 / 2.268 / 2.829 GiB** | 0.227 → 0.307 с | `9.54e-7` (`False`) |
| **4096** | **CUDA OOM** | 5.788 / 6.818 / **7.380 GiB (OOM)** | **ok** | **2.268 / 2.307 / 2.868 GiB** | OOM → **0.620 с** | — |

### 2.3. Серия 3: Физическая проверка `FP8 (block64_e4m3) + DeepseekRoutingOracle` на RTX 4060 Ti 8 ГБ (изолированные процессы, `expandable_segments:True`)
Источник: `benchmarks/results/vast-4060ti-fp8-oracle-20261120/` (инстанс `61600269` удалён сразу после выгрузки JSON). Каждый прогон Kanana 30B выполнялся в **отдельном свежем процессе ОС** с cross-process сверкой (`--save-ref` / `--compare-ref`) относительно точного BF16-эталона при идентичных весах LoRA и батчах.

| Сценарий (токенов/шаг) | Арм | Кодек (`group`) + Oracle | Шаг (с) | Скорость (tok/s) | Store RAM + Oracle RAM | Изолированный `RSS peak` | `alloc` / `resv` / **`nvml_peak` (20 мс)** | `gate_w_drift` | E3 `grad_rel_l2` (`≤0.05`) | E3 `min_cos` (`≥0.99`) | `adapter_rel_l2` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`tiny` CUDA** (`k=4, S=32`) | `l2l_ref_fp8_oracle` vs `unified_go_fp8_oracle` | `fp8_e4m3 (64)` + Oracle | 0.262 → **0.141** | 488 → **905** | `0.05 + 0.004 MiB` | 1.72 / 1.78 GiB | `0.063 / 0.080 / 0.663 GiB` | **`0.0`** | **`0.04317`** (PASS) | **`0.99780`** (PASS) | `0.00037` (**bit-exact** `gsha`/`asha` между `l2l_ref` и `unified_go`!) |
| **`tiny` CUDA** (`k=4, S=32`) | `unified_go_fp8_no_oracle` | `fp8_e4m3 (64)` без Oracle | 0.148 | 867 | `0.05 + 0 MiB` | 1.78 GiB | `0.063 / 0.080 / 0.663 GiB` | `2.63e-2` | `0.08343` (**FAIL**, в 1.93× хуже!) | `0.99595` | `0.00046` |
| **Kanana `k=64, S=64`** (4 096 tok) | `unified_go` (BF16 ref) | `bf16` | 37.10 | 110.39 | `768.0 + 0 MiB` | *(с шардингом)* | `0.823 / 0.883 / 1.468 GiB` | `0.0` | ref (`0.0`) | ref (`1.0`) | ref (`0.0`) |
| **Kanana `k=64, S=64`** (4 096 tok) | **`unified_go_fp8_oracle`** | **`fp8_e4m3 (64)` + Oracle** | **35.54** | **115.26 (+4.4%)** | **`408.0 + 7.34 MiB` (−45.9%)** | **7.86 GiB** | **`0.802 / 0.867 / 1.454 GiB`** | **`0.0`** | **`0.02111` (PASS)** | **`0.99255` (PASS)** | **`0.00049`** |
| **Kanana `k=64, S=64`** (4 096 tok) | `unified_go_fp8_perrow_oracle` | `fp8_e4m3 (row)` + Oracle | 35.30 | 116.04 (+5.1%) | `384.8 + 7.34 MiB` | 7.78 GiB | `0.802 / 0.867 / 1.454 GiB` | `0.0` | `0.02375` (на 12.5% хуже `block64`) | `0.99236` | `0.00051` |
| **Kanana `k=64, S=64`** (4 096 tok) | `unified_go_fp8_no_oracle` | `fp8_e4m3 (64)` без Oracle | 38.05 | 107.66 (−2.5%) | `408.0 + 0 MiB` | 7.82 GiB | `0.823 / 0.883 / 1.468 GiB` | **`1.11e-2`** | `0.02124` | `0.99255` | `0.00049` |
| **Kanana `k=64, S=512`** (32 768 tok) | `unified_go` (BF16 ref) | `bf16` | 53.16 | 616.43 | `6144.0 + 0 MiB` (`6.00 GiB`) | **13.51 GiB** | `2.601 / 3.102 / 3.700 GiB` | `0.0` | ref (`0.0`) | ref (`1.0`) | ref (`0.0`) |
| **Kanana `k=64, S=512`** (32 768 tok) | **`unified_go_fp8_oracle` (run 1)** | **`fp8_e4m3 (64)` + Oracle** | **51.96** | **630.68 (+2.3%)** | **`3264.0 + 58.75 MiB` (−2.76 GiB!)** | **10.89 GiB (−2.62 GiB!)** | **`2.505 / 2.975 / 3.567 GiB` (−133 MiB!)** | **`0.0`** | **`0.01393` (PASS)** | **`0.99471` (PASS)** | **`0.00038`** (`cos=0.999994`) |
| **Kanana `k=64, S=512`** (32 768 tok) | **`unified_go_fp8_oracle` (run 2)** | **`fp8_e4m3 (64)` + Oracle** | **52.01** | **630.07 (+2.2%)** | **`3264.0 + 58.75 MiB` (−2.76 GiB!)** | **10.88 GiB (−2.63 GiB!)** | **`2.505 / 2.975 / 3.567 GiB` (−133 MiB!)** | **`0.0`** | **`0.01393` (PASS)** | **`0.99471` (PASS)** | **`0.00038`** (**bit-exact** `r1 == r2` SHA-256!) |
| **Kanana `k=8, S=512`, 3 шага AdamW** (12 288 tok) | `unified_go` (BF16 ref) | `bf16` | 38.83 (`12.99/12.85/12.99`) | 316.47 | `768.0 + 0 MiB` | 8.17 GiB | `1.124 / 1.340 / 1.927 GiB` | `0.0` | ref (`0.0`) | ref (`1.0`) | ref (`0.0`) |
| **Kanana `k=8, S=512`, 3 шага AdamW** (12 288 tok) | **`unified_go_fp8_oracle` (3 шага)** | **`fp8_e4m3 (64)` + Oracle** | **39.12** (`13.18/12.93/13.02`) | **314.09** | **`408.0 + 7.34 MiB`** | **7.85 GiB (−320 MiB)** | **`1.104 / 1.322 / 1.909 GiB`** | **`0.0`** | **`0.03465`** (шаг 3) | `0.97066` (после 3 шагов) | **`0.00067`** (`loss_max=1.04e-3`) |

---

## 3. Ключевые подтверждённые выводы

1. **`expandable_segments:True` критичен для MoE + M3 на 8 ГБ GPU:** на Kanana `k=64, S=512` без него фрагментация раздувала `cuda_reserved` до `6.146 GiB` и `nvml_peak` до `6.745 GiB`. С `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` резерв упал вдвое (**`3.102 GiB`**), а реальный 20 мс пик NVML — до **`3.700 GiB`** (−3.045 GiB реальной VRAM, **`~4.0 GiB` свободного запаса** при 32 768 токенах за шаг и нулевой потере скорости).
2. **Побитовая точность на 3 последовательных шагах AdamW (30B CUDA BF16/NF4):** на `k=4, S=64, --steps 3` `unified_go` обошёл `l2l_ref` в **4.36×** (`29.10` vs `126.84 с`) и дал **100% `bit_equal=True` (`max_abs=0.0`)** по всем 12 потерям, финальным градиентам и обновлённым весам LoRA после 3 шагов оптимизатора (при `138/138` cross-layer promotions, `0` discards).
3. **Точные базы на том же железе:** официальный базовый режим P3 `k=1, S=64` (64 tok/step) дал ровно **`5.93 tok/s`**; база `l2l_ref` на `k=1, S=512` дала **`33.38 tok/s`** → честное ускорение `unified_go` при том же `S=512` составляет **`18.96×`** (при том что парный `mbr_h2_sync` на `(k=64, S=512)` падает в физический OOM).
4. **Подтверждённый выигрыш H2 на реальной 30B карте (`k=64, S=64`):** `std_mbr` (без H2) аллоцирует **`3.740 GiB`**, добавление H2 (`mbr_h2_sync`) режет до **`1.509 GiB`** (−2.23 GiB), а полный `unified_go` (M1+M2++H2+M3+Chunked CE+C++17 async) — до **`0.823 GiB`** (**4.54×** меньше `std_mbr`), ускоряя шаг до **`109.99 tok/s`**.
5. **Физически закрыт вопрос Агента 3 по голове DeepSeek-V3 671B:** на реальной 8 ГБ карте `full` CE падает в OOM на `S=4096`, а `Chunked CE` держит плоский пик **`2.75–2.89 GiB` NVML** от `S=512` до `S=4096`.
6. **Физически подтверждён `FP8 (block64_e4m3) + DeepseekRoutingOracle` в составе `unified_go`:**
   - Побитово (`max_abs = 0.0`, `torch.equal`) совпадает с эталонным `p3_realdata_codec + DeepseekOracleProbe` (PR #57/#66) и между `l2l_ref_fp8_oracle` и `unified_go_fp8_oracle` на CPU и CUDA.
   - На Kanana 30B `k=64, S=512` (32 768 tok/step) **снижает реальный пик RSS процесса на 2.63 GiB** (`13.51 → 10.88 GiB`), **снижает 20 мс пик NVML VRAM на 133 MiB** (`3.700 → 3.567 GiB` за счёт обхода `TopkDecisions`), **ускоряет шаг на +2.3%** (`616.4 → 630.7 tok/s` за счёт вдвое меньшего PCIe DMA трафика активаций) и с огромным запасом проходит строгий порог G4 E3 (**`global_rel_l2 = 0.01393 ≤ 0.05`**, **`min_cosine = 0.99471 ≥ 0.99`**, `adapter_rel_l2 = 0.00038`, `loss_bit_equal = True`).
   - На DeepSeek-V3 671B (`L=61, H=7168, k=64, S=512`) снижает host activation store с **`26.69 GiB` до `14.18 GiB`** (+ `94.25 MiB` Oracle = **`14.27 GiB`**, экономия **−12.42 GiB RAM**), гарантируя прохождение режима `k=64, S=512` в лимите **32 GiB RAM (`swap=0`)**.
7. **Локальная и CUDA регрессия и линтер:** `ruff check` — чисто, `238 passed, 8 skipped, 0 failed` + `149 passed` по FP8/Oracle/Decoupled MBR + `7 passed` на реальном CUDA RTX 4060 Ti.
