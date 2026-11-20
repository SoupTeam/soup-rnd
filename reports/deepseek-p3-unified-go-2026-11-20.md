# DeepSeek P3 — Единая GO-ветка (`feat/p3-unified-go`) и замеры на физическом RTX 4060 Ti 8 ГБ

**Дата:** 20 ноября 2026  
**База:** PR #73 (`25336300ef279a9080f0d451acfad60c8f206e58`, Triton `batch_invariant` GEMM)  
**Worktree:** `C:/Users/esenb/Desktop/soup-p3-unified-go` (локальная ветка `feat/p3-unified-go`, без пуша в remote)

---

## 1. Состав объединённого стека (`GO` из 3 параллельных треков)

| Компонент | Источник | Файлы в ветке | Эффект |
|---|---|---|---|
| **M1 + M2+ + H2 + M3** (decoupled sub-window `sub_w=1`, zero-materialization streaming backward без `WFE`/`dh_chunks`/`h_sorted`, ABA-safe window NF4 dequant, освобождение `LargeLayerBufferPool` 1.726 GiB на всех 61 слоях + фикс утечки графов при `mbr_group_size < k`) | Трек 1 (`feat/p3-decoupled-mbr`) | `src/soup_cli/utils/layer_stream_runtime.py`, `src/soup_cli/utils/expert_stream.py`, `src/soup_cli/utils/l2l_step.py` | **20.59×** меньше saved storage (`k=32`), физически снимает 8 ГБ OOM на `k=64, S=512`; `0` чтений экспертов в recompute |
| **`PackedStreamAsync` (C++17 `io_workers=4, ring_slots=8`, 2-slot async DMA, cross-layer backward pre-staging, `nf4_decode.cu`) + `merged_invariant` (`batch_invariant` Triton + фикс Triton 3.2 `tl` в `invariant_gemm.py`)** | Трек 2 (`feat/p3-io-gemm-throughput`) + PR #73 | `src/soup_cli/utils/async_expert_controller.py`, `src/soup_cli/utils/_native_expert_async/{native_stream_async.cpp,nf4_decode.cu,build_ext.py}`, `src/soup_cli/utils/invariant_gemm.py`, интеграция в `expert_stream.py` и `l2l_step.py` | **7.78×** (`k=8`) / **20.0×** (`k=64 S=64`) / **113.5×** (`k=64 S=512`) vs `l2l_ref`; `main_wait_worker` снижен в **5.7×** (`4.09 → 0.71 с`); **46/46** cross-layer promotions (`0` discards); до **76.2%** H2D overlap |
| **Chunked CE / CCE на L2L-голове** | Трек 3 (`codex/p3-deletion-and-cce`) + PR #63 | `src/soup_cli/utils/stream_chunked_ce.py`, `lora_config=None` в `layer_stream_runtime.py` | DeepSeek-голова `S=4096, V=129280`: логиты **6.90 → 0.43 GiB**, голова целиком **8.63 (OOM) → 2.16 GiB**, побитовые BF16-градиенты |

### Отсечённые направления (`NO-GO` — не включены в горячий путь)
- **VRAM Packed-NF4 Hotset на 671B** (Трек 2): 4 GiB VRAM вмещают лишь 1.12% экспертов → `<0.67%` ускорения.
- **CPU Cold-Expert / KTransformers при `k=64`** (Трек 2): $M^* = 0$, доля экспертов с $M \le 16$ равна `0.0%`.
- **`H-C1` split `gate_w`/`up_w`** (Трек 2): не нужен с `batch_invariant` (`merged_invariant`) и вносит 1 BF16 ULP сдвиг.
- **Sparse-Candidate Deletion Tuning** (Трек 3): не снижает NVMe-трафик, на 671B `active-core` дороже LoRA `r=1`, на `routed` (`p=0.2%`) даёт `8.93 GiB` OOM.

---

## 2. Реальные замеры на физическом стенде RTX 4060 Ti 8 ГБ (Vast.ai, инстанс удалён)

**Железо и стек:** `NVIDIA GeForce RTX 4060 Ti (8188 MiB, 7.65 GiB CUDA, SM 8.9)`, 62 GiB RAM, NVMe `~3.8 GB/s`, `torch 2.6.0+cu124`, `transformers 5.23.0`, `peft 0.22.0`, `bitsandbytes 0.50.2`, `triton 3.2.0`, `nvcc 12.4`.  
**Модель:** `kakaocorp/kanana-2-30b-a3b-base-2601` (`0d31365a`), NF4 (`no-double-quant`, `per_expert`), BF16, `host_cache = 3.5 GiB` (командный бюджет RAM-кэша для Kanana), LoRA `r=4` (`q_b_proj, kv_b_proj, o_proj`), `--expert-gemm batch_invariant` + `--cuda-nf4-decode`.  
**Сырые JSON:** `benchmarks/results/vast-4060ti-20261120/`.

| Прогон / Сценарий | Арм | Шаг (с) | Скорость (tok/s) | Ускорение vs `l2l_ref` | `cuda_alloc` peak | NVML peak | RSS | `recomp` reads | Побитовый паритет (loss / grad / adapter) |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Tiny CUDA** (`k=8, S=32`) | `l2l_ref` | 0.832 | 307.58 | 1.00× | 0.063 GiB | 0.647 GiB | 1.81 GiB | 0 | ref |
| **Tiny CUDA** (`k=8, S=32`) | **`unified_go_full_head`** | **0.256** | **1 001.87** | **3.26×** | 0.064 GiB | 0.647 GiB | 1.81 GiB | 0 | **`bit_equal=True` (`0.0`)** |
| **Tiny CUDA** (`k=8, S=32`) | **`unified_go`** (`chunk=8`) | **0.272** | **941.15** | **3.06×** | 0.064 GiB | 0.647 GiB | 1.81 GiB | 0 | grad/adapter **`bit_equal=True` (`0.0`)** |
| **Kanana 30B `k=8, S=64`** (512 tok) | `l2l_ref` (`python_sync`) | 93.225 | 5.49 | 1.00× | 0.752 GiB | 1.403 GiB | 7.40 GiB | 0 | ref (`30 969`×2 эксперт-чтений) |
| **Kanana 30B `k=8, S=64`** (512 tok) | **`unified_go_full_head`** | **11.981** | **42.73** | **7.78×** | 0.769 GiB | **0.897 GiB** | 7.50 GiB | 0 | **`bit_equal=True` (`0.0`)** (`5 564`×2) |
| **Kanana 30B `k=8, S=64`** (512 tok) | **`unified_go`** (`chunk=256`) | **12.104** | **42.30** | **7.70×** | 0.770 GiB | **0.893 GiB** | 7.52 GiB | 0 | **`bit_equal=True` (`0.0`)** |
| **Kanana 30B `k=64, S=64`** (4 096 tok, A–B–B–A) | `mbr_h2_sync` (A1 / A2) | 41.116 / 40.805 | 99.62 / 100.38 | 18.2× | 1.516 GiB | 2.23–2.34 GiB | 8.12–10.53 GiB | 0 | ref (`5 945`×2) |
| **Kanana 30B `k=64, S=64`** (4 096 tok, абляция) | `unified_go_qd1` (`QD=1, no CL`) | 40.419 | 101.34 | 18.5× | **0.828 GiB** | **1.356 GiB** | 10.44 GiB | 0 | **`bit_equal=True` (`0.0`)** |
| **Kanana 30B `k=64, S=64`** (4 096 tok, A–B–B–A) | **`unified_go`** (B1 / B2, `QD=4+CL`) | **37.249 / 37.240** | **109.96 / 109.99** | **20.0×** | **0.828 GiB** (−45%) | **1.358 GiB** (−0.98 GiB) | 8.25 GiB | 0 | **`bit_equal=True` (`0.0`)** |
| **Kanana 30B `k=64, S=512`** (32 768 tok, стресс 8 ГБ) | `mbr_h2_sync` (PR #72 + H2) | — | **CUDA OOM** | — | 6.680 GiB (OOM) | **7.673 GiB (OOM)** | — | — | `Tried to allocate 248 MiB` |
| **Kanana 30B `k=64, S=512`** (32 768 tok, стресс 8 ГБ) | **`unified_go`** (B1 / B2 / clean) | **52.55 / 52.61 / 52.58** | **623.5 / 622.9 / 623.1** | **113.5×** | **2.604 GiB** | 6.67 GiB* | 13.46 GiB | 0 | **`bit_equal=True` (`0.0`)** (`5 994`×2) |

### Ключевые инженерные находки с физического стенда
1. **100% побитовый паритет на реальных 30B CUDA BF16/NF4 весах:** на всех 3 сценариях Kanana (`k=8 S=64`, `k=64 S=64`, `k=64 S=512`) `unified_go` совпал побитово (`bit_equal=True`, `max_abs=0.0`) по `losses`, `gradients` и обновлённому AdamW-адаптеру с базой.
2. **Физическое подтверждение эффекта `QD=4 + cross_layer_prefetch` на Linux `O_DIRECT` NVMe (`k=64, S=64`):**
   - `main_wait_worker` упал в **5.7×** (`4.087 с` при `QD=1` → **`0.714 с`** при `QD=4 + CL`).
   - `ring_immediate_ready` вырос в **2.43×** (`4 243` → **`10 330`** из `11 890` аренд).
   - Cross-layer backward pre-staging сработал **на всех 46 переходах между 47 MoE-слоями** (`cross_layer_promotions = 46`, `cross_layer_discards = 0`).
   - На `k=64, S=512` `ring_immediate_ready = 11 601 / 11 988` (**96.8%** экспертов готовы в кольце до запроса GPU!), `main_wait_worker = 0.32–0.55 с` (<1% от шага 52.6 с), **`sampled_h2d_overlap_fraction = 74.0%–76.2%`**.
3. **Найден и исправлен реальный баг PR #73 на Triton 3.2 + Python 3.11:** в `src/soup_cli/utils/invariant_gemm.py` из-за `from __future__ import annotations` и локального импорта `tl` компилятор Triton 3.2 падал с `NameError('tl is not defined')`. Ленивая привязка `globals()["tl"] = tl` внутри `_kernel()` и `_row_dot_kernel()` решила проблему без top-level импорта `triton`.
4. **Фрагментация кэш-аллокатора PyTorch на `S=512` (`*`):** при живом пике `cuda_allocated = 2.604 GiB` `cuda_reserved` составил `6.070 GiB` (NVML `6.669 GiB`) из-за переменных размеров срезов `[M_e, H]` и реаллокации `LargeLayerBufferPool`. В `benchmarks/harness/run_p3_unified_go_4060ti.sh` добавлен `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (критично для DeepSeek-V3 671B на 8 ГБ).

---

## 3. Локальная верификация (CPU + MSVC)

- `ruff check src/soup_cli tests benchmarks/harness/p3_*.py` → **All checks passed!**
- Регрессия + сквозные тесты → **`238 passed, 8 skipped, 0 failed` (62.75 s)**.
- Парный A–B–B–A CPU (`benchmarks/results/p3-unified-go-tiny-abba-20261120.json`) → `saved_storage_peak` снижен в **4.97×** (`2.274 → 0.457 MiB`).
