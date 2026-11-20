# DeepSeek P3 — Единая GO-ветка (`feat/p3-unified-go`)

**Дата:** 20 ноября 2026  
**База:** PR #73 (`25336300ef279a9080f0d451acfad60c8f206e58`, Triton `batch_invariant` GEMM)  
**Worktree:** `C:/Users/esenb/Desktop/soup-p3-unified-go`

---

## 1. Состав объединённого стека (`GO` из 3 параллельных треков)

| Компонент | Источник | Файлы в ветке | Эффект |
|---|---|---|---|
| **M1 + M2+ + H2 + M3** (decoupled sub-window `sub_w=1`, zero-materialization streaming backward без `WFE`/`dh_chunks`/`h_sorted`, ABA-safe window NF4 dequant, освобождение `LargeLayerBufferPool` 1.726 GiB на всех 61 слоях + фикс утечки графов при `mbr_group_size < k`) | Трек 1 (`feat/p3-decoupled-mbr`) | `src/soup_cli/utils/layer_stream_runtime.py`, `src/soup_cli/utils/expert_stream.py`, `src/soup_cli/utils/l2l_step.py` | **20.59×** меньше saved storage (`k=32`), 671B `k=64`: **1.51 GiB** (`S=64`) / **2.36 GiB** (`S=512`) против OOM 25–53 GiB; `0` чтений экспертов в recompute |
| **`PackedStreamAsync` (C++17 `io_workers=4, ring_slots=8`, 2-slot async DMA, cross-layer backward pre-staging, `nf4_decode.cu`) + `merged_invariant` (`batch_invariant` Triton)** | Трек 2 (`feat/p3-io-gemm-throughput`) + PR #73 | `src/soup_cli/utils/async_expert_controller.py`, `src/soup_cli/utils/_native_expert_async/{native_stream_async.cpp,nf4_decode.cu,build_ext.py}`, интеграция в `expert_stream.py` и `l2l_step.py` | **+75% NVMe** на хосте (2–3× ожидаемо на Linux `O_DIRECT` Gen4), **−93%** ожидания старта слоя, **0.82 мс/эксперт** на RTX 4060 Ti (39.8× vs `separate`), без материализации `h_sorted`/`dy_sorted` |
| **Chunked CE / CCE на L2L-голове** | Трек 3 (`codex/p3-deletion-and-cce`) + PR #63 | `src/soup_cli/utils/stream_chunked_ce.py`, `lora_config=None` в `layer_stream_runtime.py` | DeepSeek-голова `S=4096, V=129280`: логиты **6.90 → 0.43 GiB**, голова целиком **8.63 (OOM) → 2.16 GiB**, побитовые BF16-градиенты |

### Отсечённые направления (`NO-GO` — не включены в горячий путь)
- **VRAM Packed-NF4 Hotset на 671B** (Трек 2): 4 GiB VRAM вмещают лишь 1.12% экспертов → `<0.67%` ускорения.
- **CPU Cold-Expert / KTransformers при `k=64`** (Трек 2): $M^* = 0$, доля экспертов с $M \le 16$ равна `0.0%`.
- **`H-C1` split `gate_w`/`up_w`** (Трек 2): не нужен с `batch_invariant` (`merged_invariant`) и вносит 1 BF16 ULP сдвиг.
- **Sparse-Candidate Deletion Tuning** (Трек 3): не снижает NVMe-трафик, на 671B `active-core` дороже LoRA `r=1`, на `routed` (`p=0.2%`) даёт `8.93 GiB` OOM.

---

## 2. Локальная верификация (CPU + MSVC C++17)

- **Линтер:** `ruff check src/soup_cli tests benchmarks/harness/p3_*.py` → **All checks passed!**
- **Регрессия + новые сквозные тесты:** **`238 passed, 8 skipped, 0 failed` (62.75 s)** (`test_p3_unified_go.py`, `test_l2l_chunked_ce.py`, `test_p3_decoupled_subwindow.py`, `test_b13_l2l_moe.py`, `test_p3_mbr_window_boundaries.py`, `test_p3_mbr_route_summary.py`, `test_p3_l2l_recipe.py`, `test_b13_expert_backward.py`, `test_b1_expert_forward.py`).
- **Битовая точность всех 4 компонентов вместе:** `test_unified_go_stack_bitwise_parity_two_adamw_steps` подтвердил `torch.equal` (`diff == 0.0`) за 2 шага AdamW против plain Two-Read L2L (и `atol=1e-5` с Chunked CE), `recompute_fetches == 0`, `large_pool == 0 B`, `cross_layer_prefetched > 0`, `cross_layer_promotions > 0`.
- **Парный A–B–B–A (`benchmarks/results/p3-unified-go-tiny-abba-20261120.json`, NF4 BF16, `k=8, S=32`):**
  - `saved_storage_peak`: `2.274 MiB` (`std_mbr`) → **`0.457–0.462 MiB`** (`unified_go`, **4.92–4.97×** меньше).
  - `large_pool` в forward/backward слоях: `65536 B` → **`0 B`**.
  - `native_telemetry`: `io_workers=4`, `ring_slots=8`, `ring_immediate_ready=27`, `cross_layer_prefetched=8`, `cross_layer_promotions=1`, `cross_layer_discards=0`.
  - Воспроизводимость B1 vs B2: идентичные SHA-256 потерь, градиентов (`677b0a58...`) и адаптера (`68ee64d1...`); `grad_rel_l2 = 4.66e-8` к `std_mbr`.

---

## 3. Запуск на физическом стенде RTX 4060 Ti 8 ГБ + 32 GiB RAM + NVMe

```bash
# Kanana 30B-A3B: сборка C++17+CUDA NF4, тесты, A-B-B-A k=64 S=64 + стресс k=64 S=512
P3_WORK=/mnt/nvme/kanana bash benchmarks/harness/run_p3_unified_go_4060ti.sh

# DeepSeek-V3 671B (локальный чекпойнт)
P3_CASE=custom \
P3_MODEL_DIR=/mnt/nvme/deepseek-v3/weights \
P3_WORK=/mnt/nvme/deepseek-v3 \
P3_HOST_CACHE_GIB=18.0 \
bash benchmarks/harness/run_p3_unified_go_4060ti.sh
```
