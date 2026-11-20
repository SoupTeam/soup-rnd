# DeepSeek P3 — Сравнение веток команды (`#74–#76` Алдияра, `#73` Батыра и `#77` `feat/p3-unified-go`)

**Дата:** 20 ноября 2026  
**Репозиторий:** `SoupTeam/soup-rnd`  
**Задача:** свести в одну таблицу три параллельных трека по P3, зафиксировать, где ветки дополняют друг друга, где конфликтуют в коде и в каком порядке собирать их в `main`.

---

## Главное за 30 секунд

Три ветки закрывают разные уровни одного стека и почти не дублируют работу друг друга:

- **Алдияр (`M1kado0`, PR `#74`, `#75`, `#76` — уже смержены в `#72` `p3-expert-windowing`):** решил задачу точности нативного cuBLAS (`A0`) и снял CPU/sync-накладные на `Moonlight-16B-A3B` (`k=8, S=64`, RTX 3060 12 ГБ, веса целиком в RAM-кэше). В `src/` добавлены 3 чистых opt-in механизма (`async_lease`, `fast_row_gather`, `mbr_ordered_backward`); в харнесе `a0_exact_lt` показал побитовое равенство с `A0` на 40 шагах и ускорение `−5.6% / −9.1%` к `B0` (`4.0×` к `A0`).
- **Батыр (`Batyrkhantzy111`, Draft PR `#73` поверх `#72`):** закрыл переносимый `batch_invariant` Triton GEMM прямо в `src/` (не зависит от версии PyTorch/cuBLAS, на `40.3%` быстрее сегментированного cuBLAS из `#75`, `0` расхождений на 40 шагах `soup train` на Moonlight 16B, RTX 4060 Ti 8 ГБ).
- **Мы (`kok-o`, Draft PR `#77` поверх `#73`):** сняли физический `CUDA OOM` и потолок RAM на реальной 8 ГБ карте при окне `k=64, S=512` (`32 768 tok/step`) за счёт `M1/M2+` Decoupled MBR, `H2`, `M3`, `O(N·K)` маршрутизации, `Chunked CE`, `FP8 block64 + DeepseekRoutingOracle` и C++17/CUDA `PackedStreamAsync`. Прогнали `Kanana-2-30B-A3B` (`630.7 tok/s`, `3.57 GiB` NVML, `10.88 GiB` RSS), `Qwen/Qwen3.8-27B` (`199.7 tok/s`, `4.95 GiB` NVML, `27.86 GiB` RSS), Agentic SFT До/После и контроль с Unsloth `2026.11.7`.

**Итог по сборке:** в `main` заходят `#72` (уже содержит `#74–#76`), `#63`, `#64`, поверх них ребейзится `#73` (+ наш фикс `tl` под Triton 3.2), а `#77` разбивается на 3 отдельных opt-in PR поверх общего `main` с использованием `fast_row_gather` и `mbr_ordered_backward` из `#76`.

---

## 1. Карта PR по P3 (состояние на 20 ноября 2026)

| PR | Автор | Статус | Ветка → База | Что именно делает |
|---|---|---|---|---|
| **#72** | `Artix6034` | open | `p3-expert-windowing` → `main` | Базовый route-preserving MBR в L2L |
| **#74** | `M1kado0` | merged в `#72` | `pr72-diagnostics` → `p3-expert-windowing` | Профилировка фаз MBR + 1-поточный `expert_prefetch` внутри слоя (`−11%` с диска) |
| **#75** | `M1kado0` | merged в `#74` → `#72` | `mbr-batch-invariance` → `pr72-diagnostics` | Диагностика дрейфа cuBLAS, `mbr_split_backward`, отсев segmented GEMM (`+43%`, NO-GO) |
| **#76** | `M1kado0` | merged в `#75` → `#72` | `mbr-triton-invariance` → `mbr-batch-invariance` | `async_lease`, `fast_row_gather`, `mbr_ordered_backward` в `src/` + харнес `a0_exact_lt` |
| **#73** | `Batyrkhantzy111` | draft | `feat/p3-a2-invariant-expert-gemm` → `p3-expert-windowing` | Triton `batch_invariant` GEMM в `src/` (`−40.3%` к segmented `#75`, `0` дрейфа на 40 шагах) |
| **#77** | `kok-o` | draft | `feat/p3-unified-go` → `feat/p3-a2-invariant-expert-gemm` | `M1/M2+/H2/M3`, `O(N·K)`, C++17/CUDA `PackedStreamAsync`, `Chunked CE`, `FP8+Oracle`, `Qwen3.8-27B` |
| **#63 / #64** | `nurseit14` | open | `research/zero-stream-chunked-ce`, `research/zero-l2l-qwen35` → `main` | `Chunked CE` и поддержка `qwen3_5` в L2L |

---

## 2. Прямое сравнение по осям

| Ось | Алдияр (`#74–#76`) | Батыр (`#73`) | Мы (`#77`) |
|---|---|---|---|
| **Решаемая проблема** | Убрать ~10% дрейф MBR и сделать точный MBR быстрее B0 с битами cuBLAS (`A0`) | Переносимый `batch_invariant` GEMM прямо в `src/` | Снять `CUDA OOM` и >32 ГБ RAM при `k=64, S=512`, ускорить NVMe-стриминг, проверить 27B–30B + SFT |
| **Стенд** | 2 хоста RTX 3060 12 ГБ (`sm_86`): Threadripper PRO и Xeon E5 | RTX 4060 Ti 8 ГБ (`sm_89`) + Tesla T4 (`sm_75`) | RTX 4060 Ti 8 ГБ (`sm_89`, `8188 MiB`, 20 мс поллер NVML) |
| **Модель и кэш** | `Moonlight-16B-A3B` (`9.3 ГБ` экспертов — целиком в RAM на замерах `#76`) | `Moonlight-16B-A3B` + микробенчи форм 30B и 671B | `Kanana-2-30B-A3B` (`3.5 GiB` кэш → чтение с NVMe), `Qwen/Qwen3.8-27B`, голова `671B` |
| **Макс. окно** | `k=8, S=64` (`512 tok/step`); SFT `8×128` (`1 024`) | `k=8, S=64` (`512 tok/step`) | **`k=64, S=512`** (`32 768 tok/step` — в 64× крупнее) |
| **Память (VRAM / RAM)** | Потолок памяти на больших `k, S` не меняется | Потолок памяти на больших `k, S` не меняется | `mbr_h2_sync` OOM → `3.57 GiB` NVML / `10.88 GiB` RSS (Kanana); `4.95 GiB` NVML / `27.86 GiB` RSS (Qwen 27B, `−14.88 GiB`); голова 671B `S=4096` → `2.87 GiB` |
| **I/O и CPU** | `#74` 1-поточный prefetch; `#76` `async_lease` (убрал `cuda.synchronize()`), `fast_row_gather` | Без изменений I/O | C++17 `PackedStreamAsync` (`io_workers=4, ring_slots=8`), 2-slot DMA, cross-layer `46/46`, `nf4_decode.cu` (ожидание `−5.7×`, `76%` H2D overlap) |
| **Точность / инвариантность** | Побитово равен `A0` на 40 шагах (`--deterministic`), но `a0_exact_lt` только в харнесе, привязан к PyTorch 2.14 и `H % 128 == 0` | Побитово равен L2L-Triton на 40 шагах `soup train` в `src/`, не зависит от PyTorch/cuBLAS | Наследует `#73` + фикс Triton 3.2 (`bit_equal` на 3 шагах 30B) + `FP8 block64 + Oracle` (`grad_rel_l2=0.0139`, `min_cos=0.9947`, `gate_w_drift=0.0`) |
| **Проверка SFT** | 40 шагов MetaMathQA (`160` прим.), 128 held-out GSM/MATH: `0.0` разницы NLL к `A0` (`0.2506914`) + парный bootstrap | 3 сида × 40 шагов MetaMathQA: `|ΔNLL| < 0.001` | Agentic Tool-Calling SFT До/После (`416` уник. задач `hermes-function-calling-v1`): Qwen `50% → 100%` (`$0.048`); Kanana JSON `0% → 81.25%`, Exact `0% → 62.5%` (`$0.024`) |
| **Гигиена PR** | 3 небольших opt-in PR (`#74→#75→#76`), влиты в `#72`, `A-B-C-C-B-A` + `ArmWatch` + t-тесты, `docs/` + `changelog.d/` | Чистый Draft PR по стадиям, `docs/` + `changelog.d/` | Один большой интеграционный Draft PR (`#77`); перед мержем в `main` нужно разбить на 3 PR + добавить `docs/` и `changelog.d/` |

---

## 3. Сводка замеров по веткам

*Абсолютные секунды между строками напрямую не сравниваются там, где разные карты (`RTX 3060 12 ГБ` vs `RTX 4060 Ti 8 ГБ`), модели (`16B` vs `27B` vs `30B`) и кэш (`9.3 ГБ` целиком в RAM vs `3.5 GiB` на 30B с NVMe).*

| Ветка / PR | Модель и стенд | Окно (`k×S` = tok) | Baseline → Результат | Память (VRAM / RAM) | Проверка точности / качества |
|---|---|---|---|---|---|
| **Алдияр `#74`** | Moonlight 16B, RTX 3060 12 ГБ | `8×64` (`512`) | `sync` `18.6–20.2 с` → `packed+prefetch` `16.6 с` (`−11%` с диска); в RAM-кэше `7.4 с` | `9.3 ГБ` в RAM | Побитовый дайджест градиентов совпал |
| **Алдияр `#76`** | Moonlight 16B, 2× RTX 3060 12 ГБ *(RAM-кэш)* | `8×64` (`512`) | `A0` `21.01 с` / `B0` `5.53 с` → `C (exact MBR)` **`5.22 с`** (`−5.6% / −9.1%` к `B0`, `4.0×` к `A0`) | `12 ГБ` карта | 40/40 шагов `sha256` адаптеров = `A0` (`--deterministic`), NLL `0.2506914` = `A0` |
| **Батыр `#73`** | Moonlight 16B, RTX 4060 Ti 8 ГБ | `8×64` (`512`) | L2L `81.0 с` → Triton MBR **`16.5 с`** (`4.89×`); против `#75` segmented: `14.19 → 8.29 с` (`−40.3%`) | `8 ГБ` карта | 40/40 шагов `soup train` `0` расхождений; 3 сида MetaMathQA `|ΔNLL| < 0.001` |
| **Мы `#77` (Kanana BF16)** | Kanana 30B-A3B, RTX 4060 Ti 8 ГБ *(3.5 GiB кэш)* | `64×512` (`32 768`) | `mbr_h2_sync` = **OOM** (`7.67 GiB`) → `unified_go` **`51.8 с`** (`632.9 tok/s`, `18.96×` к `k=1, S=512`) | `3.70 GiB` NVML (`2.60` alloc) / `13.52 GiB` RSS | `k=4, S=64` 3 шага AdamW: `4.36×`, `bit_equal=True` (`0.0`) |
| **Мы `#77` (Kanana FP8+Oracle)** | Kanana 30B-A3B, RTX 4060 Ti 8 ГБ *(изолир.)* | `64×512` (`32 768`) | BF16 `53.16 с` (`616.4 tok/s`) → **`51.96 с`** (`630.7 tok/s`, `+2.3%`) | `3.57 GiB` NVML (`−133 MiB`) / `10.88 GiB` RSS (`−2.63 GiB`) | `loss_bit_eq=True`, `gate_w_drift=0.0`, `grad_rel_l2=0.0139`, `min_cos=0.9947` |
| **Мы `#77` (Qwen 3.8 27B)** | Qwen3.8-27B (64 слоя), RTX 4060 Ti 8 ГБ | `64×512` (`32 768`) | BF16 `164.11 с` (`42.74 GiB` RSS) → FP8 **`164.06 с`** (`199.7 tok/s`) | `4.95 GiB` NVML / `27.86 GiB` RSS (`−14.88 GiB`) | `loss_bit_eq=True`, `grad_rel_l2=0.0058`, `min_cos=0.9978`, `adapter_cos=0.999995` |
| **Мы `#77` (Голова 671B)** | Форма DeepSeek-V3 (`H=7168, V=129280`) | `S=4096` | `full CE` = **OOM** (`7.38 GiB`) → `Chunked CE (256)` **`0.620 с`** | `2.87 GiB` NVML (`2.27` alloc) | `grad_max_abs ≤ 1.43e-6` на `S=512..2048` |
| **Мы `#77` vs Unsloth `2026.11.7`** | Qwen 27B & Kanana 30B (контроль `0.5B`), та же карта | `16×256` (`4 096`), `r=16` | Unsloth `0.5B` = `314.8 tok/s`; на **27B и 30B Unsloth падает** (`ValueError: raise_if_bnb_cpu_spill`), HF+BnB падает (`40.4–56.7 GiB` RSS) | Soup: `1.97–5.11 GiB` NVML, `7.85–13.24 GiB` RSS | Подробно в `reports/deepseek-p3-unsloth-comparison-2026-11-20.md` |

---

## 4. Как объединяем ветки в `main`

### 4.1. Что берём из каждого трека
- **Из `#76` (уже в `#72`):** `fast_row_gather` (`index_select` / `index_add_`), `mbr_ordered_backward`, `async_lease` (для fallback-пути без C++17-контроллера).
- **Из `#73`:** `Triton batch_invariant` GEMM + наш 5-строчный фикс ленивого `tl` под Triton 3.2 (остаётся основным переносимым точным GEMM в `src/`, т.к. `a0_exact_lt` из `#76` завязан на PyTorch 2.14 и `H % 128 == 0`).
- **Из `#77`:** `M1/M2+/H2/M3`, `O(N·K)` маршрутизацию, C++17/CUDA `PackedStreamAsync` + `nf4_decode.cu` + cross-layer pre-staging, `FP8 (block64_e4m3)` + `DeepseekRoutingOracle`, `expandable_segments:True`, `--train-head --head-lr`.

### 4.2. Точки конфликтов и порядок мержа
Основные пересечения — в `src/soup_cli/utils/l2l_step.py` (упорядоченный/двухфазный backward в `#75`/`#76`, `#73` и наш decoupled `sub_w=1` в `#77`) и в `src/soup_cli/utils/expert_stream.py` (`expert_prefetch`/`async_lease`/`fast_row_gather` vs `expert_controller`/`O(N·K)`).

Рекомендуемый порядок:
1. Влить `#72` (с `#74–#76`), `#63` (`Chunked CE`) и `#64` (`qwen3_5`).
2. Перебазировать `#73` на обновлённый `main` (+ фикс Triton 3.2).
3. Разбить `#77` поверх `main` на 3 отдельных opt-in PR (с `docs/` и `changelog.d/`):
   - **PR A:** Decoupled MBR (`M1/M2+/H2/M3`) + `O(N·K)` маршрутизация поверх `fast_row_gather` и `mbr_ordered_backward` из `#76`.
   - **PR B:** C++17 / CUDA `PackedStreamAsync` + `nf4_decode.cu` + cross-layer pre-staging.
   - **PR C:** `FP8 (block64_e4m3)` активации + `DeepseekRoutingOracle`.

---

## 5. Честно про ограничения и открытые риски по веткам

| Ветка | Что ещё не закрыто |
|---|---|
| **Алдияр (`#74–#76`)** | `a0_exact_lt` только в харнесе, привязан к дереву редукции PyTorch 2.14 и `H % 128 == 0`; под `--deterministic` преимущество `C` к `B0` пропадает (`+2.2% / −3.2%`); все замеры на 12 ГБ карте с тёплым 9.3 ГБ RAM-кэшем и окном `512` tok; не объяснён выброс `b0-ltfo` (`+16%` на Юте) и ~1% недетерминизм на `8×128` без `--deterministic`. |
| **Батыр (`#73`)** | Нужен ребейз на свежий `#72` (после `#75/#76` в `l2l_step.py`) и фикс `tl` под Triton 3.2; без `#77` падает в OOM на `k=64, S=512`. |
| **Мы (`#77`)** | Стенд Vast был с 62 GiB RAM, `1.4 GiB` swap и `overlay` (нужна финальная приёмка в cgroup 32 GiB `swap=0` на прямом NVMe); на образе не стоял `causal_conv1d` для Qwen3.8; агентный датасет `S=256` — компактный proof-of-learning (не замена BFCL `S=1024–2048`); в тесте Unsloth не проверялся `offload_layers="auto"`; нужно разбить `#77` на 3 PR и прогнать полный `pytest tests/`. |
