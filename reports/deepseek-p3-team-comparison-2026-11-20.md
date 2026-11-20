# DeepSeek P3 — Сравнение командных веток (`#74–#76` Алдияра, `#73` Батыра и `#77` Unified GO)

**Дата:** 20 ноября 2026  
**Репозиторий:** `SoupTeam/soup-rnd`  
**Цель документа:** быстро и честно свести в одну картину три параллельные ветки по P3 — где сильна каждая из них, чем они отличаются по масштабу и железу, где пересекаются в коде и как собрать из них единый финальный стек без лишних потерь.

---

## Главное за 30 секунд

Три направления P3 **не конкурируют, а закрывают разные уровни одной системы**:

1. **Алдияр (`M1kado0`, PR `#74` → `#75` → `#76`, уже влиты в `#72` `p3-expert-windowing`)** закрыл **точность нативного cuBLAS (A0) и снятие CPU/sync-оверхеда** на малом окне (`Moonlight-16B`, `8×64`, RTX 3060 12 ГБ): убрал глобальный `cuda.synchronize()` на каждом эксперте (`async_lease`, −0.9..1.3 с), ускорил gather (`fast_row_gather`), упорядочил backward (`mbr_ordered_backward`) и в харнесе доказал **побитовое равенство с нативным A0 на 40 шагах быстрее B0** (−5.6% / −9.1%).
2. **Батыр (`Batyrkhantzy111`, Draft PR `#73` поверх `#72`)** закрыл **переносимый batch-invariant GEMM прямо в `src/`** (фиксированный Triton-кернел, который не зависит от версии PyTorch/cuBLAS, на 40.3% быстрее сегментированного cuBLAS из `#75` и держит 0 расхождений на 40 шагах `soup train` на Moonlight 16B RTX 4060 Ti).
3. **Мы (`kok-o`, Draft PR `#77` поверх `#73`)** закрыли **главный потолок P3 — масштаб на реальной 8 ГБ GPU (`k=64, S=512` = 32 768 токенов/шаг) и 32 ГБ RAM**: сняли физический `CUDA OOM` через 7 механизмов памяти (`M1/M2+/H2/M3/O(N·K)/Chunked CE/FP8 block64 + Oracle`), добавили C++17/CUDA `PackedStreamAsync` с cross-layer prefetch (`46/46` попаданий), прогнали **30B MoE (`630.7 tok/s`, `3.57 GiB` NVML, `10.88 GiB` RAM)** и **27B Dense/Hybrid `Qwen3.8-27B` (`199.7 tok/s`, `4.95 GiB` NVML, `−14.88 GiB` RAM)**, плюс показали реальный **Agentic SFT До → После за `$0.024–$0.048`**.

**Ключевой вывод:** объединение `#77` (память, масштаб, C++17/CUDA NVMe, `FP8+Oracle`, `Chunked CE`) с `fast_row_gather` и `mbr_ordered_backward` из `#76` (плюс `async_lease` для fallback-пути) даёт лучший результат по всем осям сразу.

---

## 1. Карта PR по P3 на 20 ноября 2026

| PR | Автор | Статус | Ветка → База | Главный вклад |
|---|---|---|---|---|
| **#72** | `Artix6034` | open | `p3-expert-windowing` → `main` | Базовый route-preserving MBR (окно экспертов в L2L) |
| **#74** | `M1kado0` | **merged в `#72`** | `pr72-diagnostics` → `p3-expert-windowing` | Профиль фаз MBR + 1-поточный `expert_prefetch` внутри слоя (`−11%` с диска) |
| **#75** | `M1kado0` | **merged в `#74` → `#72`** | `mbr-batch-invariance` → `pr72-diagnostics` | Диагностика дрейфа cuBLAS + `mbr_split_backward` + отсев segmented GEMM (`+43%`, NO-GO) |
| **#76** | `M1kado0` | **merged в `#75` → `#72`** | `mbr-triton-invariance` → `mbr-batch-invariance` | 3 рычага в `src/` (`async_lease`, `fast_row_gather`, `mbr_ordered_backward`) + харнес `a0_exact_lt` (быстрее B0 и побитово = A0) |
| **#73** | `Batyrkhantzy111` | open (draft) | `feat/p3-a2-invariant-expert-gemm` → `p3-expert-windowing` | Triton `batch_invariant` GEMM в `src/` (`bit-equal` с L2L-Triton на 40 шагах, `−40.3%` к segmented `#75`) |
| **#77** | `kok-o` | open (draft) | `feat/p3-unified-go` → `feat/p3-a2-invariant-expert-gemm` | Полный стек: `M1/M2+/H2/M3`, `O(N·K)`, C++17/CUDA `PackedStreamAsync`, `Chunked CE`, `FP8 block64 + Oracle`, `Qwen3.8-27B` и Agentic SFT |
| **#63 / #64** | `nurseit14` | open | `research/zero-stream-chunked-ce`, `research/zero-l2l-qwen35` → `main` | Chunked CE / CCE и допуск гибридного `qwen3_5` в L2L (использованы концептуально в `#77`) |

---

## 2. Прямое сравнение: Алдияр (`#74–#76`) vs Батыр (`#73`) vs Мы (`#77`)

| Измерение | Алдияр (`#74` + `#75` + `#76`) | Батыр (`#73`) | Мы (`#77`, `feat/p3-unified-go`) |
|---|---|---|---|
| **Решаемая проблема** | Убрать ~10% дрейф MBR и сделать точный MBR **быстрее B0 с битами нативного cuBLAS (A0)** | Сделать **переносимый batch-invariant GEMM** прямо в `src/` для обнуления дрейфа MBR | **Снять OOM на 8 ГБ GPU / 32 ГБ RAM** при `k=64, S=512`, ускорить NVMe-стриминг и проверить на 27B–30B + SFT |
| **Железо** | 2 хоста **RTX 3060 12 ГБ** (`sm_86`, L2 3 МБ): Threadripper PRO и Xeon E5 | **RTX 4060 Ti 8 ГБ** (`sm_89`) + Tesla T4 (`sm_75`) | **RTX 4060 Ti 8 ГБ** (`sm_89`, `8188 MiB`, 20 мс фоновый поллер NVML) |
| **Модели и кэш** | **Moonlight-16B-A3B** (`9.3 ГБ` экспертов — **полностью влезли в RAM** при замерах `#76`) | **Moonlight-16B-A3B** (плюс микробенчи форм Kanana 30B и DeepSeek 671B) | **Kanana-2-30B-A3B** (кэш `3.5 GiB` → **реальный NVMe-стриминг**), **Qwen/Qwen3.8-27B**, голова **DeepSeek 671B** |
| **Максимальное окно** | **`k=8, S=64`** (`512` tok/step); обучение `8×128` (`1 024` tok/step) | **`k=8, S=64`** (`512` tok/step) | **`k=64, S=512`** (**`32 768` tok/step** — в **64× крупнее**!) |
| **Память (VRAM / RAM)** | Не меняет потолок памяти при больших `k, S`: нет `M1/M2+/H2/M3`, `O(N·K)`, `Chunked CE`, `FP8+Oracle` | Не меняет потолок памяти при больших `k, S` | **Снимает физический OOM:** `k=64, S=512` → `3.57 GiB` NVML / `10.88 GiB` RAM (Kanana); `4.95 GiB` NVML / `27.86 GiB` RAM (Qwen 27B, `−14.88 GiB`); голова 671B `S=4096` → `2.87 GiB` |
| **I/O и CPU-слой** | `#74`: 1 поток prefetch внутри слоя; `#76`: **`async_lease`** (убрал `cuda.synchronize()`), **`fast_row_gather`** (`index_select`/`index_add_`) — чистые правки в `src/` | Без изменений I/O | **C++17 `PackedStreamAsync`** (`io_workers=4, ring_slots=8`, 2-slot async DMA, **cross-layer** `46/46` попаданий, **`nf4_decode.cu`** → ожидание `−5.7×`, `76%` H2D overlap) |
| **Точность / инвариантность** | Побитово равен **нативному cuBLAS A0** на 40 шагах (в `--deterministic`), но ядро `a0_exact_lt` только в `benchmarks/harness/` и привязано к дереву редукции **PyTorch 2.14** + `H % 128 == 0` | Побитово равен L2L-Triton на **40 шагах** `soup train` прямо в `src/`, не зависит от версии PyTorch/cuBLAS | Наследует Triton `batch_invariant` (`#73`) + фикс Triton 3.2 (`bit_equal` на 3 шагах 30B) + **`FP8 block64 + DeepseekRoutingOracle`** (`grad_rel_l2 = 0.0139`, `min_cos = 0.9947`, `gate_w_drift = 0.0`) |
| **Оценка качества SFT** | 40 шагов MetaMathQA (160 примеров), 128 отложенных GSM/MATH: доказал `0.0` разницы NLL с A0 (`0.2506914`) + парный bootstrap | 3 сида × 40 шагов MetaMathQA: доказал отсутствие деградации NLL (`|Δ| < 0.001`) | **Agentic Tool-Calling SFT До → После** (416 уник. задач `hermes-function-calling-v1`): Qwen 27B **`50% → 100%`** (`$0.048`); Kanana 30B **`0% → 81.25%` JSON / `62.5%` Exact** (`$0.024`, `1.97 GiB` NVML) |
| **PR-гигиена и статистика** | **Эталон:** 3 маленьких opt-in PR (`#74→#75→#76`), все **уже смержены в `#72`**, `A-B-C-C-B-A` + `ArmWatch` + t-тесты на 2 хостах, `docs/` + `changelog.d/` | Очень чистый Draft PR с разбиением на Stage 0–2b, `docs/` + `changelog.d/` | Большой интеграционный Draft PR (`#77`) для обзора всего стека целиком; перед мержем в `main` требует разбиения + `docs/` + `changelog.d/` |

---

## 3. Сводка ключевых замеров по веткам

*Важно: абсолютные секунды между столбцами напрямую не сравниваются там, где различаются GPU (`RTX 3060 12 ГБ` vs `RTX 4060 Ti 8 ГБ`), модели (`16B` vs `27B` vs `30B`) и объём кэша (`9.3 ГБ` целиком в RAM vs `3.5 GiB` на 30B с реальным чтением с NVMe).*

| Ветка / PR | Модель и стенд | Окно (`k×S` = tok) | Baseline → Результат | Память (VRAM / RAM) | Проверка точности / качества |
|---|---|---|---|---|---|
| **Алдияр `#74`** | Moonlight 16B, RTX 3060 12 ГБ | `8×64` (`512` tok) | `sync` `18.6–20.2 с` → `packed+prefetch` `16.6 с` (`−11%` с диска); с 12 GiB RAM-кэшем `7.4 с` (`−63%`) | RAM-кэш `9.3 ГБ` | Побитовый дайджест градиентов совпал |
| **Алдияр `#76`** | Moonlight 16B, 2× RTX 3060 12 ГБ *(9.3 ГБ в RAM)* | `8×64` (`512` tok) | A0 `21.01 с` / B0 `5.53 с` → **C (`exact MBR`) `5.22 с`** (`−5.6%` Юта, `−9.1%` Норвегия, **4.0×** к A0) | `12 ГБ` карта | **40/40 шагов** `sha256` адаптеров = A0 (`--deterministic`), NLL `0.2506914` = A0 |
| **Батыр `#73`** | Moonlight 16B, RTX 4060 Ti 8 ГБ | `8×64` (`512` tok) | L2L `81.0 с` → Triton MBR `16.5 с` (`4.89×`); против `#75` segmented: `14.19 → 8.29 с` (**−40.3%**) | `8 ГБ` карта | **40/40 шагов** `soup train` `0` расхождений; 3 сида MetaMathQA `|ΔNLL| < 0.001` |
| **Мы `#77` (Kanana BF16)** | Kanana 30B-A3B, RTX 4060 Ti 8 ГБ *(3.5 GiB кэш)* | `64×512` (**`32 768` tok**) | `mbr_h2_sync` = **OOM** (`7.67 GiB`) → **`unified_go` `51.8 с` (`632.9 tok/s`, 18.96× к `k=1, S=512`)** | **`3.70 GiB` NVML** (`2.60` alloc) / `13.52 GiB` RSS | `k=4, S=64` 3 шага AdamW: **4.36×**, **`bit_equal=True` (`0.0`)** |
| **Мы `#77` (Kanana FP8+Oracle)** | Kanana 30B-A3B, RTX 4060 Ti 8 ГБ *(изолир. процессы)* | `64×512` (**`32 768` tok**) | BF16 `53.16 с` (`616.4 tok/s`) → **`51.96 с` (`630.7 tok/s`, +2.3%)** | **`3.57 GiB` NVML** (`−133 MiB`) / **`10.88 GiB` RSS (`−2.63 GiB`)** | `loss_bit_eq=True`, `gate_w_drift=0.0`, **`grad_rel_l2=0.0139`**, **`min_cos=0.9947`** (PASS) |
| **Мы `#77` (Qwen 3.8 27B)** | Qwen3.8-27B (64 слоя), RTX 4060 Ti 8 ГБ | `64×512` (**`32 768` tok**) | BF16 `164.11 с` (`42.74 GiB` RAM) → **`FP8` `164.06 с` (`199.7 tok/s`)** | **`4.95 GiB` NVML** / **`27.86 GiB` RSS (`−14.88 GiB`!)** | `loss_bit_eq=True`, **`grad_rel_l2=0.0058`**, **`min_cos=0.9978`**, `adapter_cos=0.999995` |
| **Мы `#77` (Голова 671B)** | Форма DeepSeek-V3 (`H=7168, V=129280`), RTX 4060 Ti 8 ГБ | `S=4096` | `full CE` = **CUDA OOM (`7.38 GiB`)** → **`Chunked CE (256)` `0.620 с`** | **`2.87 GiB` NVML** (`2.27` alloc) | `grad_max_abs ≤ 1.43e-6` на `S=512..2048` |
| **Мы `#77` (Agentic SFT)** | Qwen 27B & Kanana 30B, RTX 4060 Ti 8 ГБ | `16` шагов × `16×256` (`65 536` tok) | Qwen **`6.85 мин` (`$0.048`)**; Kanana **`3.44 мин` (`$0.024`)** | Qwen `5.11 GiB` NVML; Kanana **`1.97 GiB` NVML** | Qwen Exact Action **`50% → 100%`**; Kanana Valid JSON **`0% → 81.25%`**, Exact **`0% → 62.5%`**, TokAcc **`77.6% → 100%`** |
| **Контроль: Unsloth `2026.11.7` vs Мы `#77`** | Qwen 27B & Kanana 30B (и `Qwen2.5-0.5B` контроль), та же RTX 4060 Ti 8 ГБ | `16×256` (`4 096` tok/шаг), `r=16`, 4-bit NF4 | На `0.5B` Unsloth = `314.8 tok/s` (`1.17 GiB` NVML); на **27B и 30B Unsloth падает на загрузке (`ValueError: raise_if_bnb_cpu_spill`)**, а HF+BnB CPU-offload падает (`56.7 GiB` RAM / `NotImplementedError` на 27B и `40.4 GiB` RAM / `CUDA OOM` на 30B) | Unsloth: **не помещается в 8 ГБ VRAM**; HF+BnB: **`40.4–56.7 GiB` RAM (>32 ГБ!)** vs Soup **`1.97–5.11 GiB` NVML** и **`7.85–13.24 GiB` RAM** | Soup `#77` обучает 30B MoE со скоростью **`317–631 tok/s`** на 8 ГБ GPU — быстрее, чем Unsloth крутит `0.5B` (`314.8 tok/s`) |

---

## 4. Главные инженерные выводы и план объединения

### 4.1. Что обязательно берём от каждого в общий финальный стек
1. **Из `#76` (Алдияр, уже в `#72`):**
   - **`StreamedExperts.fast_row_gather`** (`index_select` + `index_add_`) — безболезненно ускоряет gather/scatter на каждом эксперте.
   - **`L2LStep.run(..., mbr_ordered_backward=True)`** — чистый детерминированный порядок градиентов LoRA в одном вызове autograd.
   - **`ExpertBufferPool.async_lease`** — снимает 3 144 лишних `cuda.synchronize()` на синхронном/fallback-пути (когда C++17-контроллер выключен).
   - **Методология приёмки:** `A-B-C-C-B-A` + `ArmWatch` + `docs/` + `changelog.d/` + маленькие opt-in PR.
2. **Из `#73` (Батыр):**
   - **Triton `batch_invariant` GEMM** (с нашим 5-строчным фиксом ленивого `tl` под Triton 3.2) — остаётся основным переносимым точным GEMM в `src/`, в отличие от харнесного `a0_exact_lt`, завязанного на PyTorch 2.14.
3. **Из `#77` (мы):**
   - **Все 7 механизмов памяти и масштаба:** decoupled sub-window `M1`, zero-materialization backward `M2+`, `H2` window NF4 dequant, `M3` освобождение `LargeLayerBufferPool`, `O(N·K)` маршрутизация без `one_hot`, `Chunked CE` (синхронизировать с `#63`), **`FP8 (block64_e4m3)` + `DeepseekRoutingOracle`**.
   - **C++17 + CUDA `PackedStreamAsync` + `nf4_decode.cu` + cross-layer backward pre-staging** — основной быстрый I/O-движок для реального NVMe (где эксперты 30B/671B не влезают в RAM).
   - **Урок по `expandable_segments:True`** и **`--train-head --head-lr`** для base-моделей с неинициализированными спецтокенами (`<tool_call>`).

### 4.2. Где будут merge-конфликты и как их чисто закрыть
Поскольку `#74`, `#75` и `#76` уже влиты в `p3-expert-windowing` (`#72`), а `#73` и `#77` ещё сидят на более ранней точке `#72`:
- Основные точки пересечения — **`src/soup_cli/utils/l2l_step.py`** (двухфазный/упорядоченный backward в `#75`/`#76`, `#73` и наш decoupled `sub_w=1` в `#77`) и **`src/soup_cli/utils/expert_stream.py`** (`expert_prefetch`/`async_lease`/`fast_row_gather` в `#74`/`#76` и `expert_controller`/`O(N·K)` в `#77`).
- **Рекомендуемый порядок мержа в `main`:**
  1. Смержить `#72` (уже включает `#74–#76`), `#63` (`Chunked CE`) и `#64` (`qwen3_5`).
  2. Перебазировать `#73` (Triton `batch_invariant`) на обновлённый `main` (+ наш фикс Triton 3.2).
  3. Разбить `#77` поверх `main` на **3 чистых opt-in PR** с `docs/` и `changelog.d/`:
     - **PR A:** Decoupled MBR (`M1/M2+/H2/M3`) + `O(N·K)` маршрутизация (интегрировано с `fast_row_gather` и `mbr_ordered_backward` из `#76`).
     - **PR B:** C++17 / CUDA `PackedStreamAsync` + `nf4_decode.cu` + cross-layer pre-staging.
     - **PR C:** `FP8 (block64_e4m3)` активации + `DeepseekRoutingOracle`.

---

## 5. Честные ограничения и открытые риски (по всем веткам)

| Ветка | Что ещё не закрыто / важные оговорки |
|---|---|
| **Алдияр (`#74–#76`)** | `a0_exact_lt` живёт только в харнесе, привязан к дереву редукции PyTorch 2.14 и `H % 128 == 0`; в `--deterministic` преимущество C к B0 исчезает (`+2.2% / −3.2%`); замеры только на RTX 3060 12 ГБ (`sm_86`) с тёплым 9.3 ГБ RAM-кэшем и окном `512` tok; не объяснён выброс `b0-ltfo` `+16%` на Юте и источник ~1% недетерминизма на `8×128` без `--deterministic`. |
| **Батыр (`#73`)** | Требует ребейза на свежий `#72` (после влития `#75/#76` в `l2l_step.py`) и фикса `tl` под Triton 3.2; без `#77` не решает OOM на `k=64, S=512`. |
| **Мы (`#77`)** | Стенд Vast был с 62 GiB RAM, `1.4 GiB` swap и `overlay` FS (нужен финальный приёмочный прогон 671B в cgroup 32 GiB `swap=0` на прямом NVMe); в образе не стоял `causal_conv1d` для Qwen3.8; агентный датасет `S=256` — компактный proof-of-learning (нужен прогон длинного `S=1024–2048` BFCL и `-instruct` Kanana); нужно разбить `#77` на 3 PR и прогнать полный `pytest tests/`. |
