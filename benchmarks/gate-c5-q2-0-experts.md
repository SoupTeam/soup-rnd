# C5: LoRA over OLMoE experts in Q2_0 versus NF4

**The decision rule was committed before the first run. No runs yet. Data: a temporary set until the dataset protocol exists (временный набор до dataset протокола).**

Spec: `.scratch/c5-q2-0-experts/issues/03-rule-and-data.md`. Training under
this rule is ticket 04, cache size and weight restore cost are ticket 05, and
the final report and G5 recommendation are ticket 06.

## Commit order

- The rule below applies from the commit that added this file, `c7f2bfcd`.
  That commit's time must be earlier than the `started` field in the first
  arm's record. To check:
  `git log --diff-filter=A --format='%H %cI' -- benchmarks/gate-c5-q2-0-experts.md`.
  A rebase or squash rewrites both the SHA and the date, so the branch history
  from this commit on is not rewritten, and the PR merges with a merge commit.
- No Soup dataset protocol was found (repo, issues, PRs, discussions, Trello).
  The question went to the team lead on 2026-10-20. If a protocol appears
  before the first arm, the rule is amended in a separate commit before the
  first arm. After the first arm the data set does not change.
- Cutting the train split to 2000 rows (if the pilot takes over 2 hours per
  run) is also a separate amendment, committed before the first real arm. The
  hash of the cut split is listed below, so the amendment needs no new script.

## Decision rule

The Russian text below is copied from the spec unchanged and is the binding
version. An English translation follows it; where the two differ, the Russian
text wins.

Принято на grill 2026-10-20. Термины по `GLOSSARY.md`, обоснование fake-quant в `docs/adr/0001-c5-fake-quant-training.md`.

**Вопрос.** Теряет ли LoRA, обученная поверх экспертов OLMoE в Q2_0, не больше допустимого по task loss против той же LoRA поверх экспертов в NF4? Меньший размер файлов без прохождения quality gate успехом не считается. Точное равенство с NF4 не заявляется.

### Модель и данные

- Модель: `allenai/OLMoE-1B-7B-0924`, revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- Данные (временный набор до dataset протокола): `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, лицензия CC BY-SA 3.0.
- Строки сортируются по SHA-256 от номера строки и текста `instruction`. Первые 512 образуют held-out панель, следующие 4000 образуют train-сплит. Хеши обоих записываются.
- Длина последовательности 1024 токена, усечение одинаково во всех arms. Loss считается только по токенам ответа.
- Если пилот на стенде покажет больше 2 часов на один прогон, train-сплит урезается до первых 2000 строк того же порядка. Это поправка к правилу, она коммитится до первого настоящего arm.

### Arms

| Arm | Эксперты | Сиды | Роль |
|---|---|---|---|
| NF4 | NF4, fake-quant | 42, 7, 123 | контроль в gate |
| Q2_0 | Q2_0, fake-quant | 42, 7, 123 | кандидат в gate |
| bf16 | без квантования | 42 | справка, в gate не входит |

- Квантуются только веса экспертов. Router, attention, embeddings, norm и output head остаются bf16 во всех arms.
- Кодек Q2_0 из тикета 01: раскладка `GGML_TYPE_Q2_0` llama.cpp (блок 64, один fp16 scale, 2.25 бита на вес), scale подбирается поиском по MSE со знаком. SHA-256 файла кодека записывается. Кодек не меняется после первого arm.
- LoRA только на attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), rank 16, alpha 32. Одна эпоха, effective batch 16, lr 2e-4, cosine. Остальные гиперпараметры по умолчанию Soup на commit замера, они записываются в конфиг каждого прогона.
- В паре NF4 и Q2_0 с одним сидом одинаковы сид, порядок батчей и все настройки, кроме кодека экспертов.

### Метрики

- **Task loss**: NLL на токенах ответа по held-out панели, взвешенный по токенам.
- **Отношение** для пары с одним сидом: task loss(Q2_0) / task loss(NF4). Среднее берётся по трём сидам.
- **Интервал**: 95% paired bootstrap по примерам панели, 2000 ресэмплов, сид 17. Пример ресэмплируется целиком, токены внутри примера отдельно не ресэмплируются. Интервал описывает разброс выборки панели в одном сиде, а не разброс обучения.
- **Gate suites**: `mini_mmlu` (26 вопросов), `mini_common_sense` (24), `mini_instruction` (24), `mini_arithmetic` (40). Считается число верных ответов, среднее по трём сидам. Suite, у которого NF4 в среднем отвечает верно на все вопросы или ни на один, отчитывается, но в gate не входит (issue #1192).

### Validity rows

Каждая строка при нарушении даёт no verdict.

| Validity row | Как проверяется |
|---|---|
| Отпечатки не-экспертных весов совпадают во всех arms | JSON из тикета 02 |
| Хеши train-сплита и held-out панели совпадают во всех arms | хеши в записи каждого прогона |
| Конфиги arms отличаются только кодеком экспертов | diff конфигов |
| Каждый прогон дошёл до последнего шага без NaN и Inf в loss и grad norm | журнал тренера |
| Grad norm на первом шаге больше 0 | журнал тренера, как в `soup bench train` (#836) |

Проверки `suspend`, `power` и `foreign_reads` для обучающих arms не обязательны: скорость здесь не меряется. Прогон, нарушивший строку 4 или 5, сохраняется и повторяется с тем же сидом один раз. Если повтор тоже нарушает, вердикта нет.

### Вердикт

Применяется по порядку:

1. **No verdict**, если не выполнена хоть одна validity row.
2. **Fail**, если выполнено хотя бы одно:
   - среднее отношение по трём сидам больше 1.03;
   - верхняя граница интервала больше 1.05 хотя бы в одном сиде;
   - в gating suite Q2_0 в среднем отвечает верно на 3 вопроса меньше, чем NF4, или ещё хуже.
3. **Pass** во всех остальных случаях.

Границы включительные: 1.03 и 1.05 проходят.

### Что вердикт значит для G5

| Исход | Рекомендация |
|---|---|
| Pass | продолжать G5, даже все эксперты в Q2_0 допустимы |
| Fail, среднее отношение ≤ 1.10 | продолжать G5 только в смешанном виде, Q2_0 для экспертов с малым трафиком |
| Fail, среднее отношение > 1.10 | остановить G5, пока нет 2-битного кодека лучше Q2_0 с поиском scale: обучаемый scale как в GSQ или формат с кодбуком вроде IQ2_XS |
| No verdict | повторить замер, по G5 не решать |

Порог 1.10 — грубая оценка: при нём даже смешанная точность вряд ли переведёт в Q2_0 больше трети экспертов без выхода за 1.03. В отчёте она так и подписывается.

### Размер кеша и восстановление весов (тикет 05)

- **Размер**: байты файлов shard-кеша на диске, по слоям и в сумме, для экспертов в Q2_0 и в NF4. Число детерминированное, validity rows не нужны.
- **Восстановление**: время dequant одного слоя (64 эксперта × 3 проекции) из упакованного вида в VRAM в bf16. A-B-B-A и число блоков по `benchmarks/METHOD.md`. Обязательные проверки `suspend` и `power`. `foreign_reads` не обязательна, потому что диск в замере не участвует.
- Вердикта нет, только отчёт с числами, GPU и частотой SM.

### Записывается в каждый прогон

GPU и его UUID, драйвер, версии Python, torch, transformers, peft, bitsandbytes, datasets; commit репо; SHA-256 кодека и harness; хеши данных; сид; полный конфиг; журнал loss и grad norm по шагам; task loss по каждому примеру панели. Синтетические прогоны (granite на CPU) подписываются отдельно и в вердикт не входят.

## Decision rule, English translation

Translation only. The Russian text above is binding.

Agreed in a grill session on 2026-10-20. Terms follow `GLOSSARY.md`; the reason for fake-quant is in `docs/adr/0001-c5-fake-quant-training.md`.

**Question.** Does a LoRA trained over OLMoE experts in Q2_0 lose no more than the allowed amount of task loss against the same LoRA over experts in NF4? Smaller files without passing the quality gate do not count as success. Exact parity with NF4 is not claimed.

### Model and data

- Model: `allenai/OLMoE-1B-7B-0924`, revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- Data (temporary set until the dataset protocol): `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, license CC BY-SA 3.0.
- Rows are sorted by SHA-256 of the row number and the `instruction` text. The first 512 form the held-out panel, the next 4000 form the train split. The hashes of both are recorded.
- Sequence length is 1024 tokens, with the same truncation in every arm. Loss is computed on response tokens only.
- If a pilot on the box shows more than 2 hours per run, the train split is cut to the first 2000 rows of the same order. This is an amendment to the rule and is committed before the first real arm.

### Arms

| Arm | Experts | Seeds | Role |
|---|---|---|---|
| NF4 | NF4, fake-quant | 42, 7, 123 | control in the gate |
| Q2_0 | Q2_0, fake-quant | 42, 7, 123 | candidate in the gate |
| bf16 | no quantization | 42 | reference, not in the gate |

- Only expert weights are quantized. Router, attention, embeddings, norms and the output head stay bf16 in every arm.
- The Q2_0 codec is from ticket 01: llama.cpp `GGML_TYPE_Q2_0` layout (block of 64, one fp16 scale, 2.25 bits per weight), with the scale chosen by a signed MSE search. The SHA-256 of the codec file is recorded. The codec does not change after the first arm.
- LoRA on attention only (`q_proj`, `k_proj`, `v_proj`, `o_proj`), rank 16, alpha 32. One epoch, effective batch 16, lr 2e-4, cosine. Other hyperparameters are Soup defaults at the measured commit and are written into each run's config.
- Within an NF4 and Q2_0 pair with the same seed, the seed, batch order and every setting are the same except the expert codec.

### Metrics

- **Task loss**: NLL on response tokens over the held-out panel, weighted by tokens.
- **Ratio** for a same-seed pair: task loss(Q2_0) / task loss(NF4). The mean is taken over the three seeds.
- **Interval**: 95% paired bootstrap over panel examples, 2000 resamples, seed 17. Each example is resampled whole; tokens inside an example are not resampled separately. The interval describes sampling spread of the panel within one seed, not training spread.
- **Gate suites**: `mini_mmlu` (26 questions), `mini_common_sense` (24), `mini_instruction` (24), `mini_arithmetic` (40). The score is the number of correct answers, averaged over the three seeds. A suite where NF4 on average answers all questions or none correctly is reported but does not enter the gate (issue #1192).

### Validity rows

Each row, if violated, gives no verdict.

| Validity row | How it is checked |
|---|---|
| Non-expert weight fingerprints match in every arm | JSON from ticket 02 |
| Train split and held-out panel hashes match in every arm | hashes in each run's record |
| Arm configs differ only in the expert codec | config diff |
| Every run reached its last step with no NaN or Inf in loss and grad norm | trainer log |
| Grad norm at the first step is above 0 | trainer log, as in `soup bench train` (#836) |

The `suspend`, `power` and `foreign_reads` checks are not required for training arms, because speed is not measured here. A run that violates row 4 or 5 is kept and repeated once with the same seed. If the repeat also violates it, there is no verdict.

### Verdict

Applied in order:

1. **No verdict** if any validity row fails.
2. **Fail** if at least one holds:
   - the mean ratio over three seeds is above 1.03;
   - the upper bound of the interval is above 1.05 in at least one seed;
   - in a gating suite, Q2_0 on average answers 3 or more questions fewer correctly than NF4.
3. **Pass** in every other case.

Bounds are inclusive: 1.03 and 1.05 pass.

### What the verdict means for G5

| Outcome | Recommendation |
|---|---|
| Pass | continue G5; even all experts in Q2_0 are acceptable |
| Fail, mean ratio ≤ 1.10 | continue G5 only in mixed form, Q2_0 for low-traffic experts |
| Fail, mean ratio > 1.10 | stop G5 until there is a 2-bit codec better than Q2_0 with scale search: a learned scale as in GSQ, or a codebook format like IQ2_XS |
| No verdict | repeat the measurement; no G5 decision |

The 1.10 threshold is a rough estimate: at that ratio, even mixed precision is unlikely to move more than a third of the experts to Q2_0 without exceeding 1.03. The report labels it as such.

### Cache size and weight restore (ticket 05)

- **Size**: bytes of the shard cache files on disk, per layer and in total, for experts in Q2_0 and in NF4. The number is deterministic, so no validity rows are needed.
- **Restore**: time to dequantize one layer (64 experts × 3 projections) from packed form in VRAM to bf16. A-B-B-A and the block count follow `benchmarks/METHOD.md`. The `suspend` and `power` checks are required. `foreign_reads` is not required, because the disk takes no part in the measurement.
- No verdict, only a report with the numbers, the GPU and the SM clock.

### Recorded for every run

GPU and its UUID, driver, versions of Python, torch, transformers, peft, bitsandbytes and datasets; repo commit; SHA-256 of the codec and the harness; data hashes; seed; full config; per-step loss and grad norm log; task loss for every panel example. Synthetic runs (granite on CPU) are labelled separately and do not enter the verdict.

## Data

The script `benchmarks/harness/c5_data.py` builds both splits and prints their hashes:

```bash
# downloads the file at the pinned revision through huggingface_hub
python benchmarks/harness/c5_data.py --out-dir DIR
# or reads a local copy of the same file
python benchmarks/harness/c5_data.py --source databricks-dolly-15k.jsonl --out-dir DIR
```

It writes `heldout.jsonl`, `train.jsonl` and `manifest.json` to `DIR`. This
is how the script reads the rule:

- Source: the only data file, `databricks-dolly-15k.jsonl` at revision
  `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, 15011 rows, SHA-256
  `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`
  (the same as the LFS oid on Hugging Face). The script rejects a file with
  any other hash and exits with code 1.
- Row number: the position of the JSON line in the file, counting from 0.
- Sort key: the hex SHA-256 of `f"{row}\n{instruction}"` in UTF-8. The
  `instruction` text is used as is, without stripping whitespace. On equal
  keys the lower row number comes first.
- No rows are filtered, because the rule sets no filter. Neither split has an
  empty response.
- Split hash: the SHA-256 of `heldout.jsonl` or `train.jsonl`. Each file has
  one JSON line per example with the keys `category`, `context`,
  `instruction`, `response`, `row` in alphabetical order, no spaces, UTF-8
  without escaping, and every line ends in `\n`. Lines follow the sort order.

| Split | Rows | SHA-256 |
|---|---:|---|
| held-out panel | 512 | `b4b2ab938ca72a52c660d810045e3f31f842e3107e00738f827abb3f932d2a37` |
| train | 4000 | `a915bbbf18ebd1fae1dd869bdc8e2a0efbd2ab522d3a0c733cc05b212befa0b7` |
| train after the 2000-row amendment (`--train-rows 2000`) | 2000 | `806910717344ff5c4545dde76e5ae50367fb9d69365d59a11e23cf32463d7f7c` |

Repeat: on 2026-10-20, on macOS with Python 3.11.0, the script ran twice,
once from a local copy of the file and once downloading it through
huggingface_hub. `diff -r` of the output directories was empty. A third run
after review fixes gave the same hashes; its manifest is
`benchmarks/results/c5-data/manifest.json`. The manifest records the
script's SHA-256 at commit `c7f2bfcd`:
`f6fa50442cae84ceabf8fc1507717f0f606f64d2b2c72644c9405f34f44710fc`.

## Facts at commit `c7f2bfcd`

They do not change the rule.

- 7 of the 512 panel `instruction` texts (after stripping whitespace) also
  appear word for word in the train split. The rule does not remove
  duplicates.
- Panel categories: `open_qa` 127, `general_qa` 81, `classification` 80,
  `brainstorming` 61, `closed_qa` 56, `information_extraction` 51,
  `summarization` 38, `creative_writing` 18.
- Gate suites in `src/soup_cli/eval/forgetting.py`: `MINI_MMLU` 26 questions,
  `MINI_COMMON_SENSE` 24, `MINI_INSTRUCTION` 24, `MINI_ARITHMETIC` 40, as in
  the rule.
- SHA-256 of the codec `benchmarks/harness/q2_0_codec.py`:
  `fbaefa2a5efdf988fbc9dbb509de339b6ea6a56de3b1c08a1b43d5305f7917ac`.
