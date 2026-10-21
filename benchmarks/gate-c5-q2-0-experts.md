# C5: LoRA over OLMoE experts in Q2_0 versus NF4

**The decision rule was committed before the first run. Ticket 04 ran all seven arms on 2026-10-20: the quality gate fails (mean ratio 1.105), and the rule's table says stop G5. Data: a temporary set until the dataset protocol exists (временный набор до dataset протокола).**

Spec: `.scratch/c5-q2-0-experts/issues/03-rule-and-data.md`. Training under
this rule is ticket 04, cache size and weight restore cost are ticket 05, and
the final report and G5 recommendation are ticket 06 (the last section).

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

## Runs under the rule (ticket 04)

Verdict: **fail**. Recommendation by the rule's table: **stop G5** until there
is a 2-bit codec better than Q2_0 with scale search (a learned scale as in
GSQ, or a codebook format like IQ2_XS). The 1.10 threshold behind that row is a
rough estimate, as the rule says. The final report and the G5 decision record
are the last section (ticket 06).

Spec: `.scratch/c5-q2-0-experts/issues/04-lora-nf4-vs-q2-0.md`. Raw records,
logs and commands: `benchmarks/results/c5-train/`. Verdict:
`benchmarks/results/c5-train/verdict.json`, computed by
`benchmarks/harness/c5_verdict.py` with both ticket-02 JSONs as
`--swap-record`.

### Box and code

- Vast.ai instance 56934950, NVIDIA GeForce RTX 3090
  `GPU-2aee27a4-1132-cfe8-518f-b8623de1c4eb`, driver 580.95.05, CUDA 13.0,
  max SM clock 2100 MHz, 24576 MiB. Rented 18:33 to 20:57 UTC.
- Commit `ff6035cf`, clean working tree in every run. SHA-256 of
  `c5_train.py` `c9ad559b…`, `c5_expert_swap.py` `e2ae3beb…`, codec
  `q2_0_codec.py` `fbaefa2a…` (the value listed above, the same in every run).
- Python 3.10.12, torch 2.14.1+cu130, transformers 5.20.2, peft 0.21.2,
  bitsandbytes 0.50.2, datasets 5.1.0, tokenizers 0.23.3, soup-cli 0.75.0.
  Full list: `pip-freeze.txt`.
- Commit order: `c7f2bfcd` was committed at 17:52:20 UTC; the first arm
  (`nf4-s42-a1`) started at 19:00:45 UTC.

### Choices the rule leaves open

Fixed in `c5_train.py` before the first arm and recorded in each run's config.

- OLMoE-0924 ships no chat template. Data goes through Soup: the Dolly row
  becomes an alpaca row (`context` as input), and `build_format_row` with
  `chat_template: chatml` gives the input ids and the response-only labels.
- Task loss counts the answer tokens only: tokens whose first character is in
  the answer text. The `<|im_start|>assistant` header and the `<|im_end|>`
  marker are 9 and 7 plain tokens on this tokenizer and are left out.
- One epoch over 4000 rows is 250 steps of 16 examples (micro batch 4,
  accumulation 4), sequence length 1024. AdamW, lr 2e-4, cosine, warmup 0.03,
  weight decay 0.01, grad clip 1.0, LoRA r 16, alpha 32, dropout 0.05 on
  `q_proj`, `k_proj`, `v_proj`, `o_proj`. Gradient checkpointing is on in
  every arm (see the pilot). The batch order depends only on the seed.
- Gate suites run through Soup's `score_bundled_suite` and `make_generator`
  (greedy, 64 new tokens). OLMoE does not stop at `<|im_end|>`, which is not a
  special token here, so an output repeats its answer until 64 tokens. Every
  output is stored in its run record.

### Pilot

Not arms; they never enter the verdict. Records in `pilot/`.

- `pilot/nf4-s42`: micro batch 4 without gradient checkpointing ran out of
  CUDA memory in the first forward pass.
- `pilot/nf4-s42-gc`: the same with gradient checkpointing completed in 14.6
  minutes (training 495 s). A run is far under 2 hours, so the train split
  stays at 4000 rows and the rule needs no amendment.

### Results

Every run completed all 250 steps with finite loss and grad norm. No run was
spoiled or repeated. Panel: 512 examples, 38178 answer tokens, one example
with none left after truncation. Train: 4000 rows, 55 truncated, 36 with no
answer tokens left.

| Run | Started (UTC) | Task loss | Step-1 grad norm | mmlu /26 | common_sense /24 | instruction /24 | arithmetic /40 |
|---|---|---:|---:|---:|---:|---:|---:|
| NF4 seed 42 | 19:00:45 | 1.6393 | 1.129 | 12 | 10 | 23 | 27 |
| Q2_0 seed 42 | 19:15:22 | 1.8123 | 0.624 | 5 | 3 | 22 | 24 |
| NF4 seed 7 | 19:33:58 | 1.6393 | 0.728 | 11 | 9 | 22 | 28 |
| Q2_0 seed 7 | 19:48:31 | 1.8126 | 0.548 | 4 | 4 | 22 | 24 |
| NF4 seed 123 | 20:07:08 | 1.6408 | 0.434 | 15 | 7 | 23 | 27 |
| Q2_0 seed 123 | 20:21:49 | 1.8108 | 0.396 | 2 | 4 | 22 | 24 |
| bf16 seed 42 (reference) | 20:40:33 | 1.6288 | 1.298 | 10 | 11 | 24 | 27 |

NF4 seeds 42 and 7 agree to 1.639324 and 1.639326. Their batch orders,
training losses and per-example panel NLL differ; only the panel totals
coincide.

| Rule check | Limit | Measured | Result |
|---|---|---|---|
| Mean ratio over three seeds | ≤ 1.03 | 1.1049 (seeds 42, 7, 123: 1.1055, 1.1057, 1.1036) | fail |
| Upper bound of the 95% interval, every seed | ≤ 1.05 | 1.1184, 1.1194, 1.1160 (lower 1.0946, 1.0945, 1.0932) | fail |
| `mini_mmlu` mean drop | < 3 | 12.67 to 3.67, drop 9.00 | fail |
| `mini_common_sense` mean drop | < 3 | 8.67 to 3.67, drop 5.00 | fail |
| `mini_arithmetic` mean drop | < 3 | 27.33 to 24.00, drop 3.33 | fail |
| `mini_instruction` mean drop | < 3 | 22.67 to 22.00, drop 0.67 | pass |

All four suites gate: NF4 answers neither all nor none of any suite on
average. All five validity rows hold: non-expert fingerprints match across the
seven runs and both ticket-02 JSONs, split hashes match, configs differ only in
`codec` and `seed`, and every run is finite with a positive first grad norm.
The analyzer also checks the codec file hash: the registered rule freezes the
codec after the first arm, so differing `codec_sha256` values give no verdict.
All seven runs share one hash (`codec_unchanged` is true in `verdict.json`).
Within each codec the expert weights were identical on every seed (expert
digest `84ae9585…` for NF4, `090f4fbd…` for Q2_0).

bf16 is reference only. NF4 task loss is 1.0064 times bf16's on seed 42.

### Synthetic run

`benchmarks/results/c5-train-synthetic/` holds the seven arms on the granite
fixture on CPU (macOS, Python 3.11.0, commit `ff6035cf`), with 64 train and 32
panel rows of the real splits and a byte-level tokenizer. It is labelled
SYNTHETIC in every record and in its `verdict.json`. It shows that the
harness and the verdict run end to end. Its numbers say nothing about OLMoE.

## Cache size and restore cost (ticket 05)

No verdict, as the rule says: this section is numbers only. The quality gate
above failed, so a smaller cache does not count as a success. Spec:
`.scratch/c5-q2-0-experts/issues/05-cache-size-and-restore-cost.md`. Raw
records, logs and commands: `benchmarks/results/c5-cache/`. Code:
`benchmarks/harness/c5_cache.py` (SHA-256 `e03ba21e…`), codec `fbaefa2a…`.

### How the caches were made

The shipped sharder (`soup_cli.utils.layer_shard.shard_checkpoint`) ran
unchanged and wrote the bf16 cache. It stacks OLMoE's per-expert weights into
`mlp.experts.gate_up_proj [64, 2048, 2048]` and `mlp.experts.down_proj
[64, 2048, 1024]` and does not quantize them: its NF4 path covers unstacked
decoder keys only. The harness copied that cache once per codec and replaced
those two tensors in each of the 16 layer files with their packed form, one
expert at a time as in ticket 04. Every other tensor and file is a byte copy.
Two checks passed in the build record: the non-expert tensors are identical
in all three caches, and each packed cache restores to exactly the experts of
ticket 04 (`expert_digest` `84ae9585…` for NF4, `090f4fbd…` for Q2_0).

Soup's runtime has no packed-expert path and cannot load these caches. They
exist to be measured. The shipped package and schema are unchanged.

### Box

Vast.ai instance 56953759, NVIDIA GeForce RTX 3090
`GPU-101261c3-8f7f-798b-267d-53e6d2b89f41`, driver 580.159.03, CUDA 13.0, max
SM clock 2100 MHz, 24576 MiB. Measured on 2026-10-20 (build 21:47 UTC, timing
21:54 to 22:01 UTC) at commit `0c0ad5f2`, clean tree. Python 3.10.12, torch
2.14.1+cu130, transformers 5.20.2, bitsandbytes 0.50.2, safetensors 0.8.0 (full list in
`pip-freeze.txt`). Model `allenai/OLMoE-1B-7B-0924` at
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`.

### Size on disk

Bytes of the files in each cache folder (`os.stat`), from `cache.json`. All
three folders hold 16 layer files, `extras.safetensors`, `index.json` and the
two vocabulary shards (`large_embed_tokens`, `large_lm_head`, 206.0 MB each,
bf16 in every variant).

| Cache | Total on disk | Expert tensors | Everything else | One layer file |
|---|---:|---:|---:|---:|
| bf16 (shipped sharder) | 13,838,342,541 | 12,884,901,888 | 953,440,653 | 839,140,408 |
| NF4 experts | 4,276,925,021 | 3,323,469,824 | 953,455,197 | 241,551,736 |
| Q2_0 experts | 2,765,385,229 | 1,811,939,328 | 953,445,901 | 147,080,576 |

All 16 layers have the same size in each variant, so the per-layer figure is
one number; the 16 values are in `cache.json`. Per weight, the expert tensors
come to 16.00 bits (bf16), 4.127 bits (NF4, the ticket-01 figure) and 2.250
bits (Q2_0).

Q2_0 against NF4: the whole cache is 64.7% of NF4's (2,765,385,229 /
4,276,925,021), the expert tensors alone 54.5%. Against bf16, NF4 is 30.9% and
Q2_0 is 20.0% of the cache. The 953 MB that is not experts (attention, router,
norms, embeddings, head) is the same in all three, which is why the whole-cache
ratio is higher than the expert-only one.

### RAM and SSD for this model

SSD: the "Total on disk" column. RAM: a RAM tier that holds the whole store
keeps the data bytes of every tensor, `ram_store_bytes` in the record, which
is 18,829 to 32,285 bytes under the file size (the safetensors headers and
`index.json` (1,261 bytes) are not tensor data): 13,838,323,712 (bf16), 4,276,892,736 (NF4),
2,765,361,152 (Q2_0). Restoring a layer in VRAM also needs working space
beyond these figures. Peak CUDA memory allocated by a restore run (packed layer
plus bf16 output plus intermediates, every run the same): 1,021,413,888 bytes
for NF4 and 2,545,942,528 bytes for Q2_0, whose path uses float32
intermediates (below).
The box had 540 GB of RAM, so no RAM tier decision was tested.

### Restore time

Only layer 0 was timed; the 16 layers have the same shapes and the same
packed size, and equal restore time is assumed, not measured. One run is a
fresh process that loads layer 0's packed experts into VRAM and
times 30 restores (after 5 warm-up) of all 64 experts of both tensors to bf16,
and of expert 0 alone, with a CUDA synchronize around each. A run's value is
the median of its 30 timed steps. The repeats rule reads the layer value.

- A-B-B-A, A = NF4 and B = Q2_0, 21 blocks of 4 runs, 84 runs, all 21 blocks
  valid. The 5 pilot blocks gave N = 21, so `blocks_to_add` asked for
  21 blocks. The recount over all 21 gives N = 20 (sigma 0.301 ms, mean of A
  1.918 ms), so nothing more was due. No run was void.
- Checks: `suspend` ok and `power` ok in all 84 runs (the box has no battery,
  so `power` is ok with the reason "no battery", which is the rule's reading for
  a machine without one and a weaker signal than a mains reading). `foreign_reads` is unknown in
  all 84 and is not required by the rule.
- Box stamp before and after each arm is in `restore.json`.

| One layer (64 experts, gate/up and down) | NF4 | Q2_0 |
|---|---:|---:|
| mean of 42 runs | 1.918 ms | 17.351 ms |
| median of 42 runs | 1.844 ms | 17.381 ms |
| min to max | 1.424 to 2.521 ms | 17.111 to 17.540 ms |
| expert 0 alone, median | 0.293 ms | 0.431 ms |

The one-expert row is a separate call for expert 0, not the layer time divided
by 64. For NF4 it is about ten times the layer time per expert (1.918 / 64 =
0.030 ms), because a single call is dominated by launch overhead.

Mean block difference (A minus B): -15.43 ms, standard deviation 0.27 ms over
21 blocks. Measured on this box, Q2_0 takes 17.35 ms and NF4 1.92 ms to restore
one layer, which is 9.0 times longer. That is the measured ratio of two
implementations, not of the two formats:

- NF4 restores through bitsandbytes' fused CUDA kernel (`dequantize_4bit`).
- Q2_0 has no kernel in this harness. `_restore_q2_0` is plain PyTorch
  (shift, mask, subtract, multiply on float32 tensors), so it writes and reads
  float32 intermediates several times larger than the packed data.

A fused Q2_0 kernel would move this number, and by how much is unmeasured.
Whether restore cost matters in a product path is not answered here: ADR 0001
keeps the product runtime out of C5.

SM clock (from `gpu_clock` in each run's record, one sample every 2 s, so a run
of about 4 s has 3 to 5 samples): median of the per-run medians 1695 MHz in
both arms (A 1230 to 1838, B 1695 to 1980), maximum seen 1980 of 2100 MHz.
Clock event reasons: `gpu_idle` in 46 runs (the process start and end fall in
the window) and `sw_power_cap` in 10. The clock was not locked and ran below the
maximum, so absolute times are for this box and will differ on another.
Throttling is on record and does not void an arm.

### Failed starts

Two launches ended before any timing and are kept, as the rule keeps every
record: `run-attempt1.log` and `run-attempt2.log` in `benchmarks/results/c5-cache/`.

1. At commit `c5f4ae38` the model download dropped (`Server disconnected`) at
   about 13 of 13.8 GB. `run.sh` did not stop, so the timing step ran against
   caches that did not exist and failed. `run.sh` now uses `set -e`.
2. The download then finished, but the shipped sharder skips symlinked weight
   files and the Hugging Face cache stores every file as a symlink, so it found
   no weights. The harness now hard-links the snapshot into a plain folder
   first (`0c0ad5f2`). The tiny-model tests had used plain files and did not
   show this.

Neither produced a number. The run above is the third launch.

### Synthetic run

`c5_cache.py build --fixture olmoe` and `compare --synthetic` run end to end on
a tiny random OLMoE on CPU (macOS). Its sizes and times say nothing about
OLMoE and are not kept in `benchmarks/results/`.

## Report and G5 decision (ticket 06)

Spec: `.scratch/c5-q2-0-experts/issues/06-report-and-g5-decision.md`. This
section adds no measurement. It restates tickets 04 and 05 under the rule.

### Recommendation

**Stop G5 until there is a 2-bit codec better than Q2_0 with scale search,
because the quality gate failed with a mean task-loss ratio of 1.105, above
the 1.10 line.** The 1.10 line is a rough estimate, as the rule says, and the
mean sits 0.005 above it, so this row of the table is close to the neighbouring
one ("continue G5 only in mixed form"). The rule fixed the thresholds before
the first run and they were not moved.

### The rule, verbatim

The binding text is the Russian rule at the top of this file, copied from the
spec unchanged in commit `c7f2bfcd` (17:52:20 UTC, 2026-10-20), before the
first arm (19:00:45 UTC). The English translation above it is not binding.
Verdict clause, in the binding wording:

> **Fail**, если выполнено хотя бы одно: среднее отношение по трём сидам больше 1.03; верхняя граница интервала больше 1.05 хотя бы в одном сиде; в gating suite Q2_0 в среднем отвечает верно на 3 вопроса меньше, чем NF4, или ещё хуже.

> Fail, среднее отношение > 1.10 → остановить G5, пока нет 2-битного кодека лучше Q2_0 с поиском scale: обучаемый scale как в GSQ или формат с кодбуком вроде IQ2_XS

### Quality (ticket 04, real runs on OLMoE-1B-7B, RTX 3090)

Verdict **fail**, computed by `benchmarks/harness/c5_verdict.py` under the
rule, no threshold changed. Equality with NF4 is not claimed: Q2_0 is worse on
task loss in every seed and scores lower on average in all four gate suites
(the drop passes the limit only in `mini_instruction`).

| Check | Limit | Measured | Result |
|---|---|---|---|
| Mean ratio, 3 seeds | ≤ 1.03 | 1.1049 | fail |
| Upper bound of 95% interval, worst seed | ≤ 1.05 | 1.1194 | fail |
| `mini_mmlu` drop | < 3 | 9.00 | fail |
| `mini_common_sense` drop | < 3 | 5.00 | fail |
| `mini_arithmetic` drop | < 3 | 3.33 | fail |
| `mini_instruction` drop | < 3 | 0.67 | pass |

Task loss, NF4 against Q2_0 (seeds 42, 7, 123): 1.6393 / 1.6393 / 1.6408
against 1.8123 / 1.8126 / 1.8108. The interval describes the spread of the
512-example panel within one seed, not training variance. Three seeds is a small
sample, so the seed-to-seed spread of the ratio (1.1036 to 1.1057) is only a
rough guide. bf16 (seed 42, 1.6288) is reference only and not in the gate.

All five validity rows hold, and so does the codec hash check. No run was spoiled or repeated, so the
spoiled-run list is empty. The only discarded runs are the two pilot runs
(`pilot/nf4-s42`, out of CUDA memory without gradient checkpointing, and
`pilot/nf4-s42-gc`), which are not arms.

### Size and restore (ticket 05, real measurement, no verdict)

The smaller cache is not presented as a success, because the quality gate
failed. Numbers only, measured on 2026-10-20 on an RTX 3090
(`GPU-101261c3-8f7f-798b-267d-53e6d2b89f41`, driver 580.159.03, SM clock not
locked, median 1695 MHz of 2100 MHz maximum):

| Cache | Total on disk | Expert tensors | One layer file |
|---|---:|---:|---:|
| bf16 | 13,838,342,541 B | 12,884,901,888 B | 839,140,408 B |
| NF4 experts | 4,276,925,021 B | 3,323,469,824 B | 241,551,736 B |
| Q2_0 experts | 2,765,385,229 B | 1,811,939,328 B | 147,080,576 B |

Q2_0 cache bytes are 64.7% of the NF4 cache bytes (experts alone: 54.5%).
Restoring one layer (64 experts, both tensors) to bf16, 21 A-B-B-A blocks,
84 runs, none void, `suspend` and `power` ok in all: NF4 1.918 ms, Q2_0
17.351 ms. That is a comparison of two implementations (NF4 fused
bitsandbytes kernel against plain PyTorch for Q2_0 with float32 intermediates),
not of two formats; a fused Q2_0 kernel would change it and by how much is not
measured. Only layer 0 was timed.

### Synthetic against real

| Kind | Where | Enters the verdict |
|---|---|---|
| Real, OLMoE on RTX 3090, 7 arms | `benchmarks/results/c5-train/` | yes |
| Real, cache size and restore on RTX 3090 | `benchmarks/results/c5-cache/` | no verdict, numbers only |
| Real, expert-swap check on OLMoE (ticket 02) | `benchmarks/results/c5-olmoe-swap/` | validity row 1 only |
| Synthetic, granite on CPU | `benchmarks/results/c5-train-synthetic/` | no |
| Synthetic, tiny OLMoE on CPU (ticket 05 smoke) | not kept | no |
| Pilot, spoiled by memory or for timing | `benchmarks/results/c5-train/pilot/` | no |
| Failed cache launches (`run-attempt1.log`, `run-attempt2.log`) | `benchmarks/results/c5-cache/` | no, no numbers |

### What this does not show

- Fake-quant training (ADR 0001) measures what the Q2_0 weights do to a
  bf16-trained LoRA. It says nothing about speed or memory of a Q2_0 runtime,
  which does not exist.
- One model (OLMoE-1B-7B), one temporary data set, one epoch, 4000 rows.
  Other MoE models and longer training are untested.
- The gate suites have 24 to 40 questions each. The drops of 9.00 and 5.00 are
  large against that size; the 3.33 drop in arithmetic is near the line.
- The mixed-precision route (Q2_0 only for low-traffic experts) was not run.
  The rule's table sends a ratio above 1.10 to "stop", so it is not
  recommended here either.

### Commands, versions and logs

- Training and verdict: `benchmarks/results/c5-train/arms.sh`, `setup.sh`,
  `pilot.sh`; verdict `python benchmarks/harness/c5_verdict.py` with the two
  ticket-02 JSONs as `--swap-record`, output in `verdict.json`.
- Cache and restore: `benchmarks/results/c5-cache/setup.sh`, `run.sh`.
- Versions: `pip-freeze.txt` in both result folders (torch 2.14.1+cu130,
  transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, datasets 5.1.0).
- Source logs: `arms.log`, `session.log`, per-run `runs/*/run.json` and
  `runs/*.log`; `run.log`, `restore.json`, `cache.json` for the cache.
- Commits: training on `ff6035cf`, cache on `0c0ad5f2`, records on `8fb0a6b0`
  and `a6f6facc`.
