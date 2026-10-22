# G4: LoRA over a frozen Mistral-7B base in NVFP4 versus NF4

**The decision rule is committed before any Mistral-7B run. Status: rule only, no 7B result yet. Data: a temporary set until the dataset protocol exists (временный набор до dataset протокола).**

Spec: `.scratch/g4-nvfp4/issues/04-mistral-harness-and-gate-rule.md`. The seven runs under
this rule are ticket 05. The speed and memory measurement and the final decision are
ticket 06. Template: `benchmarks/gate-c5-q2-0-experts.md`. Why the base is checked with a
harness and not a product path: `docs/adr/0003-g4-checks-nvfp4-with-a-harness-not-a-product-path.md`.

## Commit order

- The rule below applies from the commit that added this file. That commit's time must be
  earlier than the `started` field of every Mistral-7B arm record. To check:
  `git log --diff-filter=A --format='%H %cI' -- benchmarks/gate-g4-nvfp4.md`.
- Any change to the rule is a separate commit made before the first Mistral-7B arm. After the
  first arm the rule, the data and the thresholds do not change. A rebase or squash rewrites
  the dates, so the branch history from the rule commit on is not rewritten, and the PR merges
  with a merge commit.
- The harness refuses a real run when this file is not committed or differs from its committed
  text, and writes the commit of the rule into the run record.
- Runs on a small random model (the fixture) only check that the harness starts. They are
  labelled synthetic and never enter the verdict.

## Decision rule

The Russian text below is the binding version. An English translation follows it; where the two
differ, the Russian text wins.

Термины по `GLOSSARY.md`, обоснование harness в `docs/adr/0003-g4-checks-nvfp4-with-a-harness-not-a-product-path.md`, запрет смешивать форматы в одном arm в `docs/adr/0001-c5-fake-quant-training.md`.

**Вопрос.** Теряет ли LoRA, обученная поверх замороженной базы Mistral-7B в NVFP4, не больше допустимого по task loss против той же LoRA поверх базы в NF4? Меньшая или более быстрая база без прохождения quality gate успехом не считается. Точное равенство с NF4 не заявляется.

### Модель и данные

- Модель: `mistralai/Mistral-7B-v0.1`, revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`.
- Данные (временный набор до dataset протокола): те же, что в C5. `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, лицензия CC BY-SA 3.0, сплиты строит `benchmarks/harness/c5_data.py`. Held-out панель 512 строк, SHA-256 `b4b2ab938ca72a52c660d810045e3f31f842e3107e00738f827abb3f932d2a37`. Train-сплит 4000 строк, SHA-256 `a915bbbf18ebd1fae1dd869bdc8e2a0efbd2ab522d3a0c733cc05b212befa0b7`. Прогон с другими хешами отклоняется.
- Данные проходят через Soup: строка Dolly становится строкой alpaca (`context` это `input`), `chat_template: chatml`, loss обучения по маске ответа Soup. Task loss считается по токенам, первый символ которых лежит внутри текста ответа (так же, как в C5).
- Длина последовательности 1024 токена, усечение одинаково во всех arms. Каждый микробатч добивается справа паддингом до длины, кратной 16, одинаково во всех arms: native FP4 GEMM принимает только число токенов, кратное 16. Паддинг закрыт attention mask и не входит в loss.
- У токенизатора Mistral-7B-v0.1 нет pad-токена. Паддингу назначается `unk_token_id`, значение записывается в прогон.
- Если пилот покажет больше 2 часов на один прогон, train-сплит урезается до первых 2000 строк того же порядка (SHA-256 `806910717344ff5c4545dde76e5ae50367fb9d69365d59a11e23cf32463d7f7c`). Это поправка к правилу, она коммитится до первого настоящего arm.

### Arms

| Arm | База | Сиды | Роль |
|---|---|---|---|
| NF4 | NF4, bitsandbytes, как в QLoRA Soup | 42, 7, 123 | контроль в gate |
| NVFP4 | NVFP4, TransformerEngine `NVFP4BlockScaling`, native FP4 GEMM | 42, 7, 123 | кандидат в gate |
| bf16 | без квантования | 42 | справка, в gate не входит |

- В NF4 и NVFP4 хранятся все семь линейных слоёв каждого из 32 слоёв декодера (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), 224 тензора. Embeddings, norm и `lm_head` остаются bf16 во всех arms.
- NF4: `bnb_4bit_quant_type=nf4`, вычисления в bf16, double quant включён, как в загрузке Soup по умолчанию.
- NVFP4: вес заморожен, хранится упакованным, умножение идёт на упакованном весе. Если форма слоя не проходит проверку harness (in_features кратно 32, число токенов и out_features кратны 16), прогон отклоняется с явной ошибкой. Слой не уходит на NF4 или bf16 молча: два формата в одном arm запрещены (ADR 0001, ADR 0003).
- LoRA одним и тем же кодом harness во всех трёх arms (peft с TransformerEngine не работает) только на `q_proj`, `k_proj`, `v_proj`, `o_proj`, rank 16, alpha 32, dropout 0, адаптер в bf16, B стартует с нуля. Одна эпоха, effective batch 16, lr 2e-4, cosine, AdamW. Остальные гиперпараметры по умолчанию Soup на commit замера, они записываются в конфиг каждого прогона.
- Gate suites декодируются жадно, 64 новых токена, без KV-кеша: шаг декодирования из одного токена native FP4 GEMM не принимает, поэтому каждый шаг считает всю последовательность целиком с паддингом до кратного 16. Одинаково во всех arms.
- В паре NF4 и NVFP4 с одним сидом одинаковы сид, порядок батчей, инициализация адаптеров и все настройки, кроме формата хранения базы.

### Метрики

- **Task loss**: NLL на токенах ответа по held-out панели, взвешенный по токенам.
- **Отношение** для пары с одним сидом: task loss(NVFP4) / task loss(NF4). Среднее берётся по трём сидам.
- **Интервал**: 95% paired bootstrap по примерам панели, 2000 ресэмплов, сид 17. Пример ресэмплируется целиком, токены внутри примера отдельно не ресэмплируются. Интервал описывает разброс выборки панели в одном сиде, а не разброс обучения.
- **Gate suites**: `mini_mmlu` (26 вопросов), `mini_common_sense` (24), `mini_instruction` (24), `mini_arithmetic` (40). Считается число верных ответов, среднее по трём сидам. Suite, у которого NF4 в среднем отвечает верно на все вопросы или ни на один, отчитывается, но в gate не входит (issue #1192).

### Validity rows

Каждая строка при нарушении даёт no verdict.

| Validity row | Как проверяется |
|---|---|
| Отпечатки не квантуемых весов (embeddings, norm, `lm_head`) совпадают во всех arms | JSON прогона |
| Хеши train-сплита и held-out панели совпадают во всех arms | хеши в записи каждого прогона |
| Конфиги arms отличаются только форматом хранения базы | diff конфигов |
| Все 224 квантуемых слоя хранятся в формате arm'а: NVFP4 в arm NVFP4, NF4 в arm NF4, и ни один слой не отклонён по форме | счётчик типов весов в записи прогона |
| Каждый прогон дошёл до последнего шага без NaN и Inf в loss и grad norm | журнал тренера |
| Grad norm на первом шаге больше 0 | журнал тренера, как в `soup bench train` (#836) |

Проверки `suspend`, `power` и `foreign_reads` для обучающих arms не обязательны: скорость здесь не меряется. Прогон, нарушивший строку 5 или 6, сохраняется и повторяется с тем же сидом один раз. Если повтор тоже нарушает, вердикта нет. Нарушение строки 4 повтора не получает: формат хранения не зависит от сида.

### Вердикт качества

Применяется по порядку:

1. **No verdict**, если не выполнена хоть одна validity row.
2. **Fail**, если выполнено хотя бы одно:
   - среднее отношение по трём сидам больше 1.03;
   - верхняя граница интервала больше 1.05 хотя бы в одном сиде;
   - в gating suite NVFP4 в среднем отвечает верно на 3 вопроса меньше, чем NF4, или ещё хуже.
3. **Pass** во всех остальных случаях.

Границы включительные: 1.03 и 1.05 проходят. Пороги не меняются после первого прогона.

### Что вердикт значит для G4

Скорость шага меряется в тикете 06 по `benchmarks/METHOD.md` (A-B-B-A, число блоков по правилу повторов, обязательные проверки). Только при pass по качеству. Шаг NVFP4 «не медленнее» NF4, если медиана времени шага NVFP4 не больше медианы NF4 (отношение ≤ 1.00, граница включительная). Испорченный замер скорости решения не даёт.

| Исход | Решение |
|---|---|
| Pass и шаг не медленнее NF4 | продолжать реализацию NVFP4 в продуктовом пути |
| Pass и шаг медленнее NF4 | остановить G4 |
| Fail | остановить G4, скорость не меряется |
| No verdict | повторить замер, по G4 не решать |

### Записывается в каждый прогон

GPU и его UUID, драйвер, версии Python, torch, transformers, peft, bitsandbytes, TransformerEngine, datasets; commit репо и commit правила; SHA-256 harness; хеши данных; сид; полный конфиг; журнал loss и grad norm по шагам; число слоёв по типу весов; task loss по каждому примеру панели; выходы gate suites. Синтетические прогоны (fixture на случайной модели) подписываются отдельно и в вердикт не входят.

## Decision rule, English translation

Translation only. The Russian text above is binding.

Terms follow `GLOSSARY.md`. The reason for a harness is in `docs/adr/0003-g4-checks-nvfp4-with-a-harness-not-a-product-path.md`. The ban on mixing formats in one arm is in `docs/adr/0001-c5-fake-quant-training.md`.

**Question.** Does a LoRA trained over a frozen Mistral-7B base in NVFP4 lose no more than the allowed amount of task loss against the same LoRA over a base in NF4? A smaller or faster base that does not pass the quality gate does not count as success. Exact parity with NF4 is not claimed.

### Model and data

- Model: `mistralai/Mistral-7B-v0.1`, revision `27d67f1b5f57dc0953326b2601d68371d40ea8da`.
- Data (temporary set until the dataset protocol): the same as C5. `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, license CC BY-SA 3.0, splits built by `benchmarks/harness/c5_data.py`. Held-out panel: 512 rows, SHA-256 `b4b2ab938ca72a52c660d810045e3f31f842e3107e00738f827abb3f932d2a37`. Train split: 4000 rows, SHA-256 `a915bbbf18ebd1fae1dd869bdc8e2a0efbd2ab522d3a0c733cc05b212befa0b7`. A run with other hashes is rejected.
- The data goes through Soup: a Dolly row becomes an alpaca row (`context` is `input`), `chat_template: chatml`, training loss on Soup's assistant mask. Task loss counts tokens whose first character lies inside the answer text (as in C5).
- Sequence length is 1024 tokens, with the same truncation in every arm. Every micro batch is right-padded to a length that is a multiple of 16, the same in every arm, because the native FP4 GEMM takes only a token count that is a multiple of 16. Padding is masked from attention and from the loss.
- The Mistral-7B-v0.1 tokenizer has no pad token. Padding uses `unk_token_id`, and the value is recorded in the run.
- If a pilot shows more than 2 hours per run, the train split is cut to the first 2000 rows of the same order (SHA-256 `806910717344ff5c4545dde76e5ae50367fb9d69365d59a11e23cf32463d7f7c`). This is an amendment to the rule and is committed before the first real arm.

### Arms

| Arm | Base | Seeds | Role |
|---|---|---|---|
| NF4 | NF4, bitsandbytes, as in Soup's QLoRA | 42, 7, 123 | control in the gate |
| NVFP4 | NVFP4, TransformerEngine `NVFP4BlockScaling`, native FP4 GEMM | 42, 7, 123 | candidate in the gate |
| bf16 | no quantization | 42 | reference, not in the gate |

- NF4 and NVFP4 store all seven linear layers of each of the 32 decoder layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), 224 tensors. Embeddings, norms and `lm_head` stay bf16 in every arm.
- NF4: `bnb_4bit_quant_type=nf4`, bf16 compute, double quant on, as in Soup's default load.
- NVFP4: the weight is frozen and stored packed, and the multiply runs on the packed weight. If a layer's shape fails the harness check (in_features a multiple of 32, token count and out_features multiples of 16), the run is refused with an explicit error. The layer never falls back to NF4 or bf16 silently: two formats in one arm are banned (ADR 0001, ADR 0003).
- LoRA comes from the same harness code in all three arms (peft does not work with TransformerEngine), on `q_proj`, `k_proj`, `v_proj`, `o_proj` only, rank 16, alpha 32, dropout 0, bf16 adapter, B starts at zero. One epoch, effective batch 16, lr 2e-4, cosine, AdamW. Other hyperparameters are Soup defaults at the measured commit and are written into each run's config.
- Gate suites decode greedily, 64 new tokens, without a KV cache: the native FP4 GEMM does not take a one-token decode step, so each step runs the whole sequence, padded to a multiple of 16. The same in every arm.
- Within an NF4 and NVFP4 pair with the same seed, the seed, batch order, adapter initialization and every setting are the same except the base storage format.

### Metrics

- **Task loss**: NLL on response tokens over the held-out panel, weighted by tokens.
- **Ratio** for a same-seed pair: task loss(NVFP4) / task loss(NF4). The mean is taken over the three seeds.
- **Interval**: 95% paired bootstrap over panel examples, 2000 resamples, seed 17. Each example is resampled whole; tokens inside an example are not resampled separately. The interval describes sampling spread of the panel within one seed, not training spread.
- **Gate suites**: `mini_mmlu` (26 questions), `mini_common_sense` (24), `mini_instruction` (24), `mini_arithmetic` (40). The score is the number of correct answers, averaged over the three seeds. A suite where NF4 on average answers all questions or none correctly is reported but does not enter the gate (issue #1192).

### Validity rows

Each row, if violated, gives no verdict.

| Validity row | How it is checked |
|---|---|
| Fingerprints of the unquantized weights (embeddings, norms, `lm_head`) match in every arm | JSON of the run |
| Train split and held-out panel hashes match in every arm | hashes in each run's record |
| Arm configs differ only in the base storage format | config diff |
| All 224 quantized layers are stored in the arm's format: NVFP4 in the NVFP4 arm, NF4 in the NF4 arm, and no layer was refused for its shape | count of weight types in the run record |
| Every run reached its last step with no NaN or Inf in loss and grad norm | trainer log |
| Grad norm at the first step is above 0 | trainer log, as in `soup bench train` (#836) |

The `suspend`, `power` and `foreign_reads` checks are not required for training arms, because speed is not measured here. A run that violates row 5 or 6 is kept and repeated once with the same seed. If the repeat also violates it, there is no verdict. A violation of row 4 gets no repeat: the storage format does not depend on the seed.

### Quality verdict

Applied in order:

1. **No verdict** if any validity row fails.
2. **Fail** if at least one holds:
   - the mean ratio over three seeds is above 1.03;
   - the upper bound of the interval is above 1.05 in at least one seed;
   - in a gating suite, NVFP4 on average answers 3 or more questions fewer correctly than NF4.
3. **Pass** in every other case.

Bounds are inclusive: 1.03 and 1.05 pass. The thresholds do not change after the first run.

### What the verdict means for G4

Step speed is measured in ticket 06 by `benchmarks/METHOD.md` (A-B-B-A, block count by the repeats rule, required checks), and only after a quality pass. The NVFP4 step is "not slower" than NF4 if the median NVFP4 step time is not above the median NF4 step time (ratio ≤ 1.00, bound inclusive). A spoiled speed measurement gives no decision.

| Outcome | Decision |
|---|---|
| Pass and the step is not slower than NF4 | continue implementing NVFP4 in the product path |
| Pass and the step is slower than NF4 | stop G4 |
| Fail | stop G4; speed is not measured |
| No verdict | repeat the measurement; no G4 decision |

### Recorded for every run

GPU and its UUID, driver, versions of Python, torch, transformers, peft, bitsandbytes, TransformerEngine and datasets; repo commit and the rule's commit; SHA-256 of the harness; data hashes; seed; full config; per-step loss and grad norm log; count of layers by weight type; task loss for every panel example; gate suite outputs. Synthetic runs (the fixture on a random model) are labelled separately and do not enter the verdict.
