# G5 proposed closing comment, unposted

**Tracker status: UNPOSTED PROPOSAL.** Publishing this document is not a Trello
or GitHub comment action. G5 remains stopped under C5's rule, PR #26. No mixed
GPU experiment ran and no mixed-GPU PASS or FAIL exists. The blocks below are
proposed wording for a future tracker comment, not a record that a card closed.
The alternative mixed experiment is described in
[`gate-g5-mixed-precision-experts.md`](gate-g5-mixed-precision-experts.md).

The Russian block is the Trello text only. It is the one Russian text in this
repository's G5 documents; the rule and the runbook are English. The duration
in both blocks is the runbook's single estimate (`Duration` in the gate
document): about 3.9 h if the reproduction control passes, about 5.6 h if it
fails, from the wall clock of C5's arms.

## Trello card [G5] (Russian, unpublished)

```text
G5 остановлен по правилу C5 (PR #26, benchmarks/gate-c5-q2-0-experts.md).

Правило C5 зафиксировано до прогонов (commit c7f2bfcd, 2026-10-20 17:52 UTC; первый arm 19:00 UTC). LoRA на OLMoE-1B-7B-0924, эксперты fake-quant, 3 сида, RTX 3090.
- Среднее отношение task loss Q2_0/NF4: 1.1049 (сиды 42/7/123: 1.1055, 1.1057, 1.1036), порог 1.03 — fail.
- Верхняя граница 95% интервала: до 1.1194, порог 1.05 — fail.
- Gate suites, падение Q2_0 против NF4: mini_mmlu 9.00, mini_common_sense 5.00, mini_arithmetic 3.33 (порог < 3) — fail; mini_instruction 0.67 — pass.
- Кеш на диске: Q2_0 2,765,385,229 B против NF4 4,276,925,021 B (64.7%). Без quality gate успехом не считается.

Таблица G5 в правиле C5: fail при среднем ≤ 1.10 — продолжать G5 только в смешанном виде; > 1.10 — остановить G5. 1.1049 > 1.10, поэтому G5 остановлен, пока нет 2-битного кодека лучше Q2_0 с поиском scale (обучаемый scale как в GSQ, или кодбук вроде IQ2_XS).
Порог 1.10 в правиле C5 назван грубой оценкой; среднее выше него на 0.005. Порог не сдвигался.

Что открывает G5 снова:
1. 2-битный кодек лучше Q2_0 с поиском scale, прошедший тот же gate C5 (или заранее записанное правило для него); или
2. минимальная смешанная проверка, готова к запуску: benchmarks/gate-g5-mixed-precision-experts.md. Правило записано до прогонов и будет закоммичено до первого GPU-прогона: 25% и 50% экспертов в Q2_0, выбор по трафику и, отдельно, по чувствительности на train-данных, одинаковый бюджет байт, пороги C5 против all-NF4. Около 3.9 часа на одной RTX 3090, если контроль воспроизведения проходит (12 прогонов по ~16 мин, выбор, контроль, установка), около 5.6 часа, если нет (по времени прогонов C5).

Записи: benchmarks/results/c5-train/verdict.json, benchmarks/results/c5-cache/cache.json.
```

## PR #26 (English, unpublished)

```markdown
**G5 remains stopped under C5's rule.**

C5's rule was committed before the runs (`c7f2bfcd`, 2026-10-20 17:52 UTC; first arm 19:00 UTC). LoRA over OLMoE-1B-7B-0924 with fake-quant experts, 3 seeds, RTX 3090. From [`benchmarks/results/c5-train/verdict.json`](benchmarks/results/c5-train/verdict.json):

| Check | Limit | Measured | Result |
|---|---|---|---|
| Mean task-loss ratio Q2_0 / NF4, 3 seeds | ≤ 1.03 | **1.1049** (1.1055, 1.1057, 1.1036) | fail |
| Upper bound of the 95% interval, worst seed | ≤ 1.05 | 1.1194 | fail |
| `mini_mmlu` / `mini_common_sense` / `mini_arithmetic` drop | < 3 | 9.00 / 5.00 / 3.33 | fail |
| `mini_instruction` drop | < 3 | 0.67 | pass |

Cache on disk ([`cache.json`](benchmarks/results/c5-cache/cache.json)): Q2_0 experts 2,765,385,229 B against NF4 4,276,925,021 B (64.7%). It does not count as a success without the quality gate.

C5's G5 table: a fail with a mean ratio ≤ 1.10 continues G5 in mixed form only; above 1.10 it stops G5. **1.1049 > 1.10, so G5 is stopped until there is a 2-bit codec better than Q2_0 with scale search** (a learned scale as in GSQ, or a codebook format like IQ2_XS). The rule labels 1.10 a rough estimate; the mean is 0.005 above it, and the line was not moved.

**What reopens G5**
1. A 2-bit codec better than Q2_0 with scale search that passes C5's gate, or a rule written for it before its runs; or
2. the minimal mixed check, ready to run: [`benchmarks/gate-g5-mixed-precision-experts.md`](benchmarks/gate-g5-mixed-precision-experts.md). Its rule was written before any run and is to be committed before the first GPU arm: 25% and 50% of experts in Q2_0, chosen by traffic and, separately, by a sensitivity proxy on train data only, at equal bytes, scored against all-NF4 with C5's thresholds. About 3.9 hours on one RTX 3090 if the reproduction control passes (12 runs of ~16 min, selection, the control and setup), about 5.6 hours if it fails (from the wall clock of C5's arms).
```
