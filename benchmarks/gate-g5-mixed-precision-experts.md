# G5: per-expert precision for OLMoE experts, Q2_0 or NF4, at a fixed byte budget

**Status: STOPPED under C5 PR #26. The mixed GPU experiment was NOT run and has no verdict.** This publication preserves the proposed rule, harness and synthetic CPU pilot evidence. It does not authorize a GPU run or integrate mixed precision into the product. The rule was written before the CPU pilot on an uncommitted working tree and must be committed before any future GPU arm. The data is C5's temporary set until the dataset protocol exists.

Track G5 from the program brief, PDF p.15, and its Trello card. Base: C5
([`gate-c5-q2-0-experts.md`](gate-c5-q2-0-experts.md), PR #26), whose model,
data, LoRA, seeds, task loss, bootstrap, gate suites, codec and harness are reused.
[`g5-closing-comment.md`](g5-closing-comment.md) is a proposed, unposted tracker
comment. Publishing that file does not close or modify a Trello card or PR.

## Why run this although C5 recommended stopping

C5's rule, fixed before its runs, ends in the table "Fail, mean ratio ≤ 1.10 →
continue G5 only in mixed form; Fail, mean ratio > 1.10 → stop G5 until there
is a 2-bit codec better than Q2_0". C5 measured a mean task-loss ratio of
**1.1049** (seeds 42, 7, 123: 1.1055, 1.1057, 1.1036), so the table says
**stop**, and this rule does not change that row or its threshold.

The check is still informative for three reasons:

1. C5 labels the 1.10 line a rough estimate. The observed mean is 0.0049 above
   it, but the fixed stop threshold still applies.
2. That line rests on a guess that mixed precision cannot move more than a
   third of the experts to Q2_0 without passing 1.03. The guess was never
   measured. The proposed check would measure it at a quarter and at a half.
3. C5 tested all experts in Q2_0 against all in NF4. It did not test the
   claim of the PDF and the card, that experts differ in how much 2-bit
   rounding hurts them, nor whether traffic predicts that. The traffic-vs-
   sensitivity correlation is reported whatever the verdict.

If a future authorized experiment makes every mixed arm fail, G5 stays stopped.
Until then, the existing C5 result is the reason for the stop.

## Commit order

- This file, the harness (`benchmarks/harness/g5_*.py`, `g5_*.sh`), the
  `.gitattributes` LF lines and the tests are to be committed before the
  first GPU arm. The commit that adds this file must be earlier than the
  `started` field of the first GPU record. To check:
  `git log --diff-filter=A --format='%H %cI' --
  benchmarks/gate-g5-mixed-precision-experts.md`.
- The synthetic CPU pilot in `benchmarks/results/g5-cpu-pilot/` ran after this
  rule and the harness were written, on the uncommitted working tree (its
  records name `commit.dirty`). It says nothing about OLMoE and changes no
  number of the rule. If the committed harness hashes differ from the pilot's
  records, the pilot is re-run before the first GPU arm.
- C5's files are not changed. G5 imports them; `c5_verdict.py` still writes
  C5's published `verdict.json` unchanged.
- An amendment (for example the cut to 2000 train rows that C5's rule allows
  after a slow pilot) is a separate commit before the first real arm.

## Decision rule

This English text is the proposed rule. The Russian tracker draft is only in
`g5-closing-comment.md`.

**Question.** Does a LoRA over OLMoE with part of the experts in Q2_0 and the rest in NF4 keep all-NF4's quality within C5's thresholds, and does a traffic rule or a sensitivity rule choose that part, at the same byte budget? A smaller size without passing the quality gate does not count as success. An expert's popularity is not taken as proof of its sensitivity: traffic and sensitivity are two separate statistics, and only their correlation is reported.

### Model, data, training

Everything as in C5 (`benchmarks/gate-c5-q2-0-experts.md`), unchanged: `allenai/OLMoE-1B-7B-0924` revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; Dolly-15k revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, train split of 4000 rows (SHA-256 `a915bbbf…`) and held-out panel of 512 rows (SHA-256 `b4b2ab93…`); LoRA r 16, alpha 32 on `q_proj`, `k_proj`, `v_proj`, `o_proj`; one epoch, effective batch 16, lr 2e-4, cosine, gradient checkpointing; seeds 42, 7, 123; codec `q2_0_codec.py` SHA-256 `fbaefa2a…` and fake-quant through C5's swap, one expert at a time.

### Choosing precision (train data only)

- **Calibration**: the first 512 rows of C5's train split in file order, SHA-256 `6a62ff891333a5dc4f9386715e3424d19e64f02187ff3f1f890b5deb296e98f0` (`c5_data.split_digest`), 93,650 tokens. The held-out panel and the gate suites are not opened during selection.
- **Traffic**: on the bf16 model before any quantization, from the inputs of each layer's experts module (`top_k_index`, `top_k_weights` of OLMoE's router): routed (token, slot) assignments per expert and the sum of gate weights.
- **Sensitivity**: on the same tokens, for expert e of layer l, `d_q2_nf4 = Σ_t g[t,e]² · ‖f_e^Q2_0(x_t) − f_e^NF4(x_t)‖²` over the tokens t routed to e, where f is the expert's output with C5's fake-quant weights and g its gate weight. It is the change of the MoE block output when only e moves from NF4 to Q2_0.
- **Map**: in every layer exactly f·64 experts in Q2_0 and the rest in NF4, f ∈ {0.25, 0.5}. By traffic: the experts with the fewest assignments go to Q2_0 (then the smaller gate-weight sum, then the lower index). By sensitivity: the smallest `d_q2_nf4` (then the lower index). One fraction f in every layer gives both maps the same size in bytes.
- Recorded: the calibration hash, the statistics of every (layer, expert), their hash, the hash of each map, and the Spearman rank correlation between traffic and `d_q2_nf4` per layer.

### Arms

| Arm | Experts | Seeds | Role |
|---|---|---|---|
| all-NF4 | all NF4 | 42, 7, 123 | control, C5's runs |
| all-Q2_0 | all Q2_0 | 42, 7, 123 | reference, C5's runs |
| bf16 | no quantization | 42 | reference, C5's run |
| mixed_traffic_f25 | 16 of 64 in Q2_0 by traffic | 42, 7, 123 | candidate |
| mixed_sensitivity_f25 | 16 of 64 in Q2_0 by sensitivity | 42, 7, 123 | candidate |
| mixed_traffic_f50 | 32 of 64 in Q2_0 by traffic | 42, 7, 123 | candidate |
| mixed_sensitivity_f50 | 32 of 64 in Q2_0 by sensitivity | 42, 7, 123 | candidate |

Order: selection, bytes, the reproduction control, C5's seven arms again if the control fails, then the mixed arms seed by seed (42, 7, 123), within a seed in table order (`g5_run_all.sh`).

### Reproduction control

An NF4 seed 42 run on the new box is compared with C5's `nf4-s42-a1`. C5's runs (NF4, Q2_0, bf16) are used as baselines only if the control passes: the same config, data, batch order, model, C5 code, expert weights (`expert_digest`) and non-expert weights; task loss within 0.2% (|loss/loss_C5 − 1| ≤ 0.002); and in every gate suite the number of correct answers within 1. Otherwise all seven C5 arms are repeated on the new box and the comparison uses them.

### Validity rows

Each row, if violated, gives no verdict for its arm.

| Validity row | How it is checked |
|---|---|
| C5's five rows: non-expert weight fingerprints, split hashes, configs differ only in the codec, runs finite to the last step, first-step grad norm > 0; plus the unchanged codec | `c5_verdict.py`, unchanged |
| Selection-data hash: the calibration is the rule's above, from the same train split as the runs; the selection record holds no panel hash | `selection.json`, `run.json` |
| Map hash: every run trained over exactly the selected map (map hash, statistics hash, calibration hash, the experts' codes in the record) | `selection.json`, `run.json` |
| Byte budget equality: both maps of one fraction f have one cache size in bytes; all-NF4 and all-Q2_0 equal C5's measurements | `bytes.json` |
| One C5 code and one G5 code in every run | SHA-256 in `run.json` |
| Baselines reproduced: the control passed on the box of the mixed arms, or the baselines were repeated on it | `control.json` |

Validity comparisons use only attempts counted by the one-repeat rule, not kept
spoiled attempts. Attempt numbers must start at 1, be unique and consecutive,
with at most the permitted repeat. Reused baselines must contain the counted
reference run; repeated baselines need the complete valid seven-run set on the
mixed box. G5 code hashes must match selection across arms. Missing source
provenance gives no verdict, including on historical synthetic replay inputs.

### Verdict of each mixed arm against all-NF4

C5's thresholds, in order: no verdict if a validity row is violated; fail if the mean over three seeds of task loss(mixed)/task loss(NF4) is above 1.03, or the upper bound of the 95% paired bootstrap interval is above 1.05 in at least one seed, or in a gating suite the mean number of correct answers is 3 or more below NF4's; otherwise pass. Bounds are inclusive.

### What the outcome means for G5

| Outcome | Decision |
|---|---|
| At least one mixed arm passes | the expert precision rule: that arm's rule at its fraction; if several pass, the larger fraction, and at equal fraction the traffic rule (cheaper to compute) |
| No pass, at least one no verdict | repeat those arms; no G5 decision |
| All four fail | G5 stays stopped until there is a 2-bit codec better than Q2_0 with scale search |

A pass decides even when another arm is no verdict, including an arm at a larger fraction or the traffic arm at the same fraction: for example f25 passes and f50 is no verdict, so f25 is the rule although f50 might pass after its repeat. The larger-fraction arms are not repeated first; the no-verdict arms are listed in `decision.unresolved` and `g5_verdict.py` reports `g5: rule`. The repeat outcome applies only when no arm passes.

C5's table still holds: this measurement does not overturn it, it tests the estimate behind its 1.10 line.

### Size

The full shard-cache size on disk, including safetensors headers, metadata and the codec map: for all-NF4 and all-Q2_0, C5's measurements (`benchmarks/results/c5-cache/cache.json`), which the computation must equal to the byte; for the mixed maps, a computation with the same layout (no mixed-cache writer exists), labelled as computed.

### Recorded

Everything C5 records, plus `selection.json` (calibration, statistics, correlations, hashes), the maps, `bytes.json`, `control.json`, `verdict.json` and the chart `quality_vs_bytes.svg`. Synthetic runs (granite on CPU) never give a verdict: `g5_verdict.py --synthetic` runs the rule's code to the end, then writes `no_verdict` for every arm, for C5's baseline verdict and for `decision.g5`, sets `reportable: false` and the label `SYNTHETIC / CPU PILOT: NO VERDICT` in the JSON, the chart and the console; what the code computed is kept only under `pipeline_outcome` and `decision.pipeline`. Synthetic and real records (runs, control, selection, bytes) never mix in one verdict.

## Choices fixed before the first run

They follow from the rule and are recorded here so that nothing is decided after a number is seen.

- **Why f = 0.25 and 0.5.** C5's own estimate behind its 1.10 line is that
  mixed precision moves "no more than a third" of the experts to Q2_0 within
  1.03. A quarter is below that estimate and is the arm most likely to pass;
  a half is above it and tests whether the estimate is too pessimistic. A
  uniform-in-every-layer fraction keeps both rules at the same bytes by
  construction (no knapsack, nothing to tune). If the all-Q2_0 loss
  penalty (ratio 1.105) were spread evenly over experts, a fraction f would
  cost about 1 + 0.105·f: 1.026 at f = 0.25 (inside 1.03) and 1.052 at
  f = 0.5 (outside). A selection rule earns its keep if it beats that line.
  This is arithmetic for orientation, not a prediction.
- **Why the sensitivity proxy is the output error, not a Hessian.** It costs
  one forward pass per expert on the calibration tokens, uses the exact
  weights the training arm sees, and weights each token by its gate as the
  model does. A Fisher/Hessian proxy needs gradients through the model and is
  left to a later check (step 3 of the card: "if needed").
- **Why selection counts tokens on the bf16 model.** The card asks for routing
  statistics before quantization. The router stays bf16 in every arm, but its
  inputs change once experts are quantized, so a map drawn from a quantized
  model would depend on the codec under test.
- **Calibration size.** 512 rows (93,650 tokens) gives every expert of every
  layer on average 93,650 · 8 / 64 ≈ 11,700 assignments; a single forward pass
  per row, no padding, so no pad token is ever routed.
- **The mixed-cache layout.** No writer for a mixed cache exists, and Soup's
  runtime has no packed-expert path (C5, ticket 05). The computed layout is
  C5's: the NF4 experts of a stacked tensor under its key with C5's sidecars,
  the Q2_0 experts under `<key>::q2_0` and `<key>::q2_0::d`, and the codec map
  as a 64-character string in each layer file's metadata (`g5_codec_map`,
  `n` or `q` per expert). An expert's row in its group is its rank among the
  experts of the same codec, so the map is the whole index. The computed
  all-NF4 and all-Q2_0 sizes equal C5's measured files to the byte (16 layer
  files and the total); a test holds this.
- **Fake-quant of a mixed map** goes through C5's own per-slice swap
  (`c5_expert_swap._slice_errors`), so an all-NF4 map gives C5's NF4 weights
  bit for bit (a test holds this) and a mixed map gives, expert by expert, the
  same weights as C5's NF4 or Q2_0 arm.
- **The reproduction tolerance.** C5's NF4 runs agree to 0.09% across seeds
  (1.6393, 1.6393, 1.6408). The gate's margin is 3%. A 0.2% tolerance is
  about twice the seed spread and well under the margin, so a passing control
  cannot move a verdict across 1.03. A box with another GPU or package version
  may still pass: the control measures the number, not the box.

## Byte budget (computed before any run)

Full shard cache on disk for OLMoE-1B-7B-0924, from `g5_bytes.py` with C5's
`cache.json`. All-NF4 and all-Q2_0 equal C5's measured caches to the byte
(the `checks` of the record). The mixed rows are computed, not measured.

| Arm | Q2_0 experts | Total on disk | One layer file | Expert bits per weight | Codec map, all layers | of all-NF4 |
|---|---:|---:|---:|---:|---:|---:|
| all-NF4 (C5, measured) | 0 / 1024 | 4,276,925,021 B | 241,551,736 B | 4.127 | 0 B | 100.0% |
| mixed, f = 0.25 (both rules) | 256 / 1024 | 3,899,050,333 B | 217,934,568 B | 3.658 | 1,312 B | 91.2% |
| mixed, f = 0.50 (both rules) | 512 / 1024 | 3,521,167,709 B | 194,316,904 B | 3.188 | 1,312 B | 82.3% |
| all-Q2_0 (C5, measured) | 1024 / 1024 | 2,765,385,229 B | 147,080,576 B | 2.250 | 0 B | 64.7% |

The 953 MB outside the experts (attention, router, norms, embeddings, head)
is the same in every arm, as in C5. The codec map adds 82 bytes of header per
layer file.

## GPU runbook (RTX 3090 24 GB, Vast-style Linux box, as C5)

This is a proposed runbook, not an instruction to resume stopped G5. Any future
run needs a separate decision to reopen the experiment. Use C5's pinned versions,
gradient checkpointing on and micro batch 4, from a clean committed checkout.

**Box needs.** One CUDA GPU with 24 GB (C5's arms peaked at 16,703,848,448 B
allocated); host RAM 32 GB or more (selection keeps the calibration tokens'
hidden states for 16 layers in bf16 on the host, about 6.1 GB, beside a
14 GB model load); 60 GB free disk (13.8 GB of OLMoE weights in the Hugging
Face cache, about 10 GB of Python environment, run records about 0.2 MB each).

**Duration.** The closing-comment draft quotes this estimate. Basis: the wall
clock, `started` to `finished`, of C5's seven arms in
`benchmarks/results/c5-train/runs/*/run.json`, mean NF4 14.6 min,
Q2_0 18.6 min, bf16 14.3 min. C5's Q2_0 swap takes 339 s against 92 s for NF4,
because it also computes the llama.cpp reference error for Q2_0 slices, so a
mixed arm is estimated as NF4's 14.6 min plus 0.25 or 0.5 of that 247 s:
15.6 min at f = 0.25 and 16.6 min at f = 0.5. Selection, setup and the mixed
arms are estimates, not measurements.

| Step | Runs | Estimate |
|---|---:|---:|
| setup (`pip install`, C5 data, model download) | | 15 min (C5's setup took 4 min, plus the 13.8 GB download) |
| provenance (`g5_verdict.py provenance`) | 1 | seconds |
| selection (`g5_select.py`): one bf16 forward per calibration row, then 1024 experts × two codecs fake-quant and 3 expert forwards | 1 | 10 min (estimate) |
| bytes (`g5_bytes.py`, config only) | 1 | 1 min |
| control: C5's NF4 seed 42 | 1 | 14.6 min |
| C5's other six arms, only if the control fails (the control run is kept as `nf4-s42-a1`) | 6 | 1.65 h |
| mixed arms, 4 × 3 seeds | 12 | 3.2 h |
| verdict | 1 | seconds |
| **Total** | | **about 3.9 h (3.65 h of GPU work plus 15 min setup) if the control passes; about 5.6 h if it fails** |

**Matching provenance on the box.** The reproduction control compares
SHA-256 of C5's files as checked out (`same_c5_code`), and the guard is not
relaxed. Three things make the box's files hash as C5's records:

1. `.gitattributes` pins LF on C5's and G5's hashed Python and shell files,
   including checkouts with `core.autocrlf=true`. The historical Windows pilot
   had CRLF hashes `ba4711ea…` / `da4f94ee…` rather than C5's committed
   `c9ad559b…` / `e2ae3beb…`. Local scratch-checkout logs are not part of this
   publication; the retained JSON records preserve the recorded hashes.
2. `g5_run_all.sh real` starts with `g5_verdict.py provenance`, which hashes
   C5's files on the box (raw and LF-normalised) against C5's
   `nf4-s42-a1/run.json` and stops the session, before any GPU work, if they
   differ; `eol_only: true` in `provenance.json` names a CRLF checkout. It
   also writes `git ls-files --eol` to `eol.txt` and lists the package
   versions that differ from C5's record.
3. Every G5 record holds both hashes of every C5 and G5 file
   (`harness.sources` in `selection.json`, `swap.codec_map.harness.sources` in
   each mixed `run.json`).

The package versions are C5's pins above (transformers 5.20.2). The control
does not compare versions; it compares the number and the weights
(`same_expert_weights`, `same_non_expert_weights`). If any of its checks
fails, the rule re-runs C5's arms on the box, which costs 1.65 h, not a
verdict.

**Commands.**

```bash
# on the box, as root, PyTorch image with /venv/main (as C5)
export PATH=/venv/main/bin:$PATH
git clone https://github.com/SoupTeam/soup-rnd.git /root/soup-rnd
cd /root/soup-rnd && git checkout <commit with this rule> && git log --oneline -1
git ls-files --eol -- 'benchmarks/harness/c5_*' benchmarks/harness/q2_0_codec.py   # w/lf
mkdir -p /root/g5
pip install -q -e ".[train]" "torch==2.14.1" "transformers==5.20.2" "peft==0.21.2" \
  "bitsandbytes==0.50.2" "datasets==5.1.0" "tokenizers==0.23.3"
pip uninstall -y torchvision torchaudio
python -c "import torch,transformers,peft,bitsandbytes,datasets;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),transformers.__version__,peft.__version__,bitsandbytes.__version__,datasets.__version__)"
pip freeze > /root/g5/pip-freeze.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,pcie.link.gen.max --format=csv > /root/g5/nvidia-smi.csv
free -b > /root/g5/free.txt; df -B1 /root > /root/g5/df.txt
python benchmarks/harness/c5_data.py --out-dir /root/g5/data   # must print C5's two hashes

# everything else, in the rule's order; resumes after an interruption; exits non-zero
# (and says which step stopped it) when a step fails or leaves no record
set -o pipefail
bash benchmarks/harness/g5_run_all.sh real /root/g5/data /root/g5/out 2>&1 | tee /root/g5/session.log
```

`g5_run_all.sh real` orchestrates the steps below. These commands describe its
components, not a second copy-paste driver. Use the script for checked exit
handling, control branching and baseline repetition.

```bash
python benchmarks/harness/g5_verdict.py provenance \
  benchmarks/results/c5-train/runs/nf4-s42-a1/run.json --out /root/g5/out/provenance.json
python benchmarks/harness/g5_select.py --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda \
  --data-dir /root/g5/data --out-dir /root/g5/out/selection
python benchmarks/harness/g5_bytes.py --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --map /root/g5/out/selection/maps/mixed_traffic_f25.json \
  --map /root/g5/out/selection/maps/mixed_sensitivity_f25.json \
  --map /root/g5/out/selection/maps/mixed_traffic_f50.json \
  --map /root/g5/out/selection/maps/mixed_sensitivity_f50.json \
  --cache-json benchmarks/results/c5-cache/cache.json --out /root/g5/out/selection/bytes.json
python benchmarks/harness/c5_train.py --arm nf4 --seed 42 --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda --gradient-checkpointing \
  --data-dir /root/g5/data --out-dir /root/g5/out/baselines/nf4-s42-a1
python benchmarks/harness/g5_verdict.py control /root/g5/out/baselines/nf4-s42-a1/run.json \
  benchmarks/results/c5-train/runs/nf4-s42-a1/run.json --out /root/g5/out/control.json
# A control exit of 1 requires baseline repeats. g5_run_all.sh runs C5's fixed
# seven-arm order itself, with checked Python and tee statuses; do not use the
# historical c5_train_all.sh as a strict failure-propagating driver.
# A control exit of 2 is an error and must stop, not trigger repeats.
bash benchmarks/harness/g5_train_all.sh /root/g5/data /root/g5/out/mixed /root/g5/out/selection \
  --model allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --device cuda --gradient-checkpointing
# BASELINES: benchmarks/results/c5-train/runs if the control passed,
# /root/g5/out/baselines if C5's arms were re-run here
python benchmarks/harness/g5_verdict.py /root/g5/out/mixed --baselines "$BASELINES" \
  --control /root/g5/out/baselines/nf4-s42-a1/run.json \
  --control-reference benchmarks/results/c5-train/runs/nf4-s42-a1/run.json \
  --selection /root/g5/out/selection/selection.json --bytes /root/g5/out/selection/bytes.json \
  --swap-record benchmarks/results/c5-olmoe-swap/nf4.json \
  --swap-record benchmarks/results/c5-olmoe-swap/q2_0.json --out /root/g5/out/verdict.json
```

The control command exits 0 (passed), 1 (a decided fail, including a control run that did
not finish: `control.json` then holds `control_complete: false` and the reasons) or 2 (the
check itself could not run: unreadable or malformed JSON, a missing file, an I/O error, a
bug; `g5_verdict.py` maps every unexpected exception to 2, never to 1). `provenance` uses
the same codes. `g5_run_all.sh` reuses C5's baselines on 0, re-runs C5's seven arms only on
1 and stops on anything else. Python or `tee` failures stop the session, even if a
partial record exists. Cached selection and byte checks must pass before reuse.
A step leaving no `run.json` also stops. A spoiled attempt stays on disk: the
verdict applies the one-repeat rule, and G5's validity rows read counted attempts.

**Expected outputs** under `/root/g5/out/` for a future authorized experiment.
Review and sanitize records before publication; do not copy operational logs,
private paths or environment captures into a PR.

- `provenance.json`: `c5_code_matches` true; `eol.txt` lists `w/lf` for every
  C5 and G5 file.
- `selection/selection.json`: `calibration.sha256` equal to
  `6a62ff89…` and `calibration.tokens` 93650; all four `checks` true;
  `forward_check` per layer under 0.05; 16 layers × 64 experts of `tokens`,
  `mass`, `d_q2_nf4`; `correlation` with the per-layer Spearman and the
  busiest-quartile share; `harness.sources`; `selection/maps/*.json`, four maps.
- `selection/bytes.json`: `checks` all true, among them
  `all_nf4_ssd_bytes_match_c5` and `all_q2_0_ssd_bytes_match_c5`; mixed totals
  3,899,050,333 B (f = 0.25) and 3,521,167,709 B (f = 0.5); `synthetic` false.
- `control.json`: `passed`, with `relative_task_loss_diff` ≤ 0.002 against
  C5's 1.639324.
- `mixed/<arm>-s<seed>-a1/run.json`, twelve runs, `status` complete, each with
  `swap.codec_map`.
- `verdict.json` and `quality_vs_bytes.svg`: per arm `outcome`, `mean_ratio`,
  `intervals`, `suites`, `validity`, `ssd_bytes`; `reportable` true;
  `decision.g5` is `rule`, `repeat` or `stop`; `points` for the chart
  (all-NF4, all-Q2_0, the four mixed arms, bf16 listed off scale).
  `python benchmarks/harness/g5_verdict.py plot verdict.json --out chart.svg`
  redraws the chart from the JSON.

A spoiled run (validity row 4 or 5) is repeated once by hand with
`--attempt 2`, as the comments in `g5_train_all.sh` and `c5_train_all.sh` show.

## CPU pilot, synthetic and not reportable

[`results/g5-cpu-pilot/README.md`](results/g5-cpu-pilot/README.md) describes the
published subset: sanitized selection, maps and byte accounting, twelve mixed
`run.json` records, seven repeated C5 baseline records, and the corrected
historical acceptance verdict. All are from a random-init granite CPU fixture
with a byte-level tokenizer. They say nothing about OLMoE quality.

The source narrative calls this the 2026-10-21 pilot; record timestamps are
2026-10-22 in UTC+05:00. It used Windows 11 x86-64, Python 3.12.10 and
transformers 5.19.0 on an uncommitted working tree based on `8ba2552e`, with a
reported duration of 31 min. Selection used the first 32 train rows; training
used 64 train and 32 panel rows. These historical records are not a new run.

The control's task loss differed from C5's synthetic reference by a relative
2.7e-7, but the control failed `same_c5_code` and `same_non_expert_weights`.
CRLF checkout bytes explain the former. Nine of nineteen non-expert tensors
differed between the recorded transformers versions. The pipeline repeated
C5's seven arms on the CPU box.

The obsolete top-level verdict computed `pass` for the four synthetic arms,
ratios 0.9998 to 0.9999, and `decision.g5: rule`. That was never a mixed-GPU
result. The source keeps that legacy file unchanged; it and its chart are
omitted here to avoid publishing a misleading verdict. The retained corrected
acceptance replay has `outcome: no_verdict` for every arm,
`decision.g5: no_verdict` and `reportable: false`. Its computed values remain
under `pipeline_outcome` and `decision.pipeline`, not a G5 decision.

[`provenance.json`](results/g5-cpu-pilot/provenance.json) records the source
revision, per-file text hashes and redacted JSON pointers. Only absolute
interpreter, script and temporary-data paths were replaced. Numerical tokens,
hashes, timestamps and historical outcomes were not changed. Logs, local review
scripts, setup/rental operations, environment captures and pre-fix reproductions
were omitted. The retained verdict is historical, not proof that this publication
checkout passed validation. CPU replay commands are in the evidence README.
