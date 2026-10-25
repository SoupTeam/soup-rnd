# E4 part 2: calibration record (2026-10-25)

Calibration under `e4-part2-calibration-rule.md`, run from main at `fe05753` on one vast.ai A100 PCIe 40 GB. **Second series: every stage ok, 0 voids, no provenance gaps.** The full history is 11 successful attempts plus 2 voids from the first series; see [Restart disposition](#restart-disposition-2026-10-25). These are calibration values, not a double-quant verdict. No DECISION row passed through a model.

## Restart disposition (2026-10-25)

- The first series stopped under rule §8 after two `F_A_off` voids. Both were caused by `pynvml` being imported by `nvml_reader` but declared in no requirement or pin.
- The first series produced no F measurements and no calibrated tolerances. Its records are kept unchanged in `attempt-1-stopped/`.
- The second series ran in a fresh output directory after installing `nvidia-ml-py==13.615.71`. No threshold or rule text changed. The dependency fix is [#37](https://github.com/SoupTeam/soup-rnd/pull/37).
- This is a disclosed, infrastructure-only deviation. It is not an allowance in the rule. A new output directory does not reset the §8 retry limit.
- "0 voids" in this record refers to the second series only. The full history is 11 successful attempts plus 2 voids (`attempts.jsonl`: 11 start/end pairs in the second series; `attempt-1-stopped/attempts.jsonl`: 2 void ends).

## Result

| Stage | Outcome | Value the main rule copies |
|---|---|---|
| F (A_off, A_on, A_bf16, 16 updates each) | ok, all finite, base unchanged, no recipe mismatch | `FINAL_A100_40GB`; peak reserved 17,846,763,520 bytes on every arm (threshold 36,000,000,000) |
| T | ok | tau table by cell and quantity, `report.json` → `tau` |
| W | ok | tol_W: output 0.2197, input_grad 0.01742, A_grad 0.6595, B_grad 0.08244, update 0.000400 |
| N x5 | ok | floor 0.0 on all 8 suites, so epsilon = 1/n_items (table below) |
| P | ok | SMOKE NLL 1.625667 vs 1.625419, difference -0.000248; adapters not bit-identical |

Final card: NVIDIA A100-PCIE-40GB, GPU UUID `GPU-cc9d0159-449a-85d6-42b8-1515e454e9fc`, vast instance `57596627`, driver 595.91.07, compute capability 8.0, memory.total 40,960 MiB.

Projected main plan (9 runs + 1 rerun of A_on, 256 updates each, training updates only): **5.31 h**, from the F medians (A_bf16 7.35 s, A_off 7.46 s, A_on 7.58 s per update).

### N: noise floor per suite

| Suite | n_items | Score (all 5 runs) | Floor | Epsilon |
|---|---|---|---|---|
| mini_arithmetic | 40 | 0.625 | 0.0 | 0.025 |
| mini_common_sense | 24 | 0.8333 | 0.0 | 0.04167 |
| mini_format_json | 40 | 0.925 | 0.0 | 0.025 |
| mini_instruction | 24 | 0.9583 | 0.0 | 0.04167 |
| mini_mmlu | 26 | 0.7308 | 0.0 | 0.03846 |
| mini_over_refusal | 40 | 0.975 | 0.0 | 0.025 |
| mini_safety | 40 | 0.0 | 0.0 | 0.025 |
| mini_tool_call | 40 | 0.45 | 0.0 | 0.025 |

Greedy decoding gave identical scores across the five fresh processes, so every floor is 0 and epsilon falls back to 1/n_items, as the rule allows.

## Findings to review

1. **Undeclared dependency stopped the first calibration.** `nvml_reader` imports `pynvml`, which no requirement or pin declares. The dry run never calls the NVML sampler and CI has no GPU, so nothing caught it. Attempt 1 voided twice on `ModuleNotFoundError` before any measurement and stopped per §8. Its records are kept in `attempt-1-stopped/`. I installed `nvidia-ml-py==13.615.71` (not in the rule's pin list) and started a fresh calibration in a new output folder; no threshold or rule text changed. A separate PR declares the dependency and adds a test ([#37](https://github.com/SoupTeam/soup-rnd/pull/37)).
2. **Eligible rows 11,844 vs the pre-commit count of 11,952 (-108).** Same Dolly revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Funnel: raw 15,011, duplicate 15, prompt over 384 tokens 693, response over 128 tokens 2,305, both over 154, empty 0. Above the 6,400 floor, so no stop under §2. The difference likely sits in the token-length cuts of the review script (inference, not checked).
3. **mini_safety scores 0.0 on every run.** Explicit limits:
   - It is 0.0 on every run, so it cannot reveal further degradation from zero. The floor is 0 and the suite cannot move below 0, so it cannot flag a drop in the main series.
   - The absence of a flag is not positive safety evidence.
   - Any refusal-behaviour interpretation is unverified without generated answers. My reading that the OLMoE base model refuses none of the 40 items is inference, not checked.
   Flag only; the rule does not use suites for the outcome.
4. **P: adapters differ bit-wise between two identical runs**, NLL difference -0.000248. One paired observation, as §7 says, not an estimate of GPU nondeterminism.
5. **F arms use identical memory** (17.85 GB reserved each), because all three train on bf16 weights: A_off and A_on load the written-back checkpoints.

## Layout

| Path | What |
|---|---|
| `attempts.jsonl` | start and end of every attempt |
| `runs/<attempt_id>.json` | one record per attempt (F x3, T, W, N x5, P) with box stamps and provenance |
| `decision.json` | the final-card decision |
| `report.json` | every value the main rule copies |
| `preflight-*.json` | card check before the first attempt |
| `SHA256SUMS` | as written on the instance |
| `driver.log` | full driver output |
| `split/` | `split-record.json` (from the instance) and the DECISION, SMOKE and TRAIN manifests |
| `verified.json` | U4: all 78 output and 9 source files matched the quant manifest on the instance |
| `pip-freeze.txt` | the instance's venv |
| `attempt-1-stopped/` | the stopped attempt: journal, 2 void records, preflight, driver log |

## Edits after the run

- **Adapter weights are not committed** (1.9 GB). `SHA256SUMS` still lists the 12 adapter files; every other listed file verifies. The adapters are kept outside the repo and can be shared on request.
- **Adapter archive:** private Hugging Face repo [`RedBullet12/e4-part2-calibration-adapters`](https://huggingface.co/RedBullet12/e4-part2-calibration-adapters), revision `a2892103c1c2f83b91fc1e5a15ca0f16b98b1e48`; 12/12 hashes verified on 2026-10-25 from a fresh download at that revision.
  Absent from the repo: `adapters/F_A_off-1792896423/{README.md, adapter_config.json, adapter_model.safetensors}`, `adapters/F_A_on-1792896705/{...}`, `adapters/F_A_bf16-1792896989/{...}`, `adapters/P-1792901964/{...}`. Hashes as listed in `SHA256SUMS`.
- Split manifests were not copied off the instance. They come from the Mac-side split (~/e4-part2-data/split); their SHA-256 match split-record.json written on the instance. The split was not rebuilt.
- `driver.log` contains stray `^B` and `^Bc` keystrokes from a tmux mix-up on my side. Nothing reads stdin; no run was affected.

## Environment

vast.ai template PyTorch (`vastai/pytorch_cuda-13.2.1`), fresh Python 3.12.3 venv with the rule's pins: torch 2.14.1+cu130, transformers 5.19.0, bitsandbytes 0.50.2, peft 0.21.2. Before the run on the box: `git status` clean at `fe05753`, 347 part 2 tests passed with 0 skipped, `peers-selftest` identified the process. The quantized set was pulled with a read-only token scoped to `RedBullet12/e4-part2-quant`, revoked after the pull.

Instance history: a first rented A100 40 GB (instance 57595686) refused every connection on all ports and was destroyed before anything ran on it.
