# H3 acceptance mapping (2026-10-29)

Status: **selected technical deliverables completed; final review pending**.
The additional pipeline direction now has two passing real GPU runs and two
fresh-process reloads. Acceptance and moving the card to Done remain the
reviewer's decision; no upstream issue is silently closed by this mapping.

The selected extra direction is a bounded two-stage Qwen2 + PEFT/LoRA pipeline
smoke using native `torch.distributed.pipelining`, related to upstream
[#39](https://github.com/MakazhanAlpamys/Soup/issues/39). It is an explicit research
recipe, not a replacement of Soup's HF Trainer or complete general CLI pipeline
support. The 70B/DeepSeek/4xA100 criteria of the broader issue are not claimed.

| H3 checklist / criterion | Direct evidence | Status and exact scope |
|---|---|---|
| FSDP2 + PEFT training; finite loss/gradients, actual optimizer updates, frozen base | [FP32 report](h3-fsdp-followup-20261027.md), [math BF16](h3-bf16-result-20261027.md), [Flash BF16](h3-flash-bf16-result-20261027.md); rank records and tensor references in each linked artifact manifest | **MEASURED** on real Qwen2.5-0.5B and two equal GPUs. Initial zero A updates are allowed; B updates initially and all adapters finally. No 70B claim. |
| Synchronization, distributed data and actual batch/GAS | The same reports, actual data-window records, gradient/adapter hashes and first-step tensor comparisons; [gate tests](../tests/test_h3_fsdp_gate.py) | **MEASURED** for selected FSDP2 runs. Global8, micro1, GAS8 on one GPU / GAS4 on two. |
| Actual parameter sharding; per-rank GPU and host memory (#373) | Raw named DTensor records via `to_local()`, per-rank allocated/reserved peaks and sampled RAM in reports/artifacts; existing [probe](harness/fsdp_sharding_probe.py) and [regression tests](../tests/test_issue373_fsdp_sharding_probe.py) | **MEASURED**: 247,286,720 local elements/rank, 27 FSDP2 modules/rank; 494,573,440 unique total. Existing #373 repair reused. Larger-model upstream criteria remain unmeasured. |
| Adapter checkpoint excludes base weights (#352) | Existing [checkpoint regression](../tests/test_issue352_fsdp_peft_checkpoint.py), report checkpoint inventories, retained LoRA adapter saves and full archive manifests | **VERIFIED for tested stack/model**. Existing upstream serialization repair reused. This is not a newly reproduced/fixed 37 GB large-model incident, nor complete #352 resume acceptance. |
| Adapter save and separate-process reload | Eight accepted root/checkpoint reloads per campaign, 24 total; exact 96 adapter keys/values and zero observed eval-logit error; [FP32 artifacts](evidence/h3-fsdp-20261027/hardware/), [BF16 artifacts](evidence/h3-bf16-20261027/hardware/README.md), [Flash artifacts](evidence/h3-flash-bf16-20261027/hardware/README.md) | **MEASURED** over the same immutable base and fixed eval inputs. Each campaign has its declared precision-specific tolerance. Reload is not optimizer/scheduler resume. |
| Additional selected direction: pipeline (#39) | [GPU result](h3-pipeline-result-20261029.md), [raw references/ledger](evidence/h3-pipeline-20261029/hardware/README.md), [protocol](h3-pipeline-protocol-20261029.md), [harness](harness/h3_pipeline_smoke.py), [tests](../tests/test_h3_pipeline_smoke.py) | **MEASURED on two RTX 3090s**. Two fresh four-step native GPipe runs, global batch 8, 16 actual forwards/backwards per stage. Gradients/updates pass the unchanged ordinary-control gate; all 96 final adapter tensors update. Disjoint ownership, frozen bases and two fresh reloads pass; both reload errors are zero. This is the bounded standalone recipe, not general CLI pipeline support or all of #39. |
| Scaling baseline/formula and available 2/4/8 GPUs | Independent same-host campaigns in the three reports above; raw synchronized training times, fixed configurations and repeat order retained | **MEASURED 1→2**; **4/8 UNMEASURED**. Speedup=throughput(N)/throughput(1), efficiency=speedup/N. 4/8 absence is explicitly allowed by H3. |
| Local tests before rental and review regression | [RAM-efficient loading follow-up](h3-fsdp-review-20261028.md), [logs](evidence/h3-fsdp-review-20261028/README.md), [pre-rental pipeline checks](evidence/h3-pipeline-20261029/local/README.md), [launcher-fix log](evidence/h3-pipeline-20261029/hardware/support/launcher-fix-pytest.log) | Pre-rental expanded selection: **350 passed**. Launcher fix: **14 pipeline tests passed**, including native two-rank CPU training/reload and a real-subprocess interpreter regression. Counts overlap and are not summed. Full-scope Ruff and shell syntax passed. GPU claims use separate real hardware records. |
| Reproducibility: commands, config, source/version hashes, raw logs and failed controls | Each report links its protocol and artifact manifest; [native Flash ledger](evidence/h3-flash-bf16-20261027/hardware/README.md) and [pipeline ledger](evidence/h3-pipeline-20261029/hardware/README.md) separate accepted and rejected cases | **RETAINED**. Older large tensors/Arrow files have full local archive paths/SHA256. All pipeline numeric references, data, logs and merged adapters are public, with verified full archives retained as well. |

## Existing scaling results (separate cohorts)

| Campaign | Tokens/s, 1 GPU | Tokens/s, 2 GPUs | Speedup | Efficiency | 4/8 GPU |
|---|---:|---:|---:|---:|---|
| IEEE FP32, RTX3090 | 4216.4633 | 2075.1697 | 0.492159 | 0.246079 | UNMEASURED |
| Conservative math BF16, RTX4090 | 6932.1733 | 7584.7686 | 1.094140 | 0.547070 | UNMEASURED |
| Deterministic native Flash BF16, RTX4090 | 8178.1305 | 8102.4617 | 0.990747 | 0.495374 | UNMEASURED |

Tokens/s here counts actual non-padding input tokens from all ranks divided by
the maximum-rank sum of synchronized training-step windows. Setup, heavy
correctness probes, warmup and save/reload are excluded. These are different
arithmetic/backend/rental cohorts; no cross-cohort speed comparison is claimed.
The pipeline smoke tests correctness and does not add another scaling measurement.

## Explicit limits and review status

- The selected additional direction is now the measured two-stage pipeline
  recipe. Its GPU result is distinct from the pre-rental random CPU fixture;
  general CLI wiring and the larger upstream recipe matrix are separate work.
- Nondeterministic native Flash defaults still fail the unchanged strict
  cross-world first-step gate and are excluded from accepted scaling.
- Optimizer/scheduler/RNG resume, 70B, quality and physical multi-node execution
  remain unmeasured. Ring/context attention, AutoTP and the large recipe CI
  matrix are alternatives not selected for this submission.
- H1's authoritative card remains unavailable; the pinned environment and local
  checks are evidence, while H1 organizational completion is unknown.
- Full upstream #39/#352/#373/#41 closure is not implied by this H3 slice.
  Issue numbers above refer to MakazhanAlpamys/Soup, not similarly numbered R&D PRs.

The original GPU evidence is unchanged by this acceptance mapping. The
RAM-efficient loading bug was fixed in `301c555d`; its CPU loading regressions
do not add GPU RAM-efficient training evidence. Final owner/reviewer acceptance
is still required; the selected technical deliverables and direct evidence are
now ready for that review.
