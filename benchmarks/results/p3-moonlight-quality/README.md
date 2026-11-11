# Moonlight-16B-A3B quality comparison

## Execution authority

The user approved the restored [plan](../../p3-moonlight-quality-plan.md) on 2026-11-11, including remote pre-rental reference checks and one Vast rental within **USD 15 total**. Computation must stop at USD 14, preserving USD 1 for export and deletion. Before rental, publish the frozen protocol SHA256 and a conservative upper cost bound. If necessary, reduce only the frozen-order HellaSwag prefix, then omit the L4096 pair; keep 64 training steps, thresholds and the v2 decision rule unchanged. If the upper bound still exceeds USD 15, do not rent.

Work is on `research/p3-moonlight-quality`, based on the current reviewed parent `8bc5826d370b349993e790dcc7643aa7f4b489d4`. PR #61 and its branch are unchanged. The plan's older parenthetical `e6fa7136` predates the retained download/review fixes.

## Current status

| Result | Observed status |
| --- | --- |
| Historical tiny E3 | **FAIL 0.05028628354277727 > 0.05**, unchanged and not rerun |
| DeepSeek-V3 671B | **отложен**; full-model weights, gate, training and evaluation were not run |
| Fast tokenizer versus tiktoken | **PASS**: 348941 text checks, exact IDs/offsets, two byte-identical generations |
| Native aux-field admission | **PASS v2**: no numerical consumer; exact FP32/BF16 loss, logits, expert IDs and all 24 LoRA gradients |
| Final pinned CPU integration | **PASS**: 733 passed, 0 failed, 50 GPU cases skipped; all 14 original PR #61 test files unchanged |
| Existing v2 analysis tests and summary CLI | **PASS**: 241 passed, no skips; synthetic paired summaries, NLL direction and same-size hash-tamper refusal verified |
| Ordinary CI with Rich 14.0.0 | **PASS**: 732 passed; 50 CUDA skips plus one explicit scientific-stack skip; production admission still refuses the changed package |
| Native/reference model comparison | **PASS**: exact ordered top-6 on all 64 tokens; max error `2.086162567138672e-7` ≤ `0.00011039664745330811` |
| Moonlight quality, gated verdict | **NOT RUN**; actual cache/initial-adapter hashes and an admitted whole-workload cost bound remain unavailable |
| Moonlight quality, statistical verdict | NOT RUN |
| Moonlight numerical gate | **NOT RUN (0/6)**; A3, E1/E2/E3 and INT4 control have no Moonlight result |
| Per-step loss and adapter-change norm | NOT RUN |
| Moonlight paid resources and charges | **USD 0**; no instance or volume created by this run; one unrelated instance left untouched |

The terminated DeepSeek preparation was a separate attempt: instance `60187378` and volume `60187377` were deleted and confirmed absent. Its provider-reported charge was USD 0.159, not a Moonlight training or evaluation expense.

## Fixed comparison

Independent BF16 L2L and FP8 block64 E4M3 + Oracle runs start from one identical initial rank-2 LoRA. The full planned matrix is L1024/k4 and L4096/k1, B1, seed 23, 64 steps each. The base model is not evaluated. Frozen Granite v2 task/scoring data and the v2 statistical and informativeness decisions apply. Keep per-cell results and secondary NLL separate from the numerical diagnostic.

All six numerical gate passes must complete unless a technical failure prevents valid execution. A3, E1/E2/E3 and INT4 discrimination remain unchanged. Numerical FAIL is recorded, never retried to obtain PASS. NaN/Inf, absent Oracle VJP, RoutingGuard refusal, OOM or evidence loss invalidates the affected arm and its pair. Budget/deadline stops the entire run.

No paid execution before exact fast-tokenizer/tiktoken parity, the native aux-field no-consumer proof, and pinned native/reference model parity. Any specified mismatch stops before rental. Project execution and tests run remotely, never on the workstation. Native execution requires Ampere or newer, not Kaggle T4/P100.

## Authorized continuation

The user replaced the broad native-read stop with a no-numerical-consumer criterion. Generic initialization validation in `configuration_utils.py` and `modeling_rope_utils.py`, plus unchanged Oracle protective guards, is allowed. Native Transformers 5.19.0, Soup streaming/L2L and PEFT still require a source audit. Separate FP32 and BF16 tiny-model comparisons must show bitwise-equal loss, logits, selected experts and every LoRA gradient, without Oracle attached.

[Amended admission contract](aux-admission-contract-v2.json), SHA256 `12e9a1f548d3192035f5781e0d568642480a94a383d48fb1f5373a8f2e178cd0`, was recorded before v2 execution and must enter the final protocol before its hash. The Oracle hash binds unchanged canonical Git bytes, including their original CRLF representation. The accepted v2 proof now permits the two authorized runtime-config removals. The $15 total and $14 plus $1 split are unchanged.

### Accepted v2 evidence

The [aux v2 report](prelaunch-review-v1/aux-report-v2.json), SHA256 `3b56a594a381822bd33c11cea01d6ff5b1d2ef9508522924f148b1431f62f27a`, passed on Transformers 5.19.0 / PEFT 0.21.2 / Torch 2.11.0+cu130, CPU, without Oracle attached. Both precisions have complete and bitwise-identical loss, logits, routing and all 24 named LoRA gradients. All 495 retained source files and 35 tensor-artifact bindings were rehashed after retrieval; installed audited files match wheel RECORD. Native modeling/configuration and PEFT have no numerical consumers. Observed generic config reads are at `transformers/configuration_utils.py:558`; Oracle's unchanged protective reads remain at `p3_deepseek_oracle.py:460-462,473-474`. The Kimi-only reads at `src/soup_cli/utils/kimi_k2.py:33,85` are outside the native DeepSeek-v3 execution path.

The actual unchanged Oracle `_check_config` and pre-hook accept the normalized config and zero/None kwargs, and reject the original nonzero coefficient. Constructor bypass means this proves guard behavior only, **not CUDA attachment or Oracle VJP**. Final integration also passed two independent real CPU NF4 sessions: both model skeleton constructors received the normalized config, the shared initial adapter reloaded exactly, original source bytes stayed unchanged, and owned temporary directories were deleted.

The production tokenizer CLI independently reproduced exact parity on all 348941 checks. The training-pack manifest is `2d142c8f1641bd55173910311744455a79c4be5c9152cf181673d3965247cee4`, with 64 windows and 178545 supervised targets at each length. The public pack CLI reproduced this exact hash in integration/reference v2, before any reference worker execution.

- [Aux v2 source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-aux-v2-20261111t125603z): full retained archive has 613 members, SHA256 `9e6eb5b3aea1d7899e4283e5e30e59957addf0560a8df2f1805d30a80dc4d898`.
- [Production tokenizer/data source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-data-20261111t123842z): full retained archive has 35 members, SHA256 `214e2c9adda7dbf5bfb98a4ab32fda42f0e7a392a44be2aae9dcf4d2a632813a`.
- [Archive manifest v2](prelaunch-review-v1/archive-manifest-v2.json): every member reopened and SHA256-verified, including execution evidence and reproduction packets.

## Historical v1 pre-rental stop

On native Transformers 5.19.0, the trace observed `aux_loss_alpha` entering `PreTrainedConfig.__getattribute__` from `configuration_utils.py:558`, in `validate_token_ids()`. The installed source, verified against its wheel RECORD, contains:

```python
for name in text_config:
    value = getattr(text_config, name)
    if name.endswith("_token_id") and isinstance(value, int) and not 0 <= value < vocab_size:
```

The getter runs before the suffix filter. This is a generic validation read, **not evidence that aux loss affects training**. Nevertheless, the user's explicit stop condition covers native reads of these fields. No exemption was introduced. The probe stopped at the first observed read, before model construction or forward/backward, so bitwise loss/gradient equality remains unproved. No further claim about `seq_aux` is made.

The aux check was not weakened or repeated. Oracle and runtime configs were not changed. Reference-weight download, protocol freeze, rental, numerical gate, training and evaluation did not start. Preliminary quotes are not a frozen upper workload estimate or spending admission.

## Training and quality cells

| Length / accumulation | Arm | Completed steps | Step loss | Adapter delta norm | Primary quality / secondary NLL |
| --- | --- | ---: | --- | --- | --- |
| L1024 / k4 | BF16 L2L | 0/64 | NOT RUN | NOT RUN | NOT RUN |
| L1024 / k4 | FP8 block64 + Oracle | 0/64 | NOT RUN | NOT RUN | NOT RUN |
| L4096 / k1 | BF16 L2L | 0/64 | NOT RUN | NOT RUN | NOT RUN |
| L4096 / k1 | FP8 block64 + Oracle | 0/64 | NOT RUN | NOT RUN | NOT RUN |

No baseline model was evaluated. The numerical gate's non-blocking rule remains unchanged. Reference admission passed; final verification and artifact/cost freeze remain pending. Planned limits remain one seed and rank-2 LoRA. The planned approximately four MBPP-train cycles were not executed.

## Resources and retained evidence

| Remote CPU command | Wall seconds | Result |
| --- | ---: | --- |
| Tokenizer conversion and full-corpus parity | 381.099 | PASS |
| Native aux-field proof | 23.239 | STOP during config initialization |
| Production tokenizer conversion CLI | 25.604 | PASS |
| Production full-corpus parity CLI | 341.977 | PASS, 348941 checks |
| Candidate pack assembly | 10.266 | PASS; subsequently reproduced by the public CLI |
| Amended native aux-field proof v2 | 46.075 | PASS; worker-internal wall 42.286 s |
| CPU integration v1 | 352.886 | FAIL; 21 test modules, CLI checks, live Hub and source continuity retained |
| CPU integration v2 | 300.277 | PASS; 686 passed, 50 GPU skips; all admission requirements passed |
| Public pack CLI reproduction | 18.578 | PASS; exact previously admitted manifest SHA256 |
| First two complete reference-shard downloads | 15.215 | PASS; 2006757520 bytes, both full-file SHA256 values verified |
| Reference verifier v2 | 7.642 | Stopped at source header admission, before either model worker/comparison |
| CPU integration v3 | 327.432 | PASS; 687 passed, 50 GPU skips; source-layout correction verified |
| Successful reference preflight v3 | 329.327 | PASS; first actual numerical comparison, 216.635 s inside verifier |

These are command-wall measurements, not GPU training timings. RSS/cgroup/CUDA peaks for paid passes, training and evaluation do not exist because those stages were not run. Integration v1 observed a reaped-child lifetime RSS high-water of 1415057408 bytes; this is not a concurrent process-tree or cgroup peak. Completed CPU jobs terminated. No paid instance or volume required deletion; [resource closeout](prelaunch-review-v1/resource-closeout-v1.json) records successful provider queries, empty owned resource IDs and zero Moonlight expense.

- [Tokenizer report](prelaunch-review-v1/tokenizer-report-v1.json) and [verified retrieval](prelaunch-review-v1/tokenizer-retrieval-v1.json): all 7801 training rows and all 27795 frozen dev/eval rows covered; 348941 text checks. `len(tokenizer)=163840`, BOS 163584, EOS 163585, PAD 163838.
- [Aux report](prelaunch-review-v1/aux-report-v1.json), [call trace](prelaunch-review-v1/aux-call-trace-v1.json) and [retrieval verification](prelaunch-review-v1/aux-retrieval-v1.json): observed caller/line, failure stage, and twenty retained installed-source files verified by SHA256 and wheel RECORD.
- [Tokenizer evidence and reproduction packet](prelaunch-review-v1/tokenizer-admission-v1.zip), SHA256 `89fc45b2a9499ee60d26dc5568cb498bd5ed234843db830cc7a7025234b2d833`.
- [Aux evidence and reproduction packet](prelaunch-review-v1/aux-admission-v1.zip), SHA256 `dc221d8409b95fe8ac96dcb3d8738efcc620a3e241b51831a742018632f8e4c6`.
- [Archive manifest](prelaunch-review-v1/archive-manifest-v1.json): every archived member was checked byte-for-byte by SHA256. Source packets, executed probes, bootstrap code, provider metadata, logs, generated tokenizers and original diagnostic sources are retained.

At v1 closeout, production code and existing tests were unchanged; two incomplete new test files were removed. The user then authorized the amended criterion. Aux/tokenizer admission, CPU integration v3 and the first actual reference comparison passed. Final reviews found a derived-metric classification defect and ordinary-CI dependency coupling; their corrections await remote verification. No rental, completed Moonlight implementation PR or frozen study protocol is claimed.

## Integration v1 evidence

Kernel `batyrkhan2131123/soup-moonlight-integration-ref-v1-20261111t142239z` preserved all 14 original PR #61 test files at reviewed parent `8bc5826d`. Live inventory/32 HEAD checks transferred only the 1375-byte config, zero pretrained weight bytes. Installed-source continuity, unchanged Soup/Oracle bytes, CLI refusals, bounded success/deadline cleanup and scoring locality passed. The standalone normalized-session smoke refused an output cache outside the allowed temporary roots. Later verification corrected its directory choice, not the containment guard.

[Historical v1 source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-integration-ref-v1-20261111t142239z): full retained ZIP SHA256 `dececfa6a6d7d76b1345263cf6b55234fb4bac9a78f93197cf25965ee5127270`, 34167980 bytes, 236 verified listed artifacts plus the archive index. The source admission is `c128d3801d144fcd81ff68ee1543d73e47c9f7da8a3e5bc73596c060277cf182`. This failed integration is historical evidence, not a numerical gate or reference mismatch.

## Integration v2 and actual checkpoint headers

Kernel `batyrkhan2131123/soup-moonlight-intref-v2-20261111t144948z` passed all CPU integration requirements with original #61 tests unchanged. Its subsequent reference preparation verified the public pack, isolated native/reference environments and both complete shard payloads. The verifier then refused `model.layers.0.self_attn.rotary_emb.inv_freq` before starting model workers. There is **no numerical native/reference result** from this attempt.

[Historical v2 source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-intref-v2-20261111t144948z): full retained ZIP SHA256 `d8f376d995e767a3894a07292497dfcf9771c47e34a51290cdda448ba6e9a6b6`, 45005218 bytes; all 290 listed artifacts rehashed after download. Source admission: `15f0f12bba287ea4d14143dc745c0b4aa940f0cf9b4843e7006acd033961df23`. Both owned remote temporary roots were deleted, including downloaded weights.

The [actual headers of all 27 pinned shards](source-headers-v1.json), SHA256 `e8cdfb3617b0963bcb08340011084dde08f863120b753d6a03ced212f4825c2b`, contain 5344 BF16 tensors. All 27 legacy rotary buffers have shape `[64]`, not the assumed `[32]`; they total 1728 elements. Excluding these nonpersistent buffers leaves **15960110208 checkpoint tensor elements**, including persistent router correction biases. The earlier FP32-buffer explanation was incorrect. Only the first two full payloads were hashed here; the remaining evidence is HTTP-range headers, not full-model download qualification.

Source admission now requires the observed legacy shape. Native/reference constructors still regenerate their unchanged `[32]` nonpersistent buffers; checkpoint frequencies are not injected into either model. Shape-incomparable frequency differences are explicitly unavailable diagnostics, never an acceptance threshold. Corrected v3 packet admission: `8338566f276ad08eb56221b225a264174fa0046c7c0dde3707bce154eba9bc5d`. No numerical tolerance or gate rule was changed.

## Rental remains unadmitted

The plan requires protocol freeze before rental and actual cache/shared-initial-adapter hashes in that protocol. Those production artifacts do not exist; the supported preparation route creates them on an admitted native-BF16 GPU. No recipe hash, placeholder or post-freeze backfill is substituted.

[Read-only quote v2](prelaunch-review-v1/quote-only-v2.json) records RTX 3090 offer `56970728` and same-machine local-volume offer `56970732`: GPU + 32 GiB root + 180 GiB volume total **$0.275555556/hour**, ingress $0.005208333/GB, egress $0.0078125/GB. This is not a reserved offer or an admitted whole-workload upper estimate. Historical Granite dense-BF16 T4 timings are retained, but are not Moonlight disk-L2L measurements. No scope reduction is frozen.

[Final resource inspection](prelaunch-review-v1/resource-closeout-v3.json), `2026-11-11T16:43:53.726305+00:00`: no resources were created by this Moonlight run; provider returned one unrelated running instance and no volumes. No deletion call was made against that instance. Moonlight cost remains **$0**. The $15 total/$14 compute/$1 closeout constraints are unchanged.

## First actual native/reference comparison: PASS

[Reference admission summary](reference-admission-v1.json), SHA256 `c8b24896ef65f2c4bf42b8969eb632823d74dc1d244a7498317db3a798f04b7c`, binds the successful v3 archive and raw worker evidence. Native and reference ran CPU FP32/eager on the real embedding and first two checkpoint layers, using the same 64 admitted pack tokens. All ordered top-6 expert choices matched. Layer-1 pre-final-norm max-absolute error was `2.086162567138672e-7`; reference max-absolute value was `1.103966474533081`, so the unchanged `1e-4` relative limit was `0.00011039664745330811`. No additive tolerance or numerical retry was used.

[Successful v3 source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-intref-v3-20261111t151049z): full retained ZIP is 70121259 bytes, SHA256 `28c4332d8e32bbd8d44e5dd5206eeeaba2a68f46188ba5ece251a16f1313aba1`; all 3200 listed artifacts rehashed after retrieval. Full verifier report SHA256: `861bb9eed846b05bdbcae6c30ef3934d805230245652a7a30b7d11cf94852569`. Both owned temporary roots were deleted, including downloaded shards. This is partial-model correctness evidence, **not native CUDA/BF16 qualification or model-quality evaluation**.

The reference runner observed a reaped-child lifetime RSS high-water of 9627410432 bytes and self high-water of 147247104 bytes. These are separate lifetime maxima, not concurrent tree/cgroup peaks. The two reference-preparation attempts downloaded 4013515040 weight bytes in total; v2 stopped before workers, v3 performed the only numerical comparison. Both were free CPU runs, with no Moonlight charge.

## Final source-bound verification and publication

[Verification summary](verification-summary-v1.json) binds the final source packets, original JUnit results, CLI/process checks, installed-source admission and cleanup receipts. The [compact CPU records](prelaunch-review-v1/final-verification-records-v1.zip) retain every original evidence file except the duplicate repository source archive; `publication.json` names that omission and its SHA256. The full packet remains on [its private source kernel](https://www.kaggle.com/code/batyrkhan2131123/soup-moonlight-final-v1-20261111t161937z).

Finite raw zero-reference gradients now retain their independently reconstructed undefined relative diagnostics as JSON nulls with exact paths. The standalone public loader admitted a synthetic COMPLETE/NO_VERDICT receipt without pytest fixtures or monkeypatches. Raw NaN/Inf, missing Oracle VJP, bad routing, absent/contradictory FP8 cleanup and boolean substitutions for numeric metrics remain refusals. This API smoke is not one of the six scientific gate passes.

The first final packet omitted the existing `p3_stage2v2_results.py` dependency and correctly failed summary import. [Unchanged prerequisite provenance](analysis-prerequisites-v1.json) records the restored helper, its original tests and the historical runner they read as a byte fixture. No new controller was written or invoked. The [analysis-only verification](prelaunch-review-v1/analysis-verification-records-v1.zip) then passed full Ruff and the exact runbook summary command, including SciPy 1.15.3 wheel RECORD validation. Synthetic floor and informative cases are API checks, not Moonlight quality outcomes. No accepted tokenizer, aux or reference comparison was repeated.

[Compact raw scientific proofs](prelaunch-review-v1/scientific-proofs-v1/manifest.json) preserve all 35 auxiliary-comparison tensor artifacts, both reference workers' outputs, residuals and comparison reports. Full original archives, expanded temporary evidence and duplicate source packets are not added to Git. The source kernels are private; authorized access is required for their complete retained outputs.

The public Soup CLI/API is unchanged; no product changelog entry is needed. Full-repository pytest and native Ampere execution remain unrun. Proceeding to paid work requires an explicit amendment resolving the pre-rental artifact-hash dependency and permission for a bounded runtime calibration, or already-qualified artifacts and measurements. No final study protocol, scope cut or rental was invented.
