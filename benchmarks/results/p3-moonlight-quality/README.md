# Moonlight-16B-A3B quality comparison

## Execution authority

Current authority is **USD 30 cumulative**, including previous charges, for one more unattended whole-machine evaluation on machine140544; see the [unattended-evaluation authorization](unattended-eval-authorization-v1.json), which records the user's answers, the watchdog limits (stop request at $29, deletion at $29.8, no new scoring after 2026-11-13 20:30 UTC) and the unchanged protocol `d22871c3…`. The earlier $25 records below are historical. Earlier USD15/USD1 preparation records are not current permission.

Work is on `research/p3-moonlight-quality`, based on the current reviewed parent `8bc5826d370b349993e790dcc7643aa7f4b489d4`. PR #61 and its branch are unchanged. The plan's older parenthetical `e6fa7136` predates the retained download/review fixes.

## Current status

| Result | Observed status |
| --- | --- |
| Historical tiny E3 | **FAIL 0.05028628354277727 > 0.05**, unchanged and not rerun |
| DeepSeek-V3 671B | **отложен**; full-model weights, gate, training and evaluation were not run |
| Fast tokenizer versus tiktoken | **PASS**: 348941 text checks, exact IDs/offsets, two byte-identical generations |
| Native aux-field admission | **PASS v2**: no numerical consumer; exact FP32/BF16 loss, logits, expert IDs and all 24 LoRA gradients |
| Pre-residency pinned CPU packet | **PASS**: 733 passed, 0 failed, 50 GPU cases skipped; all 14 original PR #61 test files unchanged |
| Existing v2 analysis tests and summary CLI | **PASS**: 241 passed, no skips; synthetic paired summaries, NLL direction and same-size hash-tamper refusal verified |
| Ordinary CI with Rich 14.0.0 | **PASS**: 732 passed; 50 CUDA skips plus one explicit scientific-stack skip; production admission still refuses the changed package |
| Resident NF4 integration | **PASS**: 79 focused tests, full remote Ruff, native8192 capacity and bitwise24-task/80-choice parity with disk |
| Native/reference model comparison | **PASS**: exact ordered top-6 on all 64 tokens; max error `2.086162567138672e-7` ≤ `0.00011039664745330811` |
| Native source/cache/shared initializer | **PASS**: all 27 shards; NF4/noDQ/per-expert cache; one exported initial adapter; session closed |
| CUDA technical smoke | **COMPLETE** on RTX4090; qualification fixture only, not Moonlight quality or historical tiny E3 |
| Native resource measurements | **COMPLETE** on machine140544: four no-update training windows and resident dev measurements with valid F1; no timings transferred from another GPU |
| Moonlight quality, gated verdict | **RUNNING, not reached**: the first run was blocked by an evaluator defect (HellaSwag refused on 3 of 3 adapters after about 3510 profiler cycles). A second, unattended evaluation of all four adapters is in progress with smaller pieces. Not a degradation verdict |
| Moonlight quality, statistical verdict | **RUNNING, not reached**; no subset replaces the frozen study |
| Moonlight numerical gate | **PASS, 6/6**; complete raw closure independently readmitted and exported |
| Per-step training loss and adapter-change norm | **4/4 independent64-step arms complete**, all archives verified; BF16 L1024 timing VOID, other three blocks F1 ok |
| Moonlight paid resources and charges | First host: **USD7.189 posted** by 2026-11-12T07:18UTC, deleted and verified absent. Second host (instance60329118, volume60329113) was created at 08:14UTC and is **active**; the watchdog closes it after the results download. Charges there are not yet posted |

The terminated DeepSeek preparation was a separate attempt: instance `60187378` and volume `60187377` were deleted and confirmed absent. Its provider-reported charge was USD 0.159, not a Moonlight training or evaluation expense.

## Fixed comparison

Independent BF16 L2L and FP8 block64 E4M3 + Oracle runs start from one identical initial rank-2 LoRA. The full planned matrix is L1024/k4 and L4096/k1, B1, seed 23, 64 steps each. The base model is not evaluated. Frozen Granite v2 task/scoring data and the v2 statistical and informativeness decisions apply. Keep per-cell results and secondary NLL separate from the numerical diagnostic.

All six numerical gate passes must complete unless a technical failure prevents valid execution. A3, E1/E2/E3 and INT4 discrimination remain unchanged. Numerical FAIL is recorded, never retried to obtain PASS. NaN/Inf, absent Oracle VJP, RoutingGuard refusal, OOM or evidence loss invalidates the affected arm and its pair. Budget/deadline stops the entire run.

No paid execution before exact fast-tokenizer/tiktoken parity, the native aux-field no-consumer proof, and pinned native/reference model parity. Any specified mismatch stops before rental. Project execution and tests run remotely, never on the workstation. Native execution requires Ampere or newer, not Kaggle T4/P100.

## Earlier auxiliary-field admission

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

No baseline model was evaluated. The numerical gate's non-blocking rule remains unchanged. CPU verification and reference admission passed; actual artifact/cost freeze awaits the explicitly authorized bounded preparation and measurement. Planned limits remain one seed and rank-2 LoRA. The planned approximately four MBPP-train cycles were not executed.

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

## Historical rental blocker and explicit amendment

The original pre-rental artifact-hash requirement was circular: the production CUDA preparation creates the cache and shared initial adapter only after rental. This was reported rather than replacing artifact hashes with recipe hashes or placeholders. The user then explicitly approved two-stage freeze and at most $1 preparation/measurement within the unchanged total.

The historical [read-only quote v2](prelaunch-review-v1/quote-only-v2.json) is not a reservation or a current workload estimate. Fresh instance/volume pricing, the preparation ceiling and scientific-settings SHA256 must be recorded before creating the one rental. No final scope cut or whole-workload cost is inferred from Granite dense-BF16 T4 timings.

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

The public Soup CLI/API is unchanged; no product changelog entry is needed. Full-repository pytest and native Ampere execution remain unrun. The explicit amendment now authorizes bounded preparation/measurement after pre-rental settings/cost freeze. Final actual-artifact protocol, measured scope admission, gate, training and quality results are still pending; none is claimed by the CPU verification.

## Pre-rental freeze after explicit authorization

Scientific-settings SHA256: `f71d5e0ba4f4de381bfb20447410458a89c588572343314cb998ec031ff4c74b`.
Preparation-ledger SHA256: `0f9bfaf8a1ae15a8cbf22318c14c02155221a463b3d6ca604b11bb44782a7bd2`.
Deployment archive SHA256: `d0ff819e27bdcea710b0cdea6a4e16f3fb8121e3100d7271ecc391ce4013a1df`.

The original RTX 3090 offer became unavailable before creation. The frozen replacement is RTX 4090 offer `58143447`, machine `155543`, matching 220 GiB volume offer `59075301`, plus 32 GiB root. Combined hourly price is $0.526296296; ingress/egress $0.00390625/GB. The conservative preparation upper bound is **$0.952534722**, including at most 45 minutes from creation, 128 GB ingress, 2 GB egress and $0.05 contingency. Per-command maxima do not extend that absolute deadline. This is preparation admission only, not a whole-study estimate or quality result.

The exact resource-probe SHA256 is `1aca198c7eceb81b9856f58351091e362635c9a64ebe4b7be539a12dc8905eed`. The existing OpenSSH/SCP client will pull evidence to persistent local storage and verify each archive SHA256; no new uploader, rental controller or automatic deleter is installed.

## Paid hardware admission: technical stop, resources deleted

Run `moonlight-q-20261111t175332z` created the single authorized RTX 4090 rental. Actual mount evidence showed `/data` on **`/dev/md0p2`**, xfs, major/minor `259:7`, with sysfs path **`/sys/devices/virtual/block/md0/md0p2`**. This is Linux MD RAID, explicitly disallowed by the direct-NVMe admission. The provider's Intel NVMe model/throughput description did not prove the mounted storage topology. No guard was weakened and no replacement rental was created.

- [Hardware evidence](moonlight-q-20261111t175332z/initial-host-v2.txt) and [technical-stop receipt](moonlight-q-20261111t175332z/hardware-stop.json): actual RTX 4090, 24564 MiB VRAM; source-weight bytes downloaded **0**. Package setup, conversion, initial adapter, smoke, calibration, gate, training and quality evaluation were not executed.
- [Deletion/absence proof](moonlight-q-20261111t175332z/resource-closeout.json): instance `60241118` and volume `60241117` are absent in fresh provider listings. Unrelated resources were untouched. The first instance-delete command received EOF at its confirmation and aborted; the explicit authorized confirmation then deleted the exact instance, followed by the exact volume.
- [Provider charges](moonlight-q-20261111t175332z/charges.json): instance $0.032, volume $0.005, total **$0.037** at the recorded query. Provider reports mill precision and may post late adjustments; no resource remains billable. This is below the $1 preparation ceiling.
- Free tokenizer-only workload inventory completed without loading a model: one final adapter requires 23795 tasks, 67831 requests, 19717 batches and **20401 native forward calls** before any permitted cuts/sharding. These are source-derived counts, not measured GPU timings. Archive SHA256 `8b88c669f28985daa7b4cb18bd1aa83c710d54d0592a4097711f2a77b4a571e8`; inventory-manifest SHA256 `38e8f989d3683a2159484b828b4ba94f5dc219ab8b7446373efc087b2812dc5b`.

Quality gated/statistical: **NOT RUN**. Numerical gate: **NOT RUN, 0/6**, not numerical FAIL. Per-step loss and adapter-change norm: **NOT RUN**. Historical tiny E3 remains **FAIL 0.05028628354277727 > 0.05**. **DeepSeek-V3 671B: отложен.**

Continuation requires explicit authorization for another rental with verified direct NVMe, within the remaining original budget. The one-rental authorization has been consumed. The final artifact-bound protocol and whole-study measured cost bound do not exist.

## Explicitly authorized replacement

The user selected «Разрешить ещё одну аренду» after verified closeout. [Replacement authorization](replacement-authorization-v1.json) permits exactly one additional rental without raising any monetary/scientific limit. The existing $0.037 counts toward both the $1 preparation cap and $15 total; direct-NVMe admission remains mandatory.

Replacement [preparation ledger v2](preparation-ledger-v2.json), SHA256 `84c93d8e9c9a462912e85d42ecc253516d2500d438e5992a701f486ee494b660`, freezes RTX4090 offer `51365776`, matching volume `51365777`, machine `148949`: Ryzen 9 3900X, one GPU, advertised WD_BLACK SN850X 2TB. Combined price $0.58/hour. Upper replacement preparation cost $0.942145833, cumulative with the closed attempt **$0.979145833**; maximum replacement preparation duration 57 minutes. Actual mount topology is checked first, before packages or weights. The original scientific-settings SHA256 is unchanged; the new authorization supersedes only its one-rental count.

Replacement offer51365776 was rejected as unavailable and created no resources. Before the replacement rental, ledger v3 SHA256 `4c457258dd79c94623c90010b8a281636aaa80483e19e5e3a6dc4584c478af6c` supersedes unrented v2: offer51950781, volume51950784, machine136748, RTX4090/Core Ultra9 285K, advertised MSI M482 2TB. Combined $0.486296296296/hour; ingress/egress $0.000130208333/GB. Upper replacement preparation $0.796371527778; cumulative including the closed $0.037 attempt $0.833371527778; hard90-minute replacement deadline. Provider reliability0.9531117 is disclosed, not execution evidence. Actual direct-NVMe verification is still first.

## Replacement startup failed; all owned resources absent

The explicitly authorized replacement created instance `60244148` and linked volume `60244147`. The host repeatedly reported **`Error: Requested volume does not exist: V.60244147`** and never exposed SSH. One start of the same stopped instance after the volume contract appeared did not resolve it; this was not another rental or a scientific retry. The final observed startup interval exceeded ten minutes.

[Startup-stop receipt](moonlight-r-20261111t180727z/startup-stop.json), [provider daemon log](moonlight-r-20261111t180727z/daemon-startup-log.txt) and [same-instance recovery receipt](moonlight-r-20261111t180727z/same-instance-start.json) retain the failure. The quote advertised `MSI M482 2TB`; the rented-volume inventory instead reported `USB SanDisk`. Neither is native mount proof: **physical storage topology on this replacement was never verified**.

Instance `60244148` was destroyed. The subsequent explicit volume-delete request returned 404 because volume `60244147` was already absent; this is not reported as a successful delete response. Fresh [provider listings](moonlight-r-20261111t180727z/resource-closeout.json) confirm that **both attempts' instances and volumes are absent**. Unrelated resources were untouched.

[Cumulative posted charges](moonlight-r-20261111t180727z/charges-cumulative.json): first instance $0.047, first volume $0.008, replacement instance $0.002; **$0.057 posted so far**. The first attempt's earlier $0.037 snapshot increased due to billing lag. No charge row has yet appeared for replacement volume `60244147`; it is not asserted to cost zero. All resources are absent and the $1 preparation/$15 total caps remain unchanged.

No paid model-weight download, package setup, cache/initial-adapter preparation, native smoke/calibration, numerical gate, training or quality evaluation ran. All accepted CPU/reference/tokenizer/aux proofs remain unchanged. Quality gated/statistical: **NOT RUN**; gate **0/6, NOT RUN**, not numerical FAIL; loss/adapter delta **NOT RUN**. Tiny E3 remains **FAIL 0.05028628354277727 > 0.05**; **DeepSeek-V3 671B: отложен**.

Both authorized rental contracts have been closed. Continuation requires an actually usable direct-NVMe host and renewed rental authorization; no third rental or relaxed hardware/scientific criterion is assumed.

User authorized up to three additional technical host admissions with original $1/$15 caps unchanged. First additional attempt: offer43574919, separate volume offer43574924, machine18032, RTX4090/Samsung990PRO2TB. Ledger v4 SHA256 `1c269dd56ea31a9cd376de35b7b8fa724556dd8d53676ba8852160f00f4ae08a`; replacement upper $0.871927083333, cumulative posted-plus-upper $0.928927083333; 90-minute hard preparation allowance from volume creation. Native topology remains unverified until SSH; no guard is weakened.

## Qualified third host; measured budget rejection

The [bounded replacement authorization](host-replacements-authorization-v1.json) was used for one of three additional technical admissions. Volume `60246278` was created and registered separately, then linked to instance `60246341` on machine `18032`. Native proof found `/dev/nvme0n1p3`, XFS, PCIe NVMe, native-BF16 RTX4090 and a `97178877952`-byte cgroup memory limit. Neither RAID nor provider disk-label inference was accepted.

The pinned environment was installed. Stage a verified all **31920879456 weight bytes**; checkpoint identity SHA256 `9ee008194916494d82ce5c2d95d2028300cafcdcbc1dc0050255f37a49a24815`. Stage b produced the NF4 cache and shared initial-adapter file SHA256 `d6c9191f724ec4ca2b49253329bb492edfb45ea5efcd6fd89c44e95b1debaff1`. The CUDA smoke and seven resource cases completed; no optimizer update, scientific gate or quality scoring occurred. [Native preparation summary](moonlight-h3-20261111t184322z/native-preparation-summary.json).

Measured 4096-input-token resource windows, **not training runs**:

| Arm | Length / accumulation | Window phase, seconds | CUDA allocated peak, GiB | Sampled process RSS peak, GiB |
| --- | --- | ---: | ---: | ---: |
| BF16 L2L | 1024 / 4 | 73.232 | 2.722 | 3.692 |
| FP8 block64 + Oracle | 1024 / 4 | 96.029 | 2.722 | 3.506 |
| BF16 L2L | 4096 / 1 | 21.745 | 8.339 | 3.677 |
| FP8 block64 + Oracle | 4096 / 1 | 42.715 | 8.339 | 3.504 |

These are whole-operation resource observations, not activation-only memory; RSS sampling can miss short peaks. Native development-inference calls took 3.726–5.718 seconds. One full-v2 adapter requires **20401 native forwards**, including 16429 for HellaSwag. Replanning used the unchanged frozen selection/batching functions, without model execution or retokenization.

| Ordered budget scenario | Final adapters | Native forwards | Total planning upper, USD |
| --- | ---: | ---: | ---: |
| Full v2, both lengths | 4 | 81604 | 155.4878 |
| HellaSwag prefix 1, both lengths | 4 | 15900 | 42.8255 |
| Same prefix, omit complete L4096 pair | 2 | 7950 | 23.9493 |

[Measured budget receipt](moonlight-h3-20261111t184322z/measured-budget-admission.json), SHA256 `6624e95cc81595e2a6e9e075a49087d6a6518a72525bb8bd40f91b207ffd0249`, separates observations from planning allowances: 2× measured workload time, additional larger-shape headroom, gate/readmission and operating overhead, cumulative preparation and closeout. This is **not an observed $23.95 bill or a proven minimum achievable cost**. Even before larger-shape headroom the smallest scenario estimates $19.8575. The single short development samples do not justify reducing the allowance merely to force admission.

The prefix-1 scenarios are budget-feasibility checks, not an admitted scientific protocol. No artifact-bound final protocol was marked FROZEN; no gate FAIL was used to stop quality. The monetary stop applies first. Further rental for this budget rejection is not assumed; remaining technical-host replacement slots do not increase the financial authority.

Two operational deviations are retained. The first conversion launch used CWD=`$REPO`, so the unchanged shard containment guard refused the sibling cache before creating it. A fresh conversion from owned `$ROOT` succeeded; the runbook now uses that directory. Wrapper timeout arguments also exceeded the per-command maxima inherited in ledger v4, although every completed command stayed inside its recorded duration and the complete attempt closed inside 90 minutes. [Deviation receipt](moonlight-h3-20261111t184322z/operational-deviation.json); the pre-rental ledger was not rewritten to conceal this.

Complete evidence was exported before teardown: **465 files**, with **279 embedded manifest file hashes** verified; archive SHA256 `af72c58edee72213614ad4cbca7b92db8042e67eca1b9569611f4177e79e1d6b`. A [private Kaggle backup](https://www.kaggle.com/datasets/batyrkhan2131123/soup-moonlight-host3-evidence-20261111) was downloaded independently and all 465 expanded payload files matched. [Readback receipt](moonlight-h3-20261111t184322z/private-evidence-readback.json).

[Closeout](moonlight-h3-20261111t184322z/resource-closeout.json) confirms all three attempts' six owned resource IDs absent; unrelated resources were untouched. Third-host creation through verified absence took **53.725 minutes**. [Posted charges](moonlight-h3-20261111t184322z/charges-cumulative.json): third instance $0.425 and volume $0.055, earlier attempts $0.057, cumulative **$0.537**; billing may still settle, and volume `60244147` has no posted row. Quality gated/statistical: **NOT RUN**. Scientific gate: **0/6, NOT RUN**. Training loss/adapter-change norm: **NOT RUN**. Historical tiny **FAIL unchanged**. **DeepSeek-V3 671B: отложен**.

New resident-NF4 authorization:25USD total including prior0.537, compute stop24, reserve1,46h from first new resource. Available RTX A5000 offer55657856/volume55657862,machine53780, rate0.273333333333USD/h. Ledger SHA256 `a76d399431d2251afbe43ae8ec0cba6c771e3a95f5a37d24ee714184aa689600`. Actual direct NVMe required before setup. Measure A then B only if needed; C reference only. RTX4090 training timings are not portable: new no-update timings required on A5000. No monetary limit added to code.

## Resident NF4 qualification — host admission blocked

The two operational probe files are written; production scoring/training is unchanged. They have not executed on CUDA. Resident NF4 storage, fixed-8192 capacity, timings and C equivalence are **UNVERIFIED**, not PASS or FAIL.

The new A5000 host exposed `/dev/loop22`, the next host could not start its container (`cloud.vast.ai:443` reset; proxy SSH refused), and the Intel-labelled host exposed `/dev/md0p2`. Neither loopback nor RAID meets the unchanged direct-NVMe admission. No weights were downloaded and no A/B/C model call ran in these attempts. The previously verified machine18032 was unavailable.

All owned instances and volumes are confirmed absent in [fresh closeout evidence](moonlight-resident-20261111t201548z/intel-host/final-resource-absence.json). [Posted cumulative charges](moonlight-resident-20261111t201548z/intel-host/posted-charges.json): **$0.593**, subject to late billing. The $25 total/$24 compute/$1 reserve and original 46-hour clock were not reset between replacements.

Quality gated/statistical: **NOT RUN**. Gate diagnostics: **0/6, NOT RUN**. Scientific step losses and adapter-change norm: **NOT RUN**. Tiny **FAIL unchanged**. **DeepSeek-V3 671B: отложен**.

The user retained direct NVMe and authorized at most three new **whole single-GPU machine** rentals (`gpu_frac=1`, `num_gpus=1`), with a >=10 GB O_DIRECT read before weights and existing F1 on every timed block. No imposed32GiB cgroup or cross-machine timing transfer; this is Moonlight measurement, not a P3 x50 verdict.

Whole-machine attempt1 (machine57401) rejected registered SSH keys on direct/proxy routes; attempt2 (machine31389) failed provider container construction (`docker_build() error writing dockerfile`). Both instances and volumes are confirmed absent in their closeout receipts. No native NVMe/O_DIRECT/model result was obtained.

Before the final authorized rental: machine140544, RTX4090, WD_BLACK SN850X4000GB, full-machine quote59917583 plus volume59917588. Total quoted rate **$0.613333333333/hour**. Initial qualification envelope including conservative prior expenses, traffic and closeout: **$4.213333333333**, not a scientific-study estimate. [Ledger](moonlight-resident-20261111t201548z/whole-machine-3/rental-ledger.json) SHA256 `9395ace92da1c7ae8d966ad1e36a2f97788f7174752821d2d93cb21574ad6abb`. Original cumulative budget and absolute deadline retained; if admission fails, stop without a fourth whole-machine rental.

### Whole-machine attempt3: native storage and direct read admitted

Instance60261779/volume60261770 on machine140544 exposes `/dev/nvme0n1p3`, XFS, direct PCIe NVMe (`transport=pcie`, rotational0), not RAID/loopback. GPU RTX4090 reports24564MiB. Actual cgroup memory64644710400bytes (about60.2GiB), CPU quota23.04 cores and cpuset0–23; no32GiB cap was imposed. Provider allocation is `gpu_frac=1`, `num_gpus=1`.

| Direct-I/O phase | Bytes | Seconds | GB/s, decimal | F1 suspend/power/foreign_reads |
| --- | ---: | ---: | ---: | --- |
| Preparation write | 10737418240 | 1.965375 | 5.463293 | ok / ok / ok |
| Sequential read | 10737418240 | 1.535450 | 6.993011 | ok / ok / ok |

The owned test file was removed. [Native/direct-read receipt](moonlight-resident-20261111t201548z/whole-machine-3/nvme-read-manifest.json); [independently verified export](moonlight-resident-20261111t201548z/whole-machine-3/admission-export.json). These are Moonlight-host observations, **not a P3 x50 verdict**.

Pinned installation completed. The existing bounded weight-download stage was launched only after this qualification. Development inputs were prepared remotely without evaluation-task access; declaration SHA256 `c016d87e3b8ded06aea869d9605e4b4b259ee374c75d7ef64812112d2f3a6a91`. Remote fatal-code diagnostics passed for the four operational scripts. The unchanged precision helper confirms minimum n437 for a zero-discordance cell; that is not a power guarantee or a model result. Candidate fit/parity, scientific gate, training and quality remain unexecuted.

### Resident NF4 qualified and integrated on machine140544

Stages a+b completed. The shared initial adapter file SHA256 is `f863ccc832a66254a2089cbfc4eeb3c8f9e391539ca562607830e028708deaaf`; its parameter digest is `cf5b676c42dc0be973668d3adf6087d59a21d958bf6dca7dbc50361bfc5892ac`.

A completed the fixed8192-token input without OOM or a smaller batch. CUDA allocated/reserved peaks were10926927360/11622416384 bytes. Routed weights remain7197425664 bytes of packed NF4 plus899678208 bytes of FP32 statistics. Shared experts remain224919552 packed bytes plus28114944 statistics bytes. The separate reusable BF16 expert scratch is17301504 bytes, overwritten for one expert; it is not persistent decoded expert storage. Full category/dtype/storage inventories are retained in the qualification evidence.

B was not run because A fits. C ran only as the disk development reference. All24 task outcomes and80 choice log-probabilities match exactly, including bitwise FP32 equality and zero ULP distance. No tolerance was introduced.

| Development endpoint | Requests | Input tokens | Forwards | Warm1 seconds | Warm2 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 32 | 1363 | 6 | 5.576303 | 5.606472 |
| ARC | 32 | 777 | 4 | 3.431702 | 3.405737 |
| BoolQ | 16 | 2136 | 2 | 1.937663 | 1.917636 |

All A timing blocks and C reference blocks have F1 suspend/power/foreign_reads `ok`. These measurements belong only to this RTX4090 machine with its roughly60.2GiB cgroup allocation, not to a P3 x50 claim.

| No-update training window | BF16 L2L seconds | FP8 block64+Oracle seconds |
| --- | ---: | ---: |
| L1024/k4 | 49.072033 | 64.396104 |
| L4096/k1 | 14.731755 | 28.504821 |

All four windows completed with unchanged adapters, cleanup and valid F1. No timing was transferred from an earlier host.

Only after A/C qualification, the evaluator storage owner was integrated into `p3_deepseek_quality_residency.py`. It reuses the existing scoring restoration and reducers. Training still uses disk L2L. The protocol binds `evaluation_storage: resident_nf4`; monetary amounts are protocol metadata, not fixed implementation constants.

Remote verification:79 evaluator/study tests passed; Ruff passed. The integrated native evaluator separately passed8192 capacity and matched the retained disk reference on all24 tasks/80 choices, with immutable inputs and complete cleanup. See [integration proof](moonlight-resident-20261111t201548z/whole-machine-3/resident-integrated-equivalence.json).

Latest posted cumulative charges at22:22UTC: **$1.701**, with billing lag possible. Final measured scope/cost freeze is pending. Scientific gate remains0/6; the64-step training arms and quality verdicts have not run. Tiny FAIL remains unchanged. **DeepSeek-V3 671B: отложен**.

### Frozen measured scope before scientific execution

At 22:55 UTC, the native training and evaluation validators admitted protocol
`d22871c398d29e25c614129a1a084903fbc0dce09a5bb5d2659c08d9102b2f82`.
The [protocol](moonlight-resident-20261111t201548z/whole-machine-3/frozen-study-protocol.json)
binds the executed source overlays, source/cache/tokenizer/initial-adapter identities,
resident NF4 evaluation, unchanged v2 scoring, and the
[measured budget](moonlight-resident-20261111t201548z/whole-machine-3/frozen-measured-budget.json).
The protocol and qualification archive were exported and independently hash-checked
before launching the diagnostic gate.

| Scope | HellaSwag tasks | Final adapters | Planned evaluation forwards | Total planning upper, USD |
| --- | ---: | ---: | ---: | ---: |
| Full v2 | 10042 | 4 | 81604 | 26.2121 |
| Selected frozen prefix | 9000 | 4 | 74808 | 24.5827 |

Both L1024 and L4096 remain, with BF16 and FP8+Oracle independently trained for
64 steps each. Only the frozen-order HellaSwag prefix is reduced. The analytical
zero-discordance minimum is 437, not a power guarantee. All other endpoint counts,
decision rules, thresholds and optimizer settings remain unchanged.

The estimate uses this machine's measured training windows and resident dev forwards,
with one multiplicative factor of 1.5. LAMBADA/GSM8K/MBPP use the slowest measured
dev rate; full-evaluation throughput remains an extrapolation. Separate bounded
allowances cover process cleanup, readmission, pairing, analysis and exports.
The total includes a $2.20 accrued-cost allowance, intermediate traffic and the
reserved $1 for closeout. Posted charges at 22:53 UTC were $2.019.
Computation still stops at $24 and the original absolute deadline remains
2026-11-13T18:15:48.490959Z. No fourth whole-machine rental is authorized.

The six-pass gate completed once in 505.934 seconds. Its full closure was independently
readmitted and exported with archive SHA256
`d15b078492a8f483d0159590be0a2f535450b6dcacb7f44ed5bbc72543e6550b`.
All three F1 checks were `ok`. Moonlight's diagnostic verdict is **PASS**:
E1 RMS/worst maxima0.0251101/0.0269059, E2 maximum0.0409753,
E3 global relative L2 error0.0457640 and minimum cosine0.9921371.
See the [raw-derived gate result](moonlight-resident-20261111t201548z/whole-machine-3/d-gate-result.json)
and [export receipt](moonlight-resident-20261111t201548z/whole-machine-3/d-export.json).
All four independent64-step arms completed; frozen evaluation stopped on an evaluator defect (see final stop below).
Tiny E3 remains **FAIL, 0.05028628354277727 > 0.05**.
**DeepSeek-V3 671B: отложен**.

### First training arm: numerical completion, invalid timing block

BF16 L1024 completed64 steps from the shared initializer without a numerical or
technical refusal. First/last losses:0.7852686047554016/0.7251471132040024.
The adapter's global delta L2 is2.021127873234605, relative L2=0.12007677717314157.
All64 step records and201 manifest-listed files, including every checkpoint,
step-one raw gradients and the final adapter, were independently verified locally.
Archive SHA256:`9487acdd615ec2118cb83a6dab2363af673d268f52ee2df4eaa1e6357397094e`.

**The entire training timing block is VOID.** F1 reported18,014,482,432 bytes of
unattributed/foreign NVMe reads against its unchanged1,000,000,000-byte limit;
no foreign reader process was identified. Suspend and power were `ok`.
The raw2951.743-second observation is retained, not a valid performance result
or a calibration input. No retry, threshold change, host replacement or scientific
protocol change was made. Numeric completion is retained separately while the
other independent frozen arms continue.

See the [training summary](moonlight-resident-20261111t201548z/whole-machine-3/train-bf16_l2l-L1024-summary.json),
[all step records](moonlight-resident-20261111t201548z/whole-machine-3/train-bf16_l2l-L1024-steps.jsonl)
and [export verification](moonlight-resident-20261111t201548z/whole-machine-3/train-bf16_l2l-L1024-export.json).
The optional new private Kaggle backup was refused by authentication; complete
archives remain hash-verified on the workstation. No new private upload is claimed.

### Four independent training arms complete

Every arm started from the same initializer and completed64 steps without a retry.
For each length, every step's input and label hashes matched across BF16 and FP8+Oracle.
All four complete archives,804 manifest-listed files and256 per-step records were
independently verified locally before evaluation. The first endpoint is ARC for
the trained BF16 L1024 adapter; no base checkpoint is evaluated.

| Arm | L | First → last loss | Adapter delta L2 | F1 | Valid wall seconds | CUDA peak allocated / reserved, GiB | Sampled process RSS peak, GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 1024 | 0.785269 → 0.725147 | 2.021128 | void | VOID | VOID | VOID |
| FP8+Oracle | 1024 | 0.785269 → 0.725904 | 2.017052 | ok | 3957.657 | 2.729 / 3.270 | 3.559 |
| BF16 | 4096 | 0.683164 → 0.672251 | 1.815725 | ok | 895.675 | 8.377 / 10.223 | 3.760 |
| FP8+Oracle | 4096 | 0.683164 → 0.673061 | 1.817330 | ok | 1756.099 | 8.377 / 10.221 | 3.572 |

RSS sampling is a lower bound; it can miss short peaks. CUDA values are allocator
observations, not activation-only memory. No performance ratio involving the
BF16 L1024 void block is admitted. These losses and adapter changes are training
results, **not quality verdicts**. See [the complete matrix and archive identities](moonlight-resident-20261111t201548z/whole-machine-3/training-comparison.json).

### Frozen evaluation: BF16 L1024 technical stop

ARC3548 and BoolQ3270 completed and were fully exported with payload hashes verified.
ARC's timing block is VOID (6,336,720,896 unattributed NVMe read bytes); BoolQ's
F1 checks were all `ok`. Neither endpoint alone supplies a quality comparison.

The frozen HellaSwag9000 shard then failed during scoring:
`ValueError: attention forward lacks executed operators/CUDA kernel names`.
The unchanged attention admission guard refused absent/invalid executed-forward
operator or CUDA-kernel evidence. The underlying profiler cause is not established.
A profiler-cycle event-clearing warning is retained in the log, not asserted to be
the cause. No fallback, profiler change, threshold change or retry was applied.

The failed archive preserves3510 completed raw batch records, the incomplete
manifest, process log and cleanup/F1 receipts. It has no completed task-result shard.
SHA256:`594a2f8023b3781ef7dbc537e5a32092af1fdff3b376b07d56cf991417b2b473`.
Its timing block is also VOID. All resident ownership cleanup checks passed.

The affected BF16 L1024 evaluation stops; the L1024 pair is invalid and the frozen
whole-study gated/statistical decisions are blocked by incomplete coverage.
Other independent adapters were then evaluated without retries; see below.
The frozen matrix is not narrowed and no partial prefix or pair will be presented
as its result. All four training completions and diagnostic gate PASS remain
separate facts. See the [preserved technical-failure receipt](moonlight-resident-20261111t201548z/whole-machine-3/eval-bf16_l2l-L1024-hellaswag-0-9000-failure.json).

Full verified archives and the exact executed source overlay are also retained
outside temporary storage at
`C:\Users\sydyk\Downloads\moonlight-quality-evidence-20261111-whole3`.

### Final stop: the same refusal three times, closeout

FP8+Oracle L1024 and BF16 L4096 HellaSwag failed with the same error. The failures
came at batch indices 3510 and 3510; BF16 L1024 failed at 3509. FP8+Oracle L4096
and the remaining endpoints were not launched: no valid pair could be completed.

All three failed batches have Q/K shape `[2,16,106,192]`. Each recorded the efficient-SDPA
operators, but `cuda_kernel_names` was empty. In two runs, the immediately preceding
batch had the **same shape** and recorded the native `fmha_cutlassF` kernel. The
evaluator resets its attention observer per forward, so each batch opens a new
`torch.profiler` session. All refusals fall after3509–3511 sessions in one process.
Completed ARC used1381 sessions and BoolQ162. The failing request indices differ
between runs (17893 versus 18016). Together with the same-shape success, the evidence is
**consistent with a per-process profiler-session limit**. It is correlation from three
failures and two shorter successful runs, not a demonstrated cause. The
CUPTI/Kineto-level mechanism is unproven.

An evaluator change is required before any verdict, and either option needs user
authorization. Profiling each signature once per process does **not** work: the observer
admits at most `_MAX_SIGNATURES = 16` signatures. Evaluation shapes vary per batch, which
is why the evaluator resets the observer per forward. One retained observer would raise
"signature limit exceeded" within a few batches. Workable options:

- Keep per-forward observers and the unchanged per-forward guard. Cache admitted
  kernel evidence per signature across forwards, under its own explicit bound. Profile
  only signatures not yet in the cache. This changes the evaluator and needs remote
  re-qualification.
- Freeze smaller contiguous shards, each run in a fresh process far below ~3500 batches.
  This must cover **every** endpoint. LAMBADA's5153 tasks may exceed the limit, and
  GSM8K/MBPP batch counts are unmeasured. It changes the shard layout and the protocol hash.

No retry, guard relaxation or protocol change was made.
[Root-cause record](moonlight-resident-20261111t201548z/whole-machine-3/hellaswag-attention-evidence-root-cause.json).

The remaining remote evidence, excluding training checkpoints already archived per arm,
was exported as1607 files and hash-verified locally. Archive SHA256:
`cfcbd405588718dd3dcf3efa717080357672801e35e99ec6f72ec42db9c74d11`.
Instance60261779 and volume60261770 were then deleted. All18 owned resource IDs
across attempts were verified absent. See [closeout](moonlight-resident-20261111t201548z/whole-machine-3/resource-closeout.json).
Posted cumulative charges:**USD7.189**, within the $25 limit; billing may lag.
Tiny E3 remains **FAIL 0.05028628354277727 > 0.05**. **DeepSeek-V3 671B: отложен**.
