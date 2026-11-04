# D1 amendment E: provider-denied clock locking, before CUDA measurements

Dated 2026-10-15. The separate full-study rental is instance 56073519,
RTX 3090 UUID GPU-ba2be847-d6cd-29a4-ac92-cfe08ee3083f, driver 595.71.05,
provider power limit 370 W, approximately $0.24148148/hour including 80 GB disk.
The first offer expired (HTTP 400; no allocation). Initial source-bundle SFTP
was interrupted before extraction; fetch the exact public Git commit instead.
Both setup failures are retained. No correctness or performance measurement
ran under candidate source 08dcf600 or amendment D's clock policy.

`nvidia-smi -lgc 1500,1500` returned exit 4:
"The current user does not have permission to change clocks".
The cleanup `-rgc` was attempted and similarly denied. No clock was changed.
Amendment D required stopping at this point; setup did stop before source
extraction, package installation, model download or any CUDA fixture/timing.

## New policy, fixed before measurements

Use this dedicated physical GPU's **unchanged provider stock clock policy and
370 W power limit**. Frozen projections receive at least 20 warm-up steps AND
15 seconds of synchronized warm-up, then 200 synchronized timed steps with
combined wall/CUDA and split forward/backward CUDA events. The pilot subset's
5000-step warm-up is not pooled with this cohort. A fixed duration gives each
geometry substantial warm-up while keeping the complete grid within budget.

Every other numerical, telemetry, validity and repeat requirement stays fixed.
In particular, actual DURING-work SM-clock samples across each comparison must
still have max/min <=1.05; no missing sensor or unstable clock is accepted.
There is no post-hoc clock filtering. Any cohort-wide clock violation stops
the controller for explicit review and retains its raw evidence. Train and
ablation keep their original 20 warm-up optimizer updates; extra time-based
updates remain prohibited. The complete 96-case grid and five-block QA/ablation
plans and F1 repeat sizing are unchanged.

Normal/error/rental cleanup still attempts a clock reset, records its result,
and deletes only this exact rental. Permission denial cannot block deletion.
The existing cumulative $10 budget and independent deadline/cost guards stay
in force. Independent acceptance remains the team lead's decision.
