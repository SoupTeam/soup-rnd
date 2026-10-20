# A0 L2L step 0 reproduction

The [2026-10-19 Vast.ai RTX 3070 report](probe-vast-rtx3070-l2l-step0-a0.md)
records G1–G4 PASS. The historical session-01 ordinary-batch control with 1 warmup
and 3 measured steps remains incomplete: batches 1/2/3 exceed the 5% round-spread
limit. The separate full04 control with 2 warmup and 24 measured steps completes
valid pairs for every batch 1–4 under the unchanged 5% limit. Its next-batch check
records CUDA allocator OOM in both batch-5 processes, establishing the measured
batch bound at 4. L2L uses less allocated VRAM; a speed advantage over ordinary
batch 4 is not established. The two blocks retain their separate verdicts.
Raw data, source hashes, package versions and the A1/B1 recommendation are included.

This is a new attempt of the existing L2L experiment. It retains the G1 to G4 thresholds
in `probe-rtx5070-l2l-step0.md` and adds explicit handling of missing telemetry. The October 7
record remains historical: G1 and G2 passed, G3 and G4 had no verdict because the L2L A k1
control exceeded the 5% round-spread limit. Run the old evaluator without
`--strict-validity` to reproduce that historical decision.

## Scope and dependencies

A0 delivers the probe report and the recommendation on A1 and B1. Trainer integration,
dropout greater than zero, MoE enablement and quality claims belong to later tracks.
The existing probe always uses LoRA dropout zero. The roadmap references a private owner
handoff as implementation history. It is absent from the accessible original public Soup
main as well as this checkout; the published probe rule defines this reproduction's
acceptance criteria. The private history is not an additional acceptance blocker.

The target is one CUDA GPU with 8 GB VRAM, 32 GB accessible RAM and NVMe storage. Reserve
about 17 GB free RAM at session start. The 70B NF4 store is about 36 GB. Allocate roughly
220 to 240 GB of storage during fixture preparation: the bf16 source, NF4 cache, real 7B
fixture and spill file coexist. This is a preparation estimate, not a measured new result.

A Linux cloud run is a separate hardware record. `--power-source server` is an explicit
operator declaration of a mains-powered server, recorded as `operator-declared-server`;
it is not a physical AC sensor reading. Readers and GPU peers are visible only to the
container and its exposed NVIDIA process interface. Missing interfaces invalidate the
dependent rows. Accessible host RAM is capped by exposed cgroup v1/v2 root limits.
Available container RAM excludes inactive file cache from charged usage, as Docker's
working-set accounting does, and remains capped by host `MemAvailable`. Missing cache
counters retain the conservative estimate. The k16 preparation still tests a real pinned
allocation. See [Docker memory accounting](https://docs.docker.com/reference/cli/docker/container/stats/).
When NVIDIA exposes host PIDs without a namespace mapping, the probe can attribute the
sole compute PID only after an observed idle GPU, a confirmed live local CUDA allocation,
and exactly one visible CUDA device. The before/after evidence is recorded. Unknown or
busy starts and multiple reported processes remain invalid; GPU aliases never change
the namespace-local I/O ownership list.

## Prepare before the quiet window

Use Python 3.11 or 3.12 for the pinned benchmark environment. The Linux bootstrap installs
PyTorch 2.14.1 from its official CUDA 13.0 wheel index, the remaining dependencies, then runs
the focused CPU tests and lint. It creates a local virtualenv and freezes its packages.
It does not create a server, download models or start measurements:

```bash
bash benchmarks/a0-bootstrap.sh
```

The preparation script checks CUDA, free disk space and an aligned direct-I/O round trip
before downloading the public real `unsloth/mistral-7b-instruct-v0.3` checkpoint. It pins
the resolved Hub commit in its preparation record, generates F2 with the shipped
`synth_checkpoint.py --shape llama-70b --vocab 32768`, and prepares both NF4 CUDA caches:

```bash
.venv/bin/python benchmarks/harness/l2l_prepare.py \
  --out "$PWD/.soup/preparation-01" --power-source server
```

Inspect the preparation logs and confirm direct I/O, the pinned k16 allocation, and all
dependencies before proceeding. A dry-run exit code alone does not certify fitness.
Do not use `--allow-invalid`. Caches stay under this checkout, as required by the sharder.
On Windows use native paths and sensor power; the Python session does not depend on Bash
or a worktree name. The bootstrap specifically targets a Linux CUDA 13.0-compatible driver.

## Historical session-01 recipe (1+3)

The same invocation without `--execute` produces a plan, without CUDA work. Planning and
measurement must use different new output directories. Actual execution saves its protocol
before preflight or the first measurement. Existing output directories are refused.
For uncommitted work, the base commit identifies the starting runtime and source SHA256
hashes identify the actual modified harness. Publish the bundle/protocol hash before running;
do not describe the modified source tree as a new Git commit.

```bash
.venv/bin/python benchmarks/harness/l2l_session.py \
  --f1 "$PWD/.soup/models/mistral-7b" \
  --f2 "$PWD/.soup/models/synth-70b-v32k" \
  --f1-shards "$PWD/.soup/f1-nf4" \
  --f2-shards "$PWD/.soup/f2-nf4" \
  --spill-dir "$PWD/.soup/spill" \
  --out "$PWD/.soup/results/a0-attempt-01" \
  --power-source server --batch-max 4 --max-hours 6 --execute
```

Each correctness block compares GA with GA and L2L with GA, with nonzero gradients. An
inexact GA control triggers exactly one fresh deterministic process. A non-pass G1 stops
the session before timing. Timing uses the existing two interleaved rounds, k1/2/3/4/8/16,
one warm-up and three measured steps per arm. The ordinary batch control uses batches
1/2/3/4 in ascending then descending order, one fresh process per point. It draws the same
token rows as L2L and records their fingerprints. OOM at batch 4 is retained as a result;
other child failures stop the session. Pinned versus direct-I/O spill at k16 follows.

This historical recipe tests through batch 4; if batch 4 fits, it reports the largest
tested batch and cannot establish a maximum. It does not reproduce the completed
full04 control. Keep its results, including invalid pairs, and use the separate recipe
below for the longer batch control and the batch-5 bound.

## Separate full04 recipe (2+24 and the next-batch bound)

`l2l_session.py` has fixed 1+3 sample counts; passing `--batch-max 5` alone does not
select full04. The following Linux recipe reuses the saved
[full04 runner](results/a0-vast-rtx3070/a0-batch-control-04-full/runner.py.txt) and
[protocol template](results/a0-vast-rtx3070/a0-batch-control-04-full/protocol.json).
It retains the runner's RTX 3070 / driver 580.159.04 guard, quiet-window checks,
host-I/O recording, strict validity rules, two-hour total limit, and 30-minute child
limit. Use that hardware/driver for this recipe; another configuration needs its own
declared hardware protocol. The only runner edits are the checkout and protocol-file
paths in a new local copy. Neither archived file is edited.

First finish the historical recipe above in the same checkout and environment.
Its local `a0-attempt-01` must have G1–G4 PASS; its batch control may remain incomplete.
The recipe binds full04 to that local parent's source hashes, package versions, and
protocol hash. It does not use the archived GPU run as a claim that RND passed CUDA.
Run from the checkout root after preparation. Set `A0_QUIET_DEVICES` to the actual
visible disk-counter names for this host; the saved host used the names shown here.
All named devices must expose counters. The planner creates new files and prints their
hashes without starting CUDA work:

```bash
export A0_QUIET_DEVICES=nvme0n1,nvme1n1,sda
.venv/bin/python - <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

base = Path.cwd()
sys.path.insert(0, str(base / "benchmarks/harness"))
import l2l_session as session

archive = base / "benchmarks/results/a0-vast-rtx3070/a0-batch-control-04-full"
parent = base / ".soup/results/a0-attempt-01"
out = base / ".soup/results/a0-full04-replay-01"
plan_path = base / ".soup/a0-full04-replay-01-protocol.json"
runner_path = base / ".soup/a0-full04-replay-01-runner.py"
assert not any(p.exists() for p in (out, plan_path, runner_path)), "Use new paths"
parent_bytes = (parent / "protocol.json").read_bytes()
parent_plan = json.loads(parent_bytes)
parent_verdict = json.loads((parent / "verdict.json").read_text(encoding="utf-8"))
assert all(parent_verdict["gates"][g]["status"] == "pass" for g in ("G1", "G2", "G3", "G4"))
assert session.source_hashes() == parent_plan["source_sha256"], "Parent source changed"
assert session.versions() == parent_plan["versions"], "Parent packages changed"
template_bytes = (archive / "protocol.json").read_bytes()
plan = json.loads(template_bytes)
old_out = plan["out"]
for entry in plan["runs"]:
    entry["output"] = str(out / (entry["name"] + ".json"))
    entry["command"] = [arg.replace(old_out, str(out)).replace("/workspace/soup-a0", str(base))
                        for arg in entry["command"]]
    entry["command"][0] = sys.executable
devices = os.environ["A0_QUIET_DEVICES"].split(",")
assert devices and all(d and d in psutil.disk_io_counters(perdisk=True) for d in devices)
plan["quiet"]["devices"] = devices
plan["environment"]["PYTHONPATH"] = str(base / "src")
plan["template_source_bundle_sha256"] = plan.pop("source_bundle_sha256")
plan.update(created_utc=datetime.now(timezone.utc).isoformat(), out=str(out),
            purpose="New full04-profile replay; archived attempts retain their verdicts",
            head=session.git_head(), source_sha256=session.source_hashes(),
            versions=session.versions(),
            parent_protocol_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            template_protocol_sha256=hashlib.sha256(template_bytes).hexdigest())
runner = (archive / "runner.py.txt").read_text(encoding="utf-8")
replacements = {
    "base = Path('/workspace/soup-a0')": f"base = Path({str(base)!r})",
    "Path('/workspace/a0-control-04-full-protocol.json')": f"Path({str(plan_path)!r})",
}
for old, new in replacements.items():
    assert runner.count(old) == 1, "Runner template changed; inspect before execution"
    runner = runner.replace(old, new)
runner_bytes = runner.encode("utf-8")
plan["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
plan_bytes = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode("utf-8")
with runner_path.open("xb") as handle:
    handle.write(runner_bytes)
with plan_path.open("xb") as handle:
    handle.write(plan_bytes)
print("protocol SHA256:", hashlib.sha256(plan_bytes).hexdigest())
print("runner SHA256:", plan["runner_sha256"])
PY
```

Inspect the generated protocol and publish both hashes before execution. Round 0
tests batches **1, 2, 3, 4, 5**; round 1 tests **5, 4, 3, 2, 1**, with a fresh process
for every point. Batches 1–4 each use **2 warmup + 24 measured steps** in both rounds.
The batch-5 bound retains its predeclared **1+3** command in both processes; the
reported allocator OOM occurs before it can complete those samples. It is a bound
check, not an additional valid throughput arm. The first point requires 60 seconds
of quiet samples and subsequent points 30 seconds, with at most 180 seconds waiting
per point (disk traffic <=10 MB/s and busy <=5%, GPU idle <=500 MiB, no GPU peers,
known server power, and at least 17 GB available RAM).

Execute once during the booked quiet window; this starts GPU measurements:

```bash
.venv/bin/python .soup/a0-full04-replay-01-runner.py
```

The runner records the local parent's gates for provenance; it does not rerun G1–G4
or merge any timing rows with that parent. A finished process alone does not certify
the control. Check the complete block and both allocator-OOM records:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

out = Path(".soup/results/a0-full04-replay-01")
result = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
assert result["state"] == "finished" and result["control_complete"], result
for rnd in (0, 1):
    payload = json.loads((out / f"batch_5_r{rnd}.json").read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    row = payload["records"][0]
    assert row["kind"] == "oom" and row["memory"]["num_ooms"] > 0
print(result["verdict"])
PY
```

Claim the measured maximum of 4 only if all pairs 1–4 pass the unchanged strict
checks and both batch-5 processes confirm CUDA allocator OOM. Otherwise retain the
new block as incomplete; do not change the 5% threshold or pool rows with another
attempt. This recipe creates new evidence for the local measured source and hardware;
it does not alter session-01 or the saved full04 verdicts.

There are no automatic timing retries. Keep every failed attempt. A repeat requires a new
output directory and a newly booked quiet window under the already fixed rule; do not
combine the best rows from different attempts. A wall-time limit stops the child process,
not the rented instance. Retrieve all files and destroy the rented instance after the run;
stopping compute alone can leave billable storage.

## Acceptance

- G1: every compared LoRA gradient and per-micro-batch loss equals the reproducible GA reference.
- G2: T_A(16) is at least 0.8 times T_B(16), where B removes the read and H2D copy.
- G3: T_A(16) is at least 1.3 times T_A(1).
- G4: max allocated VRAM at A16 exceeds A1 by at most 0.5 decimal GB.
- V1 to V6 hold for every row used; unknown power/readers/peers do not count as clean.
- The ordinary-batch control is shown alongside L2L. A maximum is established only when
  the next consecutive tested batch records CUDA OOM. Otherwise label it largest tested.

The synthetic-shape timings are memory and speed evidence, not training quality. A valid
negative gate is a research result. No verdict leaves that question unresolved. Review
`session.json`, `verdict.json`, every raw block JSON and `logs/`; the gate decision alone
does not establish that the batch-control deliverable is complete. Include commands,
versions, source hashes, hardware, all invalid rows and the A1/B1 recommendation in the
report. Do not rewrite or promote the historical October 7 record.

## Local verification

```bash
python -m pytest tests/test_l2l_probe_harness.py tests/test_l2l_session.py -q \
  -o addopts='' --basetemp "$PWD/.a0-test-tmp"
ruff check src/soup_cli/ scripts/ tests/ benchmarks/
```

The full Ruff check passes. If using a sparse checkout, include `scripts/` with
`git sparse-checkout add scripts` before linting. Ruff uses the filesystem to classify
project imports; excluding `scripts/` caused 13 misleading I001 findings in unchanged
tests. Restoring that tracked directory resolved all 13 without changing the tests.

## RND integration and measured-source provenance

The RND integration branch starts at RND main
ca9be3cc9f11b91cba76d26f8a0d7ad18787c4ff. It carries the A0 harness plus the
required L2L activation/scheduling modules, shared stream-probe OOM/input-fingerprint
support, and CPU tests. It does not import the upstream production changes or CI
history. The normative original step-0 report is included as reference.

The saved GPU measurements used upstream
3da204fadbd278c01ad5a4b23906b443d15e9dd1 and the recorded harness/source manifest.
They are historical evidence for those exact source hashes, not a GPU measurement
of RND main. A new run from RND creates its own source manifest and must be judged
as a new session. Raw result files and their remote SHA256 remain unchanged.
