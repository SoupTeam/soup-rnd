# A0 L2L step 0 reproduction

The [2026-10-19 Vast.ai RTX 3070 report](probe-vast-rtx3070-l2l-step0-a0.md)
records G1–G4 PASS. The full ordinary-batch control remains incomplete because
batches 1/2/3 exceed the 5% round-spread limit. Valid batch 4 is faster than L2L k16;
L2L uses less allocated VRAM. A separate predeclared batch-5 check records CUDA
allocator OOM in both processes and establishes the measured batch bound at 4.
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

## Freeze the new protocol and run

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
