# D2 Fast-LoRA: private Kaggle T4 runbook

Notebook: [`../notebooks/d2-fast-lora-kaggle.ipynb`](../notebooks/d2-fast-lora-kaggle.ipynb).
Decision rule: [`gate-d2-fast-lora-rule.md`](gate-d2-fast-lora-rule.md).
Harness: [`harness/fast_lora_probe.py`](harness/fast_lora_probe.py).

**Execution status:** this is a procedure, not a Kaggle execution report. Local notebook
validation does not establish CUDA correctness, a speedup, or D2 acceptance. All notebook
outputs and execution counts are empty. Nothing is published, uploaded, or written to Trello.
The notebook's operator instructions are in Russian; this repository runbook is in English.

## Before allocating a GPU

1. Freeze/review the local kernels and harness first. The owner builds a dedicated
   **source ZIP**, then manually attaches it as a **private Kaggle Input**. An unpublished
   branch needs no push. This task performs no upload, login, publication, or account creation.
   The archive must place repository files at the ZIP root plus `D2-SNAPSHOT.json`; the
   older handoff archive with `source/`, `evidence/`, or a wrapper directory is not accepted.
   Leave `SOURCE_MODE = "zip"`. Set `SOURCE_ZIP` to the exact `/kaggle/input/.../*.zip`
   path. Blank selects only a unique `d2-source-*.zip` anywhere under `/kaggle/input`;
   missing or multiple candidates stop rather than guessing or falling back to Git.
   Compare the builder's ZIP SHA256 and set `EXPECTED_ZIP_SHA256` (64 lowercase hex).
   A blank expected hash records integrity without claiming authenticated source origin.
2. Import the notebook into a **private** Kaggle notebook. Select **Accelerator: None**
   and **Internet: On** in Kaggle's UI. The notebook cannot enable either setting for you.
   Availability of free T4 quota is not assumed. Do not select paid resources.
3. Leave `RUN_CUDA = False` and `RUN_8B_SHAPED_BLOCK = False`. Run sections 1–5.
   Installation and CPU tests can be completed without an active GPU allocation.
4. The kernel must use Python **3.10, 3.11, or 3.12**, matching
   `requires-python = ">=3.10,<3.13"` in `pyproject.toml`. Unsupported Python fails before
   installation. Choose a supported image manually; do not widen the project's bound or
   install packages into the unsupported interpreter to force the run.
5. Read the emitted preregistered rule. The seed, precision, step count, optimizer, and
   acceptance thresholds are not parameters to tune after observing a failure.

### Frozen-source builder contract (exact v1 schema)

After freeze/review, derive a local **Git-tracked allowlist**, adding only reviewed D2
untracked files explicitly. Preserve file bytes, not line-ending-normalized patches. Package
all tracked `src/soup_cli/**` source/package assets, the selected test files below, and the
mandatory root inputs. No `.git`, venv, cache, environment file, Kaggle credentials, private
key, model checkpoint, raw evidence, or old handoff contents belong in this source ZIP.
The filename should be `d2-source-<snapshot_id>.zip`. The archive is a **source snapshot**,
not a result report. Do not manufacture a revision or GPU result to fill its metadata.

`D2-SNAPSHOT.json` is a UTF-8 JSON object with **exactly these seven fields**; unknown or
repeated keys are rejected. This is a schematic example; placeholders are not real hashes:

```json
{
  "format": "soup-d2-snapshot-v1",
  "base_revision": "9aa43bd71afe12d3724f196202f1140e7dc409dc",
  "source_sha256": {
    "pyproject.toml": "<64 lowercase hex SHA256 of exact file bytes>",
    "src/soup_cli/utils/fast_lora.py": "<64 lowercase hex>",
    "src/soup_cli/utils/fast_lora_qkv.py": "<64 lowercase hex>",
    "src/soup_cli/utils/fast_lora_mlp.py": "<64 lowercase hex>",
    "benchmarks/harness/fast_lora_probe.py": "<64 lowercase hex>",
    "benchmarks/gate-d2-fast-lora-rule.md": "<64 lowercase hex>"
  },
  "kernel_sha256": {
    "fast_lora.py": "<same hash as source_sha256 entry>",
    "fast_lora_qkv.py": "<same hash as source_sha256 entry>",
    "fast_lora_mlp.py": "<same hash as source_sha256 entry>"
  },
  "harness_sha256": "<same hash as source_sha256 harness entry>",
  "decision_rule_sha256": "<same hash as source_sha256 rule entry>",
  "snapshot_id": "<canonical manifest SHA256 defined below>"
}
```

`source_sha256` is the **complete** exact forward-slash root-relative file map, not the
abbreviated example above. It excludes `D2-SNAPSHOT.json` itself (no recursive hash).
Its file set must equal every non-directory ZIP member other than the manifest: no missing,
extra, duplicate, or casefold-aliased entries. All digest strings are exactly 64 lowercase
hex characters. `kernel_sha256` has exactly the three **basename** keys shown, not relative
paths, and must agree with the corresponding source map. Harness/rule hashes also must agree.

Compute `snapshot_id` after all other fields are final, over the six-field object without
`snapshot_id`, using this exact Python canonicalization:

```python
canonical = json.dumps(manifest_without_snapshot_id, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=True).encode("utf-8")
snapshot_id = hashlib.sha256(canonical).hexdigest()
```

Then serialize the seven-field manifest (indentation is optional), add it at the ZIP root,
and compute SHA256 of the **finished ZIP bytes** separately for `EXPECTED_ZIP_SHA256`.
Record both values outside the archive so the operator can compare them. Do not create
`D2-SNAPSHOT.json` inside a source file that the map hashes. The historical `base_revision`
is fixed decision-rule metadata, **not a claim that the snapshot is that Git HEAD**.

Mandatory source-map inputs:

- `pyproject.toml`, `README.md` (Hatch project readme), `LICENSE`, `AGENTS.md`, `CONTRIBUTING.md`;
- `src/soup_cli/__init__.py`, all three `src/soup_cli/utils/fast_lora*.py` kernels;
- `tests/__init__.py`, the matching reviewed `tests/conftest.py`, **all twelve selected test
  files listed below**; these selected files currently have no imports of other test helpers;
- `notebooks/d2-fast-lora-kaggle.ipynb`, `benchmarks/d2-kaggle-runbook.md`,
  `benchmarks/gate-d2-fast-lora-rule.md`, `benchmarks/harness/fast_lora_probe.py`.

Other permitted members are tracked files/assets under `src/soup_cli/` and optionally
`docs/peft-and-efficiency.md`. No other root/doc/test paths are accepted. If a future test
adds helper imports, revise/review the allowlist and notebook contract before building;
do not silently omit dependencies or expand to the whole repository.

### Extraction safety and provenance

Preparation checks every name/type, schema, required input, complete inventory, and exact
file hash **before** writing a source file, executing source, or installing it. ZIP entries
must be regular files or optional necessary ancestor directory entries. Absolute/UNC paths,
drive/colon/backslash/NUL, parent/dot components, case aliases, duplicate entries, file-parent
conflicts, links/special files/reparse attributes, encrypted entries, and unrelated directory
entries are refused. Portable Windows reserved names and trailing dot/space aliases are
also refused: illegal `< > " ? * |` characters, `CONIN$`/`CONOUT$`, and `COM`/`LPT`
devices with ASCII 1–9 or superscript ¹/²/³ suffixes. Device stems remain reserved before
extensions even with intervening spaces (for example, `CON .py`). Validation does not
normalize or repair names; other legitimate Unicode source names remain unchanged.
Limits: 10,000 entries; 64 MiB per member; 256 MiB total uncompressed or input
ZIP bytes; 1 MiB manifest. Credential **filenames** such as `.env`, `kaggle.json`, `hf_token`,
`credentials.json`, `id_rsa`, and key files are excluded; legitimate `tokenizers.py` is not.
This filename policy does not replace the owner's review of source contents for secrets.

Each configuration creates fresh UTC/UUID-labelled paths, never reused or deleted:

```text
/kaggle/working/d2-source-RUN_ID/       # WORK_SOURCE; CHECKOUT alias
/kaggle/working/d2-evidence/RUN_ID/
/kaggle/working/d2-kaggle-env/RUN_ID/
```

Containment uses `os.path.realpath` + `os.path.commonpath` and checks all path ancestors.
Input/source symlinks, junctions/reparse points, hardlinks, and out-of-root targets block
execution. Extraction uses exclusive creation. Existing work is never reset, cleaned, or
modified. The entire extracted tree and manifest bytes are revalidated before and after each child;
pytest cache writing is disabled and children disable bytecode writing. Source mutation or
an added file/directory invalidates the frozen snapshot, even if the old manifest matches.
The targeted Ruff command uses `ruff check --no-cache` so lint cannot create `.ruff_cache`
inside this frozen tree. Pip preparation and import smoke retain their working directory
outside source; no cache artifacts are hidden or deleted to pass inventory validation.

`source-preparation.json` preserves transport PASS/FAIL and its original error before
propagation; it is not a numerical or GPU verdict. `input-snapshot.json` and
`source-manifest.json` retain the original manifest, ZIP hash,
manifest-byte hash, snapshot ID, complete file hashes, operator-hash comparison status,
base revision, and honest `NO_GIT_SNAPSHOT` / `head: null` provenance. The rule is copied.
The harness may report Git-query failures with `git_head: null` / `git_status: null`; the
notebook accepts that only in ZIP mode with an `environment.collection_errors` object
containing nonempty string errors under both exact keys `git rev-parse HEAD` and
`git status --short`. Missing, empty, malformed, or wrong-key diagnostics block PASS;
other optional-dependency diagnostics may coexist. Exact kernel/rule/harness hashes remain
required.
It never substitutes the base revision for Git HEAD. The production reference-precision
guard inspects **installed** Transformers/PEFT source, not a Git history, so it remains usable.

Optional legacy mode requires explicitly setting `SOURCE_MODE = "git"`. It uses only the
already public `https://github.com/SoupTeam/soup-rnd.git` branch
`d2/fast-lora-correctness`, shallow/single-branch clone, and the reviewed exact
origin/branch/remote-HEAD/clean-root checks. Missing branch is a blocker. There is no token,
GitHub-auth download, automatic fallback, push, or publication. Use a fresh configuration
for either mode; an unpublished branch is intentionally unusable in this optional mode.

Each command also gets a unique evidence subdirectory; retries retain their original
stdout/stderr, JSON/CSV, failures, and invalidation history. A successful earlier CPU snapshot
is not GPU evidence for a later source snapshot. Do not bundle evidence as source.

### Dependency policy

The notebook creates a fresh venv at `/kaggle/working/d2-kaggle-env/RUN_ID` with
`--system-site-packages`. Subprocesses, not the notebook kernel, use that interpreter.
The existing Kaggle Torch distribution is preserved via a generated exact-version
constraint, including any CUDA build suffix. Both before and after installation a CPU
subprocess checks Torch's version, CUDA build, import path, and lack of visible GPUs.
The D2 stack requires an already installed Torch `>=2.6,<3`; a missing, old, or broken
Torch stops the run rather than triggering an automatic replacement.

Install the project in editable mode without `[dev]`, `[all]`, or the complete `[train]`
extra. The notebook pins these direct dependencies:

| Package | Version |
| --- | --- |
| hatchling | 1.27.0 |
| pytest | 8.3.5 |
| ruff | 0.11.13 |
| pytest-cov | 6.1.1 |
| transformers | 5.17.0 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.1 |

`hatchling` is installed first; the editable install uses `--no-build-isolation` and
`--constraint preserve-kaggle-torch.txt`. Core project and transitive requirements are
resolved by pip, so these pins are **not** a complete lockfile. `pip freeze --all` captures
the actual resolved environment. A global `pip check` is recorded with its actual exit
code; unrelated inherited Kaggle conflicts are not represented as a clean environment.
The import smoke and selected tests still must pass. A failing resolver/import command is
not waived by a successful global check.

Every child explicitly receives `PYTHONPATH=CHECKOUT/src`. Import smoke checks the exact
`soup_cli/__init__.py` path and the three kernel module paths, ensuring an older installed
Soup is not tested. Torch/PEFT/bitsandbytes/Transformers imports occur inside child functions;
the notebook kernel does not import Torch or retain a CUDA allocation. Import smoke builds
only a random tiny Llama, never a pretrained checkpoint. Incompatible image packages such
as torchvision are reported as real errors, not silently removed or worked around by
changing Torch. Tests run with HF/model access offline and telemetry disabled.

## Correctness stages

### CPU first

All CPU children receive `CUDA_VISIBLE_DEVICES=""`. The first cell sets the kernel's
`CUDA_VISIBLE_DEVICES=0` **before any Torch import**, rejects a pre-imported Torch kernel,
and gives CUDA children `CUDA_VISIBLE_DEVICES=0`. CPU preparation does not establish CUDA
availability.

The notebook selects these test files:

- `tests/test_issue839_fast_lora_single_projection.py`
- `tests/test_issue838_fast_lora_qkv.py`
- `tests/test_issue837_fast_lora_mlp.py`
- `tests/test_d2_qkv_backward.py`
- `tests/test_d2_checkpoint_and_dtypes.py`
- `tests/test_d2_qkv_semantics.py`
- `tests/test_d2_fast_lora_probe.py`
- `tests/test_d2_mixed_precision_backward.py`
- `tests/test_d2_reference_precision.py`
- `tests/test_d2_fast_lora_probe_validity.py`
- `tests/test_d2_kaggle_notebook_validity.py`
- `tests/test_d2_kaggle_notebook.py`

These are an explicit frozen inventory, not a wildcard that silently shrinks when a file
is absent. Every selected file must be present in the source ZIP before installation.

Order:

1. Existing CPU float64 gradchecks (`-k gradcheck`) for all three path suites. JUnit must
   contain cases from each suite, not merely a nonempty overall count.
2. Harness CPU fp32 dense parity with float64 gradcheck enabled. Validate every forward
   output, requested dX, all adapter dA/dB, positive patch counts, custom grad_fn paths,
   changed-adapter negative controls, and full JSON/CSV row counts.
3. CPU regression subset (`-m 'not gpu and not smoke' -k 'not mps'`) across the selected
   files. Apple-MPS-only cases are explicitly excluded and recorded as
   `UNVERIFIED_PLATFORM_BLOCK`; they cannot execute on Linux Kaggle. Unexpected skips in
   the remaining selected cases still block PASS.
4. Separate CPU NF4 single-projection non-reentrant checkpoint subset: exactly **four**
   cases (compressed/uncompressed state × dX/no-dX).
5. Separate `CPU_MIXED_PRECISION` subset: exactly **50 CPU cases**, no skips.
6. Separate `CPU_REFERENCE_PRECISION` subset: exactly **178 CPU cases**, no skips.
   These two files also run in the common CPU regression set and are mandatory transitively
   before every CUDA stage. They validate mixed cast/scaling seams, rounded float64 bounds,
   canonical scoped arithmetic and the unchanged CPU fp16 seed-792 50-step regression.
   CPU BF16 and CPU NF4 coverage do not establish CUDA BF16/NF4 coverage.

The complete CPU regression run includes CPU streaming/checkpoint and precision tests,
QKV output subsets/cache semantics, the no-dX backward regression, harness controls, and
validity checks. CPU BF16 tests do not establish native BF16 execution on CUDA. CPU timing
unit tests are DEBUG-ONLY, not a CUDA performance result.

Pytest invocations use `-p pytest_cov --no-cov -o addopts= -o junit_family=legacy`
to retain per-test precision `record_property` evidence without xunit2 incompatibility
warnings, and subset tests do not inherit
the whole-package coverage threshold. Third-party pytest auto-loading is disabled; the
cov plugin is loaded explicitly. Available JUnit is parsed and a case summary is written
before a nonzero exit is propagated. Missing or malformed JUnit also receives a diagnostic
summary; command status, exit code, and original stdout/stderr remain intact.
A nonzero exit fails even if JUnit lists only passing cases; an empty run, a skipped
selected case, a missing expected suite,
or a wrong exact case count is UNVERIFIED and stops continuation. A skip is never PASS.

### Enable free T4 only when ready

After green CPU gates, select **GPU T4** and Internet manually in Kaggle's UI. Set
`RUN_CUDA = True`. If changing hardware restarts the session, download the existing CPU
archive first and rerun the notebook from the beginning. The notebook does not reuse
cached PASS status across sessions. Within a session, **every** upstream retry (successful
or failed) invalidates all dependent correctness/timing results before it starts. Gate
history retains earlier failures and invalidated results; each CUDA entry rechecks the
entire CPU/upstream dependency chain, not merely a stale immediate PASS. Downstream gates
must be rerun in order. If two physical T4s are allocated by the UI, only
physical GPU 0 is visible to each CUDA process; there is no distributed/multi-GPU path.

CUDA preflight requires exactly one visible GPU, a real T4 with compute capability **7.5**,
a CUDA-enabled Torch build, and a successful small CUDA allocation. Capture:

- card name, UUID, visible count, compute capability, and total memory;
- Python and interpreter path, Torch/CUDA build, PEFT/Transformers/bitsandbytes versions;
- `nvidia-smi -L`, driver, SM/memory clocks, pstate, power/limit, temperature,
  utilization, memory state, and compute-process peers;
- exact command, exit code, timestamps, raw stdout/stderr, commit, and source hashes.

The notebook keeps `nvidia-smi` failures and does not kill peer processes. Physical GPU
listings may show more cards than Torch's visible-device count; only one is used.

### CUDA correctness, before timing

1. Dense fp32 parity, dX and no-dX.
2. Dense fp16 parity, dX and no-dX.
3. Fifty-step SYNTHETIC tiny-Llama loss comparison, first fp32, then fp16, seed **792**.
4. Separate GPU-marked NF4 single-projection non-reentrant checkpoint tests in
   `tests/test_d2_checkpoint_and_dtypes.py`: **four CUDA fp16** cases. The notebook checks
   both the exit code and exact JUnit count, with no skipped cases permitted.

Harness CLI calls use the existing interface, for example (from the checked checkout,
with the notebook's interpreter and per-command output prefix):

```text
python benchmarks/harness/fast_lora_probe.py parity --device cpu --dtype fp32 --seed 792 --shapes tiny --output-prefix PREFIX
python benchmarks/harness/fast_lora_probe.py parity --device cuda --dtype fp32 --seed 792 --shapes tiny --skip-gradcheck --output-prefix PREFIX
python benchmarks/harness/fast_lora_probe.py parity --device cuda --dtype fp16 --seed 792 --shapes tiny --skip-gradcheck --no-dx --output-prefix PREFIX
python benchmarks/harness/fast_lora_probe.py loss --device cuda --dtype fp32 --seed 792 --steps 50 --shapes tiny --output-prefix PREFIX
python benchmarks/harness/fast_lora_probe.py loss --device cuda --dtype fp16 --seed 792 --steps 50 --shapes tiny --output-prefix PREFIX
```

The notebook saves the exact executable, expanded arguments, and prefixes, not just these
illustrative commands. It performs both dX variants for each CUDA precision. All inputs,
weights, token batches, and the optional 8B-shaped block are **SYNTHETIC**. Loss-model shape,
optimizer settings, tokens, per-step losses, unrounded errors, patch/execution counts, and
controls are in the harness JSON. Every finite loss pair must format identically with
`.3f` at all 50 steps. Approximate parity is not evidence of bit-exactness.

The loss-report contract additionally requires `reference_arithmetic` metadata with
`ctx_attribute: "grad_fn.reference_order"`, precision-appropriate literal `required_true`,
and `expected_modules` mapping `qkv`/`mlp` to exact observed module-name lists. Lists must
match genuine custom execution maps (three QKV projections and one MLP per model layer).
Every one of the 50 rows must contain `reference_arithmetic_status: "observed"` and a
complete `reference_arithmetic_modes` map with those exact module keys. Low precision
(fp16/bf16) requires **literal True** for every observed module on every step; fp32 requires
literal bool telemetry but permits False. Missing/partial/stale/non-bool or false required
scoped modes block PASS. Custom Function names alone cannot establish selected arithmetic.
Failed or partial 50-row curves are kept as emitted; they are never relabelled GPU-tested
based on a passing CPU run.

**Stop on incorrectness.** Missing/nonfinite gradients, a disabled fast path, changed
frozen bases, rejected negative controls, numerical parity errors, or a failing loss curve
remain FAIL with their original evidence. Do not change the seed, precision, learning rate,
step count, or threshold after viewing a failure. Do not ask for a larger GPU to explain
incorrect gradients. A genuine import/allocation/driver blocker is reported with its exact
command and logs. The final archive cell can be run separately after any failure.

### What a T4 run cannot verify

Keep these statuses explicit; do not convert them to SKIP/PASS:

- **Native CUDA BF16:** `UNVERIFIED_HARDWARE_BLOCK` on T4; fp16 is not a substitute.
- **Current Triton/Liger/Unsloth comparison:** `UNVERIFIED_HARDWARE_BLOCK`; the rule records
  current Triton's NVIDIA capability 8.0+ support requirement, excluding T4 (7.5).
- **Harness NF4:** `UNVERIFIED`; the harness has no implemented NF4 mode. The notebook
  never requests NF4 from it or fabricates NF4 rows.
- **CUDA NF4 QKV/MLP BF16:** `UNVERIFIED_HARDWARE_BLOCK`. Existing issue838/issue837 GPU
  NF4 cases request BF16; the older issue839 GPU NF4 test also uses BF16.
- **CUDA NF4 streamed execution:** `UNVERIFIED`; CPU streaming or single-projection
  checkpoint evidence is not a substitute.

Do not run the blanket `pytest tests/ -m gpu` for this T4 protocol. Besides unsupported
BF16 regimes, issue839's legacy GPU `TestMicroBenchmark` runs a large BF16 shape outside
the gated timing procedure. The notebook intentionally uses the new fp16 NF4 checkpoint
subset instead. This explicit exclusion is an unverified coverage gap, not a passing
record of the excluded tests.

## Bounded timing after correctness

Timing runs only after all notebook CPU/CUDA correctness gates are PASS. Default tiny
CUDA timing is fp32 and fp16, three paths (single/QKV/MLP), dX and no-dX, **5 warmup rounds**,
**5 measured ABBA rounds**, and **16 tokens**. Each ABBA round is A=unpatched PEFT,
B=fast, B=fast, A=unpatched PEFT: 120 raw samples per precision across all groups.

```text
python benchmarks/harness/fast_lora_probe.py timing --device cuda --dtype fp16 --seed 792 --shapes tiny --warmup 5 --repeats 5 --tokens 16 --output-prefix PREFIX
```

The harness gates its own shape/dtype/dX parity before measurement. The notebook validates
CUDA events, ABBA sample ordering, every group, and JSON/CSV row counts. Before/after
`nvidia-smi` snapshots accompany each precision; the harness records per-arm clock/power
and peer validity outside the timed regions. Keep raw forward/backward samples and peak
allocated/reserved memory separate.

Limitations of the existing interface are **not** concealed:

- Each harness invocation starts a fresh process, but its ABBA arms are **same-process**,
  not fresh-process-per-arm (the rule prefers the latter).
- Peak memory includes resident inputs and both models/arms; it is not isolated-arm VRAM.
- Clock/power/peer samples need reviewer assessment. Contaminated arms are VOID and remain
  in raw artifacts, excluded from headline summaries. Missing samples imply NO VERDICT.
- Exit zero and `passed: true` establish successful collection/correctness, not a
  performance acceptance verdict. The notebook status is `COLLECTED_NO_VERDICT`.
- No minimum speedup, full-training multiplier, real-data quality result, or upstream
  Liger/Unsloth multiplier is promised. Regressions or no gain must be reported too.

`RUN_TINY_TIMING = False` disables even the tiny timing collection. The independent
`RUN_8B_SHAPED_BLOCK = False` default prevents large shapes. Enabling it explicitly requests
only a **dense fp16 SYNTHETIC single-block shape fixture** (`--shapes llama3.1-8b`), not a
full 8B model, download, or model-quality benchmark. It still gates shape-specific parity;
OOM or any failure remains in the artifact set. It is not necessary for first T4 validation.

## Evidence preservation and manual download

Run section 9 after success **or separately after an error**. It closes interrupted gate
statuses as UNVERIFIED, inventories commands and nonzero/missing exits, hashes artifacts,
and produces a ZIP at `/kaggle/working/d2-evidence-RUN_ID.zip`. Only the current evidence
folder is archived; venvs, checkout contents, datasets, and environment secrets are not
included. Before reading the ledger, hashing, or writing summaries, the entire evidence
tree is checked with `os.path.realpath`/`os.path.commonpath` containment and `lstat`:
symlinks (including internal/broken links and linked directories), junctions/reparse points,
hardlinked files, parent traversal, and out-of-root paths block archive creation. ZIP
members come only from the checked inventory, with revalidated reads and hash comparisons.
An existing ZIP is not overwritten. No upload, Kaggle publication, GitHub posting, or
Trello write occurs.

Retain:

- `configuration.json`, `source-preparation.json`, `input-snapshot.json`, `source-manifest.json`,
  `preregistered-rule.md`, `versions.json`;
- original/installed Torch metadata, direct pins, exact constraint, freeze and pip-check logs;
- `commands.jsonl`, per-command `command.json`, complete `stdout.log` and `stderr.log`;
- all harness JSON/CSV, including failed/partial runs; all pytest JUnit and case summaries;
- `cuda-hardware.json`, raw `nvidia-smi` snapshots, per-arm validity (including VOID rows);
- `gate_status.json`, `artifact-index.json`, and `artifact-sha256.json`.

Download through Kaggle's Files/Output UI or the notebook's local FileLink. Inspect the ZIP
before voluntarily sharing it with the reviewer. Download before ending the session;
unsaved runtime files may be lost after a Kaggle shutdown. Earlier RUN_ID directories
are not deleted. A new session starts new gates rather than silently carrying evidence
forward.

**Completion remains partial:** CPU_PASS / CUDA_UNVERIFIED is not complete D2 acceptance.
A successful T4 fp32/fp16 run and NF4 single-checkpoint run leave the listed BF16/Triton/
NF4-streaming regimes unverified. T00's definition, F1 approval, and the team's reviewer
remain prerequisites; the implementer/notebook does not mark D2 Done.

## Local notebook validation (no Kaggle execution)

These stdlib-only commands check JSON and compile every Python code cell without executing
its installation, tests, CUDA probes, or model code:

```bash
python -m json.tool notebooks/d2-fast-lora-kaggle.ipynb > /dev/null
python -c '
import ast
import json
from pathlib import Path
path = Path("notebooks/d2-fast-lora-kaggle.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
embedded_count = 0
for index, cell in enumerate(cells, 1):
    source = "".join(cell["source"])
    compile(source, f"{path}:cell-{index}", "exec")
    tree = ast.parse(source, feature_version=(3, 10))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node.value.value, str) and any(name.endswith("_PROBE") for name in names):
            compile(node.value.value, names[0], "exec")
            ast.parse(node.value.value, feature_version=(3, 10))
            embedded_count += 1
assert notebook["nbformat"] == 4
assert embedded_count == 3
assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in cells)
print(f"Validated JSON; compiled {len(cells)} cells and {embedded_count} child scripts")
print("Python 3.10 syntax accepted; no remote execution")
'
```

The shell examples use bash, including Git Bash on Windows. There are no Python magics or
shell escapes in code cells. If `nbformat` is already installed, also run its schema validator;
do not install packages locally just to validate this artifact. Compilation alone does not
prove dependency resolution, Kaggle UI state, hardware validity, or remote execution.
The local Ruff regression executes actual ZIP transport and provenance/lint preparation
on an isolated frozen fixture, retaining command stdout/exit and complete before/after
file hashes plus directory inventories. Its transport-only fixture deliberately omits
dependency/model import smoke and installs. It records the installed Ruff version; a
run on another local version is not evidence that the notebook's `0.11.13` pin was exercised.
