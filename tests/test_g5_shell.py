"""CPU-only subprocess regressions for G5's shell orchestration.

Only the two shell scripts are copied. Every model-facing command is replaced
inside the temporary harness, so these tests cannot load a model or use a GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required")
ARMS = (
    "mixed_traffic_f25", "mixed_sensitivity_f25", "mixed_traffic_f50", "mixed_sensitivity_f50",
)

# Stand-ins preserve the boundary effects the shells consume: records, logs,
# decided check results and unexpected command failures. No production Python is imported.
COMMAND = r'''
import json
import os
import sys
from pathlib import Path

name = Path(__file__).name
args = sys.argv[1:]
if name == "g5_verdict.py":
    stage = args[0] if args[0] in {"control", "provenance"} else "verdict"
else:
    stage = {"g5_select.py": "selection", "g5_bytes.py": "bytes",
             "c5_train.py": "baseline", "g5_train.py": "mixed"}[name]
with open(os.environ["G5_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"stage": stage, "args": args}) + "\n")
status = int(os.environ.get("G5_STATUS_" + stage.upper(), "0"))
if stage in {"selection", "baseline", "mixed"}:
    out = Path(args[args.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    if stage == "selection":
        checks = {key: status == 0 for key in (
            "model_unchanged_by_selection", "every_token_routed_top_k",
            "forward_check_within_limit", "maps_equal_budget_per_fraction")}
        record = {"kind": "g5_selection", "checks": checks, "synthetic": "--fixture" in args}
        (out / "selection.json").write_text(json.dumps(record), encoding="utf-8")
        maps = out / "maps"
        maps.mkdir(exist_ok=True)
        for arm in ("mixed_traffic_f25", "mixed_sensitivity_f25",
                    "mixed_traffic_f50", "mixed_sensitivity_f50"):
            (maps / (arm + ".json")).write_text("{}", encoding="utf-8")
    elif not os.environ.get("G5_NO_RECORD_" + stage.upper()):
        (out / "run.json").write_text(
            json.dumps({"status": "complete" if status == 0 else "failed"}), encoding="utf-8")
else:
    out = Path(args[args.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {}
    if stage == "bytes":
        record = {"kind": "g5_bytes", "synthetic": "--fixture" in args,
                  "checks": {"mixed_arms_of_one_fraction_equal_bytes": status == 0}}
    out.write_text(json.dumps(record), encoding="utf-8")
sys.exit(status)
'''


@pytest.fixture
def shell_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    harness = tmp_path / "benchmarks" / "harness"
    harness.mkdir(parents=True)
    for name in ("g5_run_all.sh", "g5_train_all.sh"):
        shutil.copyfile(HARNESS / name, harness / name)
    for name in ("g5_verdict.py", "g5_select.py", "g5_bytes.py", "c5_train.py", "g5_train.py"):
        (harness / name).write_text(COMMAND, encoding="utf-8")
    # If orchestration regresses to C5's failure-masking wrapper, fail closed.
    (harness / "c5_train_all.sh").write_text("exit 97\n", encoding="utf-8")
    startup = tmp_path / "startup.sh"
    startup.write_text(
        'python() { "$G5_TEST_PYTHON" "$@"; }\n'
        'git() {\n'
        '  if [ "${G5_FAIL_GIT:-0}" = 1 ]; then return 29; fi\n'
        '  echo synthetic-test-checkout\n'
        '}\n'
        'tee() {\n'
        '  command tee "$@"\n'
        '  local status=$?\n'
        '  if [ -n "${G5_FAIL_TEE:-}" ] && [[ "$*" == *"$G5_FAIL_TEE"* ]]; then\n'
        '    return 23\n'
        '  fi\n'
        '  return "$status"\n'
        '}\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("G5_") or key in {"BASH_ENV", "SHELLOPTS", "BASHOPTS"}:
            env.pop(key)
    env.update(BASH_ENV=startup.as_posix(), G5_TEST_PYTHON=Path(sys.executable).as_posix(),
               G5_CALLS=(tmp_path / "calls.jsonl").as_posix(), CUDA_VISIBLE_DEVICES="")
    return tmp_path, env


def _run(root: Path, env: dict[str, str], *, train_only: bool = False,
         mode: str = "pilot") -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    harness = root / "benchmarks" / "harness"
    if train_only:
        args = [(harness / "g5_train_all.sh").as_posix(), (root / "data").as_posix(),
                (root / "mixed").as_posix(), (root / "out" / "selection").as_posix(),
                "--fixture", "granite", "--device", "cpu"]
    else:
        args = [(harness / "g5_run_all.sh").as_posix(), mode, (root / "data").as_posix(),
                (root / "out").as_posix()]
    return subprocess.run([BASH, *args], cwd=root, env=env, capture_output=True, text=True,
                          timeout=60, check=False)


def _calls(root: Path) -> list[dict]:
    path = root / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _selection(root: Path) -> Path:
    selection = root / "out" / "selection"
    (selection / "maps").mkdir(parents=True)
    for arm in ARMS:
        (selection / "maps" / f"{arm}.json").write_text("{}", encoding="utf-8")
    checks = dict.fromkeys(("model_unchanged_by_selection", "every_token_routed_top_k",
                            "forward_check_within_limit", "maps_equal_budget_per_fraction"), True)
    (selection / "selection.json").write_text(
        json.dumps({"kind": "g5_selection", "synthetic": True, "checks": checks}), encoding="utf-8")
    return selection


@pytest.mark.parametrize("stage", ["provenance", "selection", "bytes", "baseline", "control",
                                   "mixed", "verdict"])
@pytest.mark.parametrize("failure", ["python", "tee", "both"])
def test_pipeline_member_failure_stops_before_the_next_step(shell_tree, stage, failure):
    root, env = shell_tree
    if failure in {"python", "both"}:
        env[f"G5_STATUS_{stage.upper()}"] = "19"
    if failure in {"tee", "both"}:
        logs = {"baseline": "nf4-s42-a1.log", "mixed": "mixed_traffic_f25-s42-a1.log"}
        env["G5_FAIL_TEE"] = logs.get(stage, f"{stage}.log")
    done = _run(root, env)
    calls = _calls(root)
    assert done.returncode != 0, done.stdout + done.stderr
    assert calls[-1]["stage"] == stage
    assert sum(call["stage"] == stage for call in calls) == 1
    assert "G5 pilot done" not in done.stdout


@pytest.mark.parametrize("failure", ["python", "tee", "both", "no_record"])
def test_mixed_failure_stops_the_training_loop(shell_tree, failure):
    root, env = shell_tree
    _selection(root)
    if failure in {"python", "both"}:
        env["G5_STATUS_MIXED"] = "19"
    if failure in {"tee", "both"}:
        env["G5_FAIL_TEE"] = "mixed_traffic_f25-s42-a1.log"
    if failure == "no_record":
        env["G5_NO_RECORD_MIXED"] = "1"
    done = _run(root, env, train_only=True)
    assert done.returncode != 0, done.stdout + done.stderr
    assert len(_calls(root)) == 1


@pytest.mark.parametrize("control_status", [0, 1, 2, 19])
def test_only_a_decided_control_mismatch_repeats_baselines(shell_tree, control_status):
    root, env = shell_tree
    env["G5_STATUS_CONTROL"] = str(control_status)
    done = _run(root, env)
    calls = _calls(root)
    baselines = [call for call in calls if call["stage"] == "baseline"]
    if control_status in {2, 19}:
        assert done.returncode != 0
        assert calls[-1]["stage"] == "control"
        assert len(baselines) == 1
    else:
        assert done.returncode == 0, done.stdout + done.stderr
        assert len(baselines) == (7 if control_status == 1 else 1)
        assert len([call for call in calls if call["stage"] == "mixed"]) == 12
        assert calls[-1]["stage"] == "verdict"
        if control_status == 1:
            assert [(call["args"][call["args"].index("--arm") + 1],
                     call["args"][call["args"].index("--seed") + 1]) for call in baselines] == [
                ("nf4", "42"), ("q2_0", "42"), ("nf4", "7"), ("q2_0", "7"),
                ("nf4", "123"), ("q2_0", "123"), ("bf16", "42"),
            ]


@pytest.mark.parametrize("failure", ["python", "tee", "no_record"])
def test_repeated_baseline_failure_stops_before_mixed_training(shell_tree, failure):
    root, env = shell_tree
    control = root / "out" / "baselines" / "nf4-s42-a1"
    control.mkdir(parents=True)
    (control / "run.json").write_text("{}", encoding="utf-8")
    env["G5_STATUS_CONTROL"] = "1"
    if failure == "python":
        env["G5_STATUS_BASELINE"] = "19"
    elif failure == "tee":
        env["G5_FAIL_TEE"] = "q2_0-s42-a1.log"
    else:
        env["G5_NO_RECORD_BASELINE"] = "1"
    done = _run(root, env)
    calls = _calls(root)
    assert done.returncode != 0, done.stdout + done.stderr
    assert calls[-1]["stage"] == "baseline"
    assert sum(call["stage"] == "baseline" for call in calls) == 1
    assert not any(call["stage"] == "mixed" for call in calls)


@pytest.mark.parametrize("stage", ["selection", "bytes"])
@pytest.mark.parametrize("failure", ["failed_check", "missing_checks", "wrong_mode", "malformed"])
def test_invalid_cached_precondition_stops_before_training(shell_tree, stage, failure):
    root, env = shell_tree
    selection = _selection(root)
    (selection / "bytes.json").write_text(json.dumps({
        "kind": "g5_bytes", "synthetic": True,
        "checks": {"mixed_arms_of_one_fraction_equal_bytes": True},
    }), encoding="utf-8")
    path = selection / f"{stage}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if failure == "failed_check":
        record["checks"][next(iter(record["checks"]))] = False
    elif failure == "missing_checks":
        record.pop("checks")
    elif failure == "wrong_mode":
        record["synthetic"] = False
    path.write_text("{broken" if failure == "malformed" else json.dumps(record), encoding="utf-8")
    done = _run(root, env)
    assert done.returncode != 0, done.stdout + done.stderr
    assert not any(call["stage"] in {"baseline", "mixed"} for call in _calls(root))


def test_successful_resume_reuses_checked_outputs_and_keeps_attempts(shell_tree):
    root, env = shell_tree
    first = _run(root, env)
    assert first.returncode == 0, first.stdout + first.stderr
    (root / "calls.jsonl").unlink()
    resumed = _run(root, env)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert [call["stage"] for call in _calls(root)] == ["provenance", "control", "verdict"]


def test_real_provenance_mismatch_stops_before_any_model_command(shell_tree):
    root, env = shell_tree
    env["G5_STATUS_PROVENANCE"] = "1"
    done = _run(root, env, mode="real")
    assert done.returncode != 0
    assert [call["stage"] for call in _calls(root)] == ["provenance"]


@pytest.mark.parametrize("failure", ["git", "output_directory"])
def test_setup_failure_stops_before_any_python_command(shell_tree, failure):
    root, env = shell_tree
    if failure == "git":
        env["G5_FAIL_GIT"] = "1"
    else:
        (root / "out").write_text("not a directory", encoding="utf-8")
    done = _run(root, env)
    assert done.returncode != 0
    assert _calls(root) == []
