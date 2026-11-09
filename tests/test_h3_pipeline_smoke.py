"""Pipeline preparation and real CPU P2P/autograd checks, not GPU evidence."""

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parents[1] / "benchmarks/harness/h3_pipeline_smoke.py"
_SPEC = importlib.util.spec_from_file_location("h3_pipeline_smoke", _HARNESS)
pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline)


@pytest.mark.parametrize("overrides", [
    {"batch_size": "auto"}, {"batch_size": 3},
    {"pipeline_stages": 4}, {"parallelism": "data"},
])
def test_recipe_rejects_unsupported_batch_or_partition(overrides):
    from soup_cli.config.schema import SoupConfig

    training = {"parallelism": "pipeline", "pipeline_stages": 2, "batch_size": 4,
                "quantization": "none", "lora": {"dropout": 0.0}}
    config = SoupConfig(base="fixture", data={"train": "fixture.jsonl"},
                        training={**training, **overrides})
    with pytest.raises(ValueError):
        pipeline.plan(config)


def test_pipeline_batch_is_not_multiplied_by_number_of_stages():
    from soup_cli.config.schema import SoupConfig

    config = SoupConfig(base="fixture", data={"train": "fixture.jsonl"}, training={
        "parallelism": "pipeline", "pipeline_stages": 2, "batch_size": 4,
        "gradient_accumulation_steps": 2, "quantization": "none", "lora": {"dropout": 0.0},
    })
    assert pipeline.plan(config) == {
        "stages": 2, "microbatches": 2, "micro_batch": 2, "schedule_batch": 4,
        "accumulation": 2, "global_batch": 8,
    }


@pytest.mark.parametrize("shards", [[{"a": 1}, {"a": 1}], [{"a": 1}, {}],
                                    [{"a": 1}, {"c": 1}]])
def test_merge_rejects_duplicate_missing_and_unexpected_adapter_ownership(shards):
    with pytest.raises(AssertionError):
        pipeline.merge_adapter_shards(shards, {"a", "b"})


def test_merge_preserves_all_adapter_keys():
    assert pipeline.merge_adapter_shards([{"a": 1}, {"b": 2}], {"a", "b"}) == {"a": 1, "b": 2}


def test_launcher_keeps_active_python_when_global_torchrun_is_shadowed(tmp_path, monkeypatch):
    """Real subprocess/module selection; no CUDA or Torch import is needed."""
    package = tmp_path / "torch"
    (package / "distributed").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "distributed/__init__.py").write_text("")
    result = tmp_path / "worker-environment.json"
    (package / "distributed/run.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['H3_LAUNCH_RESULT']).write_text(json.dumps("
        "{'executable': sys.executable, 'args': sys.argv[1:]}))\n",
    )
    bad_console = tmp_path / "bin"
    bad_console.mkdir()
    for name in ("torchrun", "torchrun.bat"):
        (bad_console / name).write_text("#!/bin/sh\nexit 73\n")
        (bad_console / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bad_console) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("H3_LAUNCH_RESULT", str(result))
    pipeline.launch_training(tmp_path / "prepared", tmp_path / "output", "cuda", 4)
    observed = json.loads(result.read_text())
    assert os.path.samefile(observed["executable"], sys.executable)
    assert observed["args"][:2] == ["--standalone", "--nproc-per-node=2"]
    assert observed["args"][-4:] == ["--device", "cuda", "--steps", "4"]


@pytest.mark.parametrize("changed", ["config.json", "tokens.pt", "base/model.safetensors"])
def test_tampered_prepared_inputs_are_rejected(tmp_path, changed):
    from soup_cli.config.schema import SoupConfig

    (tmp_path / "base").mkdir()
    files = ["config.json", "tokens.pt", "base/model.safetensors"]
    for name in files:
        (tmp_path / name).write_bytes(b"unchanged")
    digest = hashlib.sha256(b"unchanged").hexdigest()
    pipeline.write_json(tmp_path / "prepared.json", {
        "tokens_sha256": digest, "config_sha256": digest,
        "base_files_sha256": {"model.safetensors": digest},
    })
    config = SoupConfig(base=str(tmp_path / "base"), data={"train": "fixture.jsonl"})
    (tmp_path / changed).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        pipeline.verify_prepared(tmp_path, config)


def _cpu_worker(rank, root, output, port):
    import faulthandler

    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"worker-{rank}.log").open("w", encoding="utf-8") as log:
        # Preserve the blocked Python stack before the parent's unchanged 120s deadline.
        faulthandler.dump_traceback_later(90, file=log)
        try:
            import torch

            torch.set_num_threads(1)
            log.write(f"rank={rank} torch={torch.__version__} threads={torch.get_num_threads()}\n")
            log.flush()
            pipeline.train(Path(root), directory, "cpu", 2)
            log.write("training completed\n")
        finally:
            faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def cpu_pipeline_runtime(monkeypatch):
    import torch

    # Both spawned ranks and the independent reload inherit bounded BLAS pools.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(name, "1")
    if sys.platform == "darwin":
        # This is a same-host test; don't let Gloo pick an external VM interface.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _join_pipeline(context, output, seconds=120):
    deadline = time.monotonic() + seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                diagnostics = []
                for rank, process in enumerate(context.processes):
                    log = output / f"worker-{rank}.log"
                    diagnostics.append(
                        f"rank={rank} pid={process.pid} exitcode={process.exitcode}\n"
                        + (log.read_text(encoding="utf-8")[-12000:]
                           if log.exists() else "worker log not created")
                    )
                pytest.fail(f"CPU pipeline exceeded {seconds} seconds\n" + "\n".join(diagnostics))
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.parametrize("stubborn", [False, True])
def test_pipeline_deadline_preserves_diagnostics_and_reaps_workers(tmp_path, stubborn):
    class Process:
        pid, exitcode = 123, None
        alive = True
        killed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = stubborn

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout):
            pass

    class Context:
        processes = [Process()]

        def join(self, timeout):
            return False

    context = Context()
    (tmp_path / "worker-0.log").write_text("blocked in Gloo\n", encoding="utf-8")
    with pytest.raises(
        pytest.fail.Exception, match="CPU pipeline exceeded 0 seconds",
    ) as failure:
        _join_pipeline(context, tmp_path, seconds=0)
    assert "blocked in Gloo" in str(failure.value)
    assert not context.processes[0].alive
    assert context.processes[0].killed == stubborn


def test_pipeline_child_failure_is_propagated_and_other_workers_are_reaped(tmp_path):
    class Process:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, timeout):
            pass

    class Context:
        processes = [Process()]

        def join(self, timeout):
            raise RuntimeError("child numerical failure")

    context = Context()
    with pytest.raises(RuntimeError, match="child numerical failure"):
        _join_pipeline(context, tmp_path)
    assert not context.processes[0].alive


@pytest.mark.integration
def test_real_pipeline_training_accumulation_and_fresh_reload(tmp_path, cpu_pipeline_runtime):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires Torch distributed/Gloo")
    prepared, output = tmp_path / "prepared", tmp_path / "output"
    pipeline.prepare(prepared, True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = mp.spawn(_cpu_worker, (str(prepared), str(output), port), nprocs=2, join=False)
    _join_pipeline(context, output)
    result = subprocess.run([
        sys.executable, str(_HARNESS), "reload", "--prepared", str(prepared),
        "--output", str(output), "--device", "cpu",
    ], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads((output / f"rank-{index}/record.json").read_text()) for index in range(2)]
    assert records[0]["layers"] == [0, 1] and records[1]["layers"] == [2, 3]
    assert all(record["changed_adapter_tensors"] > 0 for record in records)
    assert all(record["plan"]["global_batch"] == 8 for record in records)
    assert all(record["observed_stage_calls"] == {"forward": 8, "backward": 8}
               for record in records)
    assert records[0]["records"][0]["row_ids"] == records[1]["records"][0]["row_ids"]
    assert all(record["frozen_before"] == record["frozen_after"] for record in records)
    reload = json.loads((output / "reload.json").read_text())
    assert reload["passed"] and reload["adapter_keys"] == 16
