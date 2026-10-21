#!/usr/bin/env python3
"""Prepare local A0 fixtures before the quiet measurement window; no measured verdict.

Run after installing the CUDA torch build and benchmarks/a0-requirements.txt.
The real Mistral download is public and pinned to a Hub commit; credentials are not used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import l2l_session

MODEL_ID = "unsloth/mistral-7b-instruct-v0.3"


def direct_io_probe(out: Path) -> None:
    """Test the actual preparation filesystem before generating a 140 GB fixture."""
    import l2l_activations
    import torch

    path = out / "direct-io-probe.bin"
    if path.exists():
        raise FileExistsError(path)
    store = l2l_activations.DirectSpillFile(str(path), 65536)
    try:
        written = l2l_activations.aligned_buffer(65536, pin=False)
        written.fill_(123)
        received = l2l_activations.aligned_buffer(65536, pin=False)
        store.write(0, written)
        store.read(0, received)
        if not torch.equal(written, received):
            raise RuntimeError("direct I/O round trip differs")
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new preparation log directory")
    parser.add_argument("--power-source", choices=("sensor", "server"), default="sensor")
    args = parser.parse_args(argv)
    root = l2l_session.ROOT
    out = Path(os.path.abspath(args.out))
    out.mkdir(parents=True, exist_ok=False)
    state = {"state": "preparing", "base_commit": l2l_session.git_head(),
             "versions": l2l_session.versions()}
    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; stop before downloads or fixture generation")
        if shutil.disk_usage(root).free < 205_000_000_000:
            raise RuntimeError("need at least 205 GB free for source weights, NF4 caches and spill")
        direct_io_probe(out)
        state["direct_io_round_trip"] = True
        models = root / ".soup" / "models"
        models.mkdir(parents=True, exist_ok=True)
        f1, f2 = models / "mistral-7b", models / "synth-70b-v32k"
        # Resolve once, before download; every weight comes from the same public commit.
        revision = HfApi(token=False).model_info(MODEL_ID).sha
        state.update(model_id=MODEL_ID, model_revision=revision)
        l2l_session.write_json(out / "preparation.json", state)
        snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=f1, token=False,
                          allow_patterns=["config.json", "*.safetensors",
                                          "*.safetensors.index.json"])
        env = {**os.environ, "PYTHONPATH": str(root / "src"), "HF_HUB_OFFLINE": "1"}
        if args.power_source == "server":
            env["SOUP_L2L_POWER_SOURCE"] = "server"
        else:
            env.pop("SOUP_L2L_POWER_SOURCE", None)
        jobs = [
            ("generate", [sys.executable, str(l2l_session.HARNESS / "synth_checkpoint.py"),
                          "--shape", "llama-70b", "--vocab", "32768", "--out", str(f2)]),
        ]
        for fixture, weights, tier, k in (("f1", f1, "ram", "4"), ("f2", f2, "disk", "16")):
            jobs.append((fixture, [sys.executable, str(l2l_session.HARNESS / "l2l_probe.py"),
                "--mode", "timing" if fixture == "f2" else "correctness", "--dry-run",
                "--weights", str(weights), "--shards", str(root / ".soup" / f"{fixture}-nf4"),
                "--tier", tier, "--k", k, "--out", str(out / f"{fixture}.json")]))
        for name, command in jobs:
            with (out / f"{name}.log").open("xb") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        f2_result = json.loads((out / "f2.json").read_text())
        if not f2_result["meta"].get("v3", {}).get("ok"):
            raise RuntimeError("F2 preparation V3 failed; inspect f2.log")
        if f2_result["meta"].get("direct_io") is not True:
            raise RuntimeError("F2 weight reader did not enable direct I/O")
        if f2_result["meta"].get("dry_run_store_allocated") is not True:
            raise RuntimeError("F2 k16 pinned store allocation did not pass")
        state["state"] = "fixtures prepared; review logs before measurement"
    except (Exception, KeyboardInterrupt) as exc:
        state.update(state="stopped", reason=f"{type(exc).__name__}: {exc}")
    l2l_session.write_json(out / "preparation.json", state)
    return 0 if state["state"].startswith("fixtures prepared") else 3


if __name__ == "__main__":
    raise SystemExit(main())
