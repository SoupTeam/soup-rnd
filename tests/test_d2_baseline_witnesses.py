"""The compact failure proof rejects corrupt payloads and false numerical claims."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    path = Path(__file__).parents[1] / "benchmarks/harness/d2_verify_baseline_witnesses.py"
    spec = importlib.util.spec_from_file_location("witness_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_fixture(root):
    import torch
    from safetensors.torch import save_file

    table, witnesses = [], []
    for dtype in ("float16", "bfloat16"):
        for backend in ("peft", "soup", "liger", "unsloth"):
            folder = root / f"{dtype}-gate-{backend}"
            folder.mkdir(parents=True)
            loss = 1.0 if backend == "peft" else 0.9
            (folder / "result.json").write_text(
                json.dumps({"rows": [{"step": index, "loss": loss} for index in range(1, 51)]})
            )
            table.append(
                {
                    "dtype": dtype,
                    "backend": backend,
                    "loss_matches_3dp": 50 if backend == "peft" else 0,
                }
            )
            if backend != "peft":
                name = f"{dtype}-{backend}.safetensors"
                save_file(
                    {"reference": torch.ones(2), "candidate": torch.full((2,), 2.0)},
                    str(root / name),
                )
                witnesses.append(
                    {
                        "file": name,
                        "dtype": dtype,
                        "backend": backend,
                        "quantity": "gradient",
                        "relative_l2": 1.0,
                    }
                )
    (root / "summary.json").write_text(json.dumps({"table": table, "witnesses": witnesses}))
    refresh_manifest(root)


def refresh_manifest(root):
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.name != "compact-manifest.json"
    }
    (root / "compact-manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("defect", ["none", "payload", "false_ratio", "escape", "witness_escape"])
def test_failure_witnesses_are_independently_reproducible(replay, tmp_path, defect):
    compact_fixture(tmp_path)
    if defect == "payload":
        (tmp_path / "float16-soup.safetensors").write_bytes(b"corrupted")
    elif defect in ("false_ratio", "witness_escape"):
        path = tmp_path / "summary.json"
        data = json.loads(path.read_text())
        if defect == "false_ratio":
            data["witnesses"][0]["relative_l2"] = 0.5
        else:
            data["witnesses"][0]["file"] = "../outside.safetensors"
        path.write_text(json.dumps(data))
        refresh_manifest(tmp_path)
    elif defect == "escape":
        path = tmp_path / "compact-manifest.json"
        path.write_text(json.dumps({"../outside": "untrusted"}))
    if defect != "none":
        with pytest.raises(ValueError):
            replay.verify(tmp_path)
    else:
        result = replay.verify(tmp_path)
        assert result["loss_curves_verified"] == 8
        assert result["loss_curve_failures"] == 6
        assert len(result["failure_witnesses"]) == 6
