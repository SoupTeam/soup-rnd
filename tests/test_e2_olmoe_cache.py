"""E2 OLMoE cache series follows its prospective protocol (CPU only, no model)."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
OC = ROOT / "benchmarks" / "e2" / "olmoe_cache"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm_mod = _module(OC / "run_arm.py", "olmoe_run_arm")
ana = _module(OC / "analyze_olmoe.py", "olmoe_analyze")
speed = ana.speed


def test_order_and_commands(tmp_path):
    assert len(arm_mod.ORDER) == 13 and arm_mod.ORDER[0] == "C"
    a = arm_mod.command("A_c1_1", tmp_path / "c", tmp_path / "q", tmp_path / "x")
    b = arm_mod.command("B_c2_2", tmp_path / "c", tmp_path / "q", tmp_path / "x")
    assert "--keep-top" not in a and "--cache-dir" not in a
    assert b[b.index("--keep-top") + 1] == "4" and "--cache-dir" in b
    assert all(x[x.index("--steps") + 1] == "6000" for x in (a, b))


def test_template_is_the_mistral_recipe_with_the_moe_changes():
    mistral = yaml.safe_load((ROOT / "benchmarks/e2/recipe_check/configs/W32.yaml").read_text())
    olmoe = yaml.safe_load((OC / "configs/O16.yaml").read_text())
    assert olmoe["training"].pop("moe_aux_loss_coeff") == 0
    assert olmoe["training"]["lora"].pop("top_k_layers") == 16
    mistral["training"]["lora"].pop("top_k_layers")
    assert olmoe == mistral


def test_rendered_config_loads_with_the_router_loss_off(tmp_path):
    from soup_cli.config.loader import load_config

    path = tmp_path / "o16.yaml"
    path.write_text(arm_mod.render(tmp_path / "b", tmp_path / "o"))
    config = load_config(str(path))
    assert config.training.moe_aux_loss_coeff == 0 and config.training.lora.top_k_layers == 16


def _run(root, name, *, epochs=(550.0, 548.0, 547.0), fp="a", cache=None, hybrid=False):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(arm_mod.render(Path("/b"), d / "adapter"))
    (d / "arm.json").write_text(json.dumps({"ok": True, "process_wall_seconds": 1000.0}))
    timing = [
        {"epoch": i + 1, "start_step": 2000 * i, "end_step": 2000 * (i + 1),
         "duration_seconds": s,
         "cache_hits": None if cache is None or i == 0 else (0 if i == 1 else 2000),
         "cache_misses": None if cache is None or i == 0 else (2000 if i == 1 else 0)}
        for i, s in enumerate(epochs)
    ]
    quality = {
        "steps": 6000, "validation_loss": 0.9, "train_examples": 2000,
        "validation_examples": 200, "deterministic_algorithms": True,
        "trainable_fingerprint": fp, "training_batches_sha256": "x", "epoch_timing": timing,
    }
    if hybrid:
        quality["hybrid"] = {
            "frozen": True, "freeze_at_step": 2000, "frozen_layers": list(range(12)),
            "frozen_tensors": [f"t{i}" for i in range(48)],
            "bottom_changed_before_freeze": True, "bottom_unchanged_after_freeze": True,
            "top_changed_after_freeze": True, "requires_grad_after": ["False"],
        }
        if cache is not None:
            quality["hybrid"]["cache"] = cache
    (d / "quality.json").write_text(json.dumps(quality))


CACHE = {"installed_at_step": 2000, "cutoff": 12, "install_seconds": 9.0,
         "hits": 2000, "misses": 2000, "files": 2000, "bytes": 1 << 30}


def session(tmp_path, b_epoch3=200.0):
    root = tmp_path / "s"
    _run(root, "C", fp="h", hybrid=True)
    for c in (1, 2, 3):
        for i in (1, 2):
            _run(root, f"A_c{c}_{i}")
            _run(root, f"B_c{c}_{i}", epochs=(550.0, 360.0, b_epoch3), fp="h",
                 cache=dict(CACHE), hybrid=True)
    return root


def test_speedup_label_uses_the_olmoe_layer_counts(tmp_path):
    result = ana.analyze(session(tmp_path))
    assert result["label"] == "WARM_SPEEDUP", result.get("reasons")
    assert result["schema"] == "e2-olmoe-cache-analysis-v1"
    assert ana.analyze(session(tmp_path / "slow", b_epoch3=520.0))["label"] == "NOT_SHOWN"


def test_a_mistral_shaped_cache_is_refused(tmp_path):
    root = session(tmp_path)
    q = json.loads((root / "B_c1_1" / "quality.json").read_text())
    q["hybrid"]["cache"]["cutoff"] = 24
    (root / "B_c1_1" / "quality.json").write_text(json.dumps(q))
    reasons = ana.analyze(root)["reasons"]
    assert any("B_c1_1" in r and "layers 0-11" in r for r in reasons)


def test_notebooks_run_the_olmoe_runner_with_the_pinned_revision():
    for name, needle in (("e2_olmoe_cache.ipynb", "benchmarks/e2/olmoe_cache/run_arm.py"),
                         ("e2_olmoe_smoke.ipynb", "benchmarks/e2/olmoe_cache/smoke.py")):
        nb = json.loads((OC / name).read_text())
        code = "".join("".join(c["source"]) for c in nb["cells"])
        assert needle in code and "6d84c48581ece794365f2b8e9cfb043c68ade9c5" in code
    main = "".join("".join(c["source"]) for c in json.loads(
        (OC / "e2_olmoe_cache.ipynb").read_text())["cells"])
    for name in arm_mod.ORDER:
        assert f'"{name}"' in main
    assert "e2oc_results_" in main


def test_smoke_script_compares_the_two_runs():
    text = (OC / "smoke.py").read_text()
    assert '"--freeze-at-step", "20"' in text and "fingerprint" in text


@pytest.mark.parametrize("n", [16])
def test_protocol_states_the_router_loss_limit(n):
    text = (OC / "PROTOCOL.txt").read_text()
    assert "moe_aux_loss_coeff: 0" in text and f"{n} decoder layers" in text
