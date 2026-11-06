"""E2 hybrid seed series follows its prospective protocol (CPU only, no model)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from soup_cli.eval import gate_suites as suites

ROOT = Path(__file__).parents[1]
HS = ROOT / "benchmarks" / "e2" / "hybrid_seeds"
sys.path.insert(0, str(HS))
import analyze_seeds  # noqa: E402


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seeds_arm = _module(HS / "run_arm.py", "hybrid_seeds_run_arm")
CE = analyze_seeds.ANCHOR_CE
FREEZE = [f"{pkg}=={v}" for pkg, v in analyze_seeds.VERSIONS.items()]


def test_arm_list_is_the_protocol_list():
    assert set(seeds_arm.ARMS) == {
        "ANCHOR_s42", "H8_s42",
        "W32_s901", "H24_s901", "H16_s901", "H8_s901",
        "W32_s1234", "H24_s1234", "H16_s1234", "H8_s1234",
    }
    with pytest.raises(ValueError):
        seeds_arm.parse("H24_s42")  # reused, never rerun


def test_render_sets_both_seeds_and_nothing_else(tmp_path):
    text = seeds_arm.render(901, tmp_path / "b", tmp_path / "o")
    cfg = yaml.safe_load(text)
    assert cfg["training"]["seed"] == cfg["training"]["data_seed"] == 901
    base = yaml.safe_load(seeds_arm.render(42, tmp_path / "b", tmp_path / "o"))
    cfg["training"]["seed"] = cfg["training"]["data_seed"] = 42
    assert cfg == base


def test_commands_pass_the_seed_and_the_kept_layers(tmp_path):
    h8 = seeds_arm.command("H8", 1234, tmp_path / "c", tmp_path / "q")
    assert h8[h8.index("--seed") + 1] == "1234" and h8[h8.index("--keep-top") + 1] == "8"
    w = seeds_arm.command("W32", 901, tmp_path / "c", tmp_path / "q")
    assert "--keep-top" not in w and "benchmarks/e2/hybrid/run_hybrid.py" in w


def _arm(session, name, ce, *, kind_k=None, seed=42, delta_mmlu=0.0, folder=None):
    d = session / (folder or name)
    (d / "gate").mkdir(parents=True)
    (d / "config.yaml").write_text(seeds_arm.render(seed, Path("/b"), d / "adapter"))
    (d / "arm.json").write_text(json.dumps({"ok": True}))
    quality = {
        "steps": 4000, "validation_loss": ce, "train_examples": 2000,
        "validation_examples": 200, "parameters_changed": True,
        "deterministic_algorithms": True, "trainable_fingerprint": "f",
    }
    if kind_k:
        cutoff = 32 - kind_k
        quality["hybrid"] = {
            "frozen": True, "freeze_at_step": 2000, "frozen_layers": list(range(cutoff)),
            "frozen_tensors": [f"t{i}" for i in range(cutoff * 4)],
            "bottom_changed_before_freeze": True, "bottom_unchanged_after_freeze": True,
            "top_changed_after_freeze": True, "requires_grad_after": ["False"],
        }
    (d / "quality.json").write_text(json.dumps(quality))
    if name.startswith("ANCHOR"):
        return
    scores = {}
    for suite in suites.DEFAULT_GENERAL_SUITE:
        items = suites.MINI_BENCHMARKS.get(suite) or suites.load_suite_items(suite)
        # Answer every MMLU item correctly for the control; drop some for delta_mmlu.
        wrong = round(-delta_mmlu * len(items)) if suite == "mini_mmlu" else 0
        rows = []
        index = iter(range(len(items)))

        def gen(prompt, _suite=suite, _items=items, _wrong=wrong):
            i = next(index)
            text = "Sure. The answer is 4."
            if _suite == "mini_mmlu":
                text = "Z" if i < _wrong else f"Answer: ({_items[i]['answer']})"
            rows.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "scored_output_text": text})
            return text

        score = float(suites.score_bundled_suite(suite, gen))
        (d / "gate" / f"{suite}.generation_audit.json").write_text(json.dumps(rows))
        scores[suite] = {"score": score, "items": len(rows), "generation_errors": 0}
    (d / "gate" / "report.json").write_text(json.dumps({
        "scorer_revision": 5, "saved_adapter_weights_verified": True,
        "loaded_lora_fingerprint": "f", "scores": scores,
    }))


def build(tmp_path, gaps, *, mmlu=None, anchor=CE, freeze=FREEZE):
    """gaps/mmlu: {(k, seed): value}; returns (sessions, reuse)."""
    mmlu = mmlu or {}
    reuse = tmp_path / "reuse"
    reuse.mkdir(parents=True)
    (reuse / "session.json").write_text(json.dumps({"pip_freeze": FREEZE}))
    _arm(reuse, "W32_s42", CE, folder="W32")
    for k in (24, 16):
        _arm(reuse, f"H{k}_s42", CE * (1 + gaps[(k, 42)] / 100), kind_k=k,
             delta_mmlu=mmlu.get((k, 42), 0.0), folder=f"H{k}")
    sessions = []
    for seed, names in ((901, ["W32", "H24", "H16", "H8"]), (1234, ["W32", "H24", "H16", "H8"]),
                        (42, ["ANCHOR", "H8"])):
        s = tmp_path / f"s{seed}"
        s.mkdir()
        (s / "session.json").write_text(json.dumps({"pip_freeze": freeze}))
        for kind in names:
            name = f"{kind}_s{seed}"
            if kind == "ANCHOR":
                _arm(s, name, anchor)
            elif kind == "W32":
                _arm(s, name, CE * 1.01, seed=seed)
            else:
                k = int(kind[1:])
                control = CE if seed == 42 else CE * 1.01
                _arm(s, name, control * (1 + gaps[(k, seed)] / 100), kind_k=k, seed=seed,
                     delta_mmlu=mmlu.get((k, seed), 0.0))
        sessions.append(s)
    return sessions, reuse


def gaps_for(**per_k):
    return {(k, s): per_k[f"k{k}"][i] for k in (24, 16, 8) for i, s in enumerate((42, 901, 1234))}


def test_labels_per_k_and_the_recommended_k(tmp_path):
    gaps = gaps_for(k24=[-0.2, 0.1, 0.3], k16=[0.2, 0.6, 0.9], k8=[0.9, 1.4, 1.8])
    sessions, reuse = build(tmp_path, gaps)
    result = analyze_seeds.analyze(sessions, reuse)
    assert result["label"] == "DECIDED", result.get("reasons")
    labels = {k: v["label"] for k, v in result["per_k"].items()}
    assert labels == {"K24": "WITHIN_MARGIN", "K16": "BORDERLINE", "K8": "FAILS"}
    assert result["recommended_k_for_speed_check"] == 24


def test_suite_deltas_are_averaged_over_seeds(tmp_path):
    gaps = gaps_for(k24=[0, 0, 0], k16=[0, 0, 0], k8=[0, 0, 0])
    # K16: one seed two questions down (-0.077), the others equal: mean -0.026 passes.
    sessions, reuse = build(tmp_path, gaps, mmlu={(16, 42): -0.077})
    result = analyze_seeds.analyze(sessions, reuse)
    assert result["per_k"]["K16"]["label"] == "WITHIN_MARGIN"
    assert result["per_k"]["K16"]["mean_suite_deltas"]["mini_mmlu"] == pytest.approx(
        -2 / 26 / 3
    )
    assert result["recommended_k_for_speed_check"] == 8
    worse = build(tmp_path / "w", gaps, mmlu={(8, s): -0.077 for s in (42, 901, 1234)})
    assert analyze_seeds.analyze(*worse)["per_k"]["K8"]["label"] == "FAILS"


def test_anchor_versions_and_missing_arms_give_no_verdict(tmp_path):
    gaps = gaps_for(k24=[0, 0, 0], k16=[0, 0, 0], k8=[0, 0, 0])
    drift = analyze_seeds.analyze(*build(tmp_path / "a", gaps, anchor=CE * (1 + 1e-6)))
    assert drift["label"] == "NO_VERDICT" and any("V2" in r for r in drift["reasons"])
    other = ["torch==2.12.0", "transformers==5.20.2", "peft==0.21.2"]
    versions = analyze_seeds.analyze(*build(tmp_path / "b", gaps, freeze=other))
    assert any("V5" in r for r in versions["reasons"])
    sessions, reuse = build(tmp_path / "c", gaps)
    import shutil
    shutil.rmtree(sessions[0] / "H8_s901")
    assert "H8_s901: missing" in analyze_seeds.analyze(sessions, reuse)["reasons"]


def test_a_config_with_the_wrong_seed_is_refused(tmp_path):
    gaps = gaps_for(k24=[0, 0, 0], k16=[0, 0, 0], k8=[0, 0, 0])
    sessions, reuse = build(tmp_path, gaps)
    config = sessions[0] / "H24_s901" / "config.yaml"
    config.write_text(config.read_text().replace("data_seed: 901", "data_seed: 42"))
    reasons = analyze_seeds.analyze(sessions, reuse)["reasons"]
    assert any("H24_s901" in r and "V4" in r for r in reasons)


def test_seed_notebook_runs_the_seed_runner():
    nb = json.loads((HS / "e2_hybrid_seeds.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'ARMS = ["W32_s901", "H24_s901", "H16_s901", "H8_s901"]' in code
    assert "benchmarks/e2/hybrid_seeds/run_arm.py" in code and "e2hs_results_" in code


def test_retained_kaggle_series_replays_to_the_published_result():
    report = HS / "reports" / "20261106"
    evidence = report / "evidence"
    sessions = [evidence / s for s in ("20261105_201228", "20261105_230802", "20261106_020313")]
    result = analyze_seeds.analyze(sessions)
    published = json.loads((report / "analysis.json").read_text())
    assert result["label"] == published["label"] == "DECIDED"
    assert result["per_k"] == published["per_k"]
    assert result["recommended_k_for_speed_check"] == 8
    manifest = json.loads((report / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
