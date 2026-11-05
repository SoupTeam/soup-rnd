"""E2 recipe check follows its prospective protocol (CPU only, no model)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from soup_cli.eval import gate_suites as suites

ROOT = Path(__file__).parents[1]
RC = ROOT / "benchmarks" / "e2" / "recipe_check"
RECORD = ROOT / "benchmarks" / "e2" / "quality" / "mistral7b_two_epoch_record"
sys.path.insert(0, str(RC))
import analyze_recipe  # noqa: E402
import kit  # noqa: E402
import run_arm  # noqa: E402

CONSTANT = "Sure. The answer is 4."


def _cfg(name):
    return yaml.safe_load((RC / "configs" / f"{name}.yaml").read_text())


def test_weak_arms_are_the_recorded_configs_with_only_base_and_output_replaced():
    for arm, record in (("W32", "k32_control"), ("W24", "k24_baseline")):
        mine = _cfg(arm)
        theirs = yaml.safe_load((RECORD / f"{record}.yaml").read_text())
        assert mine.pop("base") == "__BASE__" and mine.pop("output") == "__OUTPUT__"
        theirs.pop("base"), theirs.pop("output")
        assert mine == theirs


def test_strong_arms_change_only_rank_alpha_modules_and_k():
    weak = _cfg("W32")
    for arm, k in (("S32", 32), ("S24", 24), ("S16", 16)):
        strong = _cfg(arm)
        lora = strong["training"].pop("lora")
        weak_copy = json.loads(json.dumps(weak))
        weak_lora = weak_copy["training"].pop("lora")
        assert strong == weak_copy
        assert lora.pop("top_k_layers") == k and weak_lora.pop("top_k_layers") == 32
        assert (lora.pop("r"), lora.pop("alpha")) == (16, 32)
        assert lora.pop("target_modules") == [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
        for key in ("r", "alpha", "target_modules"):
            weak_lora.pop(key)
        assert lora == weak_lora
    assert weak["training"]["frozen_prefix_cache"] is False


def test_gate_runner_is_the_saved_runner_with_only_the_declared_changes():
    saved = (ROOT / "benchmarks/e2/gates/run_saved_gate.py").read_text()
    mine = (RC / "run_gate.py").read_text()
    assert "BUNDLED_SCORER_REVISION != 5" in mine
    assert "args.suite or list(suites.DEFAULT_GENERAL_SUITE)" in mine
    kept = [line for line in saved.splitlines()[8:] if "!= 4" not in line
            and '["mini_mmlu"]' not in line and "git_commit" not in line]
    body = mine.splitlines()
    assert all(line in body for line in kept)
    assert suites.BUNDLED_SCORER_REVISION == 5


def test_render_replaces_base_and_output_once(tmp_path):
    text = run_arm.render("S24", tmp_path / "base", tmp_path / "out")
    assert "__BASE__" not in text and str(tmp_path / "base") in text
    assert yaml.safe_load(text)["training"]["lora"]["top_k_layers"] == 24


def test_kit_round_trip_and_tamper_detection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("one\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    out = tmp_path / "kit.zip"
    import os
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        built = kit.build(out)
    finally:
        os.chdir(cwd)
    unpacked = tmp_path / "unpacked"
    import zipfile
    with zipfile.ZipFile(out) as z:
        z.extractall(unpacked)
    assert kit.verify(unpacked)["commit"] == built["commit"]
    (unpacked / "a.txt").write_text("two\n")
    with pytest.raises(RuntimeError, match="differ"):
        kit.verify(unpacked)


def _audit(name, text):
    items = analyze_recipe._items(name)
    rows = []

    def gen(prompt):
        rows.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "scored_output_text": text})
        return text

    score = float(suites.score_bundled_suite(name, gen))
    return rows, score, len(items)


def make_arm(root, arm, ce, *, text=CONSTANT, tamper_score=False, config_extra=""):
    d = root / arm
    (d / "gate").mkdir(parents=True)
    config = run_arm.render(arm, Path("/tmp/base"), d / "adapter") + config_extra
    (d / "config.yaml").write_text(config)
    (d / "arm.json").write_text(json.dumps({"ok": True}))
    (d / "quality.json").write_text(json.dumps({
        "steps": 4000, "validation_loss": ce, "train_examples": 2000,
        "validation_examples": 200, "parameters_changed": True,
        "deterministic_algorithms": True, "trainable_fingerprint": "f",
    }))
    scores = {}
    for name in suites.DEFAULT_GENERAL_SUITE:
        rows, score, n = _audit(name, text)
        (d / "gate" / f"{name}.generation_audit.json").write_text(json.dumps(rows))
        scores[name] = {"score": score + (0.5 if tamper_score else 0.0), "items": n,
                        "generation_errors": 0}
    (d / "gate" / "report.json").write_text(json.dumps({
        "scorer_revision": 5, "saved_adapter_weights_verified": True,
        "loaded_lora_fingerprint": "f", "scores": scores,
    }))


def series(tmp_path, ces, **arm_kwargs):
    root = tmp_path / "session"
    root.mkdir(parents=True)
    for arm, ce in ces.items():
        make_arm(root, arm, ce, **arm_kwargs.get(arm, {}))
    return root


W = analyze_recipe.RECORD_CE_W32


def test_persisting_gap_with_reproduced_anchor(tmp_path):
    root = series(tmp_path, {"W32": W, "W24": W * 1.015, "S32": 0.85, "S24": 0.85 * 1.013})
    result = analyze_recipe.analyze([root])
    assert result["label"] == "GAP_PERSISTS", result.get("reasons")
    assert result["weak"]["ce_gap_percent"] == pytest.approx(1.5)
    assert all(v == 0 for v in result["strong"]["suite_deltas"].values())


def test_reduced_gap(tmp_path):
    root = series(tmp_path, {"W32": W, "W24": W * 1.016, "S32": 0.85, "S24": 0.85 * 1.008})
    assert analyze_recipe.analyze([root])["label"] == "GAP_REDUCED"


def test_closed_gap_needs_ce_suites_and_over_refusal():
    strong = {"ce_gap_percent": 0.4,
              "suite_deltas": {n: -0.04 for n in analyze_recipe.GENERAL},
              "over_refusal_top": 0.95}
    assert analyze_recipe.label(strong, 1.5) == "GAP_CLOSED"
    assert analyze_recipe.label({**strong, "over_refusal_top": 0.9}, 1.5) == "GAP_REDUCED"
    worse = {**strong, "suite_deltas": {**strong["suite_deltas"], "mini_mmlu": -0.2}}
    assert analyze_recipe.label(worse, 1.5) == "GAP_REDUCED"
    assert analyze_recipe.label({**strong, "ce_gap_percent": 1.2}, 1.5) == "GAP_PERSISTS"


def test_anchor_and_weak_gap_failures_give_no_verdict(tmp_path):
    drift = series(tmp_path / "a", {"W32": W * 1.01, "W24": W * 1.03, "S32": 0.85,
                                    "S24": 0.86})
    result = analyze_recipe.analyze([drift])
    assert result["label"] == "NO_VERDICT" and "V2" in result["reasons"][0]
    nogap = series(tmp_path / "b", {"W32": W, "W24": W * 1.005, "S32": 0.85, "S24": 0.86})
    result = analyze_recipe.analyze([nogap])
    assert result["label"] == "NO_VERDICT" and "V3" in result["reasons"][0]


def test_missing_arm_tampered_score_or_config_give_no_verdict(tmp_path):
    missing = series(tmp_path / "a", {"W32": W, "W24": W * 1.02, "S32": 0.85})
    assert analyze_recipe.analyze([missing])["reasons"] == ["S24: missing"]
    tampered = series(tmp_path / "b", {"W32": W, "W24": W * 1.02, "S32": 0.85, "S24": 0.86},
                      S24={"tamper_score": True})
    reasons = analyze_recipe.analyze([tampered])["reasons"]
    assert reasons and all("re-scored" in r for r in reasons)
    config = series(tmp_path / "c", {"W32": W, "W24": W * 1.02, "S32": 0.85, "S24": 0.86},
                    W24={"config_extra": "  extra: 1\n"})
    assert any("template" in r for r in analyze_recipe.analyze([config])["reasons"])


def test_two_sessions_combine_and_a_repeated_arm_is_refused(tmp_path):
    one = series(tmp_path / "1", {"W32": W, "W24": W * 1.015})
    two = series(tmp_path / "2", {"S32": 0.85, "S24": 0.85 * 1.013, "S16": 0.85 * 1.05})
    result = analyze_recipe.analyze([one, two])
    assert result["label"] == "GAP_PERSISTS"
    assert result["secondary_s16"]["label"] == "GAP_REDUCED"
    with pytest.raises(ValueError, match="twice"):
        analyze_recipe.analyze([one, one])


def test_replay_detects_a_reordered_audit(tmp_path):
    root = series(tmp_path, {"W32": W})
    audit = root / "W32" / "gate" / "mini_mmlu.generation_audit.json"
    rows = json.loads(audit.read_text())
    audit.write_text(json.dumps(rows[::-1]))
    with pytest.raises(ValueError, match="order"):
        analyze_recipe.rescore(root / "W32" / "gate", "mini_mmlu")


def test_output_is_never_overwritten(tmp_path):
    root = series(tmp_path, {"W32": W})
    out = tmp_path / "analysis.json"
    analyze_recipe.main([str(root), "--out", str(out)])
    with pytest.raises(FileExistsError):
        analyze_recipe.main([str(root), "--out", str(out)])


def test_committed_notebook_is_generated_and_requires_t4_pair_and_kit_check(tmp_path):
    committed = (RC / "e2_recipe_check.ipynb").read_text()
    subprocess.run([sys.executable, str(RC / "make_notebook.py")], check=True,
                   capture_output=True)
    assert (RC / "e2_recipe_check.ipynb").read_text() == committed
    code = "".join("".join(c["source"]) for c in json.loads(committed)["cells"])
    assert '"T4" in g' in code and '"verify"' in code and "caa1feb0e54d" in code
    assert "HF_TOKEN" in code and "print(UserSecretsClient" not in code


def test_kit_verify_tolerates_dropped_dot_files_only(tmp_path):
    root = tmp_path / "k"
    (root / ".github").mkdir(parents=True)
    files = {"a.txt": b"one\n", ".gitignore": b"x\n", ".github/ci.yml": b"y\n"}
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {"commit": "c", "files": {n: hashlib.sha256(d).hexdigest()
                                         for n, d in files.items()}}
    (root / "KIT.json").write_text(json.dumps(manifest))
    (root / ".gitignore").unlink()
    (root / ".github" / "ci.yml").unlink()
    assert sorted(kit.verify(root)["missing_dot_files"]) == [".github/ci.yml", ".gitignore"]
    (root / "a.txt").unlink()
    with pytest.raises(RuntimeError, match="differ"):
        kit.verify(root)


def test_retained_kaggle_series_replays_to_the_published_result():
    report = RC / "reports" / "20261104"
    evidence = report / "evidence"
    result = analyze_recipe.analyze(
        [evidence / "20261104_182324", evidence / "20261104_231206"]
    )
    published = json.loads((report / "analysis.json").read_text())
    assert result["label"] == published["label"] == "GAP_PERSISTS"
    assert result["weak"] == published["weak"] and result["strong"] == published["strong"]
    manifest = json.loads((report / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
