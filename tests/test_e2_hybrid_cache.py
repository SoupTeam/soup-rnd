"""E2 hybrid with the activation cache: parity on a tiny Mistral and the ABBA rule (CPU)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
HC = ROOT / "benchmarks" / "e2" / "hybrid_cache"
HY = ROOT / "benchmarks" / "e2" / "hybrid"
RC = ROOT / "benchmarks" / "e2" / "recipe_check"
sys.path.insert(0, str(HY))
import run_hybrid  # noqa: E402


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_arm = _module(HC / "run_arm.py", "hybrid_cache_run_arm")
speed = _module(HC / "analyze_speed.py", "hybrid_cache_analyze_speed")


def test_order_is_c_then_three_abba_cycles():
    assert cache_arm.ORDER == (
        "C", "A_c1_1", "B_c1_1", "B_c1_2", "A_c1_2", "A_c2_1", "B_c2_1", "B_c2_2",
        "A_c2_2", "A_c3_1", "B_c3_1", "B_c3_2", "A_c3_2",
    )
    assert cache_arm.ORDER == speed.ORDER


def test_commands(tmp_path):
    a = cache_arm.command("A_c1_1", tmp_path / "c", tmp_path / "q", tmp_path / "x")
    b = cache_arm.command("B_c2_2", tmp_path / "c", tmp_path / "q", tmp_path / "x")
    c = cache_arm.command("C", tmp_path / "c", tmp_path / "q", tmp_path / "x")
    assert "--keep-top" not in a and "--cache-dir" not in a
    assert b[b.index("--keep-top") + 1] == "8" and b[b.index("--cache-dir") + 1].endswith("x")
    assert "--keep-top" in c and "--cache-dir" not in c
    assert all(x[x.index("--steps") + 1] == "6000" for x in (a, b, c))


def _tiny(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    path = tmp_path / "tiny"
    MistralForCausalLM(MistralConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )).save_pretrained(path)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"token_{i}": i for i in range(3, 100)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(path)
    return path


def _train(tmp_path, model_dir, name, cache_dir):
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper

    rows = (ROOT / "experiments/e2_quality_data.jsonl").read_text().splitlines()[:24]
    (tmp_path / "data.jsonl").write_text("\n".join(rows) + "\n")
    cfg = yaml.safe_load((RC / "configs/W32.yaml").read_text())
    cfg["base"], cfg["output"] = str(model_dir), str(tmp_path / f"out_{name}")
    cfg["data"]["train"] = str(tmp_path / "data.jsonl")
    cfg["training"]["lora"]["top_k_layers"] = 4
    cfg["training"]["lr"] = 0.01
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    config = load_config(str(tmp_path / f"{name}.yaml"))
    wrapper = SFTTrainerWrapper(config, device="cpu")
    wrapper.setup(load_dataset(config.data))
    trainer = wrapper.trainer
    per_epoch = len(trainer.train_dataset)
    trainer.args.max_steps, trainer.args.save_strategy = 3 * per_epoch, "no"
    freezer = run_hybrid.FreezeBottomAtStep(
        trainer.model, 2, per_epoch, cache_dir=cache_dir, model_revision="tiny",
        on_install=lambda runner: setattr(wrapper, "_e2_runner", runner),
    )
    trainer.add_callback(freezer)
    try:
        wrapper.train()
    finally:
        freezer.close_cache()
    entries = run_hybrid.lora_parameters(trainer.model)
    return run_hybrid.lora_digest(entries), freezer.record, getattr(wrapper, "_e2_runner", None)


def test_cache_installed_at_the_freeze_is_bit_identical_to_no_cache(tmp_path, monkeypatch):
    model_dir = _tiny(tmp_path)
    monkeypatch.chdir(tmp_path)
    with_cache, record, runner_after = _train(tmp_path, model_dir, "b", tmp_path / "cache")
    without, plain, _ = _train(tmp_path, model_dir, "c", None)
    assert with_cache == without
    cache = record["cache"]
    assert cache["installed_at_step"] > 0 and cache["cutoff"] == 2
    assert cache["hits"] > 0 and cache["misses"] > 0 and cache["files"] == cache["misses"]
    assert "cache" not in plain and runner_after is None


def _run(root, name, *, epochs=(550.0, 548.0, 547.0), ce=0.87, fp="a", batches="x",
         cache=None, hybrid=False, wall=1800.0):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(_render(d))
    (d / "arm.json").write_text(json.dumps({"ok": True, "process_wall_seconds": wall}))
    timing = [
        {"epoch": i + 1, "start_step": 2000 * i, "end_step": 2000 * (i + 1),
         "duration_seconds": s,
         "cache_hits": None if cache is None or i == 0 else (0 if i == 1 else 2000),
         "cache_misses": None if cache is None or i == 0 else (2000 if i == 1 else 0)}
        for i, s in enumerate(epochs)
    ]
    quality = {
        "steps": 6000, "validation_loss": ce, "train_examples": 2000,
        "validation_examples": 200, "deterministic_algorithms": True,
        "trainable_fingerprint": fp, "training_batches_sha256": batches,
        "epoch_timing": timing,
    }
    if hybrid:
        quality["hybrid"] = {
            "frozen": True, "freeze_at_step": 2000, "frozen_layers": list(range(24)),
            "frozen_tensors": [f"t{i}" for i in range(96)],
            "bottom_changed_before_freeze": True, "bottom_unchanged_after_freeze": True,
            "top_changed_after_freeze": True, "requires_grad_after": ["False"],
        }
        if cache is not None:
            quality["hybrid"]["cache"] = cache
    (d / "quality.json").write_text(json.dumps(quality))


def _render(d):
    sys.path.insert(0, str(RC))
    from run_arm import render

    return render("W32", Path("/b"), d / "adapter")


CACHE = {"installed_at_step": 2000, "cutoff": 24, "install_seconds": 12.0,
         "hits": 2000, "misses": 2000, "files": 2000, "bytes": 1 << 30}


def session(tmp_path, b_epoch3=200.0, b_fp="h"):
    root = tmp_path / "s"
    _run(root, "C", epochs=(550.0, 336.0, 336.0), fp="h", hybrid=True)
    for c in (1, 2, 3):
        for i in (1, 2):
            _run(root, f"A_c{c}_{i}")
            _run(root, f"B_c{c}_{i}", epochs=(550.0, 360.0, b_epoch3), fp=b_fp,
                 cache=dict(CACHE), hybrid=True, wall=1300.0)
    return root


def test_speedup_labels(tmp_path):
    result = speed.analyze(session(tmp_path))
    assert result["label"] == "WARM_SPEEDUP", result.get("reasons")
    assert result["warm_median"] == pytest.approx(547 / 200)
    assert result["secondary_label"] == "TRAINING_SPEEDUP"
    # 547 / 520 = 1.05: faster in every cycle, but below the 1.10 median.
    assert speed.analyze(session(tmp_path / "slow", b_epoch3=520.0))["label"] == "NOT_SHOWN"


def test_parity_break_or_wrong_cache_use_gives_no_verdict(tmp_path):
    broken = speed.analyze(session(tmp_path, b_fp="different"))
    assert broken["label"] == "NO_VERDICT"
    assert any("cache parity" in r for r in broken["reasons"])
    root = session(tmp_path / "m")
    q = json.loads((root / "B_c2_1" / "quality.json").read_text())
    q["epoch_timing"][2]["cache_misses"] = 5
    (root / "B_c2_1" / "quality.json").write_text(json.dumps(q))
    reasons = speed.analyze(root)["reasons"]
    assert any("B_c2_1" in r and "epoch 3" in r for r in reasons)


def test_cache_notebook_runs_the_whole_order():
    nb = json.loads((HC / "e2_hybrid_cache.ipynb").read_text())
    code = "".join("".join(c["source"]) for c in nb["cells"])
    assert "benchmarks/e2/hybrid_cache/run_arm.py" in code and "e2hc_results_" in code
    for name in cache_arm.ORDER:
        assert f'"{name}"' in code


def test_retained_kaggle_series_replays_to_the_published_result():
    import hashlib

    report = HC / "reports" / "20261109"
    evidence = report / "evidence"
    result = speed.analyze(evidence / "20261109_070805")
    published = json.loads((report / "analysis.json").read_text())
    assert result["label"] == published["label"] == "WARM_SPEEDUP"
    assert result["secondary_label"] == "TRAINING_SPEEDUP"
    assert result["cycles"] == published["cycles"]
    assert result["warm_median"] > 3.0
    manifest = json.loads((report / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
