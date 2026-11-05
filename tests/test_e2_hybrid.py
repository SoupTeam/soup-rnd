"""E2 hybrid (all layers in epoch 1, top K from epoch 2) follows its protocol. CPU only."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
HY = ROOT / "benchmarks" / "e2" / "hybrid"
RC = ROOT / "benchmarks" / "e2" / "recipe_check"
sys.path.insert(0, str(HY))
sys.path.insert(0, str(RC))
import analyze_hybrid  # noqa: E402
import run_hybrid  # noqa: E402

from soup_cli.eval import gate_suites as suites  # noqa: E402

W = 0.876791179180


def _tiny_model(path):
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    config = MistralConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    MistralForCausalLM(config).save_pretrained(path)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"token_{i}": i for i in range(3, 100)}}
    backend = tokenizers.Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(path)


def test_freeze_at_step_stops_exactly_the_bottom_lora_on_a_tiny_mistral(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper

    model_dir = tmp_path / "tiny"
    _tiny_model(model_dir)
    rows = (ROOT / "experiments/e2_quality_data.jsonl").read_text().splitlines()[:24]
    (tmp_path / "data.jsonl").write_text("\n".join(rows) + "\n")
    cfg = yaml.safe_load((RC / "configs/W32.yaml").read_text())
    cfg["base"], cfg["output"] = str(model_dir), str(tmp_path / "out")
    cfg["data"]["train"] = str(tmp_path / "data.jsonl")
    cfg["training"]["lora"]["top_k_layers"] = 4
    cfg["training"]["lr"] = 0.01
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    config = load_config(str(tmp_path / "c.yaml"))
    wrapper = SFTTrainerWrapper(config, device="cpu")
    wrapper.setup(load_dataset(config.data))
    trainer = wrapper.trainer
    trainer.args.max_steps, trainer.args.save_strategy = 8, "no"

    _, _, bottom, top = run_hybrid.split_layers(trainer.model, keep_top=2)
    start_bottom, start_top = run_hybrid.lora_digest(bottom), run_hybrid.lora_digest(top)
    freezer = run_hybrid.FreezeBottomAtStep(trainer.model, keep_top=2, step=4)
    trainer.add_callback(freezer)
    wrapper.train()
    record = freezer.finish(start_bottom, start_top)

    assert record["frozen"] and record["frozen_layers"] == [0, 1]
    assert len(record["frozen_tensors"]) == 2 * 2 * 2  # 2 layers x q/v x lora_A/B
    assert record["bottom_changed_before_freeze"]
    assert record["bottom_unchanged_after_freeze"]
    assert record["top_changed_after_freeze"]
    assert record["requires_grad_after"] == ["False"]


def test_split_rejects_keeping_all_or_no_layers():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                torch.nn.ModuleDict({"lora_A": torch.nn.Linear(2, 2)}) for _ in range(4)
            )

    model = Tiny()
    for bad in (0, 4):
        with pytest.raises(ValueError):
            run_hybrid.split_layers(model, bad)
    layers, cutoff, bottom, top = run_hybrid.split_layers(model, 3)
    assert (layers, cutoff, len(bottom), len(top)) == (4, 1, 2, 6)


def test_runner_is_run_quality_plus_the_freeze():
    base = (ROOT / "benchmarks/e2/quality/run_quality.py").read_text().splitlines()
    mine = (HY / "run_hybrid.py").read_text()
    changed = {'"""E2 quality pilot using the real Soup SFT trainer."""'}
    trainable_block = {
        "    trainable = {",
        "        name: parameter.detach().cpu().contiguous()",
        "        for name, parameter in trainer.model.named_parameters()",
        "        if parameter.requires_grad",
        "    }",
    }
    for line in base:
        if line in changed or line in trainable_block:
            continue
        assert line in mine.splitlines(), line
    assert "--keep-top" in mine and "requires_grad_(False)" in mine


def _arm(root, name, ce, hybrid=None, text="Sure. The answer is 4."):
    from run_arm import render

    d = root / name
    (d / "gate").mkdir(parents=True)
    (d / "config.yaml").write_text(render("W32", Path("/tmp/base"), d / "adapter"))
    (d / "arm.json").write_text(json.dumps({"ok": True}))
    quality = {
        "steps": 4000, "validation_loss": ce, "train_examples": 2000,
        "validation_examples": 200, "parameters_changed": True,
        "deterministic_algorithms": True, "trainable_fingerprint": "f",
    }
    if hybrid is not None:
        quality["hybrid"] = hybrid
    (d / "quality.json").write_text(json.dumps(quality))
    scores = {}
    for suite in suites.DEFAULT_GENERAL_SUITE:
        rows = []

        def gen(prompt):
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


def _hybrid(keep_top, **overrides):
    cutoff = 32 - keep_top
    record = {
        "frozen": True, "freeze_at_step": 2000, "frozen_layers": list(range(cutoff)),
        "frozen_tensors": [f"t{i}" for i in range(cutoff * 4)],
        "bottom_changed_before_freeze": True, "bottom_unchanged_after_freeze": True,
        "top_changed_after_freeze": True, "requires_grad_after": ["False"],
    }
    return {**record, **overrides}


def session(tmp_path, h24, h16, w32=W, **hybrid24):
    root = tmp_path / "s"
    _arm(root, "W32", w32)
    _arm(root, "H24", h24, _hybrid(24, **hybrid24))
    _arm(root, "H16", h16, _hybrid(16))
    return root


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0.3, "WITHIN_MARGIN"), (0.8, "BORDERLINE"), (1.2, "FAILS")],
)
def test_labels_follow_the_gap(tmp_path, gap, expected):
    result = analyze_hybrid.analyze(session(tmp_path, W * (1 + gap / 100), W * 1.02))
    assert result["label"] == expected
    assert result["H24"]["top_k_gap_reference"] == pytest.approx(1.5449, abs=1e-4)
    assert result["H16"]["label"] == "FAILS"


def test_a_leaky_freeze_or_drifted_anchor_gives_no_verdict(tmp_path):
    leaky = session(tmp_path / "a", W * 1.001, W, bottom_unchanged_after_freeze=False)
    result = analyze_hybrid.analyze(leaky)
    assert result["label"] == "NO_VERDICT"
    assert any("bit-identical" in r for r in result["reasons"])
    drift = session(tmp_path / "b", W * 1.02, W * 1.02, w32=W * 1.01)
    assert any("V2" in r for r in analyze_hybrid.analyze(drift)["reasons"])


def test_secondary_problem_does_not_block_the_primary(tmp_path):
    root = session(tmp_path, W * 1.003, W)
    hybrid = json.loads((root / "H16" / "quality.json").read_text())
    hybrid["hybrid"]["frozen_layers"] = [0]
    (root / "H16" / "quality.json").write_text(json.dumps(hybrid))
    result = analyze_hybrid.analyze(root)
    assert result["label"] == "WITHIN_MARGIN" and result["H16"]["label"] == "NO_VERDICT"


def test_hybrid_notebook_runs_the_hybrid_arms():
    code = "".join(
        "".join(c["source"]) for c in json.loads((HY / "e2_hybrid.ipynb").read_text())["cells"]
    )
    assert 'ARMS = ["H24", "H16", "W32"]' in code
    assert "benchmarks/e2/hybrid/run_arm.py" in code and "e2hy_results_" in code
    assert '"uninstall", "-y", "-q", "torchao"' in code


def test_retained_kaggle_series_replays_to_the_published_result():
    report = HY / "reports" / "20261105"
    evidence = report / "evidence"
    result = analyze_hybrid.analyze(evidence / "20261105_082550")
    published = json.loads((report / "analysis.json").read_text())
    assert result["label"] == published["label"] == "WITHIN_MARGIN"
    assert result["H24"] == published["H24"] and result["H16"]["label"] == "FAILS"
    manifest = json.loads((report / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
