"""ZERO-2: per-layer features for an adaptive choice of the cut (utils/layer_features.py)."""
from __future__ import annotations

import json

import pytest

from tests.test_layer_select import _model

torch = pytest.importorskip("torch")


def test_update_features_match_the_dense_product():
    from soup_cli.utils.layer_features import adapter_update_features
    from soup_cli.utils.layer_select import _LAYER_RE

    model = _model("llama")
    torch.manual_seed(1)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.1)
    feats = adapter_update_features(model, 6, lambda n: int(m.group(1)) if (m := _LAYER_RE.search(
        n + ".")) else None)
    want = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            idx = int(_LAYER_RE.search(name + ".").group(1))
            delta = mod.lora_B["default"].weight @ mod.lora_A["default"].weight
            want[idx] = want.get(idx, 0.0) + float(delta.norm()) * mod.scaling["default"]
    assert set(feats) == set(want)
    for idx in want:
        assert feats[idx]["update"] == pytest.approx(want[idx], rel=1e-4)
        assert 1.0 <= feats[idx]["update_rank"] <= 4.0 + 1e-4  # r = 4


def test_cka_of_a_representation_with_itself_is_one():
    from soup_cli.utils.layer_features import linear_cka

    x = torch.randn(40, 8)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)
    assert linear_cka(x, 3.0 * x + 1.0) == pytest.approx(1.0, abs=1e-5)
    assert linear_cka(x, torch.randn(40, 8)) < 0.9


def _rows_dataset(n=6, seq=10):
    gen = torch.Generator().manual_seed(2)
    rows = [torch.randint(0, 64, (seq,), generator=gen) for _ in range(n)]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"input_ids": rows[i].tolist()}

    def collate(items):
        ids = torch.tensor([it["input_ids"] for it in items])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    return DS(), collate


@pytest.mark.parametrize("kind", ["llama", "qwen3_5"])
def test_features_are_recorded_and_training_is_unchanged(tmp_path, kind):
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils import layer_select

    ds, collate = _rows_dataset()

    def run(features, out):
        model = _model(kind)
        args = transformers.TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=1, num_train_epochs=2,
            learning_rate=1e-2, logging_steps=1, save_strategy="no", report_to=[],
            use_cpu=True, seed=0, data_seed=0)
        trainer = transformers.Trainer(model=model, args=args, train_dataset=ds,
                                       data_collator=collate)
        policy = layer_select.LayerSelectPolicy(
            "progressive", freeze_step=8, freeze_below=2, interval_steps=2,
            feature_step=6 if features else None, feature_rows=3)
        cb = layer_select.LayerSelectCallback(policy, dataset=ds, collator=collate,
                                              batch_size=1, output_dir=str(out))
        trainer.add_callback(cb)
        trainer.train()
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        state = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
        return losses, state

    plain, plain_state = run(False, tmp_path / "a")
    with_f, state = run(True, tmp_path / "b")
    assert plain == with_f
    assert all(torch.equal(state[n], plain_state[n]) for n in state)
    rec = json.loads((tmp_path / "b" / "lora_layers.json").read_text())
    feats = rec["features"]
    # the loss of step 6 is logged after its on_step_end, so 5 losses are summarised
    assert feats["step"] == 6 and feats["loss"]["steps"] == 5
    n_layers = feats["n_layers"]
    assert set(feats["layers"]) == {str(i) for i in range(n_layers)}
    one = feats["layers"]["0"]
    for key in ("update", "update_rank", "grad_mean", "grad_cv", "cka_change", "cka_own", "turn"):
        assert key in one, key
    assert all(0.0 <= v["cka_change"] <= 1.0 for v in feats["layers"].values())
    assert all(0.0 <= v["turn"] <= 1.0 for v in feats["layers"].values())


def test_schema_feature_fields():
    from tests.test_layer_select import _cfg

    cfg = _cfg(lora_layers={"method": "progressive", "freeze_step": 80, "freeze_below": 8,
                            "feature_step": 80, "feature_rows": 16})
    assert cfg.training.lora_layers.feature_step == 80
