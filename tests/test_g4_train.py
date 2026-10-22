"""Tests for the G4 Mistral-7B harness (benchmarks/harness/g4_train.py, ticket 04).

Spec: .scratch/g4-nvfp4/issues/04-mistral-harness-and-gate-rule.md. Rule:
benchmarks/gate-g4-nvfp4.md. No test touches a GPU, TransformerEngine or bitsandbytes, and
nothing is downloaded: the models are random-init configs, some of them on the meta device.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
RULE = Path(__file__).resolve().parents[1] / "benchmarks" / "gate-g4-nvfp4.md"

#: ``mistralai/Mistral-7B-v0.1`` config.json at the revision the rule pins.
MISTRAL_7B = {
    "vocab_size": 32000, "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
}


@pytest.fixture(scope="module")
def g4():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_g4_train_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "g4_train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _tiny_model(g4, **overrides):
    dims = dict(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    config = transformers.MistralConfig(**{**dims, **overrides})
    torch.manual_seed(0)
    return transformers.MistralForCausalLM(config).to(torch.float32).eval()


def _args(arm="nvfp4", **overrides):
    values = dict(
        arm=arm, seed=42, model=None, fixture="mistral", revision=None, micro_batch=4,
        gradient_checkpointing=False, gate_max_new_tokens=64, train_limit=None, panel_limit=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestMistral7BShapes:
    """Ticket criterion: every Mistral-7B layer size is accepted by the FP4 shape check."""

    def _shapes(self, g4):
        config = transformers.MistralConfig(**MISTRAL_7B)
        with torch.device("meta"):
            model = transformers.MistralForCausalLM(config)
        return g4.quantized_linear_shapes(model)

    def test_there_are_224_quantized_layers(self, g4):
        assert len(self._shapes(g4)) == 32 * 7

    def test_every_dimension_is_a_multiple_of_the_fp4_requirement(self, g4):
        for name, (in_features, out_features) in self._shapes(g4).items():
            assert in_features % g4.FP4_MULTIPLES["in_features"] == 0, name
            assert out_features % g4.FP4_MULTIPLES["out_features"] == 0, name

    @pytest.mark.parametrize("tokens", [16, 64, 4 * 1024, 16 * 1024])
    def test_the_refusal_does_not_fire_on_any_layer(self, g4, tokens):
        config = transformers.MistralConfig(**MISTRAL_7B)
        with torch.device("meta"):
            model = transformers.MistralForCausalLM(config)
        g4.check_model_shapes(model, tokens=tokens)

    def test_the_embedding_and_head_are_not_among_the_quantized_layers(self, g4):
        names = self._shapes(g4)
        assert not any("lm_head" in name or "embed" in name for name in names)

    def test_a_layer_size_off_the_multiple_is_refused_by_name(self, g4):
        config = transformers.MistralConfig(**{**MISTRAL_7B, "intermediate_size": 14340,
                                               "num_hidden_layers": 1})
        with torch.device("meta"):
            model = transformers.MistralForCausalLM(config)
        with pytest.raises(g4.UnsupportedShapeError, match="gate_proj|up_proj|down_proj"):
            g4.check_model_shapes(model, tokens=64)

    def test_a_token_count_off_the_multiple_is_refused(self, g4):
        model = _tiny_model(g4)
        with pytest.raises(g4.UnsupportedShapeError, match="tokens"):
            g4.check_model_shapes(model, tokens=60)


class TestArmConfigs:
    """Ticket criterion: the arm configs differ only in the base storage format."""

    def _flat(self, g4, arm):
        return g4.flatten(g4.run_config(_args(arm)))

    def test_nvfp4_and_nf4_differ_only_in_the_base_format(self, g4):
        a, b = self._flat(g4, "nvfp4"), self._flat(g4, "nf4")
        assert [k for k in a if a[k] != b.get(k)] == ["base.format"]
        assert (a["base.format"], b["base.format"]) == ("nvfp4", "nf4")

    def test_bf16_differs_from_nf4_only_in_the_base_format_and_its_details(self, g4):
        a, b = self._flat(g4, "bf16"), self._flat(g4, "nf4")
        assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= {
            k for k in set(a) | set(b) if k.startswith("base.")
        }

    def test_the_rule_values_are_in_the_config(self, g4):
        config = g4.run_config(_args("nf4"))
        assert config["lora"]["r"] == 16 and config["lora"]["alpha"] == 32
        assert config["lora"]["dropout"] == 0.0
        assert config["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert config["training"]["lr"] == 2e-4 and config["training"]["effective_batch"] == 16
        assert config["training"]["pad_to_multiple"] == 16

    def test_the_real_model_is_pinned_to_the_rule_revision(self, g4):
        assert g4.MODEL == "mistralai/Mistral-7B-v0.1"
        assert g4.REVISION == "27d67f1b5f57dc0953326b2601d68371d40ea8da"

    def test_the_arm_roles_are_nvfp4_nf4_and_bf16(self, g4):
        assert set(g4.ARM_FORMATS) == {"nvfp4", "nf4", "bf16"}


class TestPadding:
    def test_width_rounds_up_to_a_multiple_of_16(self, g4):
        assert [g4.padded_width(n) for n in (1, 16, 17, 31, 32, 1024)] == [
            16, 16, 32, 32, 32, 1024]

    def test_collate_pads_to_16_and_masks_the_padding(self, g4):
        examples = [
            {"input_ids": [5] * 5, "labels": [-100, 6, 6, 6, 6],
             "response_mask": [False] + [True] * 4},
            {"input_ids": [7] * 3, "labels": [-100, 8, 8], "response_mask": [False, True, True]},
        ]
        batch = g4.padded_collate(examples, "labels", pad_id=0, device="cpu")
        assert batch["input_ids"].shape == (2, 16)
        assert batch["attention_mask"].sum(dim=1).tolist() == [5, 3]
        assert (batch["targets"][:, 5:] == -100).all()
        assert batch["input_ids"][0, 5:].eq(0).all()

    def test_padding_does_not_change_the_loss(self, g4):
        model = _tiny_model(g4)
        example = {"input_ids": [3, 4, 5, 6, 7, 8], "labels": [-100, 4, 5, 6, 7, 8],
                   "response_mask": [False] + [True] * 5}
        short = g4._c5_collate([example], "labels", 0, "cpu")
        padded = g4.padded_collate([example], "labels", pad_id=0, device="cpu")
        nll_short, n_short = g4.c5.token_nll(model, short)
        nll_padded, n_padded = g4.c5.token_nll(model, padded)
        assert n_short.tolist() == n_padded.tolist()
        assert torch.allclose(nll_short, nll_padded, atol=1e-4)

    def test_train_and_panel_use_the_padded_collate(self, g4):
        seen = []
        original = g4.padded_collate

        def spy(*args, **kwargs):
            batch = original(*args, **kwargs)
            seen.append(batch["input_ids"].shape[1])
            return batch

        model = _tiny_model(g4)
        example = {"row": 1, "input_ids": [3, 4, 5, 6, 7], "labels": [-100, 4, 5, 6, 7],
                   "response_mask": [False] + [True] * 4}
        g4.padded_collate = spy
        try:
            with g4.padded_batches():
                g4.c5.evaluate_panel(model, [example], 1, "cpu", 0)
        finally:
            g4.padded_collate = original
        assert seen and all(width % 16 == 0 for width in seen)

    def test_the_c5_collate_is_restored_after_the_context(self, g4):
        before = g4.c5.collate
        with g4.padded_batches():
            assert g4.c5.collate is not before
        assert g4.c5.collate is before

    def test_the_autocast_wraps_the_forward_and_nothing_after_it(self, g4):
        import contextlib

        depth = []

        @contextlib.contextmanager
        def autocast():
            depth.append(1)
            try:
                yield
            finally:
                depth.pop()

        inside = []
        original = g4.c5.token_nll

        def spy(model, batch):
            inside.append(bool(depth))
            return original(model, batch)

        g4.c5.token_nll = spy
        try:
            with g4.padded_batches(autocast=autocast):
                example = {"input_ids": [3, 4, 5], "labels": [-100, 4, 5]}
                batch = g4.padded_collate([example], "labels", pad_id=0, device="cpu")
                g4.c5.token_nll(_tiny_model(g4), batch)
        finally:
            g4.c5.token_nll = original
        assert inside == [True] and not depth


class TestStorageCounts:
    def test_counts_quantized_layers_by_weight_type(self, g4):
        model = _tiny_model(g4)
        assert g4.storage_counts(model) == {"Parameter": 2 * 7}

    def test_replace_linears_swaps_every_quantized_layer_and_nothing_else(self, g4):
        model = _tiny_model(g4)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        made = []

        def factory(weight):
            made.append(tuple(weight.shape))
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight.data.copy_(weight)
            layer.weight.requires_grad_(False)
            return layer

        g4.replace_linears(model, factory)
        assert len(made) == 14
        assert torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"])
        assert torch.equal(model.lm_head.weight, before["lm_head.weight"])

    def test_a_refused_shape_leaves_the_model_untouched(self, g4):
        model = _tiny_model(g4, intermediate_size=130)
        factory_calls = []
        with pytest.raises(g4.UnsupportedShapeError):
            g4.replace_linears(model, factory_calls.append, tokens=64)
        assert factory_calls == []

    def test_fingerprints_cover_only_the_unquantized_weights(self, g4):
        names = set(g4.unquantized_fingerprints(_tiny_model(g4)))
        assert "model.embed_tokens.weight" in names and "lm_head.weight" in names
        assert "model.norm.weight" in names
        assert not any("q_proj" in n or "gate_proj" in n for n in names)


class TestLoRA:
    def test_adapters_are_on_attention_only_and_nothing_else_trains(self, g4):
        model = g4.add_lora(_tiny_model(g4), g4.run_config(_args("bf16")), seed=1)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)
        assert all(any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj"))
                   for n in trainable)
        assert len(trainable) == 2 * 4 * 2

    def test_adapter_starts_as_the_identity_on_the_base(self, g4):
        base = _tiny_model(g4)
        ids = torch.tensor([[3, 4, 5, 6]])
        expected = base(ids).logits
        wrapped = g4.add_lora(base, g4.run_config(_args("bf16")), seed=1)
        assert torch.equal(wrapped(ids).logits, expected)

    def test_the_same_seed_gives_the_same_adapter_init(self, g4):
        config = g4.run_config(_args("bf16"))
        first = g4.add_lora(_tiny_model(g4), config, seed=7)
        second = g4.add_lora(_tiny_model(g4), config, seed=7)
        other = g4.add_lora(_tiny_model(g4), config, seed=8)
        a = {n: p for n, p in first.named_parameters() if n.endswith("lora_a")}
        b = {n: p for n, p in second.named_parameters() if n.endswith("lora_a")}
        c = {n: p for n, p in other.named_parameters() if n.endswith("lora_a")}
        assert all(torch.equal(a[n], b[n]) for n in a)
        assert any(not torch.equal(a[n], c[n]) for n in a)


class TestGenerator:
    def test_greedy_without_cache_matches_generate_with_cache(self, g4):
        model = _tiny_model(g4)
        tokenizer = g4.c5.byte_tokenizer()
        model.config.pad_token_id = tokenizer.pad_token_id
        generate = g4.make_generator(model, tokenizer, "cpu", max_new_tokens=6, stop_id=None)
        prompt = "abc"
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            reference = model.generate(ids, max_new_tokens=6, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=None)
        expected = tokenizer.decode(reference[0, ids.shape[1]:], skip_special_tokens=True)
        assert generate(prompt) == expected

    def test_every_forward_has_a_token_count_that_is_a_multiple_of_16(self, g4):
        model = _tiny_model(g4)
        tokenizer = g4.c5.byte_tokenizer()
        widths = []
        model.register_forward_pre_hook(
            lambda _m, args, kwargs: widths.append(kwargs["input_ids"].numel()), with_kwargs=True)
        g4.make_generator(model, tokenizer, "cpu", max_new_tokens=5, stop_id=None)("hello world")
        assert widths and all(w % 16 == 0 for w in widths)

    def test_stops_at_the_stop_token(self, g4):
        model = _tiny_model(g4)
        ids = torch.tensor([[3, 4, 5]])
        first = g4.greedy_ids(model, ids, max_new_tokens=1, stop_id=None, pad_id=0)[0]
        assert g4.greedy_ids(model, ids, max_new_tokens=8, stop_id=first, pad_id=0) == []

    def test_greedy_ids_runs_the_autocast_around_every_forward(self, g4):
        import contextlib

        entered = []

        @contextlib.contextmanager
        def autocast():
            entered.append(1)
            yield

        g4.greedy_ids(_tiny_model(g4), torch.tensor([[3, 4, 5]]), max_new_tokens=3,
                      stop_id=None, pad_id=0, autocast=autocast)
        assert len(entered) == 3


class TestRuleState:
    def _git(self, repo, *args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    def _repo(self, tmp_path):
        self._git(tmp_path, "init", "-q")
        self._git(tmp_path, "config", "user.email", "t@example.com")
        self._git(tmp_path, "config", "user.name", "t")
        return tmp_path

    def test_the_committed_rule_has_a_commit_and_a_time(self, g4, tmp_path):
        repo = self._repo(tmp_path)
        rule = repo / "gate.md"
        rule.write_text("rule\n")
        self._git(repo, "add", "gate.md")
        self._git(repo, "commit", "-q", "-m", "rule")
        state = g4.rule_state(repo, rule)
        assert len(state["added_commit"]) == 40 and state["added_at"] and state["clean"]

    def test_an_edited_rule_is_not_clean(self, g4, tmp_path):
        repo = self._repo(tmp_path)
        rule = repo / "gate.md"
        rule.write_text("rule\n")
        self._git(repo, "add", "gate.md")
        self._git(repo, "commit", "-q", "-m", "rule")
        rule.write_text("changed\n")
        assert g4.rule_state(repo, rule)["clean"] is False

    def test_an_uncommitted_rule_has_no_commit(self, g4, tmp_path):
        repo = self._repo(tmp_path)
        rule = repo / "gate.md"
        rule.write_text("rule\n")
        assert g4.rule_state(repo, rule)["added_commit"] is None

    def test_a_real_run_is_refused_without_a_committed_clean_rule(self, g4):
        with pytest.raises(RuntimeError, match="rule"):
            g4.require_committed_rule({"added_commit": None, "added_at": None, "clean": False})
        with pytest.raises(RuntimeError, match="rule"):
            g4.require_committed_rule({"added_commit": "a" * 40, "added_at": "x", "clean": False})
        g4.require_committed_rule({"added_commit": "a" * 40, "added_at": "x", "clean": True})

    def test_the_repository_rule_is_committed_before_this_test_runs(self, g4):
        root = Path(__file__).resolve().parents[1]
        state = g4.rule_state(root, RULE)
        assert state["added_commit"], "benchmarks/gate-g4-nvfp4.md must be committed"
