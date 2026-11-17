"""CPU-only prototype; no CUDA or academic checkpoint qualification."""

import sys

import pytest
import torch

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.config.schema import ExpertStreamConfig
from soup_cli.utils.expert_training import validate_expert_training
from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.layer_stream_runtime import build_streamed_model


def _academic_deepseek_config(**overrides):
    from types import SimpleNamespace

    values = {"model_type": "deepseek_v3", "hidden_size": 2048,
              "moe_intermediate_size": 1408, "n_routed_experts": 64,
              "output_router_logits": False, "hidden_act": "silu"}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("backend", ["sequential", "grouped"])
def test_academic_deepseek_cuda_recipe_is_admitted(backend, monkeypatch):
    monkeypatch.setattr("soup_cli.utils.expert_training.sys.platform", "linux")
    options = ExpertStreamConfig(backend=backend)
    validate_expert_training(_academic_deepseek_config(), options,
                             device="cuda", dtype="bfloat16")


def test_deepseek_grouped_cuda_uses_restore_path_without_grouped_gemm():
    from soup_cli.utils.expert_training import _uses_grouped_gemm

    assert not _uses_grouped_gemm("deepseek_v3", "grouped", "cuda")
    assert not _uses_grouped_gemm("deepseek_v3", "sequential", "cuda")
    assert not _uses_grouped_gemm("deepseek_v3", "grouped", "cpu")
    assert _uses_grouped_gemm("olmoe", "grouped", "cuda")
    assert not _uses_grouped_gemm("olmoe", "sequential", "cuda")


@pytest.mark.parametrize(("config", "backend", "dtype", "platform", "message"), [
    (_academic_deepseek_config(), "grouped", "bfloat16", "win32", "requires Linux"),
    (_academic_deepseek_config(), "grouped", "float16", "linux", "requires native BF16"),
    (_academic_deepseek_config(hidden_size=4096), "grouped", "bfloat16", "linux",
     "H=2048, I=1408, E=64 only"),
    (_academic_deepseek_config(), "triton", "bfloat16", "linux",
     "supports sequential/grouped only"),
    (_academic_deepseek_config(output_router_logits=True), "grouped", "bfloat16", "linux",
     "router auxiliary loss"),
])
def test_unqualified_deepseek_cuda_recipe_is_refused(config, backend, dtype, platform, message,
                                                       tmp_path, monkeypatch):
    from soup_cli.utils import expert_training

    monkeypatch.setattr(expert_training.sys, "platform", platform)
    library = tmp_path / "libcublas.so.12"
    library.touch()
    options = ExpertStreamConfig(backend=backend, cublas_library=str(library))
    with pytest.raises(ValueError, match=message):
        validate_expert_training(config, options, device="cuda", dtype=dtype)


def test_missing_expert_layer_is_refused_before_reader_creation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from transformers import AutoModelForCausalLM

    from soup_cli.utils import expert_shard, expert_training
    from soup_cli.utils.expert_training import install_expert_training

    def forbidden(*args, **kwargs):
        raise AssertionError("reader must not be created for a mismatched cache")

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = AutoModelForCausalLM.from_config(fixture.moe_config())
        # Exercise the cache/model mismatch branch; this test does not qualify
        # CPU arithmetic and should not stop at the macOS platform guard.
        monkeypatch.setattr(expert_training, "_cpu_training_platform",
                            lambda: ("linux", "x86_64"))
        monkeypatch.setattr(expert_shard, "ExpertShardReader", forbidden)
        index = SimpleNamespace(expert_layout="per_expert", expert_layers=(1,))
        with pytest.raises(ValueError, match="same independently readable experts"):
            install_expert_training(model, str(tmp_path), index, ExpertStreamConfig(),
                                    device="cpu", dtype="float32", codes={})
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("storage", ["dense", "nf4", "nf4-nested"])
@pytest.mark.skipif(sys.platform == "darwin",
                    reason="CPU stream_experts exactness is not qualified on macOS")
def test_tiny_deepseek_grouped_cpu_forward_and_backward(tmp_path, storage):
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from tests.test_b13_moe_runtime import _apply_stored_nf4

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    runtime = None
    try:
        config = fixture.moe_config()
        weights = fixture.moe_weights_dir(tmp_path, config)
        shards = tmp_path / "shards"
        quant = "none" if storage == "dense" else "nf4"
        nested = storage == "nf4-nested"
        extra = {}
        if quant == "nf4":
            shell = build_meta_skeleton(str(weights), dtype="float32", quant="nf4",
                                        double_quant=nested)
            extra = {"quant_suffixes": quantised_layer_suffixes(shell), "quant_device": "cpu",
                     "double_quant": nested, "quant": quant}
        index = shard_checkpoint(str(weights), str(shards), dtype="float32", arch="deepseek_v3",
                                 expert_layout="per_expert", **extra)
        assert index.expert_layers == (1, 2)
        lora = fixture.moe_lora_config()
        streamed, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
            device="cpu", dtype="float32", buffers=2, pin=False, seed=71, quant=quant,
            double_quant=nested,
            expert_options=ExpertStreamConfig(backend="grouped", weight_budget_bytes=2**20),
        )
        fixture.randomize_adapters(streamed)
        base = fixture.resident_base(weights)
        if quant == "nf4":
            _apply_stored_nf4(base, shards, index)
        resident = fixture.resident_adapter(base, lora, source=streamed)
        batch = fixture.moe_batch(config, rows=1, seq=6)
        models = (resident, streamed)
        optimizers = [torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
                      for model in models]
        for _ in range(2):
            outputs = []
            for model, optimizer in zip(models, optimizers):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch)
                output.loss.backward()
                outputs.append(output)
            assert torch.equal(outputs[0].logits, outputs[1].logits)
            assert torch.equal(outputs[0].loss, outputs[1].loss)
            expected_grads = fixture.snapshot(resident, gradients=True)
            actual_grads = fixture.snapshot(streamed, gradients=True)
            assert expected_grads and expected_grads.keys() == actual_grads.keys()
            for name in expected_grads:
                assert expected_grads[name] is not None
                assert torch.equal(expected_grads[name], actual_grads[name]), name
            for optimizer in optimizers:
                optimizer.step()
            expected_params = fixture.snapshot(resident)
            actual_params = fixture.snapshot(streamed)
            for name in expected_params:
                assert torch.equal(expected_params[name], actual_params[name]), name
        assert runtime.stats()["expert_fetches"] > 0
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous)


@pytest.mark.skipif(sys.platform == "darwin",
                    reason="CPU stream_experts exactness is not qualified on macOS")
def test_tiny_deepseek_grouped_cpu_real_sft_trainer(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, default_data_collator
    from trl import SFTConfig, SFTTrainer

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    runtime = None
    try:
        config = fixture.moe_config()
        weights = fixture.moe_weights_dir(tmp_path, config)
        shards = tmp_path / "shards"
        index = shard_checkpoint(str(weights), str(shards), dtype="float32", arch="deepseek_v3",
                                 expert_layout="per_expert")
        model, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=str(shards), index=index,
            lora_config=fixture.moe_lora_config(), device="cpu", dtype="float32",
            buffers=2, pin=False, seed=71, quant="none",
            expert_options=ExpertStreamConfig(backend="grouped", weight_budget_bytes=2**20),
        )
        fixture.randomize_adapters(model)
        before = fixture.snapshot(model)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0")),
            pad_token="0", eos_token="1")
        rows = [[2, 3, 4, 5, 6, 1], [4, 5, 6, 7, 8, 1]]
        dataset = Dataset.from_dict({"input_ids": rows, "labels": rows,
                                     "attention_mask": [[1] * len(row) for row in rows]})
        args = SFTConfig(output_dir=str(tmp_path / "out"), use_cpu=True, bf16=False, fp16=False,
                         max_steps=2, per_device_train_batch_size=1, learning_rate=.01,
                         optim="sgd", report_to="none", save_strategy="no", logging_strategy="no",
                         disable_tqdm=True, gradient_checkpointing=False,
                         dataset_kwargs={"skip_prepare_dataset": True}, dataloader_pin_memory=False,
                         remove_unused_columns=False)
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                             processing_class=tokenizer, data_collator=default_data_collator)
        assert trainer.train().global_step == 2
        assert runtime.stats()["expert_fetches"] > 0
        after = fixture.snapshot(model)
        assert any(not torch.equal(before[name], after[name]) for name in before)
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous)
