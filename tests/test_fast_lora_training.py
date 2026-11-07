"""The public Soup SFT path must execute Fast-LoRA and keep portable adapters."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def config_payload() -> dict:
    return {
        "base": "local-llama",
        "task": "sft",
        "backend": "transformers",
        "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": {
            "fast_lora": True,
            "quantization": "none",
            "seed": 792,
            "lora": {"r": 4, "alpha": 4, "dropout": 0},
        },
    }


def test_fast_lora_is_opt_in_and_round_trips():
    default = config_payload()
    default["training"].pop("fast_lora")
    assert not SoupConfig.model_validate(default).training.fast_lora
    config = SoupConfig.model_validate(config_payload())
    assert SoupConfig.model_validate_json(config.model_dump_json()).training.fast_lora


@pytest.mark.parametrize(
    "location,key,value",
    [
        ("root", "task", "dpo"),
        ("root", "backend", "unsloth"),
        ("root", "modality", "vision"),
        ("training", "quantization", "4bit"),
        ("training", "use_liger", True),
        ("training", "use_cut_ce", True),
        ("training", "activation_offloading", "cpu"),
        ("lora", "dropout", 0.05),
        ("lora", "use_dora", True),
        ("lora", "r", 0),
        ("lora", "use_rslora", True),
    ],
)
def test_unsupported_fast_lora_config_fails_before_model_load(location, key, value):
    payload = config_payload()
    target = payload if location == "root" else payload["training"]
    if location == "lora":
        target = target["lora"]
    target[key] = value
    with pytest.raises(ValidationError):
        SoupConfig.model_validate(payload)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_soup_sft_matches_lora_and_saves_reloadable_adapter(tmp_path, checkpointing):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, TrainerCallback

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_issue691_sft_packing import _tiny_causal_model_dir

    torch.set_num_threads(1)
    base = _tiny_causal_model_dir(tmp_path)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hi yo"},
            ]
        }
    ] * 4
    results = []
    for enabled in (False, True):
        payload = config_payload()
        payload["base"] = base
        payload["output"] = str(tmp_path / ("fast" if enabled else "ordinary"))
        payload["data"]["max_length"] = 64
        payload["training"].update(
            fast_lora=enabled,
            batch_size=1,
            epochs=1,
            logging_steps=1,
            save_steps=1000,
            gradient_accumulation_steps=2,
            gradient_checkpointing=checkpointing,
            warmup_ratio=0,
        )
        payload["training"]["lora"]["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        config = SoupConfig.model_validate(payload)
        wrapper = SFTTrainerWrapper(config, device="cpu")
        wrapper.setup({"train": rows})
        wrapper.trainer.args.report_to = []
        wrapper.trainer.args.disable_tqdm = True
        initial = {
            key: value.detach().clone()
            for key, value in wrapper.model.named_parameters()
            if "lora_" in key
        }
        gradients = []

        class Capture(TrainerCallback):
            def on_pre_optimizer_step(self, _args, _state, _control, model=None, **_kwargs):
                gradients.append(
                    {
                        key: value.grad.detach().clone()
                        for key, value in model.named_parameters()
                        if "lora_" in key
                    }
                )

        wrapper.trainer.add_callback(Capture())
        with torch.autograd.profiler.profile(use_device=None) as trace:
            wrapper.train()
        final = {
            key: value.detach().clone()
            for key, value in wrapper.model.named_parameters()
            if "lora_" in key
        }
        events = {event.name for event in trace.function_events if "FastLora" in event.name}
        if enabled:
            assert all(wrapper._fast_lora_patches.values())
            assert all(
                any(marker in name for name in events)
                for marker in ("SwiGLU", "QKV", "SingleProjection")
            )
        else:
            assert wrapper._fast_lora_patches == {} and not events
        assert len(gradients) == 2
        assert any(value.abs().sum() > 0 for key, value in gradients[1].items() if "lora_A" in key)
        assert any(not torch.equal(initial[key], value) for key, value in final.items())
        loaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base),
            payload["output"],
            is_trainable=True,
        )
        reloaded = {
            key: value.detach() for key, value in loaded.named_parameters() if "lora_" in key
        }
        assert set(final) == set(reloaded)
        assert all(torch.equal(value, reloaded[key]) for key, value in final.items())
        losses = [item["loss"] for item in wrapper.trainer.state.log_history if "loss" in item]
        results.append((initial, gradients, final, losses))
    assert all(torch.equal(value, results[1][0][key]) for key, value in results[0][0].items())
    for first, second in zip(results[0][1], results[1][1], strict=True):
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-7)
    for key, value in results[0][2].items():
        torch.testing.assert_close(value, results[1][2][key], rtol=1e-5, atol=1e-7)
    assert len(results[0][3]) == len(results[1][3]) == 2
    assert all(
        f"{first:.3f}" == f"{second:.3f}"
        for first, second in zip(results[0][3], results[1][3], strict=True)
    )


def test_installation_failure_restores_forwards_and_preserves_parameter_identity(monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils import fast_lora_qkv
    from soup_cli.utils.fast_lora_training import apply_fast_lora_training

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    parameters = {name: id(value) for name, value in model.named_parameters()}
    mlp = model.get_base_model().model.layers[0].mlp
    original = mlp.forward.__func__

    def broken(_model):
        raise RuntimeError("deliberate installation failure")

    monkeypatch.setattr(fast_lora_qkv, "patch_fast_lora_qkv", broken)
    with pytest.raises(RuntimeError, match="deliberate"):
        apply_fast_lora_training(model)
    assert mlp.forward.__func__ is original and "forward" not in vars(mlp)
    assert {name: id(value) for name, value in model.named_parameters()} == parameters
    assert not getattr(model, "_soup_fast_lora_training_counts", None)


def test_bad_architecture_is_refused_without_mutation():
    from types import SimpleNamespace

    from soup_cli.utils.fast_lora_training import apply_fast_lora_training

    with pytest.raises(ValueError, match="Llama"):
        apply_fast_lora_training(SimpleNamespace(config=SimpleNamespace(model_type="gpt2")))
