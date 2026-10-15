
"""Compare normal and cached training through real TRL SFTTrainer."""

import copy
from unittest.mock import patch

import torch
from datasets import Dataset
from peft import get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner
from soup_cli.utils.peft_wiring import build_lora_config, resolve_top_k_layers


def make_model():
    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )

    base = MistralForCausalLM(config)

    cfg = LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        top_k_layers=2,
        target_modules=["q_proj", "v_proj"],
    )

    selected = resolve_top_k_layers(base, 2)

    peft_cfg = build_lora_config(
        cfg,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=selected,
    )

    return get_peft_model(base, peft_cfg)


def make_tokenizer():
    backend = Tokenizer(
        WordLevel(
            vocab={
                "[PAD]": 0,
                "[UNK]": 1,
                "[EOS]": 2,
                **{f"token_{i}": i for i in range(3, 100)},
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )


def make_trainer(model, output_dir, tokenizer):
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "labels": [[1, 2, 3, 4, 5]],
        }
    )

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        packing=False,
        use_cache=False,
        use_cpu=True,
        save_strategy="no",
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=16,
        logging_steps=1,
        learning_rate=2e-5,
        seed=42,
        data_seed=42,
        optim="adamw_torch",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def trainable_weights(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_normal_vs_cached_sft_trainer(tmp_path):
    torch.manual_seed(42)

    normal_model = make_model()
    cached_model = copy.deepcopy(normal_model)

    tokenizer = make_tokenizer()

    initial_weights = trainable_weights(normal_model)
    assert initial_weights

    normal_trainer = make_trainer(
        normal_model,
        tmp_path / "normal",
        tokenizer,
    )

    normal_result = normal_trainer.train()

    decoder = cached_model.base_model.model.model

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny-mistral-test-v1",
            frozen_prefix_fingerprint="fixed-test-weights",
            config_fingerprint="fixed-test-config",
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=2,
        metadata_factory=metadata_factory,
    )

    cached_trainer = make_trainer(
        cached_model,
        tmp_path / "cached",
        tokenizer,
    )

    frozen_calls = [0]

    def count_frozen_calls(module, inputs):
        frozen_calls[0] += 1

    handles = [
        layer.register_forward_pre_hook(count_frozen_calls)
        for layer in decoder.layers[:2]
    ]

    try:
        with patch.object(decoder, "forward", side_effect=runner.forward):
            cached_result = cached_trainer.train()
    finally:
        for handle in handles:
            handle.remove()

    assert normal_trainer.state.global_step == 2
    assert cached_trainer.state.global_step == 2

    assert runner.misses == 1
    assert runner.hits == 1

    # Two frozen layers execute only once each.
    assert frozen_calls[0] == 2

    normal_weights = trainable_weights(normal_model)
    cached_weights = trainable_weights(cached_model)

    assert normal_weights.keys() == cached_weights.keys()

    changed = 0

    for name in normal_weights:
        torch.testing.assert_close(
            normal_weights[name],
            cached_weights[name],
            rtol=1e-4,
            atol=1e-6,
            msg=lambda msg: f"{name}: {msg}",
        )

        if not torch.equal(initial_weights[name], normal_weights[name]):
            changed += 1

    assert changed > 0, "No LoRA parameters were updated"

    torch.testing.assert_close(
        torch.tensor(normal_result.training_loss),
        torch.tensor(cached_result.training_loss),
        rtol=1e-5,
        atol=1e-6,
    )

    print(
        f"\nNormal loss: {normal_result.training_loss:.6f}"
        f"\nCached loss: {cached_result.training_loss:.6f}"
        f"\nCache hits: {runner.hits}"
        f"\nCache misses: {runner.misses}"
        f"\nFrozen-layer calls: {frozen_calls[0]}"
        f"\nUpdated LoRA tensors: {changed}"
    )

