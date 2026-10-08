
"""E2: real TRL SFTTrainer two-epoch smoke test."""

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


def test_real_sft_trainer_with_cache(tmp_path):
    torch.manual_seed(42)

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
    model = MistralForCausalLM(config)

    lora_cfg = LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        top_k_layers=2,
        target_modules=["q_proj", "v_proj"],
    )

    selected = resolve_top_k_layers(model, 2)
    peft_cfg = build_lora_config(
        lora_cfg,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=selected,
    )
    model = get_peft_model(model, peft_cfg)

    decoder = model.base_model.model.model

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

    # Already-tokenized fixed examples, no external tokenizer download.
    dataset = Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
        "labels": [[1, 2, 3, 4, 5]],
    })

    args = SFTConfig(
        output_dir=str(tmp_path / "output"),
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
    )
    tokenizer_backend = Tokenizer(
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
    tokenizer_backend.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    initial = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    with patch.object(decoder, "forward", side_effect=runner.forward):
        trainer.train()

    assert runner.misses >= 1
    assert runner.hits >= 1
    assert trainer.state.global_step == 2

    updated = [
        name
        for name, param in model.named_parameters()
        if name in initial
        and not torch.equal(initial[name], param.detach())
    ]
    assert updated, "No trainable LoRA parameters were updated"

    print(
        f"E2 Trainer: hits={runner.hits}, "
        f"misses={runner.misses}, "
        f"steps={trainer.state.global_step}"
    )

