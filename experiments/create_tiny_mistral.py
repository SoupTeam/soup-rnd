
"""Create a local tiny Mistral for E2 CLI smoke testing."""

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    PreTrainedTokenizerFast,
)

OUTPUT = Path("experiments/tiny_mistral")


def main():
    torch.manual_seed(42)

    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    model = MistralForCausalLM(config)
    model.save_pretrained(OUTPUT)

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

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(OUTPUT)

    print(f"Tiny Mistral saved to {OUTPUT}")
    print(f"Decoder layers: {config.num_hidden_layers}")


if __name__ == "__main__":
    main()

