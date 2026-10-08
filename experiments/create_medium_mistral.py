
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

OUTPUT = Path("experiments/medium_mistral")


def main():
    torch.manual_seed(42)

    config = MistralConfig(
        vocab_size=100,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    model = MistralForCausalLM(config)
    model.save_pretrained(OUTPUT)

    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = [
        "What", "is", "the", "capital", "of", "France",
        "Paris", "color", "sky", "Blue",
        "2", "3", "4", "5", "8",
        "?", "+", "Answer",
    ]

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
    }

    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

    backend = Tokenizer(
        WordLevel(vocab=vocab, unk_token="[UNK]")
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

