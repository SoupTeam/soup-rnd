"""Kimi K2 under layer streaming: a ``kimi_k2`` checkpoint on transformers' DeepSeek-V3 classes.

Kimi K2 has DeepSeek-V3's decoder layer, but its ``config.json`` says ``model_type:
kimi_k2`` and names Moonshot's modeling code in ``auto_map``. transformers 5.19 has no
``kimi_k2`` config type, so ``AutoConfig`` refuses the folder unless it may run that code,
and the Hub code cannot train. This module builds the config natively
(``DeepseekV3Config``) and registers the per-expert to fused-expert conversion that
transformers keys on ``model_type`` and does not list for ``kimi_k2``. Without it a native
load leaves every routed expert at its random init, and a save writes fused keys.
"""

import base64
import json
import os
from typing import Any, Dict, Optional

KIMI_K2_MODEL_TYPE = "kimi_k2"

_OUTSIDE_STREAMING = (
    "model_type 'kimi_k2' ({model_id}) is not supported outside layer streaming yet. "
    "transformers has no kimi_k2 config type, so a plain load fails with a custom-code "
    "error or leaves the routed experts at random init. Set training.stream_layers: true "
    "to train it."
)


class KimiK2OutsideStreamingError(ValueError):
    """A ``kimi_k2`` folder reached a path that loads the base model without streaming."""

#: Fields Moonshot's config carries for the auxiliary balance loss. The native
#: ``DeepseekV3`` classes never read them. That is harmless while the router is frozen,
#: which it is under LoRA, and it is recorded here instead of being left implicit.
IGNORED_AUX_LOSS_FIELDS = ("aux_loss_alpha", "seq_aux")

_DEEPSEEK_V3 = "deepseek_v3"
_CONFLICT = (
    "a different checkpoint conversion for kimi_k2 is already registered; "
    "layer streaming will not replace it"
)


def register_kimi_k2_conversion() -> None:
    """Give ``kimi_k2`` the ``deepseek_v3`` checkpoint conversion, for load and for save.

    Three tables hold it. transformers' ``_MODEL_TO_CONVERSION_PATTERN`` names the rule set
    per model type. PEFT copies that table when it is first imported and reads its copy to
    rewrite MoE LoRA targets (``moe_lora``) onto the fused experts, so the copy needs the
    entry too. The registered mapping is what transformers applies when it loads or saves
    weights.

    Idempotent. Touches only the ``kimi_k2`` entries. A different entry registered
    under ``kimi_k2`` by someone else is an error, not something to overwrite.
    """
    from transformers import conversion_mapping
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    patterns = conversion_mapping._MODEL_TO_CONVERSION_PATTERN
    rules = get_checkpoint_conversion_mapping(_DEEPSEEK_V3)
    if not rules or _DEEPSEEK_V3 not in patterns:
        raise RuntimeError(
            "this transformers has no deepseek_v3 checkpoint conversion to reuse for kimi_k2"
        )
    existing = get_checkpoint_conversion_mapping(KIMI_K2_MODEL_TYPE)
    pattern = patterns.get(KIMI_K2_MODEL_TYPE)
    if existing is not None and repr(existing) != repr(rules):
        raise ValueError(_CONFLICT)
    if pattern is not None and pattern != patterns[_DEEPSEEK_V3]:
        raise ValueError(_CONFLICT)
    if existing is None:
        register_checkpoint_conversion_mapping(KIMI_K2_MODEL_TYPE, rules)
    patterns[KIMI_K2_MODEL_TYPE] = patterns[_DEEPSEEK_V3]
    from peft.utils import transformers_weight_conversion as peft_conversion

    peft_conversion._MODEL_TO_CONVERSION_PATTERN[KIMI_K2_MODEL_TYPE] = patterns[_DEEPSEEK_V3]


def ignored_aux_loss_fields(config: Any) -> Dict[str, Any]:
    """The auxiliary-loss fields ``config`` carries that the native classes ignore."""
    if getattr(config, "model_type", None) != KIMI_K2_MODEL_TYPE:
        return {}
    return {
        name: getattr(config, name) for name in IGNORED_AUX_LOSS_FIELDS if hasattr(config, name)
    }


def load_stream_config(model_id: str, trust_remote_code: bool) -> Any:
    """``AutoConfig.from_pretrained``, except a ``kimi_k2`` folder is read without Hub code.

    A ``kimi_k2`` config is built as a ``DeepseekV3Config`` that keeps
    ``model_type == "kimi_k2"``, and the conversion is registered so every later load or
    save of that model type converts experts. ``trust_remote_code`` is refused for it,
    because the checkpoint's own modeling code cannot train.
    """
    from transformers import AutoConfig

    try:
        from transformers import DeepseekV3Config

        raw, _ = DeepseekV3Config.get_config_dict(model_id)
    except (ImportError, OSError, ValueError):
        # Not readable as a config here. AutoConfig below raises the real error.
        raw = {}
    if raw.get("model_type") != KIMI_K2_MODEL_TYPE:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if trust_remote_code:
        raise ValueError(
            "layer streaming needs the transformers DeepSeek-V3 class for model_type="
            "'kimi_k2', but --trust-remote-code builds the checkpoint's own modeling "
            "code, which cannot train. Run again without --trust-remote-code."
        )
    register_kimi_k2_conversion()
    return DeepseekV3Config.from_pretrained(model_id)


def refuse_kimi_k2_outside_streaming(model_id: str) -> None:
    """Raise :class:`KimiK2OutsideStreamingError` if ``model_id`` is a ``kimi_k2`` folder.

    Reads ``config.json`` as plain JSON, so nothing is loaded and no Hub code runs. Any
    other folder, a missing path, a folder without ``config.json`` and a Hub id all
    pass. The caller's own loader then reports a real error.
    """
    if _local_model_type(model_id) == KIMI_K2_MODEL_TYPE:
        raise KimiK2OutsideStreamingError(_OUTSIDE_STREAMING.format(model_id=model_id))


def _local_model_type(model_id: str) -> Optional[str]:
    """``model_type`` from a local folder's ``config.json``, or None if it cannot be read."""
    config_path = os.path.join(str(model_id), "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    return raw.get("model_type") if isinstance(raw, dict) else None


def exit_if_kimi_k2_outside_streaming(console: Any, *model_ids: Any) -> None:
    """The CLI form of the refusal: print it in red and exit with code 1."""
    import typer
    from rich.markup import escape

    try:
        for model_id in model_ids:
            if model_id:
                refuse_kimi_k2_outside_streaming(str(model_id))
    except KimiK2OutsideStreamingError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc


#: Moonshot's split pattern, copied from ``TikTokenTokenizer.pat_str`` in
#: ``tokenization_kimi.py`` (Kimi-K2-Instruct). Han characters are split out first.
_K2_SPLIT_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"""
        r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"""
        r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

#: Ids after the base vocabulary that Moonshot reserves for special tokens.
_K2_RESERVED_TOKENS = 256


def _read_tiktoken_ranks(path: str) -> Dict[bytes, int]:
    """The ``tiktoken.model`` file: one ``base64(token) rank`` line per token."""
    ranks: Dict[bytes, int] = {}
    with open(path, "rb") as handle:
        for number, line in enumerate(handle.read().splitlines(), start=1):
            if not line:
                continue
            try:
                token, rank = line.split()
                ranks[base64.b64decode(token)] = int(rank)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a 'base64-token rank' line") from exc
    return ranks


def load_kimi_k2_tokenizer(model_dir: str, trust_remote_code: bool = False) -> Any:
    """Kimi K2's tokenizer as a transformers fast tokenizer, without Hub code.

    Moonshot's ``tokenization_kimi.py`` wraps a ``tiktoken`` encoding. This builds the same
    BPE from ``tiktoken.model`` with transformers' ``TikTokenConverter`` and Moonshot's split
    pattern, puts the special tokens from ``tokenizer_config.json`` at their ids, and fills
    every other id of the 256 reserved after the base vocabulary with
    ``<|reserved_token_N|>``, as Moonshot's code does. The chat template comes from
    ``chat_template.jinja``. Ids match Moonshot's tokenizer on the texts in
    ``tests/fixtures/kimi_k2_tokenizer_reference.json``.
    """
    if trust_remote_code:
        raise ValueError(
            "the native Kimi K2 tokenizer replaces Moonshot's tokenizer code, so "
            "--trust-remote-code has no use here. Run again without --trust-remote-code."
        )
    from tokenizers import AddedToken
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter

    vocab_file = os.path.join(model_dir, "tiktoken.model")
    if not os.path.isfile(vocab_file):
        raise FileNotFoundError(f"tiktoken.model not found in {model_dir}")
    with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    template_path = os.path.join(model_dir, "chat_template.jinja")
    chat_template = config.get("chat_template")
    if os.path.isfile(template_path):
        with open(template_path, encoding="utf-8") as handle:
            chat_template = handle.read()

    ranks = _read_tiktoken_ranks(vocab_file)

    class _Converter(TikTokenConverter):
        @staticmethod
        def load_tiktoken_bpe(tiktoken_url: str) -> Dict[bytes, int]:
            return ranks

    tokenizer = _Converter(vocab_file=vocab_file, pattern=_K2_SPLIT_PATTERN).converted()
    declared = {
        int(index): entry for index, entry in config.get("added_tokens_decoder", {}).items()
    }
    first = len(ranks)
    reserved = []
    for index in range(first, first + _K2_RESERVED_TOKENS):
        entry = declared.get(index)
        reserved.append(
            AddedToken(
                entry["content"] if entry else f"<|reserved_token_{index}|>",
                normalized=False,
                special=entry["special"] if entry else True,
            )
        )
    tokenizer.add_tokens(reserved)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=config.get("bos_token"),
        eos_token=config.get("eos_token"),
        unk_token=config.get("unk_token"),
        pad_token=config.get("pad_token"),
        additional_special_tokens=config.get("additional_special_tokens") or None,
        chat_template=chat_template,
        clean_up_tokenization_spaces=False,
    )


def load_stream_tokenizer(model_id: str, trust_remote_code: bool) -> Any:
    """``AutoTokenizer.from_pretrained``, except a ``kimi_k2`` folder gets the native one."""
    if _local_model_type(model_id) == KIMI_K2_MODEL_TYPE:
        return load_kimi_k2_tokenizer(model_id, trust_remote_code=trust_remote_code)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
