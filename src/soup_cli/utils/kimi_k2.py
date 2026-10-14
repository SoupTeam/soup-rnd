"""Kimi K2 under layer streaming: a ``kimi_k2`` checkpoint on transformers' DeepSeek-V3 classes.

Kimi K2 has DeepSeek-V3's decoder layer, but its ``config.json`` says ``model_type:
kimi_k2`` and names Moonshot's modeling code in ``auto_map``. transformers 5.19 has no
``kimi_k2`` config type, so ``AutoConfig`` refuses the folder unless it may run that code,
and the Hub code cannot train. This module builds the config natively
(``DeepseekV3Config``) and registers the per-expert to fused-expert conversion that
transformers keys on ``model_type`` and does not list for ``kimi_k2``. Without it a native
load leaves every routed expert at its random init, and a save writes fused keys.
"""

from typing import Any, Dict

KIMI_K2_MODEL_TYPE = "kimi_k2"

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
