"""``--trust-remote-code`` opt-in (v0.36.0 Part B).

Replaces the previous unconditional ``trust_remote_code=True`` in
sft.py / chat.py / serve.py / etc. with an explicit, auditable flag plus a
trusted-org allowlist that suppresses warning noise on first-party models.

Design:

- ``is_known_safe(model_name)``: true when the repo id starts with a known
  org prefix that does NOT ship custom modeling code (Meta, Mistral, Qwen,
  Google, etc.). Local paths and unknown orgs return False.
- ``model_requires_trust_remote_code(model_or_path)``: best-effort probe of
  ``config.json``'s ``auto_map`` field. Returns True / False / None (unknown).
  A config whose ``model_type`` has a built-in transformers class counts as
  False, because transformers ignores ``auto_map`` for it when
  ``trust_remote_code`` is off. No network calls — local-only.
- ``resolve_trust_remote_code(model_name, requested, console, requires_remote_code)``:
  the gate. Returns ``True`` / ``False`` for the kwarg, or raises ``ValueError``
  with an actionable message when the model needs custom code but the user
  did not opt in.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rich.console import Console

# Known organisations that do NOT ship custom modeling code with their HF
# checkpoints. Adding a prefix here suppresses the warning panel when the
# user passes ``--trust-remote-code`` against a safe org.
KNOWN_SAFE_PREFIXES: tuple[str, ...] = (
    "meta-llama/",
    "mistralai/",
    "Qwen/",
    "google/",
    "microsoft/",
    "deepseek-ai/",
    "01-ai/",
    "tiiuae/",
    "HuggingFaceH4/",
    "openai-community/",
    "facebook/",
    "EleutherAI/",
    "CohereForAI/",
    "stabilityai/",
    "ibm-granite/",
)


def is_known_safe(model_name: Any) -> bool:
    """Return True if ``model_name`` starts with a known-safe org prefix."""
    if not isinstance(model_name, str) or not model_name:
        return False
    return any(model_name.startswith(prefix) for prefix in KNOWN_SAFE_PREFIXES)


def _is_builtin_model_type(model_type: Any) -> bool:
    """Return True when transformers ships a config class for ``model_type``.

    Lazy import: ``transformers`` loads only when a config has ``auto_map``.
    Any failure (transformers missing, API moved) returns False so the gate
    keeps demanding the flag.
    """
    if not isinstance(model_type, str) or not model_type:
        return False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:  # noqa: BLE001 - any import problem means "not known built-in"
        return False
    return model_type in CONFIG_MAPPING_NAMES


def model_requires_trust_remote_code(model_or_path: str) -> Optional[bool]:
    """Best-effort local probe of ``config.json`` for ``auto_map``.

    Returns:
        True if config has ``auto_map`` (HF custom-code marker) and its
            ``model_type`` has no built-in transformers class.
        False if config exists but has no ``auto_map``, or has one but the
            ``model_type`` is built in (transformers uses its own class).
        None if config is missing or unreadable (unknown).

    Never makes a network request — pure local inspection.
    """
    if not isinstance(model_or_path, str) or not model_or_path:
        return None
    try:
        if os.path.isdir(model_or_path):
            config_path = os.path.join(model_or_path, "config.json")
        else:
            return None
        if not os.path.isfile(config_path):
            return None
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("auto_map"):
        return False
    try:
        return not _is_builtin_model_type(data.get("model_type"))
    except Exception:  # noqa: BLE001 - stay conservative on any lookup failure
        return True


def tokenizer_requires_trust_remote_code(model_or_path: str) -> bool:
    """Return True when a local folder's ``tokenizer_config.json`` has ``auto_map``.

    ``model_requires_trust_remote_code`` reads ``config.json`` only. Callers
    that load a tokenizer use this as well, so a built-in model type with a
    Hub tokenizer still needs the flag. Missing or unreadable files give False.
    """
    if not isinstance(model_or_path, str) or not model_or_path:
        return False
    path = os.path.join(model_or_path, "tokenizer_config.json")
    try:
        if not os.path.isfile(path):
            return False
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and bool(data.get("auto_map"))


def resolve_trust_remote_code(
    model_name: str,
    requested: bool,
    console: "Console | None" = None,
    requires_remote_code: bool = False,
) -> bool:
    """Decide whether to pass ``trust_remote_code=True`` to HF loaders.

    Args:
        model_name: HF repo id or local path.
        requested: ``True`` when the user passed ``--trust-remote-code``.
        console: Rich Console for the warning panel (optional).
        requires_remote_code: ``True`` when the model has ``auto_map`` set.
            Note: ``model_requires_trust_remote_code`` only probes LOCAL
            paths; HF Hub repo IDs return ``None`` (unknown) which the
            caller normally coerces to ``False``. In the unknown-Hub-path
            case HF's own ``from_pretrained`` will still raise loudly when
            it actually needs custom code, so the gate is defence-in-depth
            rather than the only line of defence.

    Returns:
        ``bool`` to pass directly as ``trust_remote_code=...``.

    Raises:
        ValueError: model needs custom code but user did not opt in, or
            ``model_name`` is empty.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")

    if not requested and requires_remote_code:
        raise ValueError(
            f"Model {model_name} requires trust_remote_code=True (custom "
            f"modeling code via auto_map). Re-run with --trust-remote-code "
            f"if you trust the source. This change in v0.36.0 makes the "
            f"opt-in explicit; it was previously enabled by default."
        )

    if requested and not is_known_safe(model_name) and console is not None:
        # Lazy import — keeps `import soup_cli.utils.trust_remote` cheap
        # in environments that don't otherwise pull in rich.
        from rich.panel import Panel

        from soup_cli.utils.terminal import for_terminal

        console.print(
            Panel.fit(
                f"[yellow]--trust-remote-code is enabled for[/] "
                f"[bold]{for_terminal(model_name)}[/]\n"
                f"This will execute Python code shipped in the model repo. "
                f"Only proceed if you trust the source.",
                title="[red]REMOTE CODE WARNING[/]",
                border_style="red",
            )
        )

    return bool(requested)
