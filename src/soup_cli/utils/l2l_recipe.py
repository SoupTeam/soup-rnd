"""The resolved L2L run recipe: effective k, activation codec and the snapshot on disk.

``training.gradient_accumulation_steps`` is what the user asks for. The effective k is what the
run uses. Every consumer (the HF training arguments, the loss scale the HF loop derives from
them, the activation store capacity, the log) reads the effective k from one
:class:`L2LRecipe`, so none of them can disagree. The recipe is written to the output
directory and checked on resume, because a different k changes the optimizer-step boundaries.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

RECIPE_NAME = "l2l_recipe.json"

#: Codecs the launch accepts. ``fp8_e4m3`` joins when the FP8 store exists.
AVAILABLE_CODECS = ("none",)

#: True once the memory planner can pick k at launch.
AUTO_K_AVAILABLE = False


@dataclass(frozen=True)
class L2LRecipe:
    requested_k: int
    effective_k: int
    activation_codec: str
    auto_k: bool


def resolve_l2l_recipe(tcfg: Any) -> L2LRecipe:
    """The recipe for ``tcfg`` (a ``TrainingConfig`` with ``l2l=true``).

    Raises ``ValueError`` for a codec or for ``l2l_auto_k`` the launch cannot honour yet. It
    never falls back to another value: the user asked for that one.
    """
    codec = str(tcfg.l2l_activation_codec)
    if codec not in AVAILABLE_CODECS:
        raise ValueError(
            f"training.l2l_activation_codec={codec!r} is not available yet: the launch "
            f"supports {', '.join(repr(c) for c in AVAILABLE_CODECS)}. Set it to 'none'."
        )
    if tcfg.l2l_auto_k and not AUTO_K_AVAILABLE:
        raise ValueError(
            "training.l2l_auto_k is not available yet: the memory planner is not wired "
            "into the launch. Set l2l_auto_k=false and choose gradient_accumulation_steps."
        )
    requested = int(tcfg.gradient_accumulation_steps)
    return L2LRecipe(
        requested_k=requested,
        effective_k=requested,
        activation_codec=codec,
        auto_k=bool(tcfg.l2l_auto_k),
    )


def write_recipe(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe) -> str:
    """Write the recipe to ``<output_dir>/l2l_recipe.json``; returns the path."""
    from soup_cli.utils.paths import atomic_write_text

    path = os.path.join(os.fspath(output_dir), RECIPE_NAME)
    text = json.dumps(asdict(recipe), indent=2, sort_keys=True) + "\n"
    return atomic_write_text(text, path, field="l2l recipe")


def load_recipe(output_dir: "str | os.PathLike[str]") -> Optional[L2LRecipe]:
    """The recipe saved in ``output_dir``, or None if there is none."""
    path = Path(output_dir) / RECIPE_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return L2LRecipe(**{field.name: data[field.name] for field in fields(L2LRecipe)})
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a valid L2L recipe: {exc}") from exc


def validate_resume(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe) -> None:
    """Refuse to resume a run whose saved recipe differs from ``recipe``.

    No saved recipe (a run from before the recipe existed) is accepted.
    """
    saved = load_recipe(output_dir)
    if saved is None:
        return
    for name in ("effective_k", "activation_codec"):
        before, now = getattr(saved, name), getattr(recipe, name)
        if before != now:
            raise ValueError(
                f"cannot resume: the checkpoint was trained with {name}={before!r}, but this "
                f"run has {name}={now!r}. Restore the old value in the config."
            )
