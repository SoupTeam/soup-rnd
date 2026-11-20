"""Pin recipe overrides against the live schema, not YAML text or incidental defaults.

The committed fixture stores a historical baseline and each recipe's nondefault
fields. Compare recipes against the live baseline so adding an unused opt-in
field does not require re-pinning every recipe. Consumer tests cover the behavior
of schema defaults; a snapshot of every field is not a behavior test.

Regenerate recipe deltas with ``python scripts/generate_recipe_snapshot.py`` after
a deliberate recipe change. Parsed values avoid CRLF/LF sensitivity.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from scripts.generate_recipe_snapshot import build_baseline, delta, flatten
from soup_cli.recipes.catalog import RECIPES

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_config_snapshots.json"
)
_REGENERATE_HINT = "regenerate with scripts/generate_recipe_snapshot.py"
_MISSING = object()


@functools.lru_cache(maxsize=1)
def _load_fixture() -> dict:
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not (isinstance(data, dict) and "baseline" in data and "recipes" in data):
        # A raised exception rather than an assert: assertions are stripped
        # under `python -O`, which would let a malformed fixture slip past
        # this guard and fail later with a far less clear error.
        raise ValueError(
            f"fixture must be a dict with 'baseline' and 'recipes' keys; {_REGENERATE_HINT}"
        )
    return data


# Forces a collection-time failure (not a per-test runtime error) if the
# fixture is missing or malformed — matches #621's original property that a
# deleted fixture is loud immediately, not 162 identical errors.
_load_fixture()


@functools.lru_cache(maxsize=1)
def _live_baseline() -> dict[str, object]:
    return build_baseline()


def _resolve_flat(name: str) -> dict[str, object]:
    from soup_cli.config.loader import load_config_from_string

    return flatten(load_config_from_string(RECIPES[name].yaml_str).model_dump(mode="json"))


def _diff(expected: dict, actual: dict) -> dict[str, tuple[object, object]]:
    """Paths present in either side whose values differ.

    Compares against an internal sentinel, not the string ``"<missing>"`` —
    a real config value equal to that string must never compare as
    spuriously equal to an absent one. The sentinel only ever appears in
    the returned tuples as a stand-in for display.
    """
    paths = sorted(set(expected) | set(actual))
    out: dict[str, tuple[object, object]] = {}
    for path in paths:
        e = expected.get(path, _MISSING)
        a = actual.get(path, _MISSING)
        if e != a:
            out[path] = (
                "<missing>" if e is _MISSING else e,
                "<missing>" if a is _MISSING else a,
            )
    return out


class TestFixtureCoversExactlyTheCurrentCatalog:
    def test_fixture_and_catalog_have_the_same_recipe_names(self):
        """A recipe added or removed without regenerating fails here first,
        with a clear "which names" message, rather than as a KeyError deep
        in a parametrized test."""
        fixture_names = set(_load_fixture()["recipes"])
        catalog_names = set(RECIPES)
        assert fixture_names == catalog_names, (
            f"missing from fixture: {sorted(catalog_names - fixture_names)}; "
            f"stale in fixture: {sorted(fixture_names - catalog_names)}; "
            f"{_REGENERATE_HINT}"
        )


class TestEveryRecipeMatchesItsCommittedDelta:
    """Compare recipe overrides against the live schema baseline."""

    @pytest.mark.parametrize("name", sorted(RECIPES))
    def test_recipe_delta_matches_its_snapshot(self, name: str):
        fixture = _load_fixture()
        assert name in fixture["recipes"], (
            f"{name!r} is in the catalog but missing from the snapshot fixture; "
            f"{_REGENERATE_HINT}"
        )
        live_delta = delta(_resolve_flat(name), _live_baseline())
        diff = _diff(fixture["recipes"][name], live_delta)
        assert not diff, (
            f"recipe {name!r} no longer resolves to its committed delta. "
            f"If this is a deliberate change, review the diff and {_REGENERATE_HINT}. "
            f"Differing fields (field: (snapshot, resolved)): {diff}"
        )


class TestTheSnapshotIsNotVacuous:
    """Demonstrates the comparison actually discriminates, rather than
    merely asserting it does. Exercises `_diff` directly on synthetic
    dicts, not a live recipe — an unrelated schema addition reddens the
    baseline test already, and coupling these controls to the same live
    data would redden them too, making "did the guard break, or did the
    data move?" unanswerable from the log."""

    def test_a_changed_field_value_is_detected(self):
        expected = {"training.dpo_beta": 0.1, "training.epochs": 3}
        actual = {"training.dpo_beta": 0.2, "training.epochs": 3}

        assert _diff(expected, actual) == {"training.dpo_beta": (0.1, 0.2)}

    def test_identical_dicts_have_no_diff(self):
        """Negative control: same values, must be silent."""
        expected = {"training.dpo_beta": 0.1, "training.epochs": 3}
        actual = {"training.dpo_beta": 0.1, "training.epochs": 3}

        assert _diff(expected, actual) == {}
