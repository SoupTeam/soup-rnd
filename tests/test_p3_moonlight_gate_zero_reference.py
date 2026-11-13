"""Synthetic CPU raw-tensor replay contracts, not GPU, model or quality evidence."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_gate as gate
from tests.test_p3_moonlight_gate import (
    DEEPSEEK,
    MOONLIGHT,
    PASSES,
    _commit,
    _digest,
    _receipt,
    _verify,
)

pytestmark = pytest.mark.usefixtures("moonlight_aux_admission")


def _refresh_metrics(root: Path, result: dict[str, Any]) -> None:
    """Prepare truthful derived observations after changing synthetic archived tensors."""
    import torch

    from benchmarks.harness import g4_activation_codec as metrics

    passes = result["passes"]
    gradients = {}
    for arm in PASSES:
        artifact = root / arm / "gradients.pt"
        gradients[arm] = torch.load(artifact, map_location="cpu", weights_only=True)
        passes[arm]["gradient_artifact"] = {
            "path": "gradients.pt", "bytes": artifact.stat().st_size, "sha256": _digest(artifact),
        }
        for name, value in gradients[arm].items():
            passes[arm]["gradient_checks"][name].update(
                nonzero=bool(value.ne(0).any()), shape=list(value.shape))
    result["a3"] = gate.assess_a3(
        gradients["ga"], gradients["ga_repeat"], gradients["bf16_l2l"],
        losses=passes["ga"]["losses"], repeat_losses=passes["ga_repeat"]["losses"],
        l2l_losses=passes["bf16_l2l"]["losses"], dtype="bfloat16",
        floor_gradients=gradients["ga_floor"], floor_losses=passes["ga_floor"]["losses"],
        floor_observed=True,
    )
    for arm in ("bf16_l2l", "fp8_oracle", "int4_oracle"):
        for layer in result["layers"]:
            parts = {kind: [] for kind in ("restore", "output", "branch")}
            for mb in range(gate.K):
                basename = f"layer-{layer:03d}-mb-{mb:03d}.pt"
                original = torch.load(root / "bf16_l2l/forward" / basename,
                                      map_location="cpu", weights_only=True)
                restored = torch.load(root / arm / "recompute" / basename,
                                      map_location="cpu", weights_only=True)
                for kind, key in (("restore", "input"), ("output", "output")):
                    parts[kind].append(metrics.restore_error(original[key], restored[key]))
                parts["branch"].append(metrics.restore_error(
                    original["output"].double() - original["input"].double(),
                    restored["output"].double() - restored["input"].double(),
                ))
            passes[arm]["layers"][layer].update({
                kind: metrics.merge_errors(values) for kind, values in parts.items()
            })
        if "oracle" in arm:
            passes[arm]["gradient_agreement"] = gate.gradient_agreement(
                gradients["bf16_l2l"], gradients[arm])
            assessed = gate.assess_metrics(
                passes[arm]["layers"], passes[arm]["gradient_agreement"],
                expected_layers=result["layers"], expected_tensors=result["trainable_names"],
            )
            if arm == "fp8_oracle":
                result["candidate_gates"] = assessed
            else:
                result["control"] = {
                    "gates": assessed, "discrimination": gate.assess_discrimination(assessed),
                }
    result["verdict"] = gate.overall_verdict(
        result["a3"], result["candidate_gates"], result["control"]["discrimination"], eligible=True,
    )


def _zero_reference(root: Path, result: dict[str, Any], *, all_tensors: bool) -> list[str]:
    import torch

    names = result["trainable_names"] if all_tensors else result["trainable_names"][:1]
    for arm in ("ga", "ga_repeat", "bf16_l2l", "ga_floor"):
        path = root / arm / "gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        for name in names:
            values[name].zero_()
        torch.save(values, path)
    return names


def _publish(root: Path, result: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Replace the fixture's initial wrapper with the exclusive-write production publisher."""
    for relative in ("manifest.json", "result.json", *(arm + "/pass.json" for arm in PASSES)):
        (root / relative).unlink()
    published = gate._publish(root, result)
    return root / "manifest.json", published


def _gradient_paths(names: list[str], *, all_tensors: bool) -> list[str]:
    paths = []
    for arm in ("fp8_oracle", "int4_oracle"):
        prefix = f"$.passes.{arm}.gradient_agreement"
        paths.extend(prefix + f".per_tensor.{name}.rel_l2" for name in names)
        paths.extend(prefix + f".worst[{index}].rel_l2" for index in range(min(5, len(names))))
        paths.append(prefix + ".max_rel_l2")
        if all_tensors:
            paths.append(prefix + ".global_rel_l2")
    return sorted(paths)


@pytest.mark.parametrize("all_tensors", [False, True], ids=["one-zero-reference", "all-zero"])
def test_zero_reference_gradients_publish_and_admit_exact_diagnostics(
    tmp_path: Path, all_tensors: bool,
) -> None:
    # Promoting derived infinity to FAILED, comparing raw inf to null, or demanding PASS/FAIL
    # G4 metrics would each reject this complete six-pass serialized/public-loader round trip.
    _, result = _receipt(tmp_path, verdict="PASS", production=True)
    zero_names = _zero_reference(tmp_path, result, all_tensors=all_tensors)
    _refresh_metrics(tmp_path, result)
    for arm in ("fp8_oracle", "int4_oracle"):
        agreement = result["passes"][arm]["gradient_agreement"]
        assert agreement["per_tensor"][zero_names[0]]["rel_l2"] == math.inf
        assert agreement["per_tensor"][zero_names[0]]["cosine"] == 0.0
        assert math.isinf(agreement["global_rel_l2"]) is all_tensors
    manifest, published = _publish(tmp_path, result)
    assert published["status"] == "COMPLETE"
    assert published["verdict"] == "NO_VERDICT"
    assert published["a3"]["status"] == ("NO_VERDICT" if all_tensors else "PASS")
    assert {row["status"] for row in published["candidate_gates"].values()} == {"NO_VERDICT"}
    assert published["nonfinite_evidence_fields"] == _gradient_paths(
        zero_names, all_tensors=all_tensors)
    assert not (tmp_path / "failure.json").exists()
    assert _verify(manifest) == published
    admitted = gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
    assert admitted == published
    assert set(admitted["passes"]) == set(PASSES)
    for arm in PASSES:
        row = admitted["passes"][arm]
        assert row["status"] == "COMPLETE"
        assert row["gradient_checks"][zero_names[0]]["nonzero"] is ("oracle" in arm)
    for arm in ("fp8_oracle", "int4_oracle"):
        assert admitted["passes"][arm]["gradient_agreement"]["per_tensor"][
            zero_names[0]]["rel_l2"] is None


@pytest.mark.parametrize("kind", ["restore", "branch"])
def test_zero_reference_boundary_metrics_replay_as_undefined_not_technical_failure(
    tmp_path: Path, kind: str,
) -> None:
    import torch

    _, result = _receipt(tmp_path, verdict="PASS")
    for mb in range(gate.K):
        basename = f"layer-000-mb-{mb:03d}.pt"
        original = {
            "input": torch.full((1, 64, 2), 0.0 if kind == "restore" else 1.0,
                                dtype=torch.bfloat16),
            "output": torch.ones((1, 64, 2), dtype=torch.bfloat16),
        }
        torch.save(original, tmp_path / "bf16_l2l/forward" / basename)
        torch.save(original, tmp_path / "bf16_l2l/recompute" / basename)
        for arm in ("fp8_oracle", "int4_oracle"):
            torch.save({"input": torch.ones((1, 64, 2), dtype=torch.bfloat16),
                        "output": torch.full((1, 64, 2), 2.0, dtype=torch.bfloat16)},
                       tmp_path / arm / "recompute" / basename)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    assert published["status"] == "COMPLETE"
    assert published["verdict"] == "NO_VERDICT"
    assert published["nonfinite_evidence_fields"] == sorted(
        f"$.passes.{arm}.layers[0].{kind}.{key}"
        for arm in ("fp8_oracle", "int4_oracle")
        for key in ("rel_l2", "rms_row_rel", "max_abs_over_absmax", "worst_row_rel",
                    "row_rel_sq_sum")
    )
    metric = published["passes"]["fp8_oracle"]["layers"][0][kind]
    assert metric["ref_sq"] == 0.0
    assert metric["err_sq"] == 1024.0
    assert metric["rms_row_rel"] is metric["rel_l2"] is None
    assert _verify(manifest) == published
    assert not (tmp_path / "failure.json").exists()


@pytest.mark.parametrize("singular_floor", [False, True], ids=["finite-floor", "undefined-floor"])
def test_zero_reference_a3_loss_keeps_frozen_fail_or_no_verdict(
    tmp_path: Path, singular_floor: bool,
) -> None:
    _, result = _receipt(tmp_path, verdict="PASS")
    for arm in ("ga", "ga_repeat"):
        result["passes"][arm]["losses"] = [0.0] * gate.K
    result["passes"]["ga_floor"]["losses"] = [1.0 if singular_floor else 0.0] * gate.K
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    expected_status = "NO_VERDICT" if singular_floor else "FAIL"
    assert published["status"] == "COMPLETE"
    assert published["verdict"] == published["a3"]["status"] == expected_status
    tolerance = published["a3"]["rule"]["tolerance"]
    assert tolerance["loss_rel"] == [None]
    if singular_floor:
        assert published["a3"]["rule"]["status"] == "invalid_floor"
        assert tolerance["judged"] is False
        assert tolerance.get("loss_limit") is None
        expected_paths = ["$.a3.rule.tolerance.loss_rel[0]"]
    else:
        assert tolerance["loss_limit"] == [0.0]
        assert tolerance["failing"] == [{"what": "loss", "step": 0, "error": None, "limit": 0.0}]
        expected_paths = ["$.a3.rule.tolerance.failing[0].error",
                          "$.a3.rule.tolerance.loss_rel[0]"]
    assert published["nonfinite_evidence_fields"] == expected_paths
    assert _verify(manifest) == published
    assert not (tmp_path / "failure.json").exists()


def test_public_loader_refuses_missing_or_extra_derived_diagnostic_flags(tmp_path: Path) -> None:
    _, result = _receipt(tmp_path, verdict="PASS", production=True)
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    for mutation in ("missing", "extra", "wrong-location"):
        changed = copy.deepcopy(published)
        flags = changed["nonfinite_evidence_fields"]
        if mutation == "missing":
            flags.pop()
        else:
            if mutation == "wrong-location":
                flags.pop()
            # A known report prefix is insufficient. This cosine is provably finite zero.
            flags.append("$.passes.fp8_oracle.gradient_agreement.per_tensor."
                         + changed["trainable_names"][0] + ".cosine")
            flags.sort()
        _commit(tmp_path, changed)
        with pytest.raises(ValueError, match="nonfinite|diagnostic"):
            gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
        with pytest.raises(ValueError, match="nonfinite|diagnostic"):
            _verify(manifest)


@pytest.mark.parametrize("mutation", [
    "duplicate", "not-list", "not-path", "null-finite-metric", "wrong-null", "infinite-json",
])
def test_replay_refuses_malformed_or_fabricated_derived_diagnostics(
    tmp_path: Path, mutation: str,
) -> None:
    _, result = _receipt(tmp_path, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    flags = published["nonfinite_evidence_fields"]
    agreement = published["passes"]["fp8_oracle"]["gradient_agreement"]
    if mutation == "duplicate":
        flags.append(flags[0])
    elif mutation == "not-list":
        published["nonfinite_evidence_fields"] = flags[0]
    elif mutation == "not-path":
        flags.append(None)
    elif mutation == "null-finite-metric":
        agreement["min_cosine"] = None
        flags.append("$.passes.fp8_oracle.gradient_agreement.min_cosine")
        flags.sort()
    elif mutation == "wrong-null":
        agreement["max_rel_l2"] = 0.0
    else:
        agreement["max_rel_l2"] = math.inf
    _commit(tmp_path, published)
    with pytest.raises(ValueError):
        _verify(manifest)


@pytest.mark.parametrize("location", ["gradient", "boundary", "loss"])
@pytest.mark.parametrize("nonfinite", [math.nan, math.inf], ids=["nan", "inf"])
def test_derived_diagnostics_never_exempt_nonfinite_raw_evidence(
    tmp_path: Path, location: str, nonfinite: float,
) -> None:
    import torch

    _, result = _receipt(tmp_path, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    if location == "gradient":
        path = tmp_path / "fp8_oracle/gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values[published["trainable_names"][0]][0, 0] = nonfinite
        torch.save(values, path)
    elif location == "boundary":
        path = tmp_path / "fp8_oracle/recompute/layer-000-mb-000.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values["input"] = values["input"].clone()
        values["input"][0, 0, 0] = nonfinite
        torch.save(values, path)
    else:
        published["passes"]["fp8_oracle"]["losses"][0] = nonfinite
    _commit(tmp_path, published)
    with pytest.raises(ValueError, match="nonfinite|finite|invalid raw"):
        _verify(manifest)


@pytest.mark.parametrize("location", ["loss", "unrecognized"])
@pytest.mark.parametrize("nonfinite", [math.nan, math.inf], ids=["nan", "inf"])
def test_publisher_keeps_raw_or_unrecognized_nonfinites_technical(
    tmp_path: Path, location: str, nonfinite: float,
) -> None:
    _, result = _receipt(tmp_path, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    if location == "loss":
        result["passes"]["fp8_oracle"]["losses"][0] = nonfinite
        path = "$.passes.fp8_oracle.losses[0]"
    else:
        result["measurement"] = nonfinite
        path = "$.measurement"
    manifest, published = _publish(tmp_path, result)
    assert published["status"] == "FAILED"
    assert published["verdict"] == "NO_VERDICT"
    assert path in published["nonfinite_evidence_fields"]
    assert (tmp_path / "failure.json").is_file()
    with pytest.raises(ValueError):
        _verify(manifest)


@pytest.mark.parametrize("mutation", ["missing-vjp", "routing", "cleanup", "missing-boundary"])
def test_singular_metrics_do_not_relax_executed_technical_contracts(
    tmp_path: Path, mutation: str,
) -> None:
    _, result = _receipt(tmp_path, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    row = published["passes"]["fp8_oracle"]
    if mutation == "missing-vjp":
        row["oracle"]["counts"]["oracle_vjp"] -= 1
    elif mutation == "routing":
        row["routing_guard"] = "disabled"
    elif mutation == "cleanup":
        row["oracle_after_close"]["owned_device_storage_bytes"] = 1
    else:
        (tmp_path / "fp8_oracle/recompute/layer-000-mb-000.pt").unlink()
    _commit(tmp_path, published)
    with pytest.raises(ValueError, match="VJP|routing|inventory"):
        _verify(manifest)


def test_singular_metrics_require_complete_fp8_store_cleanup(tmp_path: Path) -> None:
    _, result = _receipt(tmp_path, verdict="PASS", production=True)
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
    _verify(manifest)
    cases = [
        ("store_after_close", "missing"), ("cleanup", "missing"), ("closed", "missing"),
        ("closed", "false"),
        *[(field, mode) for field in (
            "owned_host_storage_bytes", "owned_device_storage_bytes", "charged_host_bytes",
        ) for mode in ("missing", "retained")],
        ("charged_host_bytes", "boolean"),
    ]
    for field, mode in cases:
        changed = copy.deepcopy(published)
        row = changed["passes"]["fp8_oracle"]
        target = (row if field == "store_after_close" else
                  row["store_after_close"]["cleanup"] if field == "closed" else
                  row["store_after_close"])
        if mode == "missing":
            del target[field]
        else:
            target[field] = 1 if mode == "retained" else False
        _commit(tmp_path, changed)
        with pytest.raises(ValueError, match="FP8 store cleanup"):
            gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
        with pytest.raises(ValueError, match="FP8 store cleanup"):
            _verify(manifest)


def test_singular_metrics_refuse_booleans_in_numeric_reconstructions(tmp_path: Path) -> None:
    _, result = _receipt(tmp_path, verdict="PASS", production=True)
    names = _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
    _verify(manifest)
    for location in ("zero-cosine", "unit-cosine", "optimizer-updates", "observations"):
        changed = copy.deepcopy(published)
        if location.endswith("cosine"):
            name = names[0] if location == "zero-cosine" else changed["trainable_names"][1]
            target = changed["passes"]["fp8_oracle"]["gradient_agreement"]["per_tensor"][name]
            expected = 0.0 if location == "zero-cosine" else 1.0
            assert target["cosine"] == expected and type(target["cosine"]) is float
            target["cosine"] = bool(expected)
        else:
            field = ("optimizer_updates" if location == "optimizer-updates"
                     else "comparison_observations")
            assert type(changed["a3"][field]) is int
            changed["a3"][field] = bool(changed["a3"][field])
        _commit(tmp_path, changed)
        with pytest.raises(ValueError, match="A3 raw tensor recomputation|E3 raw tensor metrics"):
            gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
        with pytest.raises(ValueError, match="A3 raw tensor recomputation|E3 raw tensor metrics"):
            _verify(manifest)


@pytest.mark.parametrize("location", ["a3", "boundary", "gradient"])
def test_replay_rejects_unverified_members_even_inside_derived_metric_reports(
    tmp_path: Path, location: str,
) -> None:
    _, result = _receipt(tmp_path, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    if location == "a3":
        target = result["a3"]["rule"]
    elif location == "boundary":
        target = result["passes"]["fp8_oracle"]["layers"][0]["restore"]
    else:
        target = result["passes"]["fp8_oracle"]["gradient_agreement"]
    target["unverified_metric"] = math.inf
    manifest, published = _publish(tmp_path, result)
    assert any(path.endswith(".unverified_metric")
               for path in published["nonfinite_evidence_fields"])
    with pytest.raises(ValueError, match="recomputation|metrics differ"):
        _verify(manifest)


def test_legacy_deepseek_still_refuses_zero_reference_derived_nonfinites(tmp_path: Path) -> None:
    _, result = _receipt(tmp_path, profile_id=DEEPSEEK, verdict="PASS")
    _zero_reference(tmp_path, result, all_tensors=False)
    _refresh_metrics(tmp_path, result)
    manifest, published = _publish(tmp_path, result)
    assert published["verdict"] == "NO_VERDICT"
    assert published["nonfinite_evidence_fields"]
    with pytest.raises(ValueError, match="nonfinite"):
        _verify(manifest, profile_id=DEEPSEEK)
    with pytest.raises(ValueError, match="PASS"):
        gate.load_gate(manifest, _digest(manifest), profile_id=DEEPSEEK)
