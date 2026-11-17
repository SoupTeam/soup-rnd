"""Independent-review additions: pin the attempt018 gates whose removal no existing test detected.

Each test targets one mutation that survived the review's scoped mutation run (CPU, synthetic).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
sys.path.insert(0, str(Path(__file__).parent))
import h2_ce_confirmation as runner  # noqa: E402
import h2_ce_ref64 as ce  # noqa: E402
import h2_ce_ref64_evidence as gates  # noqa: E402
import h2_ce_ref64_review as review  # noqa: E402
from test_h2_ce_ref64_evidence import reseal, sealed_tree, write  # noqa: E402

torch = pytest.importorskip("torch")


def _binding_fixture(tmp_path, monkeypatch):
    spec = tmp_path / "spec"
    spec.mkdir()
    monkeypatch.setattr(runner, "SPEC", spec)
    before = {"trainable": {"x.lora_A": {}, "x.lora_B": {}}}
    write(
        spec / "operator_study.json",
        {"reference_identity": {"before_sha256": gates.identity(before)}},
    )
    batches = [
        {"input_ids": [[10 + i, 2, 3]], "labels": [[-100, 2, 3]], "attention_mask": [[1, 1, 1]]}
        for i in range(4)
    ]
    write(spec / "ce_confirmation_workset.json", {"probes": batches})

    def values(index):
        out = {"input." + k: torch.tensor(v) for k, v in batches[index].items()}
        out.update({"gradient." + k: torch.zeros(1) for k in before["trainable"]})
        return out

    return {"compute_policy": "bf16-autocast-cuda", "before": before}, values


def test_probe_is_bound_to_its_own_workset_entry(tmp_path, monkeypatch):
    """Distinct frozen probes: probe k and perturbation child 0k must carry entry k only."""
    data, values = _binding_fixture(tmp_path, monkeypatch)
    for phase, key, index in [
        ("baseline_probes", "probe_01", 1),
        ("ref64_probes", "probe_03", 3),
        ("ref64_perturbed_probe02", "pass_4_1", 2),
        ("ref64_perturbed_probe03", "pass_0_0", 3),
    ]:
        path = tmp_path / phase / "runner" / key / "model_probe"
        gates.validate_probe(path, data, values(index))
        for other in {0, 1, 2, 3} - {index}:
            with pytest.raises(ValueError, match="frozen workset"):
                gates.validate_probe(path, data, values(other))


@pytest.mark.parametrize("fault", ["nan", "ignored_row"])
def test_review_runs_the_dlogits_gate(tmp_path, monkeypatch, fault):
    from safetensors.torch import load_file, save_file

    root = sealed_tree(tmp_path, monkeypatch)
    child = root / "ref64_probes" / "runner"
    path = child / "hooks" / "probe_01.safetensors"
    dlogits = load_file(str(path))["dlogits"].clone()
    if fault == "nan":
        dlogits.view(-1)[0] = float("nan")
    else:
        dlogits[..., -1, 0] = 1.0  # the last position has no next-token label after the shift
    save_file({"dlogits": dlogits}, str(path))
    reseal(child)
    result = review.review_ref64(root)
    assert result["label"] == result["run_status"] == "VOID", result
    assert "dlogits" in " ".join(result["reasons"])


@pytest.mark.parametrize("fault", ["level", "usable_below_count", "usable_above_total"])
def test_flip_record_fields_are_gated(tmp_path, monkeypatch, fault):
    root = sealed_tree(tmp_path, monkeypatch)
    child = root / "ref64_perturbed_probe03" / "runner"
    data = json.loads((child / "evidence.json").read_text())
    item = data["ref64"]["flips"][5]
    if fault == "level":
        item["level"] = ce.GRID[(5 // ce.SEEDS_PER_LEVEL) + 1]
    elif fault == "usable_below_count":
        item["eligible_usable"] = item["flipped"] - 1
    else:
        item["eligible_usable"] = item["eligible_total"] + 1
    write(child / "evidence.json", data)
    reseal(child)
    result = review.review_ref64(root)
    assert result["label"] == result["run_status"] == "VOID", result


def test_source_pin_mismatch_alone_is_void(tmp_path, monkeypatch):
    """All contract copies agree; only a pinned artifact hash differs from the source."""
    root = sealed_tree(tmp_path, monkeypatch)
    contract = json.loads((root / "contract.json").read_text())
    contract["artifacts_sha256"]["benchmarks/harness/h2_ce_ref64_review.py"] = "0" * 64
    write(root / "contract.json", contract)
    for phase in runner.REF64_PHASES:
        child = root / phase / "runner"
        write(child / "contract.json", contract)
        reseal(child)
    result = review.review_ref64(root)
    assert result["label"] == result["run_status"] == "VOID", result
    assert "authority" in " ".join(result["reasons"])


def test_child_evidence_changed_between_reads_is_void(tmp_path, monkeypatch):
    root = sealed_tree(tmp_path, monkeypatch)
    original = runner.read_child

    def changed(child, phase):
        data = dict(original(child, phase))
        if phase == "ce_only_probes":
            data["status"] = "edited"
        return data

    monkeypatch.setattr(runner, "read_child", changed)
    result = review.review_ref64(root)
    assert result["label"] == result["run_status"] == "VOID", result
    assert "changed during review" in " ".join(result["reasons"])


def test_ce64_scales_by_grad_output():
    cross_entropy = ce.make_cross_entropy(ce.State(50, False))
    target = torch.tensor([3, -100, 7, 49])
    torch.manual_seed(5)
    x = (torch.randn(4, 50) * 3).requires_grad_(True)
    (cross_entropy(x, target) * 2.5).backward()
    ref = x.detach().double().requires_grad_(True)
    (torch.nn.functional.cross_entropy(ref, target, ignore_index=-100) * 2.5).backward()
    assert torch.equal(x.grad, ref.grad.float())
