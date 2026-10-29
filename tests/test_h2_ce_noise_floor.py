"""CPU controls for the attempt016 noise-floor series: phases, contract, review, classes."""

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
sys.path.insert(0, str(Path(__file__).parent))
import h2_ce_confirmation as runner  # noqa: E402
import h2_numerics  # noqa: E402
from h2_gpu_correctness import sha_file  # noqa: E402
from test_h2_ce_confirmation import authority  # noqa: E402

NAMES = ("m.lora_A.w", "m.lora_B.w")
BEFORE = {"trainable": {name: "x" for name in NAMES}}
BEFORE_SHA = hashlib.sha256(
    json.dumps(BEFORE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
WORKSET = json.loads((runner.SPEC / "ce_confirmation_workset.json").read_text())
FOUR = runner.NOISE_FLOOR_PHASES


def test_phase_map_and_series_constants():
    assert runner.PHASES == (
        "baseline_probes",
        "ce_only_probes",
        "baseline_training",
        "ce_only_training",
    )
    assert runner.SERIES["ce_confirmation"] is runner.PHASES
    assert FOUR == (
        "baseline_probes",
        "ce_only_probes",
        "baseline_probes_repeat",
        "ce_only_probes_repeat",
    )
    assert set(runner.PHASE_ARMS) == set(runner.PHASES) | set(FOUR)
    for repeat in ("baseline_probes_repeat", "ce_only_probes_repeat"):
        # A repeat uses exactly the base arm's flags and kind.
        assert runner.PHASE_ARMS[repeat] == runner.PHASE_ARMS[repeat.removesuffix("_repeat")]
    assert runner.PHASE_ARMS["ce_only_probes_repeat"] == ("ce_only", "probes")


def test_contract_series_selection(tmp_path):
    data = authority(tmp_path)
    data["deadline_unix_s"] = data["series_started_unix_s"] + 2400
    assert runner.series_name(data) == "ce_confirmation"
    runner.validate_contract(data)
    data["series"] = "noise_floor"
    runner.validate_contract(data)
    assert runner.series_phases(data) == FOUR
    for bad in ("other", "", None, ["baseline_probes"], list(FOUR), "CE_CONFIRMATION"):
        data["series"] = bad
        with pytest.raises(ValueError, match="Unknown CE series"):
            runner.validate_contract(data)


def test_repeat_phase_refused_under_the_default_series(tmp_path, monkeypatch):
    data = authority(tmp_path)
    for series in (None, "ce_confirmation"):
        contract = dict(data)
        if series:
            contract["series"] = series
        monkeypatch.setattr(runner, "_USED", False)
        with pytest.raises(ValueError, match="declared phase"):
            runner.run_child(contract, "baseline_probes_repeat", tmp_path / "x")
    contract = dict(data, series="noise_floor")
    monkeypatch.setattr(runner, "_USED", False)
    with pytest.raises(ValueError, match="declared phase"):
        runner.run_child(contract, "baseline_training", tmp_path / "y")


@pytest.mark.parametrize("failure", [None, "child", "review_void"])
def test_noise_floor_launch_order_fresh_children_and_void(tmp_path, monkeypatch, failure):
    data = authority(tmp_path)
    data["series"] = "noise_floor"
    data["deadline_unix_s"] = data["series_started_unix_s"] + 2400
    launched, outs = [], []
    monkeypatch.setattr(runner, "validate_contract", lambda value: {})

    def child(argv, out, **kwargs):
        phase = argv[argv.index("--phase") + 1]
        launched.append(phase)
        outs.append(argv[argv.index("--out") + 1])
        assert "--child" in argv
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True)
        (target / "evidence.json").write_text(json.dumps({"status": "COMPLETE_CE_CAPTURE"}))
        return {
            "status": "CHILD_EXITED",
            "group_cleanup_confirmed": True,
            "child_status": {"returncode": 2 if failure == "child" and len(launched) == 3 else 0},
        }

    def review(root, **kwargs):
        raise AssertionError("The per-pair review belongs to the default series only")

    def noise_review(root):
        if failure == "review_void":
            raise ValueError("Child identity differs from authority")
        return {"classification": "PASS"}

    monkeypatch.setattr(runner, "bounded_process", child)
    monkeypatch.setattr(runner, "review_pair", review)
    monkeypatch.setattr(runner, "review_noise_floor", noise_review)
    result = runner.run_series(copy.deepcopy(data), Path(data["evidence_root"]))
    assert launched == list(FOUR[:3] if failure == "child" else FOUR)
    assert len(set(outs)) == len(outs)  # Each phase has its own evidence directory and process.
    if failure is None:
        assert result["status"] == "COMPLETE_CE_NOISE_FLOOR_SERIES"
        assert result["classification"] == "PASS"
    elif failure == "child":
        assert result["status"] == "BLOCKED"
    else:
        assert result["status"] == "VOID" and "identity" in result["error"]


@pytest.mark.parametrize(
    "phase,arm", [("baseline_probes_repeat", "baseline"), ("ce_only_probes_repeat", "ce_only")]
)
def test_read_child_accepts_repeat_phase_with_base_arm_receipts(tmp_path, phase, arm):
    from h2_operator_patch import patch_flags
    from test_h2_ce_controls import good_record

    (tmp_path / "probe_00").mkdir()
    kernel = "liger_cross_entropy_kernel" if arm == "ce_only" else "ordinary_gemm"
    (tmp_path / "probe_00/kernel_trace.json").write_text(
        json.dumps({"traceEvents": [{"cat": "kernel", "name": kernel}]})
    )
    count = int(arm != "baseline")
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "status": "COMPLETE_CE_CAPTURE",
                "patch_after": {
                    "status": "PASS_EXPLICIT_PATCH_BINDINGS",
                    "arm": arm,
                    "flags": patch_flags(arm),
                    "patch_calls": {"class": count, "instance": count},
                },
            }
        )
    )
    controls = good_record()
    controls["phase"] = phase
    (tmp_path / "controls.json").write_text(json.dumps(controls))
    runner.tree_seal(tmp_path)
    assert runner.read_child(tmp_path, phase)["phase"] == phase
    other = "ce_only" if arm == "baseline" else "baseline"
    with pytest.raises(ValueError):  # Wrong arm's patch evidence under this phase is refused.
        evidence = json.loads((tmp_path / "evidence.json").read_text())
        evidence["patch_after"]["arm"] = other
        (tmp_path / "evidence.json").write_text(json.dumps(evidence))
        (tmp_path / "files_sha256.json").unlink()
        runner.tree_seal(tmp_path)
        runner.read_child(tmp_path, phase)


def test_determinism_receipt_is_recording_only_and_complete():
    receipt = runner.determinism_receipt({"attention_implementation": "sdpa"})
    assert set(receipt) == {
        "attention_implementation",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "sdp_flash",
        "sdp_mem_efficient",
        "sdp_math",
        "sdp_cudnn",
        "matmul_allow_tf32",
        "float32_matmul_precision",
    }
    assert receipt["attention_implementation"] == "sdpa"


def test_policy_and_addendum_pins_are_untouched(tmp_path):
    study = json.loads((runner.SPEC / "operator_study.json").read_text())
    assert sha_file(runner.SPEC / "numerical_policy.json") == study["numerical_policy_sha256"]
    assert "benchmarks/h2/CE_NOISE_FLOOR_ADDENDUM.txt" in runner.ARTIFACTS
    assert runner.noise_floor_k() == 3.0 and runner.noise_floor_f() == 2.0**-8


@pytest.mark.parametrize(
    "line,expected",
    [
        ("K = 3.5", 3.5),
        ("K = UNSET", None),
        ("K = -1", None),
        ("K = nan", None),
        ("K = 0", None),
        ("K=3", None),
    ],
)
def test_k_is_read_only_from_a_well_formed_line(tmp_path, monkeypatch, line, expected):
    (tmp_path / "CE_NOISE_FLOOR_ADDENDUM.txt").write_text("text\n" + line + "\nF = 0.5\n")
    monkeypatch.setattr(runner, "SPEC", tmp_path)
    assert runner.noise_floor_k() == expected
    assert runner.noise_floor_f() == 0.5


THRESHOLD = 3 * 2.0**-8  # 0.01171875 when the repeats are bit-identical


@pytest.mark.parametrize(
    "g_cross,g_within,ok,marginal",
    [
        (0.018, 0.0, False, False),  # bit-identical repeats: threshold 0.01172, rejected
        (0.018, 0.006, True, True),  # threshold 0.018 exactly: accepted, marginal
        (0.0181, 0.0, False, False),  # attempt015 value, zero within-arm noise: rejected
        (0.0128, 0.0, False, True),  # attempt015 value: rejected, within 10% of 0.01172
        (0.0, 0.0, True, False),  # G_cross = 0 accepted; the denominator is F, never 0
        (0.0, 0.006, True, False),
        (0.9001 * THRESHOLD, 0.0, True, True),  # just inside the 10% band
        (0.8999 * THRESHOLD, 0.0, True, False),  # just outside the band
        (THRESHOLD, 0.0, True, True),
        (1.0999 * THRESHOLD, 0.0, False, True),  # above the threshold but within 10%: rejected
        (1.1001 * THRESHOLD, 0.0, False, False),
        (0.005, 0.002, True, False),  # small within-arm noise: the floor F still sets 3 * F
    ],
)
def test_equivalence_rule_table_for_k3(g_cross, g_within, ok, marginal):
    verdict = runner.equivalence_for_probe(g_cross, g_within, 3.0, 2.0**-8)
    assert (verdict["ok"], verdict["marginal"]) == (ok, marginal), verdict
    assert verdict["threshold"] == pytest.approx(3.0 * max(g_within, 2.0**-8))


def test_equivalence_rule_refuses_undefined_values():
    for args in ((None, 0.01), (0.01, None)):
        verdict = runner.equivalence_for_probe(*args, 3.0, 2.0**-8)
        assert verdict["ok"] is False and verdict["marginal"] is False


# --- review and classification on synthetic probe tensors with the REAL policy and compare code ---
def grads(index, noise, seed):
    import torch

    generator = torch.Generator().manual_seed(100 + index)
    base = {n: torch.randn(256, generator=generator).abs() + 0.5 for n in NAMES}
    if not noise:
        return base
    other = torch.Generator().manual_seed(seed * 1000 + index)
    return {n: v * (1 + noise * torch.randn(256, generator=other)) for n, v in base.items()}


SCENARIOS = {
    "far_cross": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.058, 1),
        "baseline_probes_repeat": (0.06, 2),
        "ce_only_probes_repeat": (0.058, 1),
    },
    "unequal_within": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.02, 1),
        "baseline_probes_repeat": (0, 0),
        "ce_only_probes_repeat": (0.02, 3),
    },
    "b_clean": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.005, 1),
        "baseline_probes_repeat": (0, 0),
        "ce_only_probes_repeat": (0.005, 1),
    },
    "uniform_scale": {phase: (0, 0) for phase in FOUR},
    "pass": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (1e-3, 1),
        "baseline_probes_repeat": (1e-3, 2),
        "ce_only_probes_repeat": (1e-3, 3),
    },
    "blocked": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.1, 1),
        "baseline_probes_repeat": (0, 0),
        "ce_only_probes_repeat": (0.1, 1),
    },
    "noisy_baseline": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0, 0),
        "baseline_probes_repeat": (0.1, 2),
        "ce_only_probes_repeat": (0, 0),
    },
    "noisy_ce": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0, 0),
        "baseline_probes_repeat": (0, 0),
        "ce_only_probes_repeat": (0.1, 3),
    },
    "all_noisy": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.1, 1),
        "baseline_probes_repeat": (0.1, 2),
        "ce_only_probes_repeat": (0.1, 3),
    },
    "equivalent_but_failing": {
        "baseline_probes": (0, 0),
        "ce_only_probes": (0.022, 1),
        "baseline_probes_repeat": (0.022, 2),
        "ce_only_probes_repeat": (0.022, 3),
    },
}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    contract = authority(tmp_path)  # Built from the real study before SPEC is redirected.
    spec = tmp_path / "spec"
    spec.mkdir()
    for name in (
        "numerical_policy.json",
        "ce_confirmation_workset.json",
        "CE_NOISE_FLOOR_ADDENDUM.txt",
    ):
        shutil.copyfile(runner.SPEC / name, spec / name)
    (spec / "operator_study.json").write_text(
        json.dumps({"reference_identity": {"before_sha256": BEFORE_SHA}})
    )
    monkeypatch.setattr(runner, "SPEC", spec)
    root = tmp_path / "series"
    contract["series"] = "noise_floor"
    root.mkdir()
    (root / "contract.json").write_text(json.dumps(contract))
    for phase in FOUR:
        (root / phase / "runner").mkdir(parents=True)
        (root / phase / "runner/contract.json").write_text(json.dumps(contract))
    state = {
        "scenario": SCENARIOS["pass"],
        "drift": {},
        "loss": {},
        "a_noise": {},
        "scale": {},
        "hook": None,
    }

    def fake_read_child(path, phase):
        evidence = {k: contract[k] for k in ("source_commit", "gpu_uuid", "boot_id")}
        evidence["initial_identity"] = BEFORE_SHA
        evidence.update(state["drift"].get(phase, {}))
        return evidence

    def fake_read_probe(path):
        phase, index = path.parents[2].name, int(path.parent.name.split("_")[1])
        noise, seed = state["scenario"][phase]
        data = {
            "schema": "h2-model-probe-v1",
            "compute_policy": "bf16-autocast-cuda",
            "before": BEFORE,
            "work": WORKSET["probes"][index],
            "loss": state["loss"].get(phase, 1.0),
        }
        values = {"input." + k: torch.tensor(v) for k, v in WORKSET["probes"][index].items()}
        gradient = grads(index, noise, seed)
        if state["a_noise"].get(phase):  # Extra noise on lora_A only: outside the LoRA-B G.
            extra = torch.Generator().manual_seed(777 + index)
            gradient[NAMES[0]] = gradient[NAMES[0]] * (
                1 + state["a_noise"][phase] * torch.randn(256, generator=extra)
            )
        a_scale, b_scale = state["scale"].get(phase, (1.0, 1.0))
        gradient[NAMES[0]] = gradient[NAMES[0]] * a_scale
        gradient[NAMES[1]] = gradient[NAMES[1]] * b_scale
        values.update({"gradient." + n: v for n, v in gradient.items()})
        if state["hook"]:
            state["hook"](phase, index, data, values)
        return data, values

    monkeypatch.setattr(runner, "read_child", fake_read_child)
    monkeypatch.setattr(runner, "read_probe", fake_read_probe)
    monkeypatch.setattr(h2_numerics, "read_probe", fake_read_probe)
    return root, state, contract


@pytest.mark.parametrize(
    "name,expected",
    [
        ("pass", "PASS"),
        ("blocked", "BLOCKED"),
        ("noisy_baseline", "INCONCLUSIVE"),
        ("noisy_ce", "INCONCLUSIVE"),
        ("all_noisy", "INCONCLUSIVE"),
        ("equivalent_but_failing", "INCONCLUSIVE"),
    ],
)
def test_classification_uses_the_unchanged_policy(rig, name, expected):
    root, state, contract = rig
    state["scenario"] = SCENARIOS[name]
    review = runner.review_noise_floor(root)
    assert review["classification"] == expected
    equivalence = review["noise_floor_equivalent"]
    assert equivalence["can_produce_pass"] is False
    if expected == "PASS":
        assert equivalence["status"] == "NOT_EVALUATED_POLICY_PASS"
    else:
        assert equivalence["status"] in {"NOISE_FLOOR_EQUIVALENT", "NOT_EQUIVALENT_UNDER_K"}
    assert review["pairs"]  # Per-tensor metrics of every comparison are kept.
    assert len(review["pairs"]) == 6
    primary = review["primary_pair_passes_policy"]
    within = review["within_arm_pairs_pass_policy"]
    assert (expected == "PASS") == (primary and all(within.values()))
    if name == "noisy_baseline":
        assert primary is True and within["baseline_probes vs baseline_probes_repeat"] is False
    if name == "blocked":
        assert primary is False and all(within.values())
    json.dumps(review)  # The review is serializable evidence.


def test_equivalence_field_never_produces_pass(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["equivalent_but_failing"]
    review = runner.review_noise_floor(root)
    assert review["classification"] == "INCONCLUSIVE"  # Policy failure stays a failure.
    equivalence = review["noise_floor_equivalent"]
    assert equivalence["status"] == "NOISE_FLOOR_EQUIVALENT", equivalence["probes"]
    assert equivalence["marginal"] is False and equivalence["can_produce_pass"] is False
    assert all(
        p["threshold"] == pytest.approx(3 * max(p["g_within"], 2.0**-8))
        for p in equivalence["probes"]
    )
    state["scenario"] = SCENARIOS["blocked"]  # Bit-identical repeats: threshold 3 * F only.
    review = runner.review_noise_floor(root)
    assert review["classification"] == "BLOCKED"
    assert review["noise_floor_equivalent"]["status"] == "NOT_EQUIVALENT_UNDER_K"


def test_equivalence_needs_loss_agreement_across_arms(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["equivalent_but_failing"]
    state["loss"] = {
        "ce_only_probes": 2.0,
        "ce_only_probes_repeat": 2.0,
    }  # Cross pairs differ by 1.
    equivalence = runner.review_noise_floor(root)["noise_floor_equivalent"]
    assert equivalence["status"] == "NOT_EQUIVALENT_UNDER_K"
    assert not any(p["losses_ok"] for p in equivalence["probes"])


def test_equivalence_needs_tensor_cosine_even_when_lora_b_g_is_small(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["equivalent_but_failing"]
    state["a_noise"] = {"ce_only_probes_repeat": 0.3}  # lora_A cosine drops below 0.999
    equivalence = runner.review_noise_floor(root)["noise_floor_equivalent"]
    assert equivalence["status"] == "NOT_EQUIVALENT_UNDER_K"
    assert all(p["g_cross"] <= p["threshold"] for p in equivalence["probes"])  # G alone would pass
    assert not any(p["cosines_ok"] for p in equivalence["probes"])


def test_equivalence_not_evaluated_when_k_or_f_unset(rig, monkeypatch):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["equivalent_but_failing"]
    monkeypatch.setattr(runner, "noise_floor_f", lambda: None)
    equivalence = runner.review_noise_floor(root)["noise_floor_equivalent"]
    assert equivalence["status"] == "NOT_EVALUATED_K_OR_F_UNSET"


@pytest.mark.parametrize(
    "phase,fault",
    [
        ("baseline_probes_repeat", {"gpu_uuid": "GPU-other"}),
        ("ce_only_probes_repeat", {"boot_id": "other-boot"}),
        ("baseline_probes_repeat", {"source_commit": "b" * 40}),
        ("ce_only_probes_repeat", {"initial_identity": "0" * 64}),
    ],
)
def test_identity_drift_is_void_not_a_classification(rig, phase, fault):
    root, state, contract = rig
    state["drift"][phase] = fault
    with pytest.raises(ValueError, match="identity|initial state"):
        runner.review_noise_floor(root)


def test_changed_input_or_state_between_children_raises(rig):
    root, state, contract = rig
    original = runner.read_probe

    def tampered(path):
        data, values = original(path)
        if path.parents[2].name == "baseline_probes_repeat":
            values["input.input_ids"] = values["input.input_ids"] + 1
        return data, values

    runner.read_probe = tampered
    try:
        with pytest.raises(ValueError, match="frozen workset"):
            runner.review_noise_floor(root)
    finally:
        runner.read_probe = original


def test_missing_repeat_evidence_is_refused(rig):
    root, state, contract = rig
    shutil.rmtree(root / "ce_only_probes_repeat")
    with pytest.raises((ValueError, OSError)):
        runner.review_noise_floor(root)


def test_noise_review_refuses_a_default_series_contract(rig):
    root, state, contract = rig
    contract.pop("series")
    (root / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Not a noise-floor series"):
        runner.review_noise_floor(root)


def test_strict_review_still_raises_on_policy_failure(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["blocked"]
    with pytest.raises(ValueError, match="Probe comparison failed"):
        runner.review_pair(root, training=False)
    report = runner.review_pair(root, training=False, phases=runner.NOISE_PRIMARY, strict=False)
    assert report["status"] == "FAIL_PROBE_POLICY" and report["failures"]


def lora_b_g(index, scenario, reference, other):
    """Independent G: FP64 relative L2 of the lora_B gradient, first-named run as reference."""
    first = grads(index, *scenario[reference])[NAMES[1]].double()
    second = grads(index, *scenario[other])[NAMES[1]].double()
    return float((first - second).norm() / first.norm())


def test_g_cross_is_the_largest_cross_pair_and_g_within_the_largest_within_pair(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["far_cross"]
    equivalence = runner.review_noise_floor(root)["noise_floor_equivalent"]
    for probe in equivalence["probes"]:
        i = probe["probe"]
        cross = [
            lora_b_g(i, SCENARIOS["far_cross"], a, c)
            for a, c in [
                ("baseline_probes", "ce_only_probes"),
                ("baseline_probes", "ce_only_probes_repeat"),
                ("baseline_probes_repeat", "ce_only_probes"),
                ("baseline_probes_repeat", "ce_only_probes_repeat"),
            ]
        ]
        within = [
            lora_b_g(i, SCENARIOS["far_cross"], "baseline_probes", "baseline_probes_repeat"),
            lora_b_g(i, SCENARIOS["far_cross"], "ce_only_probes", "ce_only_probes_repeat"),
        ]
        assert max(cross) > 1.03 * min(cross) and max(within) > 1.03 * min(within)
        assert probe["g_cross"] == pytest.approx(max(cross), rel=1e-6)
        assert probe["g_within"] == pytest.approx(max(within), rel=1e-6)


def test_unequal_within_pairs_use_the_larger_one(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["unequal_within"]
    equivalence = runner.review_noise_floor(root)["noise_floor_equivalent"]
    for probe in equivalence["probes"]:
        small, large = sorted(probe["within_pairs"].values())
        assert small == 0.0 and large > 0.02
        assert probe["g_within"] == pytest.approx(large)
        assert probe["threshold"] == pytest.approx(3.0 * large)  # not 3 * F from the zero pair
    assert equivalence["status"] == "NOISE_FLOOR_EQUIVALENT"


def test_noisy_lora_a_does_not_enter_g_but_lora_b_clean_stays_equivalent(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["b_clean"]
    state["a_noise"] = {"ce_only_probes": 0.03, "ce_only_probes_repeat": 0.03}
    review = runner.review_noise_floor(root)
    assert review["classification"] == "BLOCKED"  # lora_A differs beyond the policy
    equivalence = review["noise_floor_equivalent"]
    assert all(p["g_cross"] < 3 * 2.0**-8 for p in equivalence["probes"])  # lora_B only
    assert all(p["cosines_ok"] for p in equivalence["probes"])
    assert equivalence["status"] == "NOISE_FLOOR_EQUIVALENT"


def test_g_uses_the_baseline_run_as_reference(rig):
    root, state, contract = rig
    state["scenario"] = SCENARIOS["uniform_scale"]
    # C = 1.0118 * A on lora_B: G(A, C) = 0.0118 > 3F = 0.01172, but G(C, A) = 0.01166 would pass.
    state["scale"] = {
        "ce_only_probes": (1.1, 1.0118),
        "ce_only_probes_repeat": (1.1, 1.0118),
    }
    review = runner.review_noise_floor(root)
    assert review["classification"] == "BLOCKED"
    equivalence = review["noise_floor_equivalent"]
    assert all(p["g_cross"] == pytest.approx(0.0118, rel=1e-4) for p in equivalence["probes"])
    assert equivalence["status"] == "NOT_EQUIVALENT_UNDER_K"


@pytest.mark.parametrize(
    "defect",
    ["drop_tensor", "extra_tensor", "nonfinite_loss", "missing_loss", "state_changed", "dtype"],
)
def test_evidence_defects_raise_instead_of_becoming_a_classification(rig, defect):
    import torch

    root, state, contract = rig
    state["scenario"] = SCENARIOS["pass"]

    def hook(phase, index, data, values):
        if phase != "baseline_probes_repeat" or index != 0:
            return
        if defect == "drop_tensor":
            del values["gradient." + NAMES[0]]
        elif defect == "extra_tensor":
            values["gradient.extra"] = torch.ones(4)
        elif defect == "nonfinite_loss":
            data["loss"] = float("nan")
        elif defect == "missing_loss":
            data["loss"] = None
        elif defect == "state_changed":
            data["before"] = {"trainable": {NAMES[0]: "x"}}
        elif defect == "dtype":
            values["gradient." + NAMES[1]] = values["gradient." + NAMES[1]].to(torch.float64)

    state["hook"] = hook
    with pytest.raises(ValueError):
        runner.review_noise_floor(root)


def test_run_child_and_read_child_use_the_phase_map_not_suffixes():
    import ast
    import inspect

    for function in (runner.run_child, runner.read_child):
        tree = ast.parse(inspect.getsource(function).lstrip())
        subscripts = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == "PHASE_ARMS"
        ]
        assert subscripts, function.__name__
        assert not [
            n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "removesuffix"
        ], function.__name__


@pytest.mark.parametrize("phase", sorted(set(FOUR) | set(runner.PHASES)))
def test_command_line_accepts_every_declared_phase(tmp_path, monkeypatch, phase):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"series": "noise_floor"}))
    seen = []

    def fake_run_child(contract_data, name, out):
        seen.append(name)
        return {"status": "COMPLETE_CE_CAPTURE"}

    monkeypatch.setattr(runner, "run_child", fake_run_child)
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--contract", str(contract), "--child", "--phase", phase, "--out", str(tmp_path)],
    )
    runner.main()
    assert seen == [phase]


def test_command_line_refuses_an_unknown_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--contract", "c", "--child", "--phase", "ce_only_probes_again", "--out", "o"],
    )
    with pytest.raises(SystemExit):
        runner.main()
