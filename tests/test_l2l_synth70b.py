"""CPU tests for the A5 ticket 05 harness: the synthetic 70B L2L run (l2l_synth70b.py).

The run uses random weights of the 70B shape, so it checks mechanics and memory, not quality
or speed of a real model. These tests pin the parts a rented GPU cannot check by itself: the
layer shapes, the size of the activation store, the resource plan, the refusal when RAM or
disk is short, and the report that must label every table and heading.
"""

from __future__ import annotations

import copy
import json

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def synth():
    return _load("l2l_synth70b")


GIB = 2**30


# --------------------------------------------------------------------------
# layer shapes
# --------------------------------------------------------------------------
def test_layer_shapes_are_the_llama_70b_decoder_layer(synth):
    shapes = synth.layer_shapes(3)
    prefix = "model.layers.3."
    assert shapes[prefix + "self_attn.q_proj.weight"] == (8192, 8192)
    assert shapes[prefix + "self_attn.k_proj.weight"] == (1024, 8192)
    assert shapes[prefix + "self_attn.v_proj.weight"] == (1024, 8192)
    assert shapes[prefix + "self_attn.o_proj.weight"] == (8192, 8192)
    assert shapes[prefix + "mlp.gate_proj.weight"] == (28672, 8192)
    assert shapes[prefix + "mlp.up_proj.weight"] == (28672, 8192)
    assert shapes[prefix + "mlp.down_proj.weight"] == (8192, 28672)
    assert shapes[prefix + "input_layernorm.weight"] == (8192,)
    assert len(shapes) == 9


def test_the_model_has_80_layers_and_about_69_billion_parameters(synth):
    assert synth.SHAPE["layers"] == 80
    assert synth.layer_param_count() == 855_654_400
    assert 68e9 < synth.total_params() < 70e9


def test_the_shape_matches_the_checkpoint_writer(synth):
    writer = _load("synth_checkpoint")
    assert synth.layer_shapes(0) == writer.layer_tensors(synth.SHAPE, 0)


# --------------------------------------------------------------------------
# activation store size
# --------------------------------------------------------------------------
def test_one_chunk_at_seq_512_is_8_mib_and_needs_no_rounding(synth):
    out = synth.activation_store(k=4, batch=1, seq=512)
    assert out["chunk_bytes"] == 8 * 2**20
    assert out["block_bytes"] == out["chunk_bytes"]


def test_store_at_k4_seq_512_is_two_and_a_half_gib(synth):
    out = synth.activation_store(k=4, batch=1, seq=512)
    assert out["logical_bytes"] == 80 * 4 * 8 * 2**20
    assert out["pinned_bytes"] == out["logical_bytes"]
    assert out["pinned_bytes"] / GIB == 2.5


def test_a_pinned_chunk_rounds_up_to_a_power_of_two(synth):
    out = synth.activation_store(k=2, batch=1, seq=300)
    assert out["chunk_bytes"] == 300 * 8192 * 2
    assert out["block_bytes"] == 8 * 2**20
    assert out["pinned_bytes"] == 80 * 2 * out["block_bytes"]
    assert out["logical_bytes"] == 80 * 2 * out["chunk_bytes"]


def test_the_store_grows_linearly_with_k(synth):
    one = synth.activation_store(k=1, batch=1, seq=256)["pinned_bytes"]
    assert synth.activation_store(k=7, batch=1, seq=256)["pinned_bytes"] == 7 * one


@pytest.mark.parametrize("bad", [0, -1])
def test_k_below_one_is_refused(synth, bad):
    with pytest.raises(ValueError):
        synth.activation_store(k=bad, batch=1, seq=512)


# --------------------------------------------------------------------------
# weight bytes and the resource plan
# --------------------------------------------------------------------------
def test_bf16_source_is_two_bytes_per_parameter(synth):
    out = synth.weight_bytes()
    assert out["source_bf16"] == 2 * synth.total_params()
    assert 136e9 < out["source_bf16"] < 140e9


def test_nf4_cache_is_about_a_quarter_of_the_source(synth):
    out = synth.weight_bytes()
    assert 0.25 < out["nf4_cache"] / out["source_bf16"] < 0.30


def test_plan_counts_source_cache_and_margin_on_disk(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    disk = plan["disk"]
    assert disk["source_bf16"] == synth.weight_bytes()["source_bf16"]
    assert disk["nf4_cache"] == synth.weight_bytes()["nf4_cache"]
    assert disk["spill"] == 0
    assert disk["total"] >= disk["source_bf16"] + disk["nf4_cache"]
    assert disk["total"] == pytest.approx(
        (disk["source_bf16"] + disk["nf4_cache"] + disk["spill"]) * synth.DISK_MARGIN
    )


def test_plan_disk_tier_asks_ram_for_activations_only(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    ram = plan["ram"]
    assert ram["weights"] == 0
    store = synth.activation_store(k=4, batch=1, seq=512)["pinned_bytes"]
    assert ram["activation_need"] == pytest.approx(store / 0.7)
    assert ram["total"] == ram["activation_need"] + ram["workspace"]


def test_plan_ram_tier_adds_the_nf4_weights_to_ram(synth):
    disk = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    ram = synth.resource_plan(k=4, batch=1, seq=512, tier="ram")
    assert ram["ram"]["weights"] == synth.weight_bytes()["nf4_cache"]
    assert ram["ram"]["total"] == disk["ram"]["total"] + ram["ram"]["weights"]


def test_plan_spill_adds_the_spilled_layers_to_disk_and_drops_them_from_ram(synth):
    plain = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    spilled = synth.resource_plan(k=4, batch=1, seq=512, tier="disk", spill_layers=40)
    chunk = 8 * 2**20
    assert spilled["disk"]["spill"] == 40 * 4 * chunk
    assert spilled["ram"]["activation_need"] == pytest.approx(
        plain["ram"]["activation_need"] / 2
    )


def test_plan_refuses_spill_layers_beyond_the_model(synth):
    with pytest.raises(ValueError):
        synth.resource_plan(k=4, batch=1, seq=512, tier="disk", spill_layers=81)


def test_plan_refuses_an_unknown_tier(synth):
    with pytest.raises(ValueError):
        synth.resource_plan(k=4, batch=1, seq=512, tier="tape")


# --------------------------------------------------------------------------
# refusal on short RAM or disk
# --------------------------------------------------------------------------
def test_check_passes_when_both_fit(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    synth.check_resources(
        plan, free_ram=plan["ram"]["total"] + 1, free_disk=plan["disk"]["total"] + 1
    )


def test_check_refuses_short_ram_and_names_the_numbers(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    with pytest.raises(synth.SynthRefusedError) as err:
        synth.check_resources(plan, free_ram=GIB, free_disk=plan["disk"]["total"] * 2)
    text = str(err.value)
    assert "RAM" in text and "disk" not in text.lower()
    assert f"{plan['ram']['total'] / GIB:.1f}" in text


def test_check_refuses_short_disk(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    with pytest.raises(synth.SynthRefusedError) as err:
        synth.check_resources(plan, free_ram=plan["ram"]["total"] * 2, free_disk=100 * 10**9)
    assert "disk" in str(err.value).lower()
    assert f"{plan['disk']['total'] / GIB:.1f}" in str(err.value)


def test_check_reports_both_shortages_together(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    with pytest.raises(synth.SynthRefusedError) as err:
        synth.check_resources(plan, free_ram=1, free_disk=1)
    text = str(err.value)
    assert "RAM" in text and "disk" in text.lower()


@pytest.mark.parametrize("ram,disk", [(None, 10**13), (10**13, None)])
def test_check_refuses_an_unknown_amount(synth, ram, disk):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    with pytest.raises(synth.SynthRefusedError):
        synth.check_resources(plan, free_ram=ram, free_disk=disk)


def test_check_accepts_exactly_the_planned_amount(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=512, tier="disk")
    synth.check_resources(
        plan, free_ram=plan["ram"]["total"], free_disk=plan["disk"]["total"]
    )


# --------------------------------------------------------------------------
# price
# --------------------------------------------------------------------------
def test_rental_cost_is_rate_times_hours(synth):
    assert synth.rental_cost(rate_per_hour=0.5, hours=3.0) == 1.5


@pytest.mark.parametrize("rate,hours", [(-0.1, 1), (0.5, 0), (0.5, -1)])
def test_rental_cost_refuses_nonsense(synth, rate, hours):
    with pytest.raises(ValueError):
        synth.rental_cost(rate_per_hour=rate, hours=hours)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def _run(**override):
    run = {
        "steps": 3, "step_s_mean": 41.5, "real_tokens": 3000, "tokens_per_s": 24.1,
        "peak_vram_allocated_bytes": 9_000_000_000, "peak_vram_reserved_bytes": 10_000_000_000,
        "peak_rss_bytes": 30_000_000_000, "pinned_store_bytes": 2_684_354_560,
        "os_read_bytes": 5_000_000_000_000, "os_write_bytes": 1_000,
        "layer_loads": 960, "nf4_restores": 0, "large_loads": 0, "layer_loads_per_step": 320.0,
    }
    run.update(override)
    return run


def _payload(**override):
    payload = {
        "model": "synthetic-70b",
        "run": _run(),
        "weight_read_bytes": 4_900_000_000_000,
        "spill_written_bytes": 0,
        "spill_read_bytes": 0,
        "plan": None,
        "rental": {"instance": 1, "offer": 2, "rate_per_hour": 0.4, "hours": 3.0,
                   "cost_usd": 1.2, "budget_usd": 8.0, "destroyed": True},
        "excluded": [],
        "logs": ["runs/a.log"],
        "gpu": {"name": "RTX 3090"},
    }
    payload.update(override)
    return payload


def test_report_labels_every_heading(synth):
    text = synth.render_report(_payload())
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings
    assert all(synth.LABEL in line for line in headings)


def test_report_puts_the_label_above_every_table(synth):
    lines = synth.render_report(_payload()).splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|"))]
    assert starts
    for i in starts:
        assert synth.LABEL in lines[i - 2], lines[i - 2:i + 1]


def test_report_records_the_measured_fields(synth):
    text = synth.render_report(_payload())
    for needle in ("VRAM", "RSS", "закреплённая", "чтение весов", "спилл", "загрузок слоя",
                   "время шага"):
        assert needle in text, needle


def test_report_says_no_claim_about_a_real_model(synth):
    text = synth.render_report(_payload())
    assert "не доказательство" in text
    assert "качества" in text and "скорости" in text


def test_report_refuses_a_7b_record(synth):
    with pytest.raises(synth.MixedRecordsError):
        synth.render_report(_payload(model="mistral-7b-nf4"))


def test_report_refuses_a_record_without_a_model(synth):
    payload = _payload()
    del payload["model"]
    with pytest.raises(synth.MixedRecordsError):
        synth.render_report(payload)


def test_report_refuses_extrapolation_fields(synth):
    with pytest.raises(synth.MixedRecordsError):
        synth.render_report(_payload(extrapolated_1t={"step_s": 1.0}))


def test_report_prints_null_with_its_reason_not_zero(synth):
    payload = _payload(weight_read_bytes=None,
                       unavailable={"weight_read_bytes": "container hides disk reads"})
    text = synth.render_report(payload)
    assert "container hides disk reads" in text
    assert "| чтение весов | 0 " not in text


def test_report_refuses_null_without_a_reason(synth):
    with pytest.raises(ValueError):
        synth.render_report(_payload(weight_read_bytes=None))


def test_report_lists_excluded_f1_runs_and_logs(synth):
    payload = _payload(excluded=[{"run": "r2", "reason": "F1 void: suspend"}])
    text = synth.render_report(payload)
    assert "r2" in text and "F1 void: suspend" in text
    assert "runs/a.log" in text


def test_report_states_instance_destroyed(synth):
    text = synth.render_report(_payload())
    assert "уничтожен" in text
    open_payload = _payload()
    open_payload["rental"] = dict(open_payload["rental"], destroyed=False)
    assert "НЕ подтверждено" in synth.render_report(open_payload)


def test_report_does_not_mutate_its_input(synth):
    payload = _payload()
    before = copy.deepcopy(payload)
    synth.render_report(payload)
    assert payload == before


# --------------------------------------------------------------------------
# F1: spoiled runs never reach the table
# --------------------------------------------------------------------------
def test_split_runs_drops_spoiled_runs_and_keeps_the_reason(synth):
    entries = [
        {"name": "r1", "spoiled": False, "unverified": False, "void_reasons": {}},
        {"name": "r2", "spoiled": True, "unverified": False,
         "void_reasons": {"suspend": "machine slept"}},
        {"name": "r3", "spoiled": False, "unverified": True, "void_reasons": {}},
    ]
    kept, excluded = synth.split_runs(entries)
    assert [e["name"] for e in kept] == ["r1", "r3"]
    assert excluded == [{"run": "r2", "reason": "suspend: machine slept"}]


# --------------------------------------------------------------------------
# physical RAM ceiling of the RAM tier (review finding)
# --------------------------------------------------------------------------
def test_ram_tier_refuses_weights_above_55_percent_of_physical_ram(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=256, tier="ram")
    total = int(plan["ram"]["weights"] / 0.55) - 1
    with pytest.raises(synth.SynthRefusedError) as err:
        synth.check_resources(
            plan, free_ram=10**13, free_disk=10**14, total_ram=total
        )
    assert "55%" in str(err.value)


def test_ram_tier_passes_when_weights_are_within_the_ceiling(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=256, tier="ram")
    total = int(plan["ram"]["weights"] / 0.55) + 1
    synth.check_resources(plan, free_ram=10**13, free_disk=10**14, total_ram=total)


def test_disk_tier_ignores_the_physical_ceiling(synth):
    plan = synth.resource_plan(k=4, batch=1, seq=256, tier="disk")
    synth.check_resources(plan, free_ram=10**13, free_disk=10**14, total_ram=1)


# --------------------------------------------------------------------------
# build_record: the arm's run.json into the report payload
# --------------------------------------------------------------------------
def _arm_json(**override):
    arm = {
        "failure": None,
        "gpu": {"name": "RTX 3090"},
        "run": _run(os_read_bytes=7_000_000_000_000),
        "steps": [],
    }
    arm.update(override)
    return arm


def test_build_record_sets_the_model_and_reads_weights_from_the_run(synth):
    record = synth.build_record(_arm_json(), plan=None, rental=None, logs=["a.log"], entries=[])
    assert record["model"] == synth.MODEL_KIND
    assert record["weight_read_bytes"] == 7_000_000_000_000
    assert record["run"]["tokens_per_s"] == 24.1
    assert record["logs"] == ["a.log"]


def test_build_record_marks_spill_unavailable_with_a_reason(synth):
    record = synth.build_record(_arm_json(), plan=None, rental=None, logs=[], entries=[])
    assert record["spill_written_bytes"] is None and record["spill_read_bytes"] is None
    assert "not wired" in record["unavailable"]["spill_written_bytes"]
    synth.render_report(record)


def test_build_record_takes_excluded_runs_from_f1_entries(synth):
    entries = [{"name": "r2", "spoiled": True, "void_reasons": {"suspend": "slept"}}]
    record = synth.build_record(_arm_json(), plan=None, rental=None, logs=[], entries=entries)
    assert record["excluded"] == [{"run": "r2", "reason": "suspend: slept"}]


def test_build_record_fills_the_cost_from_rate_and_hours(synth):
    rental = {"instance": 1, "offer": 2, "rate_per_hour": 0.4, "hours": 2.5,
              "budget_usd": 8.0, "destroyed": True}
    record = synth.build_record(_arm_json(), plan=None, rental=rental, logs=[], entries=[])
    assert record["rental"]["cost_usd"] == 1.0


def test_build_record_refuses_a_failed_run(synth):
    with pytest.raises(ValueError, match="failed"):
        synth.build_record(_arm_json(failure="OOM"), plan=None, rental=None, logs=[], entries=[])


def test_build_record_refuses_an_arm_without_a_run_record(synth):
    with pytest.raises(ValueError, match="run"):
        synth.build_record(_arm_json(run=None), plan=None, rental=None, logs=[], entries=[])


def test_report_prints_notes_under_a_labelled_heading(synth):
    text = synth.render_report(_payload(notes=["F1 checks were not run"]))
    assert "F1 checks were not run" in text
    heading = [line for line in text.splitlines() if "Оговорки" in line]
    assert heading and synth.LABEL in heading[0]


def test_report_without_f1_checks_does_not_claim_no_spoiled_runs(synth):
    text = synth.render_report(_payload(f1_checked=False))
    assert "Испорченных по F1 замеров нет" not in text
    assert "F1 не проверялся" in text


def test_campaign_runs_the_same_l2l_config_in_both_arms_and_watches_the_synthetic_base(tmp_path):
    camp = _load("l2l_synth70b_campaign")
    work, out = tmp_path / "w", tmp_path / "o"
    work.mkdir()
    out.mkdir()
    campaign = camp.Campaign(work, out, "24gb", str(tmp_path / "synth-llama-70b"))
    assert set(campaign.arm_config.values()) == {"l2l.yaml"}
    assert (work / "l2l.yaml").is_file() and not (work / "ga.yaml").exists()
    assert campaign.model_path.endswith("synth-llama-70b")
    assert "synthetic 70B" in campaign.label


def test_campaign_summary_is_speed_and_memory_without_a_loss_verdict(tmp_path):
    camp = _load("l2l_synth70b_campaign")
    work, out = tmp_path / "w", tmp_path / "o"
    work.mkdir()
    out.mkdir()
    campaign = camp.Campaign(work, out, "24gb", str(tmp_path))
    campaign.write_summary()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["model"] == "synthetic 70B" and "loss" not in summary
