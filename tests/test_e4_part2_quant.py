"""E4 part 2, Mac-side quantization (benchmarks/harness/e4_part2_quant.py), on a tiny OLMoE.

The tiny model is saved in the original per-expert checkpoint layout, so the run exercises
the same load, fuse, slice, quantize, write-back and verify path as the real 3,136 units.
"""

import json
import shutil
from pathlib import Path

import pytest

from tests.e4_part2_support import tiny_model, write_legacy_checkpoint  # noqa: F401 (path setup)

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("transformers")

import e4_double_quant_prevalence as part1  # noqa: E402
import e4_part2_common as _common  # noqa: E402
import e4_part2_quant as quant  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(_common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to torch/transformers/bitsandbytes/peft of the rule",
)

LAYERS, EXPERTS = 2, 4


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    model = tiny_model(seed=3, dtype=torch.bfloat16)
    return write_legacy_checkpoint(model, root / "source")


def _part1_rows(source: Path, units):
    """Part 1's rows for the tiny checkpoint, from part 1's own metric code."""
    from safetensors import safe_open

    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    rows = {}
    for unit in units:
        with safe_open(str(source / index[unit.key]), framework="pt") as handle:
            ref = handle.get_tensor(unit.key).to(torch.bfloat16)
        row = {"key": unit.key}
        for tag, dq in (("dq_off", False), ("dq_on", True)):
            metrics = part1.nf4_metrics(ref, dq)
            row[tag] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        rows[unit.key] = row
    return rows


class TestUnits:
    def test_the_real_model_has_3136_units(self):
        units = quant.enumerate_units()
        assert len(units) == 3136
        counts = {k: sum(u.kind == k for u in units) for k in quant.KIND_ORDER}
        assert counts == {"q_proj": 16, "k_proj": 16, "v_proj": 16, "o_proj": 16,
                          "gate_proj": 1024, "up_proj": 1024, "down_proj": 1024}

    def test_unit_names_are_checkpoint_keys_and_unique(self):
        units = quant.enumerate_units()
        assert len({u.key for u in units}) == len(units)
        assert units[0].key == "model.layers.0.self_attn.q_proj.weight"
        assert units[4].key == "model.layers.0.mlp.experts.0.gate_proj.weight"

    def test_every_fused_slice_equals_its_checkpoint_tensor(self, tiny_checkpoint):
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(tiny_checkpoint, dtype=torch.bfloat16)
        index = json.loads(
            (tiny_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        units = quant.enumerate_units(LAYERS, EXPERTS)
        for unit in units:
            with safe_open(str(tiny_checkpoint / index[unit.key]), framework="pt") as handle:
                want = handle.get_tensor(unit.key)
            got = quant.fused_view(model, unit, model.config.intermediate_size)
            assert quant.bitwise_equal(got, want), unit.key


class TestCodec:
    @pytest.mark.parametrize("double_quant", [False, True])
    def test_block_metrics_match_part1_exactly(self, double_quant):
        gen = torch.Generator().manual_seed(1)
        ref = (torch.randn(64, 192, generator=gen) * torch.rand(64, 1, generator=gen) * 3).to(
            torch.bfloat16)
        _, _, back = quant.quantize_unit(ref, double_quant)
        mine = quant.block_metrics(ref, back)
        theirs = part1.nf4_metrics(ref, double_quant)
        assert mine["frac_over"] == theirs["frac_over"]
        assert mine["worst"] == theirs["worst"]

    def test_quant_state_roundtrips_through_the_packed_tensors(self):
        from bitsandbytes.functional import QuantState, dequantize_4bit

        ref = torch.randn(32, 128, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
        for double_quant in (False, True):
            packed, state, back = quant.quantize_unit(ref, double_quant)
            rebuilt = QuantState.from_dict(quant.quant_state_tensors(state), device="cpu")
            assert quant.bitwise_equal(dequantize_4bit(packed, rebuilt), back)
            assert rebuilt.nested == double_quant

    def test_bitwise_equal_tells_nan_payloads_and_signed_zero_apart(self):
        a = torch.tensor([0.0], dtype=torch.bfloat16)
        b = torch.tensor([-0.0], dtype=torch.bfloat16)
        assert a.eq(b).all() and not quant.bitwise_equal(a, b)


class TestBackendProbe:
    def test_records_which_kernels_ran(self):
        ref = torch.randn(16, 128, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
        with quant.BackendProbe() as probe:
            quant.quantize_unit(ref, True)
        report = probe.report()
        assert report["ops"]["quantize_4bit"]["cpu_kernel_registered"] is False
        assert any("quantize_4bit" in name for name in report["ops"]["quantize_4bit"]["calls"])
        assert report["ops"]["dequantize_4bit"]["cpu_kernel_registered"] is True
        if report["native_library_loaded"]:
            assert report["native_functions_called"], "native CPU kernels must be counted"


class TestDiskPlan:
    def test_stops_when_outputs_do_not_fit(self, tiny_checkpoint, tmp_path, monkeypatch):
        units = quant.enumerate_units(LAYERS, EXPERTS)
        monkeypatch.setattr(quant.shutil, "disk_usage",
                            lambda _p: shutil._ntuple_diskusage(10, 9, 1))
        plan = quant.disk_plan(tiny_checkpoint, tmp_path / "out", units)
        assert plan["fits"] is False

    def test_run_returns_1_without_writing_when_disk_is_short(self, tiny_checkpoint, tmp_path,
                                                              monkeypatch):
        monkeypatch.setattr(quant.shutil, "disk_usage",
                            lambda _p: shutil._ntuple_diskusage(10, 9, 1))
        code = quant.run(tiny_checkpoint, tmp_path / "out", tmp_path / "m.json", None,
                         LAYERS, EXPERTS, log=lambda *_: None)
        assert code == 1
        assert not (tmp_path / "out" / "codes-dqoff").exists()


@pytest.fixture(scope="module")
def finished(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    units = quant.enumerate_units(LAYERS, EXPERTS)
    part1_path = out / "part1.jsonl"
    part1_path.write_text("".join(
        json.dumps(r) + "\n" for r in _part1_rows(tiny_checkpoint, units).values()))
    manifest = out / "manifest.json"
    code = quant.run(tiny_checkpoint, out / "data", manifest, part1_path, LAYERS, EXPERTS,
                     log=lambda *_: None)
    return {"code": code, "out": out / "data", "manifest": json.loads(manifest.read_text()),
            "units": units, "part1": part1_path}


class TestEndToEnd:
    def test_u1_u2_u3_pass_for_both_settings(self, finished):
        assert finished["code"] == 0
        for tag in ("off", "on"):
            checks = finished["manifest"]["checks"][tag]
            assert checks["U1"]["pass"] and checks["U2"]["pass"] and checks["U3"]["pass"]
            assert checks["U3"]["units_changed"] == len(finished["units"])
            assert checks["codes_decode_to_writeback"]["pass"]

    def test_manifest_names_the_backend_for_every_op(self, finished):
        ops = finished["manifest"]["bnb_cpu_backends"]["quantize_pass_dq_on"]["ops"]
        assert set(ops) == set(quant.BNB_OPS)
        assert finished["manifest"]["bnb_cpu_backends"]["verify"]["native_library_loaded"] in (
            True, False)

    def test_manifest_hashes_every_output_file_and_sha256sums_agree(self, finished):
        out = finished["out"]
        files = finished["manifest"]["files"]
        on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS"}
        assert set(files) == on_disk
        sums = dict(line.split("  ", 1)[::-1] for line in (out / "SHA256SUMS").read_text().split(
            "\n") if line)
        assert {k: v["sha256"] for k, v in files.items()} == sums

    def test_writeback_dir_loads_through_from_pretrained_to_the_written_back_values(
            self, finished):
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        wb = finished["out"] / "writeback-dqon"
        model = AutoModelForCausalLM.from_pretrained(wb, dtype=torch.bfloat16)
        index = json.loads((wb / "model.safetensors.index.json").read_text())["weight_map"]
        for unit in finished["units"]:
            view = quant.fused_view(model, unit, model.config.intermediate_size)
            with safe_open(str(wb / index[unit.key]), framework="pt") as handle:
                assert quant.bitwise_equal(view, handle.get_tensor(unit.key)), unit.key

    def test_only_unit_tensors_differ_from_the_source(self, finished, tiny_checkpoint):
        from safetensors import safe_open

        src_index = json.loads(
            (tiny_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        wb = finished["out"] / "writeback-dqoff"
        wb_index = json.loads((wb / "model.safetensors.index.json").read_text())["weight_map"]
        unit_keys = {u.key for u in finished["units"]}
        assert set(src_index) == set(wb_index)
        for key in src_index:
            with safe_open(str(tiny_checkpoint / src_index[key]), framework="pt") as a, \
                    safe_open(str(wb / wb_index[key]), framework="pt") as b:
                same = quant.bitwise_equal(a.get_tensor(key), b.get_tensor(key))
            assert same != (key in unit_keys), key

    def test_a_corrupted_codes_file_fails_u1(self, finished, tiny_checkpoint):
        from safetensors.torch import load_file, save_file

        path = finished["out"] / "codes-dqon" / "layer-00.safetensors"
        original = path.read_bytes()
        try:
            tensors = load_file(str(path))
            key = quant.enumerate_units(LAYERS, EXPERTS)[0].key
            tensors[key] = tensors[key].clone()
            tensors[key][0, 0] ^= 0x01
            save_file(tensors, str(path), metadata={"format": "pt"})
            units = finished["units"]
            result = quant.verify(tiny_checkpoint, finished["out"], units, None, lambda *_: None)
            assert not result["on"]["U1"]["pass"]
            assert result["off"]["U1"]["pass"]
        finally:
            path.write_bytes(original)

    def test_a_changed_non_unit_tensor_fails_u3(self, finished, tiny_checkpoint):
        from safetensors.torch import load_file, save_file

        path = finished["out"] / "writeback-dqoff" / "model-other.safetensors"
        original = path.read_bytes()
        try:
            tensors = load_file(str(path))
            tensors["model.norm.weight"] = tensors["model.norm.weight"] + 1
            save_file(tensors, str(path), metadata={"format": "pt"})
            result = quant.verify(tiny_checkpoint, finished["out"], finished["units"], None,
                                  lambda *_: None)
            assert not result["off"]["U3"]["pass"]
            assert "model.norm.weight" in result["off"]["U3"]["non_unit_tensors_differing"]
        finally:
            path.write_bytes(original)

    def test_part1_disagreement_fails_u2_with_no_verdict(self, finished, tiny_checkpoint):
        rows = quant.read_part1_rows(finished["part1"])
        key = finished["units"][0].key
        rows[key]["dq_on"]["worst"] += 1e-3
        result = quant.verify(tiny_checkpoint, finished["out"], finished["units"], rows,
                              lambda *_: None)
        assert not result["on"]["U2"]["pass"]
        assert result["on"]["U2"]["mismatches"][0]["key"] == key
        assert result["off"]["U2"]["pass"]
