"""G5: full cache bytes per arm, metadata included (benchmarks/harness/g5_bytes.py).

Rule: benchmarks/gate-g5-mixed-precision-experts.md. Sizes are computed from
shapes; these tests write real safetensors files and compare, and check the
model against C5's measured OLMoE cache (benchmarks/results/c5-cache/cache.json).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness"
C5_CACHE = ROOT / "benchmarks" / "results" / "c5-cache" / "cache.json"
EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
#: One OLMoE-1B-7B-0924 layer file as the shipped sharder writes it (C5's cache.json).
OLMOE_LAYER = {
    "self_attn.q_proj.weight": ("BF16", (2048, 2048)),
    "self_attn.k_proj.weight": ("BF16", (2048, 2048)),
    "self_attn.v_proj.weight": ("BF16", (2048, 2048)),
    "self_attn.o_proj.weight": ("BF16", (2048, 2048)),
    "self_attn.q_norm.weight": ("BF16", (2048,)),
    "self_attn.k_norm.weight": ("BF16", (2048,)),
    "mlp.gate.weight": ("BF16", (64, 2048)),
    "mlp.experts.gate_up_proj": ("BF16", (64, 2048, 2048)),
    "mlp.experts.down_proj": ("BF16", (64, 2048, 1024)),
    "input_layernorm.weight": ("BF16", (2048,)),
    "post_attention_layernorm.weight": ("BF16", (2048,)),
}


@pytest.fixture(scope="module")
def harness() -> Iterator[ModuleType]:
    name = "_g5_bytes_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "g5_bytes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(name, None)


def _written_size(entries, metadata, path: Path) -> int:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    dtypes = {"U8": torch.uint8, "F16": torch.float16, "BF16": torch.bfloat16,
              "F32": torch.float32}
    blob = {name: torch.zeros(shape, dtype=dtypes[dtype]) for name, (dtype, shape)
            in entries.items()}
    safetensors.save_file(blob, str(path), metadata=metadata)
    return os.path.getsize(path)


SMALL_LAYER = {
    "self_attn.q_proj.weight": ("BF16", (64, 64)),
    "mlp.gate.weight": ("BF16", (8, 64)),
    "input_layernorm.weight": ("BF16", (64,)),
    "mlp.experts.gate_up_proj": ("BF16", (8, 128, 64)),
    "mlp.experts.down_proj": ("BF16", (8, 64, 64)),
}


@pytest.mark.parametrize("codes", [
    ["nf4"] * 8,
    ["q2_0"] * 8,
    ["q2_0", "nf4", "nf4", "q2_0", "nf4", "nf4", "nf4", "nf4"],
    ["nf4", "q2_0", "q2_0", "q2_0", "q2_0", "q2_0", "q2_0", "q2_0"],
])
def test_the_computed_file_size_is_the_written_files(harness, tmp_path, codes):
    entries, metadata = harness.layer_entries(SMALL_LAYER, EXPERT_KEYS, codes)
    computed = harness.layer_file(SMALL_LAYER, EXPERT_KEYS, codes)
    assert computed["file"] == _written_size(entries, metadata, tmp_path / "layer.safetensors")


def test_a_mixed_layer_stores_its_codec_map_in_the_metadata(harness):
    codes = ["q2_0", "nf4", "nf4", "q2_0", "nf4", "nf4", "nf4", "nf4"]
    entries, metadata = harness.layer_entries(SMALL_LAYER, EXPERT_KEYS, codes)
    assert metadata["c5_codec"] == "mixed"
    assert metadata["g5_codec_map"] == "qnnqnnnn"
    assert entries["mlp.experts.down_proj"] == ("U8", (6, 64 * 64 // 2))
    assert entries["mlp.experts.down_proj::q2_0"] == ("U8", (2, 64, 16))
    assert entries["mlp.experts.down_proj::q2_0::d"] == ("F16", (2, 64))


def test_the_metadata_bytes_are_counted(harness):
    codes = ["q2_0", "nf4", "nf4", "q2_0", "nf4", "nf4", "nf4", "nf4"]
    entry = harness.layer_file(SMALL_LAYER, EXPERT_KEYS, codes)
    assert entry["codec_map"] == len('"g5_codec_map":"qnnqnnnn",')
    # payload, scales and the non-expert tensors are the data; the rest of the file is header
    assert entry["file"] == entry["header"] + entry["payload"] + entry["scales"] + entry[
        "non_expert"]
    assert entry["header"] > entry["codec_map"]


@pytest.mark.parametrize("variant,codec", [("nf4", "nf4"), ("q2_0", "q2_0")])
def test_the_olmoe_layer_file_is_c5s_measured_size(harness, variant, codec):
    measured = json.loads(C5_CACHE.read_text(encoding="utf-8"))["variants"][variant]
    computed = harness.layer_file(OLMOE_LAYER, EXPERT_KEYS, [codec] * 64)
    assert {layer["file_bytes"] for layer in measured["layers"]} == {computed["file"]}
    assert {layer["expert_bytes"] for layer in measured["layers"]} == {
        computed["payload"] + computed["scales"]}


def test_expert_bits_per_weight_are_c5s(harness):
    # 4.127 bits NF4 (codes, absmax, nested absmax and offset) and 2.25 Q2_0, as C5 reports.
    weights = 64 * 2048 * 3072
    nf4 = harness.layer_file(OLMOE_LAYER, EXPERT_KEYS, ["nf4"] * 64)
    q2 = harness.layer_file(OLMOE_LAYER, EXPERT_KEYS, ["q2_0"] * 64)
    assert 8 * (nf4["payload"] + nf4["scales"]) / weights == pytest.approx(4.127, abs=5e-4)
    assert 8 * (q2["payload"] + q2["scales"]) / weights == 2.25


def _inventory(n_layers: int = 16) -> dict:
    return {"layers": [dict(OLMOE_LAYER) for _ in range(n_layers)],
            "rest": {"model.embed_tokens.weight": ("BF16", (50304, 2048))},
            "expert_keys": list(EXPERT_KEYS)}


def _codes(q2_experts: set[int], n_layers: int = 16) -> list[list[str]]:
    return [["q2_0" if e in q2_experts else "nf4" for e in range(64)] for _ in range(n_layers)]


@pytest.fixture(scope="module")
def record(harness):
    """OLMoE's accounting with two f = 0.25 maps that pick opposite experts."""
    cache = json.loads(C5_CACHE.read_text(encoding="utf-8"))
    maps = {
        "mixed_traffic_f25": {"codes": _codes(set(range(16))), "fraction_q2_0": 0.25,
                              "map_sha256": "a"},
        "mixed_sensitivity_f25": {"codes": _codes(set(range(48, 64))),
                                  "fraction_q2_0": 0.25, "map_sha256": "b"},
    }
    return harness.accounting(_inventory(), maps, cache)


class TestAccounting:
    def test_all_nf4_and_all_q2_0_equal_c5s_measured_caches(self, record):
        assert record["checks"]["all_nf4_ssd_bytes_match_c5"]
        assert record["checks"]["all_q2_0_ssd_bytes_match_c5"]
        assert record["arms"]["all_nf4"]["ssd_bytes"] == 4_276_925_021
        assert record["arms"]["all_q2_0"]["ssd_bytes"] == 2_765_385_229
        assert record["arms"]["all_nf4"]["ssd_bytes_source"].startswith("measured")

    def test_mixed_arms_are_labelled_computed(self, record):
        assert record["arms"]["mixed_traffic_f25"]["ssd_bytes_source"] == "computed"

    def test_two_maps_with_the_same_count_cost_the_same_bytes(self, record):
        traffic = record["arms"]["mixed_traffic_f25"]
        sensitivity = record["arms"]["mixed_sensitivity_f25"]
        assert traffic["ssd_bytes"] == sensitivity["ssd_bytes"]
        assert record["checks"]["mixed_arms_of_one_fraction_equal_bytes"]

    def test_the_mixed_cache_includes_its_map_and_sits_between_the_codecs(self, record):
        arms = record["arms"]
        mixed = arms["mixed_traffic_f25"]
        assert arms["all_q2_0"]["ssd_bytes"] < mixed["ssd_bytes"] < arms["all_nf4"]["ssd_bytes"]
        assert mixed["codec_map_bytes"] == 16 * len('"g5_codec_map":"' + "q" * 64 + '",')
        expected_payload = 16 * (48 * 2048 * 3072 // 2 + 16 * (2048 * 3072 // 64) * 16)
        assert mixed["expert_payload_bytes"] == expected_payload

    def test_unequal_budgets_are_caught(self, harness):
        maps = {"mixed_traffic_f25": {"codes": _codes(set(range(16)), 2),
                                      "fraction_q2_0": 0.25, "map_sha256": "a"},
                "mixed_sensitivity_f25": {"codes": _codes(set(range(15)), 2),
                                          "fraction_q2_0": 0.25, "map_sha256": "b"}}
        record = harness.accounting(_inventory(2), maps)
        assert not record["checks"]["mixed_arms_of_one_fraction_equal_bytes"]

    def test_without_c5s_record_every_file_is_computed(self, harness):
        record = harness.accounting(_inventory(2), {})
        rest = record["arms"]["all_nf4"]["rest"]
        assert rest["source"] == "computed"
        # the NF4 code tables (16 + 256 fp32) are stored once whenever an expert is NF4
        q2 = record["arms"]["all_q2_0"]["rest"]
        assert rest["bytes"] - q2["bytes"] >= 4 * (16 + 256)


def test_the_cli_counts_the_fixtures_expert_bytes(harness, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    out = tmp_path / "bytes.json"
    assert harness.main(["--fixture", "granite", "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    layers, n_experts = 2, 4
    weights = {"gate_up_proj": 256 * 128, "down_proj": 128 * 128}  # per expert, granite fixture
    nf4 = record["arms"]["all_nf4"]
    assert nf4["expert_payload_bytes"] == layers * n_experts * sum(weights.values()) // 2
    # per expert: one uint8 absmax per 64 weights, one fp32 nested absmax per 256 of
    # them, one fp32 offset
    scales = sum(n // 64 + 4 * math.ceil(n / 64 / 256) + 4 for n in weights.values())
    assert nf4["expert_scale_bytes"] == layers * n_experts * scales
