"""C5 ticket 05: shard-cache size and expert restore cost (benchmarks/harness/c5_cache.py).

Spec: .scratch/c5-q2-0-experts/issues/05-cache-size-and-restore-cost.md. Every
model here is a tiny random-init OLMoE built from a config, so nothing is
downloaded.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


@pytest.fixture(scope="module")
def harness():
    original_path = sys.path[:]
    names = ("_c5_cache_test_module", "q2_0_codec", "c5_expert_swap", "arm_validity",
             "compare_shard_caches")
    original_modules = {name: sys.modules.get(name) for name in names}
    spec = importlib.util.spec_from_file_location(names[0], HARNESS / "c5_cache.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[names[0]] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def caches(harness, tmp_path_factory):
    """The shipped bf16 cache of a tiny OLMoE and both packed variants, built once."""
    root = tmp_path_factory.mktemp("c5-cache")
    weights = root / "weights"
    model = harness.olmoe_fixture(seed=0)
    model.save_pretrained(weights)
    record = harness.build_caches(str(weights), str(root / "caches"), device="cpu")
    return {"root": root / "caches", "model": model, "record": record}


def _fake_quant(weight, codec_name, harness):
    """Ticket 04's expert weights: each expert quantized, restored and cast to bf16."""
    codec = harness.codec
    return torch.stack(
        [codec.dequantize(codec.quantize(expert, codec_name)).to(weight.dtype)
         for expert in weight]
    )


@pytest.mark.parametrize("codec_name", ["nf4", "q2_0"])
def test_restored_layer_equals_ticket_04_fake_quant(harness, caches, codec_name):
    model = caches["model"]
    for layer in range(model.config.num_hidden_layers):
        packed = harness.load_layer(str(caches["root"] / codec_name), layer, device="cpu")
        restored = harness.restore_layer(packed)
        experts = model.model.layers[layer].mlp.experts
        for key in EXPERT_KEYS:
            original = getattr(experts, key.rsplit(".", 1)[1]).detach()
            assert restored[key].dtype == torch.bfloat16
            assert torch.equal(restored[key], _fake_quant(original, codec_name, harness))


@pytest.mark.parametrize("codec_name", ["nf4", "q2_0"])
def test_one_expert_restores_to_its_slice_of_the_layer(harness, caches, codec_name):
    packed = harness.load_layer(str(caches["root"] / codec_name), 1, device="cpu")
    whole = harness.restore_layer(packed)
    for expert in (0, 3):
        alone = harness.restore_layer(packed, slice(expert, expert + 1))
        for key in EXPERT_KEYS:
            assert torch.equal(alone[key], whole[key][expert:expert + 1])


def test_sizes_are_read_from_the_files_on_disk(caches):
    record = caches["record"]
    for variant in ("bf16", "nf4", "q2_0"):
        folder = caches["root"] / variant
        sizes = record["variants"][variant]
        on_disk = {path.name: path.stat().st_size for path in folder.iterdir()}
        assert sizes["files"] == on_disk
        assert sizes["ssd_bytes"] == sum(on_disk.values())
        assert [layer["file_bytes"] for layer in sizes["layers"]] == [
            on_disk["layer_000.safetensors"], on_disk["layer_001.safetensors"]]


def test_expert_bytes_match_each_format(caches):
    """Fixture experts per layer: gate_up [4, 128, 128] and down [4, 128, 64], 98304 weights.

    Q2_0 is 18 bytes per 64 weights: 27648. NF4 is 49152 bytes of nibbles, 1536
    of uint8 absmax, 4 fp32 nested absmax per tensor (one nested block per
    expert) and 4 fp32 offsets per tensor: 49152 + 1536 + 32 + 32 = 50752.
    bf16 is 2 bytes per weight: 196608.
    """
    expected = {"q2_0": 27648, "nf4": 50752, "bf16": 196608}
    for variant, nbytes in expected.items():
        layers = caches["record"]["variants"][variant]["layers"]
        assert [layer["expert_bytes"] for layer in layers] == [nbytes, nbytes]


def test_ram_store_is_every_tensor_byte_of_the_cache(caches):
    from safetensors.torch import load_file

    for variant in ("bf16", "nf4", "q2_0"):
        folder = caches["root"] / variant
        loaded = sum(
            tensor.numel() * tensor.element_size()
            for path in folder.glob("*.safetensors")
            for tensor in load_file(path).values()
        )
        assert caches["record"]["variants"][variant]["ram_store_bytes"] == loaded


def test_non_expert_tensors_are_identical_in_every_variant(caches):
    record = caches["record"]
    assert record["non_expert_mismatch"] == []
    assert record["checks"]["non_expert_identical"] is True


def test_mismatch_names_a_touched_non_expert_tensor(harness, caches, tmp_path):
    from safetensors.torch import load_file, save_file

    touched = tmp_path / "q2_0"
    shutil.copytree(caches["root"] / "q2_0", touched)
    layer = touched / "layer_001.safetensors"
    blob = load_file(layer)
    blob["self_attn.q_proj.weight"][0, 0] += 1
    save_file(blob, layer)
    mismatch = harness.non_expert_mismatch(str(caches["root"] / "nf4"), str(touched))
    assert mismatch == ["layer_001.safetensors:self_attn.q_proj.weight"]


def test_expert_digest_matches_ticket_04(harness, caches):
    """The cache restores to the weights ticket 04 trained over, digested the same way."""
    import copy
    import hashlib
    import json

    swap = harness.swap
    for codec_name in ("nf4", "q2_0"):
        model = copy.deepcopy(caches["model"])
        swap.run_arm(model, codec_name)
        params = dict(model.named_parameters())
        fingerprints = {name: swap._sha256(params[name])
                        for name in swap.expert_tensor_names(model)}
        expected = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
        assert caches["record"]["variants"][codec_name]["expert_digest"] == expected


class _FakeWatch:
    """Stands in for ArmWatch: records the call and returns a fixed outcome."""

    def __init__(self, calls, outcomes, **kwargs):
        self.kwargs = kwargs
        self.outcome = outcomes.get(kwargs["run"], "ok")
        calls.append(kwargs)
        self.record = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.record = {"outcome": self.outcome, "run": self.kwargs["run"]}
        return False


def _fake_comparison(harness, outcomes=None, max_blocks=20):
    calls, ran = [], []

    def runner(arm, round, run):
        ran.append((arm, round, run))
        layer = {"A": 0.0100, "B": 0.0050}[arm] + 0.00001 * (run % 3)
        return {"layer_s": layer, "expert_s": layer / 64}

    def watch(**kwargs):
        return _FakeWatch(calls, outcomes or {}, **kwargs)

    return harness.run_comparison(runner, watch, max_blocks=max_blocks), calls, ran


def test_comparison_runs_the_pilot_in_abba_blocks(harness):
    result, calls, ran = _fake_comparison(harness)
    assert len(result["blocks"]) == 5
    assert [arm for arm, _, _ in ran] == ["A", "B", "B", "A"] * 5
    assert [(r, n) for _, r, n in ran][:8] == [(1, 1), (1, 2), (2, 3), (2, 4),
                                               (3, 5), (3, 6), (4, 7), (4, 8)]
    assert result["complete"] is True
    assert result["repeats"]["blocks_needed"] == 2
    assert result["summary"]["A"]["layer_s"]["mean"] == pytest.approx(0.01001, abs=1e-5)
    assert result["summary"]["B"]["layer_s"]["mean"] == pytest.approx(0.00501, abs=1e-5)


def test_a_void_run_keeps_its_block_and_adds_a_replacement(harness):
    result, _, ran = _fake_comparison(harness, outcomes={6: "void"})
    assert len(result["blocks"]) == 6
    assert len(ran) == 24
    assert [run["outcome"] for run in result["blocks"][1]] == ["ok", "void", "ok", "ok"]
    assert result["valid_blocks"] == 5
    assert result["complete"] is True


def test_comparison_stops_at_max_blocks_and_says_it_is_incomplete(harness):
    void_everything = {run: "void" for run in range(1, 100)}
    result, _, ran = _fake_comparison(harness, outcomes=void_everything, max_blocks=3)
    assert len(ran) == 12
    assert result["complete"] is False
    assert result["repeats"] is None


def test_restore_runs_require_suspend_and_power_only(harness, tmp_path):
    watch = harness.make_watch(label="test", cache_dir=str(tmp_path))(arm="A", round=1, run=1)
    assert sorted(watch.required) == ["power", "suspend"]
