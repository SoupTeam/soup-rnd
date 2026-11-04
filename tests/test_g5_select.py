"""G5: routing statistics, the sensitivity proxy and the codec maps
(benchmarks/harness/g5_select.py), and the mixed swap (benchmarks/harness/g5_train.py).

Rule: benchmarks/gate-g5-mixed-precision-experts.md. The models are tiny
random-init granitemoe and OLMoE built from a config, so nothing is downloaded.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
NAMES = ("g5_select", "g5_train", "c5_train", "c5_expert_swap", "c5_data", "q2_0_codec",
         "l2l_box", "arm_validity")


@pytest.fixture(scope="module")
def modules() -> Iterator[dict[str, ModuleType]]:
    original_path = sys.path[:]
    original = {name: sys.modules.get(name) for name in NAMES}
    sys.path.insert(0, str(HARNESS))
    try:
        loaded = {}
        for name in ("g5_select", "g5_train"):
            spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        sys.path[:] = original_path
        for name, previous in original.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def select(modules: dict[str, ModuleType]) -> ModuleType:
    return modules["g5_select"]


def _layer(tokens: list[int], d: list[float], layer: int = 0,
           mass: list[float] | None = None) -> dict[str, Any]:
    return {"layer": layer, "tokens": tokens, "mass": mass or [float(t) for t in tokens],
            "d_q2_nf4": d}


class TestRanks:
    def test_ties_share_their_average_rank(self, select):
        assert select.ranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]

    def test_spearman_of_a_monotone_map_is_one_and_of_its_reverse_minus_one(self, select):
        assert select.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
        assert select.spearman([1, 2, 3, 4], [9, 8, 7, 1]) == pytest.approx(-1.0)

    def test_spearman_of_a_constant_side_is_undefined(self, select):
        assert select.spearman([1, 1, 1], [1, 2, 3]) is None


class TestMaps:
    def test_traffic_and_sensitivity_pick_different_experts_when_they_disagree(self, select):
        # Expert 0 has the least traffic but the largest error: popularity is not sensitivity.
        layer = _layer(tokens=[1, 50, 60, 70], d=[9.0, 1.0, 2.0, 3.0])
        assert select.layer_codes(layer, "traffic", 0.25) == ["q2_0", "nf4", "nf4", "nf4"]
        assert select.layer_codes(layer, "sensitivity", 0.25) == ["nf4", "q2_0", "nf4", "nf4"]

    def test_every_map_of_one_fraction_has_the_same_q2_0_count_in_every_layer(self, select):
        layers = [_layer([5, 1, 9, 3, 7, 2, 8, 4], [0.1, 0.9, 0.3, 0.2, 0.8, 0.5, 0.4, 0.7], i)
                  for i in range(3)]
        for fraction in select.FRACTIONS:
            counts = {rule: select.build_map(layers, rule, fraction)["q2_0_per_layer"]
                      for rule in select.RULES}
            assert counts["traffic"] == counts["sensitivity"] == [int(8 * fraction)] * 3

    def test_traffic_ties_break_on_gate_mass_then_index(self, select):
        layer = _layer(tokens=[4, 4, 4, 9], d=[0.0] * 4, mass=[2.0, 1.0, 2.0, 5.0])
        assert select.layer_codes(layer, "traffic", 0.5) == ["q2_0", "q2_0", "nf4", "nf4"]

    def test_a_fraction_that_is_not_a_whole_number_of_experts_is_refused(self, select):
        with pytest.raises(ValueError, match="whole number"):
            select.layer_codes(_layer([1, 2, 3], [1.0, 2.0, 3.0]), "traffic", 0.25)

    def test_the_map_hash_names_its_codes(self, select):
        layers = [_layer([1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0])]
        record = select.build_map(layers, "traffic", 0.5)
        assert record["map_sha256"] == select.map_sha256(record["codes"])
        assert record["map_sha256"] != select.build_map(layers, "sensitivity", 0.5)["map_sha256"]

    def test_the_correlation_report_counts_the_busiest_quartile(self, select):
        report = select.correlation_report([_layer([10, 10, 10, 70], [1.0, 2.0, 3.0, 4.0])])
        assert report["per_layer"][0]["busiest_quartile_traffic_share"] == pytest.approx(0.7)
        assert report["fractions"]["0.25"]["jaccard_traffic_vs_sensitivity"] == pytest.approx(1.0)


# --- routing on real (tiny) MoE models ------------------------------------------------------


def _olmoe():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.OlmoeConfig(
        vocab_size=64, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
        eos_token_id=None,
    )
    torch.manual_seed(1)
    return transformers.OlmoeForCausalLM(config).to(torch.bfloat16).eval()


def _granite(modules):
    pytest.importorskip("torch")
    return modules["g5_select"].swap.granite_fixture(seed=0, vocab_size=64)


def _brute_force(model, examples) -> dict[int, dict[str, list]]:
    """Per (layer, expert) counts and gate mass from the model's own router, token by token."""
    import torch

    out = {}
    blocks = {}
    for name, module in model.named_modules():
        if type(module).__name__ in ("OlmoeSparseMoeBlock", "GraniteMoeMoE"):
            blocks[int(name.split(".layers.")[1].split(".")[0])] = module
    inputs = {layer: [] for layer in blocks}
    hooks = [block.register_forward_pre_hook(
        lambda module, args, layer=layer: inputs[layer].append(args[0].detach()))
        for layer, block in blocks.items()]
    with torch.no_grad():
        for example in examples:
            model(input_ids=torch.tensor([example["input_ids"]]), use_cache=False)
    for hook in hooks:
        hook.remove()
    for layer, block in blocks.items():
        router = getattr(block, "gate", None) or block.router
        n_experts = router.weight.shape[0]
        tokens, mass = [0] * n_experts, [0.0] * n_experts
        for hidden in inputs[layer]:
            for row in hidden.reshape(-1, hidden.shape[-1]):
                result = router(row[None])
                if type(block).__name__ == "OlmoeSparseMoeBlock":
                    _, weights, index = result
                else:
                    # Granite's router return order differs across transformers releases
                    # ((index, weights, logits) vs (logits, weights, index)): tell the three
                    # apart by dtype and shape instead of position.
                    index = next(t for t in result if not t.dtype.is_floating_point)
                    floats = [t for t in result if t.dtype.is_floating_point]
                    weights = next(t for t in floats if t.shape == index.shape)
                    assert len(floats) == 2 and weights.shape[-1] != n_experts
                for expert, weight in zip(index[0].tolist(), weights[0].float().tolist()):
                    tokens[expert] += 1
                    mass[expert] += weight
        out[layer] = {"tokens": tokens, "mass": mass}
    return out


@pytest.mark.parametrize("build", ["olmoe", "granite"])
def test_routing_counts_match_a_brute_force_count(select, modules, build):
    model = _olmoe() if build == "olmoe" else _granite(modules)
    examples = [{"input_ids": [(7 * i + j) % 64 for j in range(5 + i)]} for i in range(6)]
    routed = select.capture_routing(model, examples, "cpu")
    expected = _brute_force(model, examples)
    assert set(routed) == set(expected)
    for layer, stats in expected.items():
        n_experts = len(stats["tokens"])
        got = select.traffic(routed[layer]["index"], routed[layer]["weight"], n_experts)
        assert got["tokens"] == stats["tokens"]
        assert got["mass"] == pytest.approx(stats["mass"], rel=1e-5)
        assert sum(got["tokens"]) == sum(len(e["input_ids"]) for e in examples) * 2


def test_the_experts_forward_of_the_proxy_matches_the_models(select):
    model = _olmoe()
    examples = [{"input_ids": list(range(3, 40))}]
    routed = select.capture_routing(model, examples, "cpu")
    for layer, module in select.experts_modules(model).items():
        assert select.forward_check(module, routed[layer], "cpu") < select.FORWARD_CHECK_LIMIT


def test_sensitivity_is_zero_for_an_expert_nobody_routes_to(select):
    torch = pytest.importorskip("torch")
    model = _olmoe()
    module = select.experts_modules(model)[0]
    routed = {"x": torch.randn(6, 64).to(torch.bfloat16),
              "index": torch.tensor([[0, 1]] * 6), "weight": torch.full((6, 2), 0.5)}
    result = select.layer_sensitivity(module, routed, "cpu")
    assert result["d_q2_nf4"][0] > 0 and result["d_q2_nf4"][1] > 0
    assert result["d_q2_nf4"][2:] == [0.0] * 6
    assert result["d_q2_bf16"][0] > result["d_nf4_bf16"][0]


# --- the CLI on the granite fixture ---------------------------------------------------------


def _write_split(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in rows), encoding="utf-8")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("g5data")
    rows = [{"row": i, "instruction": f"Question {i}?", "context": "",
             "response": f"Answer {i}.", "category": "open_qa"} for i in range(24)]
    _write_split(directory / "heldout.jsonl", rows[:8])
    _write_split(directory / "train.jsonl", rows[8:])
    return directory


@contextlib.contextmanager
def panel_guard(panel_path: Path) -> Iterator[list[str]]:
    """Fail the moment anything opens ``panel_path``; yields every path opened meanwhile."""
    panel = os.path.normcase(os.path.abspath(panel_path))
    opened: list[str] = []
    real_open = builtins.open
    real_read_text, real_read_bytes = Path.read_text, Path.read_bytes

    def guard(target: Any) -> None:
        if isinstance(target, (str, os.PathLike)):
            path = os.path.normcase(os.path.abspath(os.fspath(target)))
            opened.append(path)
            if path == panel:
                raise AssertionError("the held-out panel was read")

    def guarded_open(file, *args, **kwargs):
        guard(file)
        return real_open(file, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        guard(self)
        return real_read_text(self, *args, **kwargs)

    def guarded_read_bytes(self):
        guard(self)
        return real_read_bytes(self)

    patch = pytest.MonkeyPatch()
    patch.setattr(builtins, "open", guarded_open)
    patch.setattr(io, "open", guarded_open)
    patch.setattr(Path, "read_text", guarded_read_text)
    patch.setattr(Path, "read_bytes", guarded_read_bytes)
    try:
        yield opened
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def selection(select, data_dir, tmp_path_factory) -> dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    out = tmp_path_factory.mktemp("sel")
    with panel_guard(data_dir / "heldout.jsonl") as opened:
        code = select.main(["--fixture", "granite", "--data-dir", str(data_dir),
                            "--calibration-rows", "8", "--out-dir", str(out)])
    record = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    return {"code": code, "record": record, "out": out, "opened": opened}


class TestSelectionRun:
    def test_it_reads_the_train_split_and_never_the_panel(self, selection, data_dir):
        assert selection["code"] == 0
        train = os.path.normcase(os.path.abspath(data_dir / "train.jsonl"))
        assert train in selection["opened"]

    @pytest.mark.parametrize("read", ["c5_train.load_split", "open", "read_bytes"])
    def test_the_guard_fails_a_panel_read(self, select, data_dir, read):
        # The guard is only a test if reading the panel through any of these trips it.
        panel = data_dir / "heldout.jsonl"
        with pytest.raises(AssertionError, match="held-out panel"):
            with panel_guard(panel):
                if read == "open":
                    with open(panel, encoding="utf-8") as handle:
                        handle.read()
                elif read == "read_bytes":
                    panel.read_bytes()
                else:
                    select.c5_train.load_split(panel, None)

    def test_calibration_is_the_first_train_rows_with_their_hash(self, selection, select,
                                                                  data_dir):
        calibration = selection["record"]["calibration"]
        rows = [json.loads(line) for line in
                (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()][:8]
        assert calibration["row_ids"] == [row["row"] for row in rows]
        assert calibration["sha256"] == select.c5_data.split_digest(rows)
        assert "heldout_sha256" not in calibration

    def test_it_is_labelled_synthetic_and_leaves_the_model_unchanged(self, selection):
        record = selection["record"]
        assert record["synthetic"] is True
        assert all(record["checks"].values()), record["checks"]

    def test_it_writes_one_map_per_arm_with_equal_budgets(self, selection, select):
        maps = {arm: json.loads((selection["out"] / "maps" / f"{arm}.json")
                                .read_text(encoding="utf-8")) for arm in select.ARMS}
        for fraction in select.FRACTIONS:
            pair = [maps[select.arm_name(rule, fraction)] for rule in select.RULES]
            assert pair[0]["q2_0_per_layer"] == pair[1]["q2_0_per_layer"]
            assert pair[0]["q2_0_per_layer"] == [int(4 * fraction)] * 2
        for record in maps.values():
            assert record["stats_sha256"] == selection["record"]["stats_sha256"]

    def test_a_real_model_must_use_the_rules_calibration_rows(self, select, data_dir, tmp_path):
        with pytest.raises(SystemExit):
            select.main(["--model", "m", "--revision", "r", "--data-dir", str(data_dir),
                         "--calibration-rows", "8", "--out-dir", str(tmp_path)])

    def test_a_real_model_calibrates_on_the_first_512_train_rows_and_never_the_panel(
        self, select, tmp_path, monkeypatch,
    ):
        # 600 train rows whose first 512 hash as the rule's: the run gets past the hash
        # check, and the rows it encodes are exactly those 512, never a panel row.
        rows = [{"row": 10_000 + i, "instruction": f"Train {i}?", "context": "",
                 "response": f"A {i}.", "category": "open_qa"} for i in range(600)]
        panel = [{"row": i, "instruction": f"Panel {i}?", "context": "",
                  "response": f"P {i}.", "category": "open_qa"} for i in range(8)]
        data = tmp_path / "data"
        data.mkdir()
        _write_split(data / "train.jsonl", rows)
        _write_split(data / "heldout.jsonl", panel)
        train_sha = select.c5_train._sha256_file(data / "train.jsonl")
        monkeypatch.setattr(select, "CALIBRATION_SHA256",
                            select.c5_data.split_digest(rows[:select.CALIBRATION_ROWS]))
        monkeypatch.setattr(select.c5_train, "TRAIN_SHA256", {train_sha: len(rows)})
        encoded: list[int] = []

        class StopAfterEncodingError(Exception):
            pass

        def encode(row, tokenizer, max_length):
            encoded.append(row["row"])
            return {"input_ids": [1]}

        def load_model(args, tokenizer):
            raise StopAfterEncodingError

        monkeypatch.setattr(select.c5_train, "load_tokenizer", lambda args: object())
        monkeypatch.setattr(select.c5_train, "encode_example", encode)
        monkeypatch.setattr(select.c5_train, "load_model", load_model)
        with panel_guard(data / "heldout.jsonl"), pytest.raises(StopAfterEncodingError):
            select.main(["--model", "m", "--revision", "r", "--data-dir", str(data),
                         "--out-dir", str(tmp_path / "sel")])
        assert select.CALIBRATION_ROWS == 512
        assert encoded == [row["row"] for row in rows[:512]]

    def test_a_real_model_refuses_calibration_rows_that_are_not_the_rules(
        self, select, data_dir, tmp_path,
    ):
        # the fixture's train split is not C5's: its first rows do not hash as the rule's
        with pytest.raises(SystemExit, match="calibration data does not match"):
            select.main(["--model", "m", "--revision", "r", "--data-dir", str(data_dir),
                         "--out-dir", str(tmp_path)])

    def test_the_record_holds_raw_and_lf_hashes_of_every_harness_file(self, selection,
                                                                        select):
        sources = selection["record"]["harness"]["sources"]
        assert set(sources) == set(select.SOURCES)
        for name, entry in sources.items():
            assert entry == select.source_digest(select.HARNESS_DIR / name)


def test_a_crlf_file_keeps_its_lf_hash(select, tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert select.source_digest(lf)["sha256"] != select.source_digest(crlf)["sha256"]
    assert select.source_digest(crlf)["sha256_lf"] == select.source_digest(lf)["sha256"]


# --- the mixed swap ---------------------------------------------------------------------


def test_the_map_changes_only_the_experts_it_names(modules):
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    train, swap = modules["g5_train"], modules["g5_select"].swap
    codes = [["q2_0", "nf4", "nf4", "nf4"], ["nf4", "nf4", "q2_0", "nf4"]]
    reference = {codec: swap.granite_fixture(seed=0, vocab_size=64) for codec in ("nf4", "q2_0")}
    for codec, model in reference.items():
        swap.run_arm(model, codec)
    model = swap.granite_fixture(seed=0, vocab_size=64)
    before = swap.fingerprints(model)
    record = train.apply_map(model, codes)
    after = swap.fingerprints(model)
    assert all(record["checks"].values())
    experts = set(swap.expert_tensor_names(model))
    assert {name: after[name] for name in after if name not in experts} == {
        name: before[name] for name in before if name not in experts}
    params = dict(model.named_parameters())
    refs = {codec: dict(m.named_parameters()) for codec, m in reference.items()}
    for name in experts:
        layer = modules["g5_select"].layer_of(name)
        for expert, codec in enumerate(codes[layer]):
            assert torch.equal(params[name][expert], refs[codec][name][expert]), (name, expert)
        assert record["tensors"][name]["codes"] == codes[layer]


def test_an_all_nf4_map_gives_c5s_nf4_weights_bit_for_bit(modules):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    train, swap = modules["g5_train"], modules["g5_select"].swap
    c5_model, g5_model = (swap.granite_fixture(seed=0, vocab_size=64) for _ in range(2))
    swap.run_arm(c5_model, "nf4")
    train.apply_map(g5_model, [["nf4"] * 4, ["nf4"] * 4])
    assert swap.fingerprints(c5_model) == swap.fingerprints(g5_model)


def test_a_map_with_a_tampered_hash_is_refused(modules, tmp_path):
    select = modules["g5_select"]
    record = select.build_map([_layer([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])], "traffic", 0.5)
    record["codes"][0][3] = "q2_0"
    path = tmp_path / "map.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="map_sha256"):
        modules["g5_train"].load_map(path)


def test_the_arm_is_registered_with_c5_train_only_for_the_run(modules):
    train = modules["g5_train"]
    c5_train = modules["g5_select"].c5_train
    select = modules["g5_select"]
    record = select.build_map([_layer([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])], "traffic", 0.5)
    original = c5_train.prepare_experts
    with train.mixed_arm(record) as arm:
        assert c5_train.ARM_CODECS[arm] == f"map:{record['map_sha256']}"
        assert c5_train.prepare_experts is not original
    assert arm not in c5_train.ARM_CODECS
    assert c5_train.prepare_experts is original
    assert set(c5_train.ARM_CODECS) == {"nf4", "q2_0", "bf16"}


@pytest.mark.parametrize("codec_name", ["nf4", "q2_0"])
def test_the_proxy_measures_the_weights_the_training_arm_sees(modules, codec_name):
    # d_q2_nf4 is only about training if fake_quant is exactly C5's swap.
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    select, swap = modules["g5_select"], modules["g5_select"].swap
    model = swap.granite_fixture(seed=0, vocab_size=64)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    swap.run_arm(model, codec_name)
    for name, param in model.named_parameters():
        if name in swap.expert_tensor_names(model):
            for expert in range(param.shape[0]):
                assert torch.equal(select.fake_quant(original[name][expert], codec_name),
                                   param[expert])
