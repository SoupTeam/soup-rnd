"""CPU tests for the G4 harness around the codec: the Dolly split and packing
(g4_data.py), the probe's schedule, task loss and error experiment (g4_probe.py), and
the A-B-B-A timing driver (g4_timing.py)."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager

import pytest

from tests import test_l2l_probe_harness as l2l_tests

torch = pytest.importorskip("torch")
streamed = l2l_tests.streamed


@pytest.fixture()
def data():
    return l2l_tests._load("g4_data")


@pytest.fixture()
def probe():
    return l2l_tests._load("g4_probe")


@pytest.fixture()
def timing():
    return l2l_tests._load("g4_timing")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _dolly_rows(n):
    return [{"instruction": f"q{i}", "context": "" if i % 2 else f"c{i}",
             "response": f"a{i}", "category": "x"} for i in range(n)]


def test_split_is_sorted_by_hash_disjoint_and_sized(data):
    rows = _dolly_rows(40)
    splits = data.build_splits(rows, heldout_rows=8, train_rows=20)
    held = [row["row"] for row in splits["heldout"]]
    train = [row["row"] for row in splits["train"]]
    assert len(held) == 8 and len(train) == 20 and not set(held) & set(train)
    keys = [data.sort_key(i, rows[i]["instruction"]) for i in held + train]
    assert keys == sorted(keys)


def test_parse_rows_does_not_split_on_unicode_line_separators(data):
    payload = (json.dumps({"instruction": "a\u2028b"}, ensure_ascii=False) + "\n").encode()
    assert data.parse_rows(payload + b"\n") == [{"instruction": "a\u2028b"}]


def test_a_file_other_than_the_pinned_one_is_refused(data, tmp_path):
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(b"".join(
        (json.dumps(row) + "\n").encode() for row in _dolly_rows(600)))
    with pytest.raises(ValueError, match="is not the pinned"):
        data.load_splits(str(path))


class _CharTokenizer:
    """One token per character; the template is 'U:<user>|A:' and '<answer>$'."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"U:{m['content']}|" if m["role"] == "user" else f"A:{m['content']}$"
                       for m in messages)
        return text + ("A:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


def test_pack_train_cuts_one_stream_into_full_rows(data):
    rows = [{"row": i, "instruction": f"q{i}", "context": "", "response": "ab"}
            for i in range(5)]
    packed = data.pack_train(_CharTokenizer(), rows, seq=7)
    stream = "".join(f"U:q{i}|A:ab$" for i in range(5))
    assert ["".join(map(chr, row)) for row in packed] == [
        stream[i:i + 7] for i in range(0, len(stream) - 6, 7)]
    assert all(len(row) == 7 for row in packed)
    assert len(data.pack_train(_CharTokenizer(), rows, seq=7, limit=2)) == 2


def test_panel_scores_the_response_after_the_prompt(data):
    rows = [{"row": 3, "instruction": "q", "context": "ctx", "response": "yes"},
            {"row": 4, "instruction": "x" * 20, "context": "", "response": "no"}]
    panel = data.encode_panel(_CharTokenizer(), rows, max_length=12)
    prompt = "U:q\nctx|A:"
    assert panel[0]["response_from"] == len(prompt)
    assert "".join(map(chr, panel[0]["ids"]))[:len(prompt)] == prompt
    assert panel[0]["response_tokens"] == 12 - len(prompt)
    assert panel[1]["response_tokens"] == 0  # the prompt fills the window


def test_ids_digest_separates_row_boundaries(data):
    assert data.ids_digest([[1, 2], [3]]) != data.ids_digest([[1], [2, 3]])


# --------------------------------------------------------------------------
# probe helpers
# --------------------------------------------------------------------------
def test_lr_schedule_matches_hf_cosine_with_warmup(probe):
    transformers = pytest.importorskip("transformers")
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=2e-4)
    sched = transformers.get_cosine_schedule_with_warmup(opt, 3, 20)
    for step in range(20):
        assert probe.lr_at(step, total=20, warmup=3, peak=2e-4) == pytest.approx(
            sched.get_last_lr()[0], abs=1e-12)
        opt.step()
        sched.step()


def test_warmup_steps_round_up(probe):
    assert probe.warmup_steps(394, 0.03) == 12
    assert probe.warmup_steps(6, 0.03) == 1


def test_task_loss_is_token_weighted(probe):
    per = [{"nll": 3.0, "tokens": 1}, {"nll": 10.0, "tokens": 9}, {"nll": 0.0, "tokens": 0}]
    assert probe.task_loss(per) == pytest.approx(13.0 / 10)
    assert probe.task_loss([{"nll": 0.0, "tokens": 0}]) is None


def test_cpu_runs_are_labelled_as_pilots(probe):
    args = probe.parse_args(["--mode", "error", "--device", "cpu", "--weights", "w",
                             "--out", "o.json", "--label", "smoke"])
    assert args.label == "SYNTHETIC/CPU PILOT smoke"
    assert args.deterministic is True
    timing = probe.parse_args(["--mode", "timing", "--weights", "w", "--out", "o.json"])
    assert timing.deterministic is False and timing.steps == 8


def test_int4_is_not_a_training_codec(probe):
    with pytest.raises(SystemExit):
        probe.parse_args(["--mode", "train", "--codec", "int4", "--weights", "w",
                          "--out", "o.json"])


def test_suites_refuses_an_adapter_that_is_not_the_trained_one(probe, tmp_path):
    adapter = tmp_path / "custom" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"trained")
    arm = tmp_path / "arms" / "train-int8-seed1.json"
    arm.parent.mkdir()
    args = probe.parse_args(["--mode", "train", "--weights", "w", "--out", str(arm),
                             "--adapter-out", str(adapter)])
    found = probe.adapter_dir_for(args.out, args.adapter_out)
    record = {"adapter": {"dir": "../custom/adapter", "sha256": probe.tree_sha256(found)}}
    arm.write_text(json.dumps(record), encoding="utf-8")
    assert probe.recorded_adapter_dir(arm, record) == adapter.resolve()
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    suites = probe.parse_args(["--mode", "suites", "--device", "cpu", "--weights", "w",
                               "--arm", str(arm), "--out", str(tmp_path / "s.json")])
    with pytest.raises(RuntimeError, match="does not match the adapter hashes"):
        probe.suites_main(suites)


def test_panel_nll_scores_only_response_tokens(probe, streamed):
    model, _runtime = streamed
    ctx = argparse.Namespace(model=model, device="cpu")
    ids = [5, 9, 3, 7, 1, 2]
    per = probe.panel_nll(ctx, [{"row": 0, "ids": ids, "response_from": 4},
                                {"row": 1, "ids": [1, 2], "response_from": 2}])
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    expected = torch.nn.functional.cross_entropy(logits[3:5].float(), torch.tensor([1, 2]),
                                                 reduction="sum")
    assert per[0]["tokens"] == 2
    assert per[0]["nll"] == pytest.approx(float(expected), rel=1e-6)
    assert per[1] == {"row": 1, "nll": 0.0, "tokens": 0}


def test_base_parity_resident_reproduces_the_streamed_model_and_detects_a_change(
        probe, streamed, tmp_path):
    """V13's measurement on the tiny float32 fixture: the resident base loaded with the
    streamed run's saved adapter gives the streamed NLL, the live adapter equals its
    file, and both controls move: the adapter switched off, and a perturbed adapter."""
    pytest.importorskip("peft")
    from soup_cli.utils.live_eval import load_model_and_tokenizer
    from tests.test_v07204 import _write_tiny_tokenizer

    sched = l2l_tests._load("l2l_schedule")
    model, _runtime = streamed
    sched.make_non_vacuous(model)
    gen = torch.Generator().manual_seed(4)
    rows = [{"row": i, "ids": torch.randint(0, 64, (12,), generator=gen).tolist(),
             "response_from": 5} for i in range(4)]
    trained = probe.panel_nll(argparse.Namespace(model=model, device="cpu"), rows)
    adapter = tmp_path / "adapter"
    model.save_pretrained(str(adapter))
    weights = str(tmp_path / "model")
    _write_tiny_tokenizer(weights)
    resident, _tok, device = load_model_and_tokenizer(
        weights, adapter=str(adapter), device="cpu", dtype="float32")
    ctx = argparse.Namespace(model=resident, device=device)
    with resident.disable_adapter():
        off = probe.panel_nll(ctx, rows)
    parity = probe.base_parity(probe.panel_nll(ctx, rows), trained,
                               adapter=probe.adapter_mismatch(resident, adapter), disabled=off)
    assert parity["adapter"] == {"tensors": 24, "mismatch": [], "mismatched": 0}
    assert parity["tokens_match"] and parity["rows"] == 4
    assert parity["task_loss_resident"] == parity["task_loss_streamed"]
    assert parity["task_loss_resident_adapter_disabled"] != parity["task_loss_streamed"]
    with torch.no_grad():
        param = next(p for n, p in resident.named_parameters() if "lora_B" in n)
        param.add_(0.5)
    assert probe.adapter_mismatch(resident, adapter)["mismatched"] == 1
    assert probe.task_loss(probe.panel_nll(ctx, rows)) != parity["task_loss_streamed"]


def test_base_parity_refuses_rows_the_training_record_never_scored(probe):
    with pytest.raises(RuntimeError, match="no streamed NLL for panel rows"):
        probe.base_parity([{"row": 7, "nll": 1.0, "tokens": 1}],
                          [{"row": 1, "nll": 1.0, "tokens": 1}], adapter={})


def test_a_cuda_run_without_cuda_fails_instead_of_skipping(probe, monkeypatch, tmp_path, capsys):
    """g4-run.sh marks a step done on exit 0, so a missing GPU must not exit 0."""
    monkeypatch.setattr(probe.stream_probe, "cuda_available", lambda: False)
    out = tmp_path / "error.json"
    code = probe.main(["--mode", "error", "--weights", "w", "--out", str(out)])
    assert code == probe.NO_CUDA_EXIT != 0
    assert not out.exists()
    assert "CUDA is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the error experiment's recording store, on the tiny streamed model
# --------------------------------------------------------------------------
def test_recording_store_measures_without_changing_the_step(probe, streamed):
    sched = l2l_tests._load("l2l_schedule")
    acts = l2l_tests._load("l2l_activations")
    codec = l2l_tests._load("g4_activation_codec")
    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = l2l_tests._micro_batches(2)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    plain_losses = step.run(mbs, l2l_tests._cpu_store(acts, 2))
    plain = sched.collect_grads(model)
    names = ("int8", "int4")
    parts = probe._empty_parts(names, 3)
    recorder = probe.RecordingStore(l2l_tests._cpu_store(acts, 2), step, names, parts)
    sched.zero_grads(model)
    losses = step.run(mbs, recorder)
    assert sched.compare(plain, sched.collect_grads(model))["equal"]
    assert sched.losses_equal(plain_losses, losses)
    assert recorder.sample is not None
    table = probe._layer_table(parts, 3)
    for name in names:
        assert [row["layer"] for row in table[name]] == [0, 1, 2]
        assert all(row["restore"]["rows"] == 2 * l2l_tests.SEQ for row in table[name])
    int8 = max(row["restore"]["rms_row_rel"] for row in table["int8"])
    int4 = max(row["restore"]["rms_row_rel"] for row in table["int4"])
    assert 0 < int8 < 0.02 < int4
    # The recorded output error is the layer re-run on the codec's restore.
    x = recorder.sample
    assert codec.restore_error(x, codec.round_trip(x, "int8"))["rms_row_rel"] > 0


# --------------------------------------------------------------------------
# timing driver
# --------------------------------------------------------------------------
class _Watch:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    @contextmanager
    def __call__(self, *, arm, round, run):
        self.calls.append((arm, round, run))
        holder = argparse.Namespace(record=None)
        yield holder
        holder.record = {"outcome": self.outcomes.get(run, "ok")}


def test_comparison_runs_abba_blocks_until_the_repeats_rule_is_met(timing):
    values = {"bf16": 10.0, "int8": 10.4}
    seen = []

    def runner(arm, codec, rnd, run):
        seen.append((arm, codec))
        return {"value": values[codec] + 0.01 * (run % 3), "report": f"r{run}"}

    watch = _Watch({3: "void"})
    result = timing.run_comparison("int8", runner, watch, max_blocks=12)
    assert seen[:4] == [("A", "bf16"), ("B", "int8"), ("B", "int8"), ("A", "bf16")]
    assert result["stopped"] is None
    assert len(result["blocks"]) == 6  # five valid pilot blocks + the voided one replaced
    assert result["blocks_needed"] == 2
    assert [call[2] for call in watch.calls] == list(range(1, 25))


def test_comparison_stops_at_max_blocks(timing):
    def runner(arm, codec, rnd, run):
        return {"value": 10.0 + run, "report": ""}

    result = timing.run_comparison("fp8_e4m3", runner, _Watch({}), max_blocks=5)
    assert len(result["blocks"]) == 5
    assert result["stopped"].startswith("max blocks (5)")


def test_a_second_timing_attempt_keeps_the_first_attempts_files(timing, tmp_path, monkeypatch):
    """A restarted timing step must not write over an earlier attempt: before, a crash
    in the first comparison left no timing.json, and the restart reopened runs/run001-*
    with mode "w"."""
    made = []

    def fake(codec, runner, watch, *, max_blocks, first_run):
        made.append(runner.runs_dir)
        (runner.runs_dir / "run001-A-bf16.log").write_text(f"attempt {len(made)}")
        return {"codec": codec, "blocks": [], "stopped": "fake", "next_run": first_run}

    monkeypatch.setattr(timing, "run_comparison", fake)
    first = tmp_path / "first" / "timing.json"
    first.parent.mkdir()
    assert timing.main(["--codecs", "int8", "--out", str(first)]) == 0
    with pytest.raises(SystemExit):
        timing.main(["--codecs", "int8", "--out", str(first)])
    assert timing.main(["--codecs", "int8", "--out", str(tmp_path / "first" / "t2.json")]) == 0
    assert [path.name for path in made] == ["attempt-01", "attempt-02"]
    assert (made[0] / "run001-A-bf16.log").read_text() == "attempt 1"
    assert json.loads(first.read_text())["attempt"] == 1
    runner = timing.ProbeRunner([], made[0])
    (made[0] / "run002-A-bf16.json").write_text("{}")
    with pytest.raises(FileExistsError):
        runner("A", "bf16", 0, 2)


def test_per_token_arithmetic_is_labelled_and_never_a_speed(probe):
    codec = l2l_tests._load("g4_activation_codec")
    rows = codec.per_token_table()
    mistral = {r["codec"]: r for r in rows if r["shape"] == "mistral-7b-v0.3"}
    assert mistral["bf16"]["bytes_per_token"] == 32 * 4096 * 2
    assert mistral["int8"]["tokens_per_step_factor_at_fixed_ram"] == pytest.approx(
        8192 / 4100)
    assert not any("speed" in key for row in rows for key in row)
    assert math.isclose(mistral["fp8_e4m3"]["bytes_per_token"], 32 * 4100)
