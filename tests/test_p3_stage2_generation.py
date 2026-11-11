"""Uncertain Stage2 generation boundaries, without downloads or generated-code execution.

Only the integration owner runs these tests. These fixtures do not qualify GPU
kernels, native BF16 accumulation, checkpoint identity, or final task coverage.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest


@pytest.fixture
def generation():
    from benchmarks.harness import p3_stage2_generation

    return p3_stage2_generation


@pytest.fixture
def tokenizer():
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<|endoftext|>": 0, "<unk>": 1}
    vocabulary.update({
        character: index + 2
        for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    })
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        pad_token="<|endoftext|>", unk_token="<unk>", model_max_length=2048,
    )


def _record(index=0):
    from benchmarks.harness.p3_stage2_apps import GENERATION_POLICY

    prompt = f"QUESTION:\nSynthetic fixture {index}\nANSWER:\n"
    return {
        "benchmark": "apps", "task_id": f"apps/fixture/{index}", "source_id": index,
        "source_ref": f"synthetic@fixture-v1/test/records.jsonl#row={index}",
        "source_row_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation_policy": copy.deepcopy(GENERATION_POLICY),
    }


def test_left_padding_masks_prompt_eos_and_slices_after_full_batch_width(generation, tokenizer):
    # Padding by token value erases a legitimate prompt EOS; unpadded-length slicing leaks input.
    torch = pytest.importorskip("torch")
    inputs, mask = generation._left_pad([[7, 0, 9], [5]], device=torch.device("cpu"))
    assert inputs.tolist() == [[7, 0, 9], [0, 0, 5]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    answer = tokenizer.encode("answer", add_special_tokens=False)
    row = generation._extract_continuation(
        tokenizer, [0, 0, 5, *answer, 0, 0, 0], input_width=3,
        terminal_step=len(answer) + 1, budget=32, stop_strings=(),
    )
    assert row["raw_continuation_token_ids"] == [*answer, 0]
    assert row["returned_continuation_token_ids"] == [*answer, 0, 0, 0]
    assert row["batch_padding_token_ids"] == [0, 0]
    assert row["scored_continuation"] == "answer"
    assert row["new_token_count"] == len(answer) + 1
    assert row["stop_reason"] == "eos"
    assert row["eos_emitted"] is True
    assert row["cap_hit"] is False
    assert row["scored_continuation_sha256"] == hashlib.sha256(b"answer").hexdigest()


@pytest.mark.parametrize(
    ("text", "append_eos", "budget", "stops", "expected", "reason", "cap_hit"),
    [
        ("ok\nQuestion:ignored", True, 64, ("\nQuestion:",), "ok", "literal_stop", False),
        ("ok", True, 3, ("\nQuestion:",), "ok", "eos", True),
        ("ok", False, 2, (), "ok", "max_new_tokens", True),
        ("", True, 1, (), "", "eos", True),
        ("x\nclass Later\ndef EarlierByListOrder", False, 64,
         ("\ndef", "\nclass"), "x", "literal_stop", False),
    ],
)
def test_stop_prefix_uses_first_text_offset_and_eos_precedes_cap(
    generation, tokenizer, text, append_eos, budget, stops, expected, reason, cap_hit,
):
    # Taking the last stop or treating a cap/EOS tie as a cap changes the frozen scored bytes.
    ids = tokenizer.encode(text, add_special_tokens=False) + ([0] if append_eos else [])
    row = generation._extract_continuation(
        tokenizer, [99, *ids], input_width=1, terminal_step=len(ids), budget=budget,
        stop_strings=stops,
    )
    assert row["scored_continuation"] == expected
    assert row["stop_reason"] == reason
    assert row["cap_hit"] is cap_hit
    assert row["eos_emitted"] is append_eos
    assert row["new_token_count"] == len(ids)
    assert row["raw_continuation_token_ids"] == ids


def test_literal_stop_inside_one_token_beats_cap_without_inventing_scored_ids(
    generation, tokenizer,
):
    # Literal trimming can split a token; retokenization would misstate what the model generated.
    torch = pytest.importorskip("torch")
    tokenizer.add_tokens(["ok\nclassTail"])
    ids = tokenizer.encode("ok\nclassTail", add_special_tokens=False)
    assert len(ids) == 1
    tracker = generation._make_stop_tracker(
        tokenizer, ("\nclass",), input_width=1, batch_size=1, budget=1,
        device=torch.device("cpu"),
    )
    assert tracker(torch.tensor([[7, *ids]], dtype=torch.long), None).tolist() == [True]
    row = generation._extract_continuation(
        tokenizer, [7, *ids], input_width=1, terminal_step=1, budget=1,
        stop_strings=("\nclass",),
    )
    assert row["raw_continuation"] == "ok\nclassTail"
    assert row["scored_continuation"] == "ok"
    assert row["raw_continuation_token_ids"] == ids
    assert row["scored_token_ids"] is None
    assert row["new_token_count"] == 1
    assert row["stop_reason"] == "literal_stop"
    assert row["stop_character_offset"] == 2
    assert row["cap_hit"] is True
    assert row["eos_emitted"] is False


def test_generated_eos_excludes_literal_text_after_eos(generation, tokenizer):
    # Looking for stops in the entire returned tensor misclassifies EOS padding and later bytes.
    ids = tokenizer.encode("ok", add_special_tokens=False)
    tail = tokenizer.encode("\nQuestion:", add_special_tokens=False)
    with pytest.raises(ValueError, match="non-padding tokens after termination"):
        generation._extract_continuation(
            tokenizer, [13, *ids, 0, *tail], input_width=1, terminal_step=len(ids) + 1,
            budget=32, stop_strings=("\nQuestion:",),
        )


def test_hf_stops_ignore_prompt_suffix_and_finish_rows_independently(generation, tokenizer):
    # Whole-input stopping can complete a stop across the prompt boundary and kill the row early.
    torch = pytest.importorskip("torch")
    prefix = tokenizer.encode("prompt\n", add_special_tokens=False)
    width = len(prefix)
    tracker = generation._make_stop_tracker(
        tokenizer, ("\nclass",), input_width=width, batch_size=2, budget=32,
        device=torch.device("cpu"),
    )
    boundary_only = tokenizer.encode("class", add_special_tokens=False)
    other = tokenizer.encode("xxxxx", add_special_tokens=False)
    first = torch.tensor([prefix + boundary_only, prefix + other], dtype=torch.long)
    assert tracker(first, None).tolist() == [False, False]
    continuation = tokenizer.encode("class\nclass", add_special_tokens=False)
    second_other = tokenizer.encode("xxxxxxxxxxx", add_special_tokens=False)
    second = torch.tensor([prefix + continuation, prefix + second_other], dtype=torch.long)
    assert tracker(second, None).tolist() == [True, False]
    first_step = len(continuation)
    third = torch.cat((second, torch.tensor([[0], [0]], dtype=torch.long)), dim=1)
    assert tracker(third, None).tolist() == [True, True]
    assert tracker.terminal_steps.tolist() == [first_step, first_step + 1]
    assert tracker.literal_stops.tolist() == [True, False]
    assert tracker.eos_stops.tolist() == [False, True]


def test_context_admission_reserves_entire_budget_for_longest_padded_input(generation):
    # Prompt-only checks admit beyond-context generation; reducing the budget is forbidden.
    assert generation._check_context([[3, 4], [5]], context_limit=6, budget=4) == 2
    with pytest.raises(ValueError, match="context budget exceeded"):
        generation._check_context([[3, 4, 5], [6]], context_limit=6, budget=4)


@pytest.mark.parametrize("field", [
    "task_id", "source_id", "source_ref", "source_row_sha256", "prompt_sha256",
])
def test_missing_identity_refuses_shard_without_loading_model(generation, tmp_path, field):
    # Missing provenance must invalidate the shard rather than generate a smaller denominator.
    record = _record()
    del record[field]
    report = generation.run_generation(
        None, None, [record], output_dir=tmp_path / "missing", benchmark="apps", seed=23,
        length=1024, arm="bf16_l2l", protocol_sha256="a" * 64, checkpoint_sha256="b" * 64,
    )
    assert report["status"] == "FAILED"
    assert report["completed_rows"] == 0
    assert report["selected_shard_complete"] is False
    failure = json.loads((tmp_path / "missing" / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "record_admission"
    assert field in failure["message"]
    assert (tmp_path / "missing" / "generations.jsonl").read_bytes() == b""


@pytest.mark.parametrize(
    "duplicate", ["task_id", "source_row_sha256", "prompt_sha256", "source_id"],
)
def test_distinct_ids_do_not_hide_duplicate_source_or_prompt_hashes(generation, duplicate):
    # Only checking task IDs admits duplicate source problems under renamed identities.
    first, second = _record(0), _record(1)
    second[duplicate] = first[duplicate]
    if duplicate == "prompt_sha256":
        second["prompt"] = first["prompt"]
    with pytest.raises(ValueError, match="duplicate"):
        generation._validated_records([first, second], "apps")


def test_identity_hash_mismatch_is_not_silently_recomputed(generation):
    # Replacing a stale hash erases evidence that the frozen prompt was changed.
    record = _record()
    record["prompt"] += " changed"
    with pytest.raises(ValueError, match="prompt_sha256 mismatch"):
        generation._validated_records([record], "apps")


def test_frozen_generation_policy_cannot_change_budget_or_add_answer_selection(generation):
    # Accepting arbitrary caller policies permits an undeclared endpoint after final outcomes.
    for changes in ({"max_new_tokens": 512}, {"postprocessing": "remove_fences"}):
        record = _record()
        record["generation_policy"].update(changes)
        with pytest.raises(ValueError, match="frozen generation_policy"):
            generation._validated_records([record], "apps")


def test_configuration_and_mixed_module_modes_restore_on_failure(generation, tokenizer):
    # Restoring only model.training recursively destroys a caller's mixed submodule modes.
    torch = pytest.importorskip("torch")
    from transformers import GenerationConfig, GraniteMoeConfig

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    model.train()
    model[0].eval()
    model.config = GraniteMoeConfig(use_cache=False)
    model.generation_config = GenerationConfig(use_cache=False)
    tokenizer.padding_side = "right"
    tokenizer.backend_tokenizer.enable_padding(length=11, pad_id=0, pad_token="<|endoftext|>")
    tokenizer.backend_tokenizer.enable_truncation(max_length=17)
    modes = [module.training for module in model.modules()]
    config = copy.deepcopy(model.config.__dict__)
    generation_config = copy.deepcopy(model.generation_config.__dict__)
    padding = tokenizer.backend_tokenizer.padding
    truncation = tokenizer.backend_tokenizer.truncation
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(RuntimeError, match="deliberate generation failure"):
        with generation._inference_session(model, tokenizer):
            assert all(not module.training for module in model.modules())
            assert torch.is_grad_enabled() is False
            assert model.config.use_cache is False
            model.config.use_cache = True
            model.generation_config.do_sample = True
            tokenizer.padding_side = "left"
            tokenizer.backend_tokenizer.no_padding()
            tokenizer.backend_tokenizer.no_truncation()
            raise RuntimeError("deliberate generation failure")
    assert [module.training for module in model.modules()] == modes
    assert model.config.__dict__ == config
    assert model.generation_config.__dict__ == generation_config
    assert tokenizer.padding_side == "right"
    assert tokenizer.backend_tokenizer.padding == padding
    assert tokenizer.backend_tokenizer.truncation == truncation
    for original, parameter in zip(parameters, model.parameters(), strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_compute_log_parser_uses_typed_value_not_descriptor(generation, tmp_path):
    raw = (
        "computeType: type=cublasComputeType_t; val=CUBLAS_COMPUTE_32F(68)\n"
        "computeType: type=cublasComputeType_t;\n"
        "computeType=COMPUTE_32F\n"
        '"compute_type":68\n'
    ).encode()
    source = tmp_path / "library.log"
    source.write_bytes(raw)
    output = tmp_path / "capture"
    output.mkdir()
    result = generation._capture_logs(
        [{"library": "cublas", "path": str(source), "byte_offset": 0}], output, "phase",
    )
    fields = {entry["verbatim_value"]: entry["count"] for entry in result["compute_type_fields"]}
    assert fields == {"CUBLAS_COMPUTE_32F": 1, "COMPUTE_32F": 1, "68": 1}
    assert (output / "phase.cublas.log").read_bytes() == raw
