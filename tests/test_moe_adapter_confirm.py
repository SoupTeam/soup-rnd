"""The E1 confirmation wrapper (benchmarks/harness/moe_adapter_confirm.py, record section 11).

C must share no source question with an E1 split, also after normalising and as a
near-duplicate, and the per-row NLL it records must add up to the measured harness's
own NLL, or the confirmation rule refuses the record.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

WRAPPER = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_confirm.py"
USED = "Tom has 3 apples and buys 4 more apples at the shop. How many apples does Tom have now?"


@pytest.fixture(scope="module")
def confirm():
    original_path = sys.path[:]
    spec = importlib.util.spec_from_file_location("_moe_adapter_confirm", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(spec.name, None)
        sys.modules.pop("_e1_measured_harness", None)


@pytest.fixture(scope="module")
def harness(confirm):
    return confirm.load_measured_harness()


def _rows_in_walk_order(harness, questions):
    """Rows whose position in E1's shuffle is the index in ``questions``."""
    order = list(range(len(questions)))
    random.Random(harness.SPLIT_SEED).shuffle(order)
    rows = [None] * len(questions)
    for position, index in enumerate(order):
        rows[index] = {"query": questions[position], "response": "The answer is: 7",
                       "original_question": questions[position], "type": "GSM_AnsAug"}
    return rows, order


def test_c_skips_used_normalised_and_near_duplicate_questions(confirm, harness):
    questions = [
        USED, "A train travels 60 miles in 1 hour. How far does it go in 5 hours?",
        "What is 12 times 12?",
        USED,                                                      # exact
        ("  tom HAS 3 apples, and buys 4 more apples at the shop -- how many apples "
         "does Tom have now "),                                    # normalised
        USED.replace("now?", "now in total?"),                     # Jaccard 16/18
        "Sara reads 20 pages a day. How many pages does she read in a week?",
        "Sara reads 20 pages a day. How many pages does she read in a week?",
        "If a square has side 9 cm, what is its area?",
        "Tom has 3 pears. How many pears are left after he eats one?",
    ]
    rows, order = _rows_in_walk_order(harness, questions)
    split = {"calibration": [order[0]], "train": [order[1]], "eval": [order[2]],
             "examined": 3}

    chosen = confirm.select_confirmation(harness, rows, split, [], lambda row: None,
                                         wanted=3)

    assert chosen["eval"] == [order[6], order[8], order[9]]
    assert chosen["dropped"] == {"same_group": 1, "used_source_question": 1,
                                 "used_after_normalising": 1, "near_duplicate": 1}
    assert chosen["walk_start"] == 3 and chosen["examined"] == 7


def test_c_also_avoids_an_excluded_split(confirm, harness):
    other = "How many legs do 4 spiders have?"
    questions = [USED, "What is 2 + 2?", "What is 3 + 3?", other, "What is 5 + 5?"]
    rows, order = _rows_in_walk_order(harness, questions)
    split = {"calibration": [order[0]], "train": [order[1]], "eval": [order[2]],
             "examined": 3}
    withdrawn = {"calibration": [], "train": [], "eval": [order[3]]}

    chosen = confirm.select_confirmation(harness, rows, split, [withdrawn],
                                         lambda row: None, wanted=1)

    assert chosen["eval"] == [order[4]]
    assert chosen["dropped"]["used_source_question"] == 1


def test_a_split_reaching_past_its_examined_count_is_refused(confirm, harness):
    questions = [USED, "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?"]
    rows, order = _rows_in_walk_order(harness, questions)
    split = {"calibration": [order[0]], "train": [order[1]], "eval": [order[2]],
             "examined": 2}

    with pytest.raises(ValueError, match="past its own 'examined'"):
        confirm.select_confirmation(harness, rows, split, [], lambda row: None, wanted=1)


def test_check_confirmation_refuses_a_shared_source_question(confirm, harness):
    questions = [USED, "What is 2 + 2?", "What is 3 + 3?", USED]
    rows, order = _rows_in_walk_order(harness, questions)
    split = {"calibration": [order[0]], "train": [order[1]], "eval": [order[2]]}

    with pytest.raises(ValueError, match="shares a source question"):
        confirm.check_confirmation(harness, rows, split, {"eval": [order[3]]})
    confirm.check_confirmation(harness, rows, split, {"eval": []})


def test_per_row_nll_adds_up_to_the_measured_evaluate_nll(confirm, harness):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(vocab_size=96, hidden_size=32, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         num_experts=8, num_experts_per_tok=2, pad_token_id=0, bos_token_id=1,
                         eos_token_id=2)
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    generator = random.Random(3)
    items = []
    for prompt_len, answer_len in ((5, 3), (9, 1), (4, 12), (7, 6), (3, 2)):
        ids = [generator.randrange(3, 96) for _ in range(prompt_len + answer_len)]
        items.append((ids[:prompt_len], ids))

    rows = confirm.per_row_nll(harness, model, items, 0, "cpu", micro=2)
    whole = harness.evaluate_nll(model, items, 0, "cpu", 2)

    assert rows["tokens"] == [3, 1, 12, 6, 2]
    assert sum(rows["tokens"]) == whole["tokens"]
    assert sum(rows["nll_sum"]) / sum(rows["tokens"]) == pytest.approx(whole["nll"], rel=1e-6)
