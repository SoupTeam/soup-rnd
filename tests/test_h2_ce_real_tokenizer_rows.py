"""Frozen 128-row prepared-batch audit with the real Llama-3.1 tokenizer, both arms, on CPU.

Env-gated: skipped unless H2_TOKENIZER_DIR holds the snapshot's tokenizer.json,
tokenizer_config.json and special_tokens_map.json (sha256 checked against the snapshot receipt).
Plumbing only: a tiny random Llama with the real 128256-token vocabulary stands in for the 8B
model, quantization is 'none', the device is CPU and Liger is a spy that only makes Soup set
use_liger_kernel=True. It exercises the real Soup data path and TRL _prepare_dataset, not Liger,
CUDA, NF4 or 8B behavior. The audit is the 5d0cd345 validate_prepared_batches.
Declared deviations from baseline.json: auto_mixed_precision forced False and
bnb_4bit_quant_storage removed (valid only with 4bit); no PatchController is installed; the
1-layer, hidden-16 model does not exercise LoRA r16 at 8B shapes. Recipe settings otherwise
(max_length 512, llama3 template, LoRA targets, GC, batch/accumulation) are the frozen ones.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TOKENIZER_DIR = os.environ.get("H2_TOKENIZER_DIR")
# sha256 from H2_SNAPSHOT_RECEIPT.json of unsloth/Meta-Llama-3.1-8B e9a141a2...
RECEIPT_SHA256 = {
    "tokenizer.json": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "tokenizer_config.json": "590b9837d3c2a8790e2aaf920f4c69f9181c3ca8fe52e6579ad3a3f654e59a9b",
    "special_tokens_map.json": "32f404d626cf7b1b6eea36c241ae7cbd6ec29c777a8701ea48c0fe9ebb94c9b1",
}

pytestmark = pytest.mark.skipif(
    not TOKENIZER_DIR, reason="Set H2_TOKENIZER_DIR to the verified snapshot tokenizer files"
)

CHILD = r"""
import copy, json, shutil, sys
from pathlib import Path
from types import SimpleNamespace

import soup_cli.utils.liger as soup_liger
import torch
from h2_ce_confirmation import validate_prepared_batches
from soup_cli.config.schema import SoupConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper
from transformers import LlamaConfig, LlamaForCausalLM

root, tokenizer_dir = Path(sys.argv[1]), Path(sys.argv[2])
torch.manual_seed(42)
model_dir = root / 'model'
LlamaForCausalLM(LlamaConfig(
    vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    bos_token_id=128000, eos_token_id=128009, pad_token_id=128004,
    attention_dropout=0.0, tie_word_embeddings=False)).save_pretrained(model_dir)
for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
    shutil.copy(tokenizer_dir / name, model_dir / name)
soup_liger.apply_liger_kernel = lambda model_name: True  # Only sets use_liger_kernel=True.
frozen = [json.loads(line) for line in
          Path('benchmarks/h2/prepared_fixture_candidate.jsonl').read_text().splitlines()]


def build(arm):
    class CPUWrapper(SFTTrainerWrapper):
        @staticmethod
        def _as_sft_config(args, max_length, packing=False):
            args.use_cpu = True
            return SFTTrainerWrapper._as_sft_config(args, max_length, packing)

    recipe = json.loads(Path('benchmarks/h2/candidate_configs/baseline.json').read_text())
    recipe['base'] = str(model_dir)
    recipe['data']['train'] = str(Path('benchmarks/h2/synthetic_fixture.jsonl').resolve())
    recipe['output'] = str(root / ('output_' + arm))
    recipe['training'].pop('bnb_4bit_quant_storage')  # Only valid with 4bit.
    recipe['training'].update(quantization='none', auto_mixed_precision=False,
                              use_liger=(arm == 'ce_only'))
    wrapper = CPUWrapper(SoupConfig.model_validate(recipe), device='cpu', report_to='none')
    wrapper.setup(load_dataset(wrapper.config.data))
    return wrapper.trainer


def failure(trainer_like, expected):
    try:
        validate_prepared_batches(trainer_like, expected)
    except Exception as error:
        return type(error).__name__ + ': ' + str(error)
    return None


result = {'rows': len(frozen), 'arms': {}}
trainers = {arm: build(arm) for arm in ('baseline', 'ce_only')}
for arm, trainer in trainers.items():
    result['arms'][arm] = {
        'use_liger_kernel': trainer.args.use_liger_kernel,
        'columns': sorted(trainer.train_dataset.column_names),
        'audit': validate_prepared_batches(trainer, frozen),
    }
corrupted = copy.deepcopy(frozen)
corrupted[64]['labels'][-2] = 1
result['corrupted_label'] = failure(trainers['ce_only'], corrupted)
corrupted = copy.deepcopy(frozen)
corrupted[7]['input_ids'][5] = 1
result['corrupted_token'] = failure(trainers['baseline'], corrupted)
result['dropped_row'] = failure(trainers['ce_only'], frozen[:-1])
(root / 'receipt.json').write_text(json.dumps(result, indent=2))
"""


def test_all_128_frozen_rows_pass_for_both_arms_and_corruption_is_rejected(tmp_path):
    source = Path(TOKENIZER_DIR)
    for name, expected in RECEIPT_SHA256.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    env = dict(
        os.environ,
        PYTHONPATH="src:benchmarks/harness",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ACCELERATE_USE_CPU="true",
        ACCELERATE_TORCH_DEVICE="cpu",
        TOKENIZERS_PARALLELISM="false",
    )
    root = tmp_path / "run"
    root.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(root), str(source)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["rows"] == 128
    fields = ["attention_mask", "input_ids", "labels"]
    baseline, liger = receipt["arms"]["baseline"], receipt["arms"]["ce_only"]
    assert baseline["use_liger_kernel"] is False and liger["use_liger_kernel"] is True
    for arm in (baseline, liger):
        audit = arm["audit"]
        assert audit["status"] == "PASS_EXACT_COLLATED_PREPARED_ROWS"
        assert audit["rows"] == 128 and audit["fields"] == fields
        assert audit["dataset_columns"] == arm["columns"]
        assert audit["collator"]["pad_token_id"] == 128004
        assert audit["collator"]["padding_free"] is False
    assert liger["columns"] == ["assistant_masks", "input_ids"]
    assert {"attention_mask", "labels"} <= set(baseline["columns"])
    for name in ("corrupted_label", "corrupted_token", "dropped_row"):
        assert receipt[name] is not None and receipt[name].startswith("ValueError: "), name
        assert "prepared" in receipt[name]
