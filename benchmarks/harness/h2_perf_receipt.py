"""Freeze the real token accounting of the H2-close performance workload (CPU, no weights).

    H2_TOKENIZER_DIR=<snapshot tokenizer files> python benchmarks/harness/h2_perf_receipt.py write
    H2_TOKENIZER_DIR=... python benchmarks/harness/h2_perf_receipt.py check

Runs the real Soup data path and TRL dataset preparation for the frozen perf configs with the
real Llama-3.1 tokenizer and a tiny random Llama of the real vocabulary standing in for the 8B
model (quantization none, CPU). Records, per row, the formatted length and the useful
(supervised, shifted) target count, and refuses the workload if any row reaches max_length
(truncation) or if the baseline and Liger arms prepare different token ids or targets.
Unsloth prepares its data inside its own runtime and is not covered here; stage S of
H2_CLOSE_PROTOCOL.txt compares its observed per-step token counts against this receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "benchmarks" / "h2" / "perf_workload_receipt.json"
WORKLOAD = ROOT / "benchmarks" / "h2" / "perf_workload.jsonl"
CONFIGS = {
    "baseline": ROOT / "benchmarks" / "h2" / "candidate_configs" / "perf_baseline.json",
    "liger_ce": ROOT / "benchmarks" / "h2" / "candidate_configs" / "perf_liger_ce.json",
    "unsloth": ROOT / "benchmarks" / "h2" / "candidate_configs" / "perf_unsloth.json",
}
TOKENIZER_SHA256 = {  # H2_SNAPSHOT_RECEIPT.json, unsloth/Meta-Llama-3.1-8B e9a141a2...
    "tokenizer.json": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "tokenizer_config.json": "590b9837d3c2a8790e2aaf920f4c69f9181c3ca8fe52e6579ad3a3f654e59a9b",
    "special_tokens_map.json": "32f404d626cf7b1b6eea36c241ae7cbd6ec29c777a8701ea48c0fe9ebb94c9b1",
}
MIN_TOKENS, MAX_LENGTH = 256, 1024

CHILD = r"""
import json, shutil, sys
from pathlib import Path
import soup_cli.utils.liger as soup_liger
import torch
from soup_cli.config.schema import SoupConfig
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper
from transformers import LlamaConfig, LlamaForCausalLM

root, tokenizer_dir, configs = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
torch.manual_seed(42)
model_dir = root / 'model'
LlamaForCausalLM(LlamaConfig(
    vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
    bos_token_id=128000, eos_token_id=128009, pad_token_id=128004,
    attention_dropout=0.0, tie_word_embeddings=False)).save_pretrained(model_dir)
for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
    shutil.copy(tokenizer_dir / name, model_dir / name)
soup_liger.apply_liger_kernel = lambda model_name: True  # Only sets use_liger_kernel=True.


class CPUWrapper(SFTTrainerWrapper):
    @staticmethod
    def _as_sft_config(args, max_length, packing=False):
        args.use_cpu = True
        return SFTTrainerWrapper._as_sft_config(args, max_length, packing)


out = {}
for arm in ('baseline', 'liger_ce'):
    recipe = json.loads(Path(configs[arm]).read_text())
    recipe['base'] = str(model_dir)
    recipe['data']['train'] = str(Path(recipe['data']['train']).resolve())
    recipe['output'] = str(root / ('output_' + arm))
    recipe['training'].pop('bnb_4bit_quant_storage')
    recipe['training'].update(quantization='none', auto_mixed_precision=False)
    wrapper = CPUWrapper(SoupConfig.model_validate(recipe), device='cpu', report_to='none')
    wrapper.setup(load_dataset(wrapper.config.data))
    trainer = wrapper.trainer
    rows = []
    for row in trainer.train_dataset:
        ids = list(row['input_ids'])
        if 'labels' in row:
            mask = [int(t != -100) for t in row['labels']]
        else:
            mask = [int(m) for m in row['assistant_masks']]
        rows.append({'tokens': len(ids), 'useful_targets': sum(mask[1:]),
                     'ids_sha256': __import__('hashlib').sha256(
                         json.dumps([ids, mask]).encode()).hexdigest()})
    out[arm] = {'columns': sorted(trainer.train_dataset.column_names),
                'use_liger_kernel': bool(trainer.args.use_liger_kernel),
                'max_length': recipe['data']['max_length'], 'rows': rows}
# The actual step order: one real 20-step run of arm A through the stage P runner on CPU.
import h2_perf_abba
_original = SFTTrainerWrapper._as_sft_config


def _cpu_sft_config(args, max_length, packing=False):
    args.use_cpu = True
    return _original(args, max_length, packing)


SFTTrainerWrapper._as_sft_config = staticmethod(_cpu_sft_config)
recipe = h2_perf_abba.load_recipe('A', str(model_dir), root / 'output_steps')
recipe['training'].pop('bnb_4bit_quant_storage')
recipe['training'].update(quantization='none', auto_mixed_precision=False)
report = h2_perf_abba.run_arm('A', recipe, device='cpu')
out['steps'] = {'valid': report['valid'], 'failures': report['failures'],
                'steps_measured': report['steps_measured'],
                'step_tokens': report['h2']['step_tokens'],
                'counted_tokens': report['tokens']}
(root / 'receipt.json').write_text(json.dumps(out))
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(tokenizer_dir: Path, scratch: Path) -> dict:
    for name, expected in TOKENIZER_SHA256.items():
        if _sha(tokenizer_dir / name) != expected:
            raise ValueError(f"tokenizer file {name} does not match the snapshot receipt")
    env = dict(
        os.environ,
        PYTHONPATH="src:benchmarks/harness",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ACCELERATE_USE_CPU="true",
        ACCELERATE_TORCH_DEVICE="cpu",
        TOKENIZERS_PARALLELISM="false",
    )
    scratch.mkdir(parents=True, exist_ok=False)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(scratch),
            str(tokenizer_dir),
            json.dumps({k: str(v) for k, v in CONFIGS.items()}),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])
    arms = json.loads((scratch / "receipt.json").read_text())
    base, liger = arms["baseline"], arms["liger_ce"]
    if len(base["rows"]) != 96 or len(liger["rows"]) != 96:
        raise ValueError("expected 96 prepared rows per arm")
    if [r["ids_sha256"] for r in base["rows"]] != [r["ids_sha256"] for r in liger["rows"]]:
        raise ValueError("baseline and liger_ce prepare different token ids or targets")
    tokens = [r["tokens"] for r in base["rows"]]
    useful = [r["useful_targets"] for r in base["rows"]]
    if max(tokens) >= MAX_LENGTH or min(tokens) < MIN_TOKENS:
        raise ValueError(
            f"row lengths {min(tokens)}..{max(tokens)} outside "
            f"[{MIN_TOKENS}, {MAX_LENGTH}) -- truncation or too short"
        )
    steps = arms["steps"]
    if not steps["valid"] or steps["steps_measured"] != 20 or len(steps["step_tokens"]) != 20:
        raise ValueError(f"CPU step-order run invalid: {steps['failures']}")
    if sum(s["total"] for s in steps["step_tokens"]) > sum(tokens):
        raise ValueError("20 steps saw more tokens than the workload holds")
    configs = {k: {"path": str(v.relative_to(ROOT)), "sha256": _sha(v)} for k, v in CONFIGS.items()}
    return {
        "status": "PERF_WORKLOAD_TOKEN_ACCOUNTING_FROZEN_CPU",
        "scope": "Real tokenizer and Soup/TRL data path, tiny stand-in model, CPU; "
        "no weights, GPU, Liger kernels or Unsloth runtime",
        "workload": {
            "path": str(WORKLOAD.relative_to(ROOT)),
            "sha256": _sha(WORKLOAD),
            "rows": len(tokens),
        },
        "generator_sha256": _sha(Path(__file__).with_name("h2_perf_workload.py")),
        "configs": configs,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "max_length": MAX_LENGTH,
        "tokens": {
            "total": sum(tokens),
            "min": min(tokens),
            "max": max(tokens),
            "mean": round(sum(tokens) / len(tokens), 2),
        },
        "useful_targets": {"total": sum(useful), "min": min(useful), "max": max(useful)},
        "useful_share": round(sum(useful) / sum(tokens), 4),
        "columns": {"baseline": base["columns"], "liger_ce": liger["columns"]},
        "use_liger_kernel": {
            "baseline": base["use_liger_kernel"],
            "liger_ce": liger["use_liger_kernel"],
        },
        "steps": {
            "source": "one 20-step CPU run of arm A through h2_perf_abba.run_arm (tiny model)",
            "steps": 20,
            "warmup": 4,
            "runner_sha256": _sha(Path(__file__).with_name("h2_perf_abba.py")),
            "step_tokens": steps["step_tokens"],
            "counted_useful": steps["counted_tokens"]["useful"],
            "counted_total": steps["counted_tokens"]["total"],
        },
        "rows": [
            {"row": i, "tokens": t, "useful_targets": u, "ids_sha256": r["ids_sha256"]}
            for i, (t, u, r) in enumerate(zip(tokens, useful, base["rows"]))
        ],
        "performance_verdict": "NO_VERDICT",
    }


def main(argv: list[str]) -> int:
    import tempfile

    command = argv[1] if len(argv) > 1 else "check"
    tokenizer_dir = Path(os.environ["H2_TOKENIZER_DIR"])
    with tempfile.TemporaryDirectory(prefix="h2-perf-receipt-") as tmp:
        receipt = build(tokenizer_dir, Path(tmp) / "run")
    text = json.dumps(receipt, indent=2) + "\n"
    if command == "write":
        RECEIPT.write_text(text)
    elif RECEIPT.read_text() != text:
        print("receipt differs from a fresh build", file=sys.stderr)
        return 1
    print(
        json.dumps({k: receipt[k] for k in ("status", "tokens", "useful_targets", "useful_share")})
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
