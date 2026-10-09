"""Inference-only Soup bundled scores for a saved E2 experiment adapter.

Run from the soup-rnd root. This does not train, merge, publish, or install E2.
The caller sets CUBLAS_WORKSPACE_CONFIG before starting Python.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from soup_cli.eval import gate_suites as suites
from soup_cli.eval.gate import write_baseline_file

REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", action="append", choices=suites.DEFAULT_GENERAL_SUITE)
    args = parser.parse_args()
    requested = args.suite or ["mini_mmlu"]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate suite names")

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This protocol expects two visible CUDA GPUs")

    cfg_path = Path(args.training_config)
    report_path = Path(args.quality_report)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    quality = json.loads(report_path.read_text(encoding="utf-8"))
    base_path = Path(cfg["base"])
    adapter_path = Path(cfg["output"])
    if base_path.name != REVISION or not (base_path / "config.json").is_file():
        raise ValueError("The original pinned Mistral checkpoint is not available")
    if quality["steps"] != 4000:
        raise ValueError("Expected a completed 4000-step quality run")
    if cfg["training"]["quantization"] != "none":
        raise ValueError("Expected the original non-quantized experiment")
    adapter_cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    if Path(adapter_cfg["base_model_name_or_path"]).resolve() != base_path.resolve():
        raise ValueError("Adapter declares a different base checkpoint")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Expected saved adapter_model.safetensors")
    weights_path = report_path.with_suffix(".weights.pt")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if suites.BUNDLED_SCORER_REVISION != 4:
        raise RuntimeError("Scorer revision differs from the reviewed source")
    if suites.bundled_scorer_fingerprint() != suites.BUNDLED_SCORER_FINGERPRINT:
        raise RuntimeError("Bundled scorer self-check failed")

    out = Path(args.output_dir).resolve()
    out.relative_to(Path.cwd().resolve())
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), local_files_only=True, trust_remote_code=False,
    )
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("Saved tokenizer has no single chat template; do not substitute one")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Saved tokenizer must define EOS and PAD")

    print("Loading original base + SAVED adapter:", adapter_path, flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path), dtype=torch.float16, device_map="auto",
        max_memory={0: "13GiB", 1: "13GiB"},
        local_files_only=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base, str(adapter_path), is_trainable=False,
        autocast_adapter_dtype=True, local_files_only=True,
    )
    model.eval()
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("CPU/disk/meta placement detected; stop and inspect the load")
    if model.active_adapters != ["default"]:
        raise RuntimeError("The saved default adapter is not the active adapter")

    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    loaded = dict(model.named_parameters())
    if set(saved) != {name for name in loaded if ".lora_" in name}:
        raise RuntimeError("Saved and loaded LoRA tensor names differ")
    digest = hashlib.sha256()
    adapter_dtypes = set()
    for name in sorted(saved):
        actual = loaded[name].detach().cpu().contiguous()
        expected = saved[name].detach().cpu().contiguous()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(f"Saved adapter shape/dtype changed: {name}")
        if not torch.isfinite(actual).all().item() or not torch.equal(actual, expected):
            raise RuntimeError(f"Saved adapter values changed: {name}")
        adapter_dtypes.add(str(actual.dtype))
        digest.update(name.encode())
        digest.update(actual.float().numpy().tobytes())
    if digest.hexdigest() != quality["trainable_fingerprint"]:
        raise RuntimeError("Loaded LoRA fingerprint differs from the training report")
    print("Saved adapter weights verified:", len(saved), "tensors", flush=True)
    del saved, loaded

    budget = suites.BEHAVIOURAL_MAX_NEW_TOKENS
    generation = GenerationConfig(
        max_new_tokens=budget, do_sample=False, num_beams=1,
        use_cache=True, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
    )
    input_device = model.get_input_embeddings().weight.device
    records = []

    def generate(prompt: str) -> str:
        row = {"prompt_sha256": sha(prompt.encode())}
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            batch = tokenizer(
                text, return_tensors="pt", add_special_tokens=False, truncation=False,
            )
            length = int(batch["input_ids"].shape[1])
            if length > 2048 or length + budget > model.config.max_position_embeddings:
                raise ValueError("Prompt exceeds the evaluation budget; no truncation applied")
            row["input_ids_sha256"] = sha(batch["input_ids"].numpy().tobytes())
            batch = {key: value.to(input_device) for key, value in batch.items()}
            with torch.inference_mode():
                output = model.generate(**batch, generation_config=generation)
            new_ids = output[0, length:].detach().cpu()
            answer = tokenizer.decode(new_ids, skip_special_tokens=True)
            row.update({
                "input_tokens": length,
                "new_tokens": int(new_ids.numel()),
                "output_sha256": sha(answer.encode()),
                "empty_output": not bool(answer.strip()),
                "hit_generation_limit": int(new_ids.numel()) >= budget,
            })
            return answer
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            records.append(row)

    results = {}
    for name in requested:
        items = (
            suites.MINI_BENCHMARKS[name]
            if name in suites.MINI_BENCHMARKS
            else suites.load_suite_items(name)
        )
        if not items:
            raise RuntimeError(f"Empty suite: {name}")
        start = len(records)
        print("Scoring:", name, "items:", len(items), flush=True)
        try:
            score = float(suites.score_bundled_suite(name, generate))
        finally:
            (out / f"{name}.generation_audit.json").write_text(
                json.dumps(records[start:], indent=2) + "\n", encoding="utf-8",
            )
        trace = records[start:]
        if any("error_type" in row for row in trace) or len(trace) != len(items):
            raise RuntimeError(f"Invalid generation run for {name}; inspect the audit file")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"Invalid score: {name}")
        results[name] = {
            "score": score, "items": len(items), "generation_errors": 0,
            "empty_outputs": sum(row["empty_output"] for row in trace),
            "outputs_at_token_limit": sum(row["hit_generation_limit"] for row in trace),
            "fixture_sha256": sha(json.dumps(items, sort_keys=True).encode()),
        }
        print(json.dumps({name: results[name]}, indent=2), flush=True)

    payload = {
        "role": (
            "all-layer LoRA control"
            if cfg["training"]["lora"]["top_k_layers"] == 32
            else "Top-K candidate"
        ),
        "k": cfg["training"]["lora"]["top_k_layers"],
        "source_config": str(cfg_path), "source_quality_report": str(report_path),
        "source_quality_report_sha256": sha(report_path.read_bytes()),
        "base_revision": REVISION, "adapter_path": str(adapter_path),
        "saved_adapter_weights_verified": True,
        "loaded_lora_fingerprint": digest.hexdigest(),
        "adapter_dtypes": sorted(adapter_dtypes), "base_dtype": str(base.dtype),
        "device_map": {key: str(value) for key, value in base.hf_device_map.items()},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": 42, "deterministic_algorithms": True,
        "generation": generation.to_dict(),
        "prompt_template_sha256": sha(tokenizer.chat_template.encode()),
        "prompt_token_limit": 2048, "prompt_truncation": False,
        "e2_activation_cache_installed": False,
        "scorer_revision": suites.BUNDLED_SCORER_REVISION,
        "scorer_fingerprint": suites.BUNDLED_SCORER_FINGERPRINT,
        "scores": results, "research_verdict": "NOT_ASSESSED",
    }
    (out / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    write_baseline_file(str(out / "scores.json"), {k: v["score"] for k, v in results.items()})
    print("Saved:", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
