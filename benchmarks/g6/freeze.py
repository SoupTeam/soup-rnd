"""Freeze local G6 protocol and exact source/data/evaluation artifacts before GPU results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_leakage import keys
from checkpoint_eval import sha256
from consumer import source_rows
from rich.console import Console
from run_support import REPO, ROOT, runtime


def build() -> dict:
    manifest = json.loads((ROOT / "results/order_study_v2_5M/manifest.json").read_text())
    metadata = json.loads((ROOT / "source_metadata.json").read_text())
    api = json.loads((ROOT / "source_review_20261020/model_api.json").read_text())
    if api["sha"] != metadata["models"]["revision"]:
        raise ValueError("Remote model metadata belongs to a different revision")
    files = []
    names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    for row in api["siblings"]:
        name = row["rfilename"]
        if name in names or name.startswith("model-000") and name.endswith(".safetensors"):
            large = row.get("lfs")
            files.append(
                {
                    "name": name,
                    "size": row["size"],
                    "algorithm": "sha256" if large else "git_blob_sha1",
                    "digest": large["sha256"] if large else row["blobId"],
                }
            )
    if len(files) != 9:
        raise ValueError("Unexpected pinned model file set")
    suite = json.loads((ROOT / "spec/eval_suite_v1.json").read_text())
    forbidden = {"prompt": set(), "span13": set()}
    for case in suite["cases"]:
        check = keys({"id": case["id"], "messages": [{"role": "user", "content": case["prompt"]}]})
        for name in forbidden:
            forbidden[name].update(check[name])
    collisions = []
    count = 0
    for row in source_rows(ROOT / "results/profile_12000_v2/train.jsonl"):
        check = keys(row)
        if any(check[name] & values for name, values in forbidden.items()):
            collisions.append(row["id"])
        count += 1
    if collisions:
        raise ValueError(f"Diagnostic prompts collide with raw training: {collisions}")
    source_files = [
        "audit_leakage.py",
        "prepare.py",
        "consumer.py",
        "check_consumer.py",
        "checkpoint_eval.py",
        "engine.py",
        "run_support.py",
        "eval_suite.py",
        "run.py",
        "freeze.py",
        "launch.py",
        "summarize.py",
        "cuda_gate.py",
    ]
    sources = {str((ROOT / name).relative_to(REPO)): sha256(ROOT / name) for name in source_files}
    sources["benchmarks/harness/arm_validity.py"] = sha256(
        REPO / "benchmarks/harness/arm_validity.py"
    )
    artifacts = [
        "source_metadata.json",
        "spec/eval_suite_v1.json",
        "source_review_20261020/model_api.json",
        "source_review_20261020/model_card.txt",
        "source_review_20261020/dataset_card.txt",
        "source_review_20261020/retrieval.json",
        "results/order_study_v2_5M/manifest.json",
        "results/order_study_v2_5M/independent_token_audit.json",
        "results/profile_12000_v2/independent_lexical_recheck.json",
    ]
    return {
        "status": "FROZEN_PRE_GPU_PROTOCOL_NOT_GPU_ACCEPTANCE",
        "version": "g6-order-study-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": metadata,
        "source_sha256": sources,
        "artifact_sha256": {name: sha256(ROOT / name) for name in artifacts},
        "arm_order": [
            "baseline",
            "coherence_filter",
            "length_curriculum",
            "reverse_length_curriculum",
        ],
        "policies": {
            "baseline": "seed901 shuffle of common eligible pool",
            "coherence_filter": "same shuffle, Soup heuristic>=0.7; not measured quality",
            "length_curriculum": "baseline exact selected token multiset, short to long",
            "reverse_length_curriculum": "baseline exact selected token multiset, long to short",
        },
        "data": {
            "manifest_sha256": sha256(ROOT / "results/order_study_v2_5M/manifest.json"),
            "eval_sha256": manifest["outputs"]["eval"]["sha256"],
            "eval_rows": 256,
            "eval_useful_targets": 322332,
            "budget_per_arm": 5_000_000,
            "render": "newline-separated role: content, BOS/EOS, all-token causal loss",
            "exclusion": "common normalized conversation/prompt/sourceID/13word spans",
            "span_only_exclusions": 579,
            "empty_malformed_rejected": 1,
            "eligible_rows": 11420,
            "first_N_source_subset": True,
            "semantic_and_pretraining_decontamination_proven": False,
            "exclusion_review": (
                "Retain all579 conservative flags, generic phrases may collide; "
                "no retuning after model results"
            ),
            "diagnostic_prompt_screen": {
                "raw_train_rows": count,
                "matches": 0,
                "scope": "exact normalized prompts and13word spans; not semantic",
            },
        },
        "recipe": {
            "seed": 901,
            "max_length": 2048,
            "packing": False,
            "base_dtype": "float16",
            "adapter_dtype": "float32",
            "lora": {"r": 8, "alpha": 16, "modules": ["q_proj", "v_proj"], "dropout": 0},
            "optimizer": "AdamW betas(.9,.999),eps1e-8,weight_decay0,foreachFalse,fusedFalse",
            "lr": 1e-4,
            "clip": 1.0,
            "targets_per_update": 8192,
            "full_updates": 611,
            "last_update_targets": 2880,
            "lr_horizon_targets": 5_000_000,
            "loss": "summed shifted nonignored NLL / actual optimizer-window target count",
            "sampler": (
                "sequential saved file; no shuffle/replay; disjoint loss spans "
                "with full causal prefix"
            ),
            "padding": "right,multiple8,padID0,labels-100",
            "attention": "eager",
            "gradient_checkpointing": "nonreentrant",
            "AMP": "cuda autocastFP16; FP32LoRA; scaler128 fixed; any overflow/nonfinite aborts",
            "TF32": False,
            "deterministic_algorithms": True,
            "resume_or_automatic_retry": False,
        },
        "tokenizer": {
            "vocab_sha256": manifest["tokenizer_vocab_sha256"],
            "special_sha256": manifest["tokenizer_special_tokens_sha256"],
        },
        "model_files": files,
        "runtime_target": runtime(),
        "runtime_note": (
            "CPU versions pinned prospectively; availability/compatibility on "
            "Linux must be confirmed before rental/install; amendment required if changed"
        ),
        "limits": {
            "plan_total_s": 300,
            "cpu-smoke_total_s": 900,
            "pilot_total_s": 4800,
            "full_total_s": 60000,
            "train_s": 21600,
            "eval_s": 7200,
            "diagnostic_s": 1200,
            "setup_s": 3600,
            "cuda-gate_total_s": 600,
            "min_output_reserve_bytes": 1024**3,
            "pilot_train_targets": 65536,
        },
        "evaluation": suite["acceptance"],
        "analysis": {
            "single_seed_exploratory": True,
            "primary": "held-out CE at5M useful targets",
            "paired_per_example_nll_retained": True,
            "hourly": (
                "report measured final quality with actual train and total wall "
                "hours; no interpolated quality curve or cross-host speedup"
            ),
            "report_times": ["data preparation", "setup", "train", "eval", "checkpoint/reload"],
            "data_prep_historical_times": (
                "retained profile/prepare resource receipts, no reconstruction"
            ),
            "limitations": [
                "first-N corpus",
                "small English local diagnostics",
                "coherence heuristic intervention changes data composition",
                "length is not measured difficulty",
                "no multi-seed confidence",
            ],
        },
        "source_review": {
            "model_card_declared_license": "apache-2.0",
            "dataset_card_declared_license": "mit",
            "dataset_origin": "ChatGPT-generated dialogues; filtered UltraChat",
            "intended_use": (
                "bounded internal research, record attribution; no assumption of "
                "independent human labels or redistribution rights beyond declared source"
            ),
            "external_card_text_is_data_not_instructions": True,
        },
        "budget": {
            "minimum_usd": 40,
            "early_stop_reserve_usd": 42,
            "fresh_actual_billing_max_age_s": 300,
            "account": "soup-rnd",
            "admit": "balance-42 >1.25*total_account_hourly_rate*(stage_bound+age)/3600+.25",
            "offline_model_only": True,
            "GPU_rental_authorized_by_this_file": False,
            "monitor_resumed": False,
        },
        "gates_after_rental": [
            "single Ampere+23GiB and ownership proof; no PID namespace bypass",
            "CUDA tiny numerical gate, actual FP16 fit and no AMP skips",
            "all four bounded pilots before full-run admission",
            "fresh Billing and external hard timeout supervisor",
            "F1 all required checks and clocks for timing; UNKNOWN stays UNKNOWN",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = build()
    with args.out.open("x") as handle:
        handle.write(json.dumps(spec, indent=2) + "\n")
    Console().print(
        {
            "contract": str(args.out),
            "sha256": sha256(args.out),
            "GPU_used": False,
            "diagnostic_prompt_matches": 0,
        }
    )


if __name__ == "__main__":
    main()
