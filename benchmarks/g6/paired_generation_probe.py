"""Separate prospective inference-only probe; never retrains or replaces v1 scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from eval_suite import score_task
from turn_boundary_candidate import inspect_prefix


def admit_probe(spec: dict, admission: dict, spec_sha: str, now: float) -> None:
    """Separate12USD authority; do not modify or trick the historical5USD guard."""
    if admission.get("account") != "soup-rnd" or admission.get("mode") != "generation-probe":
        raise ValueError("Wrong account or stage")
    if admission.get("contract_sha256") != spec_sha:
        raise ValueError("Admission contract mismatch")
    for key in ("observed_unix_s", "balance_usd", "total_account_rate_usd_h"):
        value = admission.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Invalid actual Billing numbers")
    age = now - admission["observed_unix_s"]
    if not 0 <= age <= 300:
        raise ValueError("Fresh actual Billing required")
    budget = spec["budget"]
    if type(spec["total_s"]) is not int or not 0 < spec["total_s"] <= 2400:
        raise ValueError("Invalid bounded inference duration")
    opening = budget["opening_balance_usd"]
    if (type(opening) not in (int, float) or not math.isfinite(opening)
            or opening <= 12 or budget["total_cap_usd"] != 12):
        raise ValueError("This user authority covers only the fixed total12USD")
    rate = admission["total_account_rate_usd_h"]
    if rate <= 0 or admission["balance_usd"] > opening + 0.01:
        raise ValueError("Invalid rate or unreviewed top-up")
    required = 1.25 * rate * (spec["total_s"] + age) / 3600 + 0.50
    if admission["balance_usd"] - (opening - 12) <= required:
        raise ValueError("Bounded stage exceeds remaining authorization")
    if (admission.get("instance_id") != spec["instance_id"]
            or admission.get("gpu_uuid") != spec["gpu_uuid"]
            or admission.get("authorized_for_this_stage") is not True):
        raise ValueError("Explicit stage and host authorization required")


def stopping(tokenizer: object, prompt_length: int, guard: object, candidate: bool) -> object:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            guard()
            if input_ids.shape[0] != 1:
                raise ValueError("This probe requires batch1")
            ids = input_ids[0, prompt_length:].tolist()
            stop = candidate and inspect_prefix(ids, tokenizer)["stop"]
            return torch.tensor([stop], device=input_ids.device, dtype=torch.bool)

    return StoppingCriteriaList([Stop()])


def generate_pass(model: object, tokenizer: object, suite: dict, out: Path,
                  device: str, guard: object, candidate: bool) -> dict:
    import torch

    was_training = model.training
    model.eval()
    passes = []
    try:
        with out.open("x") as handle, torch.inference_mode():
            for case in suite["cases"]:
                guard()
                prompt = f"user: {case['prompt']}\nassistant:"
                ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
                limit = suite["generation"]["max_new_tokens"]
                if limit != 128 or len(ids) + limit > 2048:
                    raise ValueError("Frozen128token/context bounds differ")
                inputs = torch.tensor([ids], device=device)
                result = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, num_beams=1, max_new_tokens=limit, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                    stopping_criteria=stopping(tokenizer, len(ids), guard, candidate),
                )
                guard()
                tokens = result[0, len(ids):].tolist()
                if not tokens or len(tokens) > limit:
                    raise ValueError("Unexpected completion length")
                inspected = inspect_prefix(tokens, tokenizer)
                matched = inspected["boundary"] if candidate else None
                reason = "role_boundary" if matched else (
                    "eos" if tokens[-1] == tokenizer.eos_token_id else
                    "max_new_tokens" if len(tokens) == limit else "unexpected"
                )
                if reason == "unexpected":
                    raise ValueError("Generation stopped without a declared reason")
                scored = inspected["candidate_scored_text"] if candidate else inspected[
                    "raw_completion"
                ]
                score = score_task(case, scored) if case["kind"] == "task" else None
                row = {
                    "id": case["id"], "kind": case["kind"], "prompt_ids": ids,
                    "raw_completion_ids": tokens, "raw_completion": inspected["raw_completion"],
                    "scored_completion": scored, "stop_reason": reason, "boundary": matched,
                    "decoder": "role_boundary_v1" if candidate else "original_v1",
                    "task_pass": score, "safety_review": "PENDING" if score is None else None,
                    "replaces_historical_result": False,
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if score is not None:
                    passes.append(score)
    finally:
        model.train(was_training)
    return {"task_passes": sum(passes), "task_total": len(passes),
            "safety_status": "PENDING_INDEPENDENT_BLIND_REVIEW", "quality_verdict": "NO_VERDICT"}


def execute(args: argparse.Namespace) -> None:
    from check_consumer import state_hash
    from checkpoint_eval import load_adapter, sha256
    from run import build_model, phase, tokenizer_for, write
    from run_support import Guard, contained, gpu_inventory, runtime, verify_model

    if sha256(args.spec) != args.spec_sha:
        raise ValueError("Spec SHA mismatch")
    spec = json.loads(args.spec.read_text())
    if sha256(args.budget_authority) != spec["budget"]["authority_sha256"]:
        raise ValueError("Private user budget authority differs from prospective pin")
    authority = json.loads(args.budget_authority.read_text())
    if authority["total_cap_usd"] != spec["budget"]["total_cap_usd"]:
        raise ValueError("Private budget cap mismatch")
    spec["budget"]["opening_balance_usd"] = authority["opening_balance_usd"]
    repo = Path(__file__).parents[2]
    for name, digest in spec["source_sha256"].items():
        if sha256(contained(repo, name)) != digest:
            raise ValueError(f"Source SHA mismatch: {name}")
    suite_path = contained(repo, spec["suite"]["path"])
    if sha256(suite_path) != spec["suite"]["sha256"]:
        raise ValueError("Frozen suite mismatch")
    suite = json.loads(suite_path.read_text())
    wheels = spec["runtime_wheels"]
    if sha256(contained(repo, wheels["path"])) != wheels["sha256"]:
        raise ValueError("Historical wheel lock differs")
    if len(suite["cases"]) != 24 or len({c["id"] for c in suite["cases"]}) != 24:
        raise ValueError("Expected exact frozen24cases")
    if not re.fullmatch(r"[0-9a-f]{40}", args.protocol_commit):
        raise ValueError("User-created prospective protocol commit required")
    # The orchestrating shell must enforce this hard deadline as well as the guard.
    if os.environ.get("G6_PROBE_EXTERNAL_TIMEOUT_S") != str(spec["total_s"]):
        raise ValueError("Use the bounded launch shell")
    if os.environ.get("SOUP_G6_INSTANCE_ID") != str(spec["instance_id"]):
        raise ValueError("Instance mismatch")
    admission = json.loads(args.admission.read_text())
    admit_probe(spec, admission, args.spec_sha, time.time())
    if sys.version.split()[0] != spec["python"] or runtime() != spec["runtime_target"]:
        raise ValueError("Exact historical runtime required; no silent version change")
    from importlib.metadata import version

    if any(version(name) != expected for name, expected in wheels["packages"].items()):
        raise ValueError("One of the56 recorded package versions differs")
    if sha256(args.adapter) != spec["adapter"]["sha256"]:
        raise ValueError("Historical adapter bytes differ")
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "launch.json", {"spec_sha256": args.spec_sha,
          "protocol_commit": args.protocol_commit, "scope": "paired inference only"})
    guard = Guard(args.out, spec["total_s"], gpu=True, uuid=spec["gpu_uuid"])
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        inventory = gpu_inventory(spec["gpu_uuid"])
        guard()
        verify_model(args.model_dir, spec["model_files"], guard)
        tokenizer = tokenizer_for(spec, args.model_dir)
        model = build_model(spec, args.model_dir, synthetic=False)
        if state_hash(model) != spec["adapter"]["initial_state"]:
            raise ValueError("Initial model state differs from preserved baseline")
        phases = {}
        for state, order in (("initial", (False, True)), ("adapted", (True, False))):
            if state == "adapted":
                load_adapter(model, args.adapter, spec["adapter"]["sha256"],
                             spec["adapter"]["frozen_base"])
                if state_hash(model) != spec["adapter"]["final_state"]:
                    raise ValueError("Restored adapter state differs")
            before = state_hash(model)
            for candidate in order:
                name = state + ("_boundary" if candidate else "_original")
                phases[name] = phase(
                    args.out, name, guard, 1200,
                    lambda c=candidate, n=name: generate_pass(
                        model, tokenizer, suite, args.out / (n + ".jsonl"), "cuda:0", guard, c
                    ), gpu=True, model_dir=args.model_dir, arm=name,
                )
                observed = json.loads((args.out / (name + "_phase.json")).read_text())["F1"]
                if (observed["outcome"] != "ok"
                        or observed.get("gpu_clock", {}).get("outcome") != "ok"):
                    raise ValueError("Required guest F1/clock check did not pass")
            if state_hash(model) != before:
                raise ValueError("Inference changed model weights")
        write(args.out / "receipt.json", {
            "status": "PAIRED_INFERENCE_COMPLETE_REVIEW_PENDING", "phases": phases,
            "inventory": inventory, "model_weights_unchanged_during_each_pair": True,
            "guest_only": True, "historical_scores_replaced": False,
            "performance_claim": False, "quality_verdict": "NO_VERDICT",
        })
    except Exception as error:
        write(args.out / "failure.json", {"error": repr(error), "no_retry": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "admission", "model-dir", "adapter", "out", "budget-authority"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--spec-sha", required=True)
    parser.add_argument("--protocol-commit", required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
