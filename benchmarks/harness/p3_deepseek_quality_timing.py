"""Time native streamed inference without computing base-model quality outcomes.

Only development inputs are opened. Native position chunks and unlabelled FP32
log-softmax scratch match the frozen scorer's resource workload; no gold gather,
accuracy, NLL, predictions or paired decisions are computed or retained. This is
resource evidence, not by itself admission of a complete evaluation budget.
Use p3_deepseek_quality_stage around this CLI to bound in-flight native calls;
the internal deadline alone can only refuse between operations.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from . import p3_deepseek_quality_eval as scoring
from . import p3_stage2v2_loglik as frozen
from .p3_fp8_smoke import MemoryObservation


def time_inputs(
    session: Any, tokenizer: Any, encoded: list[dict], *, max_tokens_per_batch: int,
    deadline: float, logits_memory_limit_bytes: int = frozen.LOGITS_MEMORY_LIMIT_BYTES,
    reduction_chunk_tokens: int = frozen.REDUCTION_CHUNK_TOKENS,
    test_fixture: bool = False, observations: list[dict] | None = None,
    evidence_prefix: Path | None = None,
) -> list[dict]:
    """Use input IDs/positions only; targets are deliberately not copied or accessed."""
    import torch

    rows = [{"input_ids": row["input_ids"], "positions": row["positions"]} for row in encoded]
    scoring._require(bool(rows), "nonempty resource inputs required")
    vocabulary = session.model.config.vocab_size
    for row in rows:
        ids, positions = row["input_ids"], row["positions"]
        scoring._require(bool(ids) and bool(positions)
                         and all(type(value) is int and 0 <= value < len(tokenizer)
                                 for value in ids)
                         and all(type(value) is int and 0 <= value < len(ids)
                                 for value in positions)
                         and positions == sorted(set(positions)), "invalid resource input")
        scoring._require(len(ids) + 1 <= frozen.generation._context_limit(session.model, tokenizer),
                         "untruncated resource input exceeds context limit")
    plan = frozen._batch_plan(
        rows, max_tokens_per_batch=max_tokens_per_batch, vocabulary=vocabulary,
        logits_memory_limit_bytes=logits_memory_limit_bytes,
        reduction_chunk_tokens=reduction_chunk_tokens,
    )
    if observations is None:
        observations = []
    with scoring._scoring_session(session, tokenizer, test_fixture=test_fixture) as model:
        device = model.device
        for indices in plan:
            scoring._require(time.monotonic() < deadline, "resource timing deadline exceeded")
            batch = [rows[index] for index in indices]
            width = max(len(row["input_ids"]) for row in batch)
            positions = sorted({position for row in batch for position in row["positions"]})
            chunks = frozen._position_chunks(
                positions, batch_size=len(batch), vocabulary=vocabulary,
                logits_memory_limit_bytes=logits_memory_limit_bytes,
                reduction_chunk_tokens=reduction_chunk_tokens,
            )
            measurement = MemoryObservation(device.type, scope=(
                "Development inputs, native streamed forward and unlabelled FP32 log-softmax "
                "scratch. Includes model/allocator residency. No quality outcome or gradient."
            ))
            started = time.monotonic()
            largest_logits = 0
            record = {
                "status": "FAILED", "input_indices": indices,
                "input_tokens": sum(len(row["input_ids"]) for row in batch),
                "batch_size": len(batch), "width": width, "padded_input_tokens": len(batch) * width,
                "native_forward_calls": 0, "planned_native_forward_calls": len(chunks),
                "position_count": len(positions),
                "position_chunk_sizes": [len(part) for part in chunks],
                "logits_estimated_peak_bytes": max(frozen._logits_bound(
                    len(batch), len(part), vocabulary, reduction_chunk_tokens) for part in chunks),
            }
            observations.append(record)
            evidence_name = None if evidence_prefix is None else evidence_prefix.with_name(
                f"{evidence_prefix.name}-batch-{len(observations):04d}")
            if evidence_name is not None:
                scoring._write_json(evidence_name.with_name(evidence_name.name + "-start.json"), {
                    "admission_sufficient": False, "observation": record,
                })
            try:
                with measurement:
                    inputs = torch.full((len(batch), width), tokenizer.pad_token_id,
                                        dtype=torch.long, device=device)
                    attention = torch.zeros_like(inputs)
                    for index, row in enumerate(batch):
                        length = len(row["input_ids"])
                        inputs[index, :length] = torch.tensor(row["input_ids"], device=device)
                        attention[index, :length] = 1
                    for part in chunks:
                        scoring._require(time.monotonic() < deadline,
                                         "resource timing deadline exceeded")
                        output = model(
                            input_ids=inputs, attention_mask=attention, use_cache=False,
                            output_attentions=False, output_hidden_states=False,
                            output_router_logits=False,
                            logits_to_keep=torch.tensor(part, device=device),
                            return_dict=True,
                        )
                        record["native_forward_calls"] += 1
                        logits = output.logits
                        scoring._require(
                            tuple(logits.shape) == (len(batch), len(part), vocabulary),
                            "native selected-logit shape mismatch",
                        )
                        largest_logits = max(largest_logits, logits.numel() * logits.element_size())
                        flattened = logits.reshape(-1, vocabulary)
                        # Full position union, not gold-selected targets.
                        for first in range(0, flattened.shape[0], reduction_chunk_tokens):
                            scratch = flattened[first:first + reduction_chunk_tokens].float()
                            normalized = scratch.log_softmax(-1)
                            scoring._require(bool(torch.isfinite(normalized).all().item()),
                                             "nonfinite resource-only inference")
                            del scratch, normalized
                        del flattened, logits, output
                    del inputs, attention
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                record["status"] = "COMPLETE"
            except BaseException as error:
                record["failure"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                record.update(
                    largest_native_logits_bytes=largest_logits,
                    wall_seconds=time.monotonic() - started, memory=measurement.result,
                    attention=model.attention_reports,
                )
                model.attention_reports = []
                if evidence_name is not None:
                    scoring._write_json(evidence_name.with_suffix(".json"), {
                        "admission_sufficient": False, "observation": record,
                    })
    scoring._require(time.monotonic() < deadline, "resource timing deadline exceeded")
    return observations


def run_timing(
    *, source: Path, cache: Path, initial: Path, initial_sha256: str,
    data_root: Path, output: Path, max_tokens_per_batch: int, deadline_seconds: int,
    gate: Path, gate_sha256: str, device: str = "cuda",
) -> dict:
    """Measure the first frozen development task per endpoint, never evaluation rows."""
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "p3-deepseek-quality-forward-timing-v1", "status": "FAILED",
        "include_base_model_evaluation": False, "quality_outcomes_computed": False,
        "evaluation_rows_opened": 0, "budget_admission_sufficient": False,
        "sample_policy": "first frozen development task per endpoint, no outcome selection",
        "forecast_limit": "Three development tasks are resource samples, not a complete cost bound",
        "fp8": False, "oracle": False, "endpoints": {},
    }
    session = None
    started = time.monotonic()
    try:
        from .p3_deepseek_quality_data import load_tokenizer
        from .p3_deepseek_quality_gate import load_gate
        from .p3_deepseek_quality_model import (
            adapter_digest,
            load_initial_adapter,
            open_session,
            require_production_source,
            sha256,
            trainable_parameters,
        )
        from .p3_deepseek_quality_train import _execution_identity, require_gate_binding

        scoring._require(type(deadline_seconds) is int and 0 < deadline_seconds <= 5400,
                         "resource timing deadline must be in (0,5400]")
        result["deadline_seconds"] = deadline_seconds
        scoring._require(type(max_tokens_per_batch) is int and max_tokens_per_batch > 0,
                         "positive resource token budget required")
        gate_result = load_gate(gate, gate_sha256)
        result["gate"] = {"manifest_sha256": gate_sha256, "overall": "PASS"}
        result["hardware"] = scoring._admit_native_execution(device)
        execution_identity = _execution_identity(device)
        result["environment"] = execution_identity["gate_environment"]
        result["attention_policy"] = scoring.attention_policy.policy_identity()
        result["source_files_sha256"] = {
            **scoring._source_files(), __name__: scoring._hash(Path(__file__)),
        }
        scoring._require(sha256(initial) == initial_sha256, "initial adapter SHA256 mismatch")
        development, data_identity = frozen._admit_pilot_data(
            data_root, scoring.DATA_MANIFEST_SHA256)
        tokenizer, tokenizer_identity = load_tokenizer(source)
        result.update(data=data_identity, tokenizer=tokenizer_identity,
                      initial_adapter_sha256=initial_sha256,
                      max_tokens_per_batch=max_tokens_per_batch)
        session = open_session(source, cache, device=device, dtype="bfloat16", prepare=False)
        result["source_files_sha256"].update(scoring._source_files())
        require_production_source(session.metadata["source"])
        require_gate_binding(
            gate_result, model_metadata=session.metadata, initial_sha256=initial_sha256,
            execution_identity=execution_identity,
        )
        result["model"] = copy.deepcopy(session.metadata)
        scoring._require(load_initial_adapter(session.model, initial) == initial_sha256,
                         "initial adapter changed during admission")
        before = adapter_digest(trainable_parameters(session.model))[0]
        scoring._write_json(output / "execution-start.json", {
            "admission_sufficient": False, "partial_execution": result,
        })
        for endpoint in frozen.DEV_COUNTS:
            tasks = development[endpoint][:1]
            requests, _ = frozen._requests(tasks)
            encoded = frozen._request_tokens(tokenizer, requests)
            result["endpoints"][endpoint] = {
                "task_ids": [row["task_id"] for row in tasks],
                "input_sha256": frozen.generation._object_hash([
                    {key: row[key] for key in ("input_ids", "positions")} for row in encoded]),
                "batches": [],
            }
            time_inputs(
                session, tokenizer, encoded, max_tokens_per_batch=max_tokens_per_batch,
                deadline=started + deadline_seconds,
                observations=result["endpoints"][endpoint]["batches"],
                evidence_prefix=output / endpoint,
            )
        scoring._require(adapter_digest(trainable_parameters(session.model))[0] == before,
                         "resource timing changed the shared adapter")
        scoring._require(sha256(initial) == initial_sha256, "initial adapter file changed")
        scoring._require(result["attention_policy"] == scoring.attention_policy.policy_identity()
                         and result["source_files_sha256"] == {
                             **scoring._source_files(), __name__: scoring._hash(Path(__file__)),
                         }, "attention policy/executing source changed during timing")
        scoring._require(scoring._hash(data_root / "manifest.json") == scoring.DATA_MANIFEST_SHA256,
                         "data manifest changed during timing")
        result.update(status="COMPLETE", unchanged_adapter_sha256=before)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result.update(status="FAILED", cleanup_error=f"{type(error).__name__}: {error}")
        result["wall_seconds"] = time.monotonic() - started
        scoring._manifest(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "cache", "initial", "data-root", "output", "gate"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--initial-sha256", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    parser.add_argument("--device", default="cuda")
    result = run_timing(**vars(parser.parse_args()))
    Console().print_json(data=result)
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
