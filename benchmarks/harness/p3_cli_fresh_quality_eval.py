"""Evaluate two Soup CLI adapters on fresh heldout examples with resumable progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from b5_real_moe import _meta_construction
from p3_moonlight_soup_quality_screen import _bootstrap_mean_delta, _make_batch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "cache", "data", "control", "mbr", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed result {args.output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one CUDA GPU")
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    examples = data["heldout_examples"]
    if len(examples) != args.count or len({r["group_sha256"] for r in examples}) != args.count:
        raise RuntimeError("unexpected heldout count or duplicate group")
    train_groups = {r["group_sha256"] for w in data["train_steps"] for r in w}
    if train_groups & {r["group_sha256"] for r in examples}:
        raise RuntimeError("heldout groups overlap training")

    run_id = {
        "data_sha256": digest(args.data),
        "model_config_sha256": digest(args.source / "config.json"),
        "cache_index_sha256": digest(args.cache / "index.json"),
        "runner_sha256": digest(Path(__file__)),
        "count": args.count,
        "control_adapter_sha256": digest(args.control / "adapter_model.safetensors"),
        "mbr_adapter_sha256": digest(args.mbr / "adapter_model.safetensors"),
        "precision": "BF16 CUDA autocast",
        "evaluation_operator": "common single-example streamed forward",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    with _meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(args.source),
            shard_dir=str(args.cache),
            index=read_shard_index(str(args.cache)),
            lora_config=config,
            device="cuda",
            dtype="bfloat16",
            buffers=2,
            pin=False,
            seed=42,
            quant="nf4",
            double_quant=False,
            tier="disk",
            read_ahead=1,
            two_read_experts=True,
            grouped_expert_routes=True,
        )
    try:
        model.config.use_cache = False
        results = {}
        for name, directory in (("control", args.control), ("mbr", args.mbr)):
            saved = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
            if {x.dtype for x in saved.values()} != {torch.bfloat16}:
                raise RuntimeError(f"{name}: expected BF16 adapter")
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(dtype=torch.bfloat16)
            if set(saved) != set(get_peft_model_state_dict(model)):
                raise RuntimeError(f"{name}: adapter keys mismatch")
            set_peft_model_state_dict(model, saved)
            restored = get_peft_model_state_dict(model)
            if any(not torch.equal(restored[k].detach().cpu(), v) for k, v in saved.items()):
                raise RuntimeError(f"{name}: adapter restore is not exact")
            progress_path = args.output.with_name(args.output.stem + f".{name}.progress.json")
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("run_id") != run_id or progress.get("branch") != name:
                    raise RuntimeError(f"{name}: progress identity does not match")
                rows = progress["rows"]
                if [x["group_sha256"] for x in rows] != [
                    x["group_sha256"] for x in examples[: len(rows)]
                ]:
                    raise RuntimeError(f"{name}: progress example order differs")
            else:
                rows = []
            model.eval()
            started = time.monotonic()
            print(f"EVAL {name}: resuming {len(rows)}/{len(examples)}", flush=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for index in range(len(rows), len(examples)):
                    example = examples[index]
                    ids, labels, target_count = _make_batch(example, torch.device("cuda"))
                    output = model(input_ids=ids, use_cache=False, return_dict=True)
                    logits = output.logits[:, :-1, :].float().reshape(-1, output.logits.shape[-1])
                    targets = labels[:, 1:].reshape(-1)
                    nll = float(
                        functional.cross_entropy(
                            logits, targets, ignore_index=-100, reduction="sum"
                        ).item()
                    )
                    if not math.isfinite(nll) or target_count < 1:
                        raise RuntimeError(f"{name}: non-finite NLL or no target tokens at {index}")
                    rows.append(
                        {
                            "group_sha256": example["group_sha256"],
                            "type": example["type"],
                            "nll_sum": nll,
                            "target_tokens": target_count,
                            "nll_token_mean": nll / target_count,
                        }
                    )
                    if len(rows) % 8 == 0 or len(rows) == len(examples):
                        save_atomic(progress_path, {"run_id": run_id, "branch": name, "rows": rows})
                        print(
                            f"EVAL {name}: {len(rows)}/{len(examples)} "
                            f"elapsed_this_invocation={time.monotonic() - started:.1f}s",
                            flush=True,
                        )
            results[name] = {
                "per_example": rows,
                "mean_per_example_nll": sum(x["nll_token_mean"] for x in rows) / len(rows),
                "token_weighted_nll": sum(x["nll_sum"] for x in rows)
                / sum(x["target_tokens"] for x in rows),
            }
        left, right = results["control"]["per_example"], results["mbr"]["per_example"]
        if [(x["group_sha256"], x["target_tokens"]) for x in left] != [
            (x["group_sha256"], x["target_tokens"]) for x in right
        ]:
            raise RuntimeError("paired eval rows or target counts differ")
        results["paired"] = _bootstrap_mean_delta(
            [x["nll_token_mean"] for x in left], [x["nll_token_mean"] for x in right]
        )
        results["families"] = {}
        for family in ("GSM", "MATH"):
            pairs = [(a, b) for a, b in zip(left, right) if a["type"].startswith(family + "_")]
            results["families"][family] = _bootstrap_mean_delta(
                [a["nll_token_mean"] for a, _ in pairs], [b["nll_token_mean"] for _, b in pairs]
            )
        results["run_id"] = run_id
        results["dataset_manifest"] = data["manifest"]
        results["scope"] = (
            "exploratory heldout NLL only; not general quality or non-inferiority acceptance"
        )
        save_atomic(args.output, results)
        print("PAIRED", json.dumps(results["paired"]), flush=True)
        print("SAVED", args.output, flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
