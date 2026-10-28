"""Prepare public-trainer inputs/caches without executing a pretrained forward."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import traceback
from pathlib import Path

from benchmarks.harness.d3_trainer_gate import make_config, public_workspace, save, software
from benchmarks.harness.d3_trainer_protocol import CODECS, file_hash, read_rule


def metadata(config: object) -> dict:
    from soup_cli.utils.data_pipeline import (
        make_preprocess_cache_key,
        preprocess_dataset_key_input,
        preprocess_mask_mode,
    )

    data, training = config.data, config.training
    dataset_key = preprocess_dataset_key_input(data)
    mask = preprocess_mask_mode(data, training)
    return {
        "format": data.format,
        "chat_template": None,
        "mask_mode": mask,
        "key_schema": 1,
        "dataset_key": json.loads(dataset_key),
        "cache_key": make_preprocess_cache_key(
            dataset_path=dataset_key,
            tokenizer_name=config.base,
            max_length=data.max_length,
            format_name=data.format,
            mask_mode=mask,
            task="sft",
            chat_template=None,
        ),
    }


def prepare(checkpoint: Path, out: Path) -> dict:
    """Dataset, public sharding/setup, initial factors; no pretrained loss/training."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from datasets import Dataset
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoConfig

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.layer_shard import resolve_shard_dir

    rule = read_rule()
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "preparing",
        "formal_execution": False,
        "workspace": str(out.resolve()),
        "pretrained_model_executed": False,
        "codecs": {},
        "library": (
            "/root/d3-trainer-venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13"
        ),
        "rows_path": str(out / "input-rows.jsonl"),
    }
    wrapper = None
    try:
        if any(
            file_hash(checkpoint / name) != expected
            for name, expected in rule["checkpoint_files"].items()
        ):
            raise ValueError("pinned original checkpoint differs")
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if (
            config.model_type != "olmoe"
            or config.num_hidden_layers != 16
            or (config.hidden_size, config.intermediate_size) != (2048, 1024)
        ):
            raise ValueError("not the declared pretrained OLMoE recipe")
        generator = torch.Generator().manual_seed(rule["input_seed"])
        ids = torch.randint(
            3, config.vocab_size, (rule["rows"], rule["tokens"]), generator=generator
        )
        labels = ids.clone()
        labels[:, ::11] = -100
        rows = [
            {"input_ids": x.tolist(), "labels": y.tolist(), "attention_mask": [1] * len(x)}
            for x, y in zip(ids, labels, strict=True)
        ]
        Path(record["rows_path"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
        save(out / "progress.json", record)
        for codec in CODECS:
            base = out / codec
            base.mkdir()
            alias = base / "checkpoint"
            alias.mkdir()
            for path in checkpoint.iterdir():
                if path.is_file():
                    os.link(path, alias / path.name)
            data = base / "dataset"
            Dataset.from_list(rows).save_to_disk(str(data))
            row = {
                "checkpoint": str(alias),
                "dataset": str(data),
                "initial_adapters": str(base / "initial-adapters.safetensors"),
                "shards": resolve_shard_dir(str(alias)),
            }
            record["codecs"][codec] = row
            setup = make_config(rule, record, base / "setup", codec, "sequential", "ordinary")
            save(data / "metadata.json", metadata(setup))
            wrapper = SFTTrainerWrapper(setup, device="cuda", report_to="none")
            with public_workspace(record, base / "setup"):
                wrapper.setup({"train": [], "val": []})
            if len(wrapper.trainer.train_dataset) != rule["rows"]:
                raise ValueError("public Trainer changed the prepared row count")
            save_file(
                {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in get_peft_model_state_dict(wrapper.model).items()
                },
                row["initial_adapters"],
            )
            row["initial_adapters_sha256"] = file_hash(Path(row["initial_adapters"]))
            row["runtime"] = wrapper._stream_runtime.stats()
            row["attention"] = wrapper.model.config._attn_implementation
            wrapper._close_stream_runtime()
            wrapper = None
            gc.collect()
            torch.cuda.empty_cache()
            save(out / "progress.json", record)
        identity = {}
        inode_hashes = {}
        path = Path(record["rows_path"])
        info = path.stat()
        identity[str(path)] = {
            "sha256": file_hash(path), "bytes": info.st_size, "st_dev": info.st_dev,
            "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
        }
        for codec, row in record["codecs"].items():
            paths = [Path(row["initial_adapters"])]
            for key in ("checkpoint", "dataset", "shards"):
                paths += [
                    path
                    for path in Path(row[key]).rglob("*")
                    if path.is_file() and ".cache" not in path.parts
                ]
            for path in paths:
                info = path.stat()
                key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                digest = inode_hashes.setdefault(key, None)
                if digest is None:
                    digest = inode_hashes[key] = file_hash(path)
                identity[str(path)] = {
                    "sha256": digest,
                    "bytes": info.st_size,
                    "st_dev": info.st_dev,
                    "inode": info.st_ino,
                    "mtime_ns": info.st_mtime_ns,
                }
        record["files"] = identity
        record["software"] = software()
        record["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        record["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"],
            text=True,
        ).strip()
        record["checkpoint_files"] = dict(rule["checkpoint_files"])
        record["status"] = "prepared_mapping_and_controls_pending"
        save(out / "prepared.json", record)
        return record
    except BaseException:
        record.update(status="preparation_failed", traceback=traceback.format_exc())
        raise
    finally:
        if wrapper is not None:
            wrapper._close_stream_runtime()
        save(out / "progress.json", record)
