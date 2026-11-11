"""Bounded real-weight qualification; never a full-model quality result.

Copies a prefix through the first native MoE layer, preserving every selected
checkpoint tensor byte, expert and hidden dimension. Only layer count changes.
The production sharder has no layer-subset argument, so a bounded source is made
before using its unchanged NF4 conversion path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CONFIG_SHA256 = "4422016f41c506339e0510a32812a63da554dc9febd5f1368e7240ded14e4a27"


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _child(root: Path, name: str) -> Path:
    candidate = root / name
    if os.path.commonpath((os.path.realpath(root), os.path.realpath(candidate))) != (
        os.path.realpath(root)
    ):
        raise ValueError("source path escapes root")
    return candidate


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > 100_000_000:
            raise ValueError("invalid safetensors header length")
        raw = handle.read(size)
        if len(raw) != size:
            raise ValueError("truncated safetensors header")
        return json.loads(raw), 8 + size


def select_prefix(source: Path, layers: int) -> tuple[dict, dict, int, int]:
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = {}
    for key, filename in index["weight_map"].items():
        match = re.match(r"^model\.layers\.(\d+)\.", key)
        if (match and int(match[1]) < layers) or key.startswith((
            "model.embed_tokens.", "model.norm.", "lm_head.",
        )):
            mapping[key] = filename
    if not mapping:
        raise ValueError("checkpoint prefix has no weights")
    headers = {}
    payload_bytes = elements = 0
    for name in sorted(set(mapping.values())):
        header, offset = _header(_child(source, name))
        selected = {key: header[key] for key in mapping if mapping[key] == name}
        for key, entry in selected.items():
            begin, end = entry["data_offsets"]
            if begin < 0 or end < begin or offset + end > _child(source, name).stat().st_size:
                raise ValueError(f"invalid data range for {key}")
            payload_bytes += end - begin
            elements += math.prod(entry["shape"])
        headers[name] = (selected, offset)
    return mapping, headers, payload_bytes, elements


def copy_prefix(source: Path, destination: Path, *, config: dict, layers: int,
                mapping: dict, headers: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    source_config = dict(config)
    source_config["num_hidden_layers"] = layers
    _write(destination / "config.json", source_config)
    records = {}
    for name, (entries, data_start) in headers.items():
        output_header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for key in sorted(entries):
            entry = entries[key]
            length = entry["data_offsets"][1] - entry["data_offsets"][0]
            output_header[key] = dict(entry, data_offsets=[cursor, cursor + length])
            cursor += length
        encoded = json.dumps(output_header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        target = _child(destination, name)
        with _child(source, name).open("rb") as src, target.open("xb") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for key in sorted(entries):
                begin, end = entries[key]["data_offsets"]
                src.seek(data_start + begin)
                remaining = end - begin
                tensor_hash = hashlib.sha256()
                while remaining:
                    data = src.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise ValueError("truncated source tensor")
                    tensor_hash.update(data)
                    dst.write(data)
                    remaining -= len(data)
                records[key] = {
                    "source_shard": name, "source_data_offsets": [begin, end],
                    "source_tensor_bytes_sha256": tensor_hash.hexdigest(),
                    "dtype": entries[key]["dtype"], "shape": entries[key]["shape"],
                }
        # Verify the copied byte ranges independently, not only the write-side digest.
        copied_header, copied_start = _header(target)
        with target.open("rb") as verify:
            for key in entries:
                begin, end = copied_header[key]["data_offsets"]
                verify.seek(copied_start + begin)
                digest = hashlib.sha256()
                remaining = end - begin
                while remaining:
                    data = verify.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise ValueError("truncated copied tensor")
                    digest.update(data)
                    remaining -= len(data)
                if digest.hexdigest() != records[key]["source_tensor_bytes_sha256"]:
                    raise ValueError("copied tensor changed")
    _write(destination / "model.safetensors.index.json", {
        "metadata": {"total_size": sum(
            entry["data_offsets"][1] - entry["data_offsets"][0]
            for entries, _ in headers.values() for entry in entries.values()
        )}, "weight_map": mapping,
    })
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    return {"layers": layers, "source_layers": config["num_hidden_layers"],
            "config_changes": {"num_hidden_layers": layers}, "tensors": records}


def _one_pair(source: Path, cache: Path, output: Path) -> dict:
    import torch

    from benchmarks.harness.p3_deepseek_quality_model import (
        load_initial_adapter,
        open_session,
        trainable_parameters,
        write_initial_adapter,
    )
    from benchmarks.harness.p3_deepseek_quality_train import WindowFailureError, run_window

    records = {}
    with open_session(source, cache, prepare=True) as session:
        session.model.train()
        initial = output / "initial.safetensors"
        write_initial_adapter(session.model, initial, seed=23)
        ids = torch.randint(
            session.config.vocab_size, (1, 64),
            generator=torch.Generator().manual_seed(23),
        ).tolist()
        labels = [[-100, *ids[0][1:]]]
        window = {"window_id": "r1-real-weight-qualification",
                  "micro_batches": [{"input_ids": ids, "labels": labels}],
                  "valid_target_count": 63}
        for arm in ("bf16_l2l", "oracle_block64_e4m3"):
            initial_hash = load_initial_adapter(session.model, initial)
            parameters = trainable_parameters(session.model)
            # Code-only update, not a study step or its initial LR0 warmup.
            optimizer = torch.optim.AdamW(parameters.values(), lr=2e-4)
            try:
                record, _ = run_window(
                    session, window, arm=arm, optimizer=optimizer, host_limit_bytes=2 * 1024**3,
                )
            except WindowFailureError as error:
                _write(output / f"{arm}.json", error.record)
                raise
            record["initial_adapter_sha256"] = initial_hash
            record["qualification_optimizer"] = "AdamW lr2e-4 without study scheduler"
            records[arm] = record
            _write(output / f"{arm}.json", record)
    return records


def qualify(source: Path, output: Path, scratch: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "FAILED", "full_model_acceptance": False,
        "input_policy": "seed23 synthetic token IDs; code/weights check, not quality data",
        "source": str(source),
    }
    owned = None
    try:
        report["config_sha256"] = _sha(source / "config.json")
        if report["config_sha256"] != CONFIG_SHA256:
            raise ValueError("mounted R1 config differs from pinned qualification Input")
        for name, args in (
            ("df", ["df", "-B1", "/kaggle/working", "/tmp", str(source)]),
            ("mount", ["findmnt", "-T", str(source), "-J"]),
        ):
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            (output / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        layers = int(config["first_k_dense_replace"]) + 1
        mapping, headers, source_bytes, elements = select_prefix(source, layers)
        free = shutil.disk_usage(scratch).free
        owned = Path(tempfile.mkdtemp(prefix="p3-real-weight-", dir=scratch))
        # Conservatively allow BF16 for every selected element plus source and workspace.
        cache_upper = 2 * elements + 2 * 1024**3
        report.update({"free_bytes": free, "subset_source_payload_bytes": source_bytes,
                       "subset_cache_upper_bytes": cache_upper,
                       "full_cache_reference_bytes": 380_276_447_824,
                       "full_cache_disk_admitted": free >= 519_579_082_752,
                       "stage_deadline_seconds": 1200,
                       "full_cache_reference_build_seconds": 2568.94,
                       "full_cache_time_admitted": False,
                       "full_cache_refusal": (
                           "B6 reference conversion alone exceeds the bounded code-check stage; "
                           "full DeepSeek experiment remains a Vast stage")})
        if free >= source_bytes + cache_upper + 4 * 1024**3:
            chosen = owned / "source"
            report["subset"] = copy_prefix(source, chosen, config=config, layers=layers,
                                             mapping=mapping, headers=headers)
            report["scope"] = "first native MoE prefix, all experts, unchanged tensor bytes"
        else:
            report["status"] = "REFUSED_DISK"
            report["reason"] = "conservatively bounded prefix does not fit scratch space"
            return report
        report["arms"] = _one_pair(chosen, owned / "cache", output)
        report["status"] = "COMPLETE"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if owned is not None:
                shutil.rmtree(owned)
        except OSError as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "FAILED"
        report["temporary_weights_removed"] = owned is None or not owned.exists()
        _write(output / "result.json", report)
        _write(output / "manifest.json", {"status": report["status"], "files": {
            str(path.relative_to(output)): _sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        }})
    return report


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.source, args.output, args.scratch)
    Console().print_json(data=result)
    if result["status"] not in ("COMPLETE", "REFUSED_DISK"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
