"""Plan the real cache from pinned source headers and the actual meta layout."""

import argparse
import gc
import hashlib
import io
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from soup_cli.utils.expert_cache_budget import (
    expert_cache_budget_from_headers,
    new_expert_cache_budget,
    read_source_header,
)


def source_file(source: Path, name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
        raise ValueError("source plan requires top-level model shard names")
    root = os.path.realpath(source)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([root, path]) != root:
        raise ValueError("planned source escapes checkpoint directory")
    return Path(path)


def validate_plan(source: Path, path: Path, revision: str) -> dict:
    """Bind a plan to full downloaded sources before trusting its disk bound."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("version") != 1 or plan.get("source_revision") != revision
            or hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
            != plan.get("config_sha256")):
        raise ValueError("source plan revision/config mismatch")
    records = plan["files"]
    names = [row["name"] for row in records]
    actual_names = sorted(p.name for p in source.glob("*.safetensors"))
    if not names or names != sorted(set(names)) or names != actual_names:
        raise ValueError("source plan shard selection mismatch")
    paths = []
    for row in records:
        file = source_file(source, row["name"])
        size = file.stat().st_size
        with file.open("rb") as handle:
            raw, _ = read_source_header(handle, size)
        if size != row["size"] or hashlib.sha256(raw).hexdigest() != row["header_sha256"]:
            raise ValueError("downloaded source differs from planned header/size")
        paths.append(str(file))
    if sum(row["size"] for row in records) != plan["source_bytes"]:
        raise ValueError("source plan byte total mismatch")
    budget = new_expert_cache_budget(
        paths, quant_suffixes=plan["quant_suffixes"], n_layers=plan["n_layers"],
    )
    if (type(plan["cache_budget_bytes"]) is not int or budget != plan["cache_budget_bytes"]
            or new_expert_cache_budget(paths) != plan["generic_cache_budget_bytes"]):
        raise ValueError("source plan cache bound mismatch")
    return plan


def validate_writer_plan(plan: dict, suffixes: Sequence[str], n_layers: int, dtype: str) -> None:
    if (sorted(suffixes) != plan["quant_suffixes"] or n_layers != plan["n_layers"]
            or dtype != plan["dtype"]):
        raise ValueError("source plan quantization/layout differs from the actual writer")


def read_remote_header(url: str, size: int) -> tuple[bytes, dict]:
    """Refuse an ignored Range before reading any weight payload into memory."""
    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
        })
        expected = end - start + 1
        with urllib.request.urlopen(request, timeout=60) as response:
            if (response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"):
                raise ValueError("source server did not honor the bounded header range")
            raw = response.read(expected + 1)
        if len(raw) != expected:
            raise ValueError("source server returned a truncated or oversized header range")
        return raw

    prefix = fetch(0, 7)
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= min(64 * 2**20, size - 8):
        raise ValueError("invalid remote safetensors header length")
    raw = fetch(8, 7 + length)
    return read_source_header(io.BytesIO(prefix + raw), size)


def collect_plan(args: argparse.Namespace) -> dict:
    import torch
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "config"
    hf_hub_download(args.model, "config.json", revision=args.revision,
                    local_dir=config_dir, token=False)
    info = HfApi(token=False).model_info(args.model, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Hub metadata does not match the pinned source commit")
    files = sorted((r for r in info.siblings if r.rfilename.endswith(".safetensors")),
                   key=lambda r: r.rfilename)
    if len(files) != 163 or sum(r.size for r in files) != 688_586_727_753:
        raise ValueError("B6 requires the complete pinned 163-file DeepSeek-V3 source")
    dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float32"
    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        skeleton = build_meta_skeleton(
            str(config_dir), dtype=dtype, quant="nf4", double_quant=False,
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
    suffixes = sorted(quantised_layer_suffixes(skeleton))
    layout = shard_layout_of(skeleton)
    del skeleton
    gc.collect()
    def fetch(row: Any) -> tuple[dict, tuple[int, int, dict]]:
        source_file(output, row.rfilename)  # Validate the untrusted remote basename.
        remote = hf_hub_url(args.model, row.rfilename, revision=args.revision)
        raw, header = read_remote_header(remote, row.size)
        (output / (row.rfilename + ".header.json")).write_bytes(raw)
        return ({"name": row.rfilename, "size": row.size,
                 "header_sha256": hashlib.sha256(raw).hexdigest()},
                (row.size, len(raw), header))

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, files))
    budget = expert_cache_budget_from_headers(
        (r[1] for r in rows), quant_suffixes=suffixes, n_layers=layout.n_layers,
    )
    generic = expert_cache_budget_from_headers(r[1] for r in rows)
    plan = {"version": 1, "model": args.model, "source_revision": args.revision,
            "config_sha256": hashlib.sha256((config_dir / "config.json").read_bytes()).hexdigest(),
            "source_bytes": sum(r.size for r in files), "files": [r[0] for r in rows],
            "dtype": dtype, "n_layers": layout.n_layers, "quant_suffixes": suffixes,
            "cache_budget_bytes": budget, "generic_cache_budget_bytes": generic,
            "scope": "header-only planning; all source payloads must still be downloaded; "
                     "the runtime must independently match this actual writer layout"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {k: v for k, v in plan.items() if k != "files"}


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    Console().print(collect_plan(parser.parse_args()))
