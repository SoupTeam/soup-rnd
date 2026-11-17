"""Prepare trained or strictly pinned synthetic inputs without running model forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.harness import d3_fused_trainer_protocol as protocol
from benchmarks.harness import d3_trainer_protocol as historical
from benchmarks.harness.d3_trainer_gate import save, software
from benchmarks.harness.d3_trainer_prepare import prepare


def library_path():
    candidates = list(
        Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu13/lib/libcublas.so.13")
    )
    if len(candidates) != 1:
        raise ValueError("one explicit installed CUDA 13 cuBLAS library required")
    return str(candidates[0])


def fixture(out):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(71)
    out.mkdir(exist_ok=False, parents=True)
    config = OlmoeConfig(
        hidden_size=2048,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=1024,
        output_router_logits=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        eos_token_id=1,
        bos_token_id=2,
        pad_token_id=0,
    )
    model = OlmoeForCausalLM(config).to(dtype=torch.bfloat16)
    model.save_pretrained(out)
    raw = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(out)
    expected = protocol.read_rule()["required_launcher_fixture"]["weight_sha256"]
    if historical.file_hash(out / "model.safetensors") != expected:
        raise ValueError("regenerated synthetic fixture differs from its fixed identity")
    return out


def prepare_inputs(checkpoint, out, *, synthetic=False):
    rule = protocol.read_rule()
    protocol.validate_rule(rule)
    if synthetic:
        checkpoint = fixture(out.parent / "synthetic-fixture")
        rule = dict(
            rule,
            checkpoint_files={
                "model.safetensors": rule["required_launcher_fixture"]["weight_sha256"]
            },
        )
    elif checkpoint is None:
        raise ValueError("trained preparation requires --checkpoint")
    result = prepare(
        checkpoint,
        out,
        rule=rule,
        codecs=protocol.CODECS,
        library=library_path(),
        expected_layers=4 if synthetic else 16,
    )
    result["preparation_source_sha256"] = {
        path.relative_to(protocol.ROOT).as_posix(): historical.file_hash(path)
        for base in ("src", "benchmarks/harness")
        for path in (protocol.ROOT / base).rglob("*.py")
    }
    result["preparation_rule_sha256"] = historical.file_hash(protocol.ROOT / protocol.PROTOCOL)
    save(out / "prepared.json", result)
    return result


def controls(prepared_path, out):
    """Sibling foreign/watch processes: foreign is not in the watched tree."""
    sys.path.insert(0, str(protocol.ROOT / "benchmarks/harness"))
    from benchmarks.harness.d3_f1_mapping_probe import discover_guest

    prepared = json.loads(prepared_path.read_text())
    paths = sorted(
        {
            path
            for row in prepared["codecs"].values()
            for path in Path(row["shards"]).rglob("*.safetensors")
            if path.stat().st_size >= 2 * 2**20
        }
    )
    if not paths:
        raise ValueError("no actual prepared weight shards")
    out.mkdir(exist_ok=False, parents=True)
    mapping = discover_guest(paths)
    save(out / "mapping.json", mapping)
    if mapping["status"] != "mapping_observed_controls_still_required":
        raise ValueError("actual guest mapping rejected")
    session = out / "controls"
    command = [sys.executable, "-u", "-m", "benchmarks.harness.d3_f1_controls"]
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (
                str(protocol.ROOT),
                str(protocol.ROOT / "src"),
                str(protocol.ROOT / "benchmarks/harness"),
            )
        ),
    )
    with (out / "watch.log").open("x") as watch_log, (out / "foreign.log").open("x") as foreign_log:
        watch = subprocess.Popen(
            command + ["watch", "--mapping", str(out / "mapping.json"), "--session", str(session)],
            stdout=watch_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        foreign = subprocess.Popen(
            command + ["foreign", "--session", str(session)],
            stdout=foreign_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            code = watch.wait(timeout=300)
            other = foreign.wait(timeout=150)
            if code != 0 or other != 0:
                raise ValueError("actual-shard F1 controls rejected; preserve logs")
        finally:
            for process in (watch, foreign):
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=20)
    return json.loads((session / "report.json").read_text())


def host_rule(prepared):
    import torch

    versions = software()
    line = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True
        )
        .strip()
        .split(",")
    )
    versions.update(
        driver=line[1].strip(), cublas_sha256=historical.file_hash(Path(prepared["library"]))
    )
    return {
        "software": versions,
        "hardware": {
            "gpu_uuid": line[0].strip(),
            "boot_id": prepared["boot_id"],
            "minimum_vram_bytes": 23 * 2**30,
            "actual_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }


def bind_synthetic(prepared_path, mapping):
    prepared = json.loads(prepared_path.read_text())
    rule = protocol.read_rule()
    if prepared["checkpoint_files"] != {
        "model.safetensors": rule["required_launcher_fixture"]["weight_sha256"]
    }:
        raise ValueError("refuse pretrained data in synthetic binding")
    control = json.loads((mapping.parent / "controls/report.json").read_text())
    if control["status"] != "controls_passed":
        raise ValueError("synthetic actual-shard controls incomplete")
    names = {
        protocol.PROTOCOL,
        *protocol.ENGINES,
        *(
            path.relative_to(protocol.ROOT).as_posix()
            for base in ("src", "benchmarks/harness")
            for path in (protocol.ROOT / base).rglob("*.py")
        ),
    }
    receipt = {
        "kind": "synthetic packed-fused launcher only",
        "prepared": str(prepared_path),
        "prepared_sha256": historical.file_hash(prepared_path),
        "mapping": str(mapping),
        "mapping_sha256": historical.file_hash(mapping),
        "host_rule": host_rule(prepared),
        "protocol_utf8": (protocol.ROOT / protocol.PROTOCOL).read_text(),
        "source_files": {
            name: {
                "lf_sha256": hashlib.sha256(
                    (protocol.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
            }
            for name in sorted(names)
            if name.endswith(".py")
        },
        "git_blob_sha256": {
            name: historical.file_hash(protocol.ROOT / name) for name in sorted(names)
        },
    }
    save(protocol.ROOT / "D3-FUSED-SYNTHETIC-SOURCE.json", receipt)
    return receipt


def main(argv=None):
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "controls", "bind-synthetic"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.mode == "prepare":
        if args.out is None:
            parser.error("prepare requires --out")
        result = prepare_inputs(args.checkpoint, args.out, synthetic=args.synthetic)
    elif args.mode == "controls":
        if args.out is None or args.prepared is None:
            parser.error("controls requires --out and --prepared")
        result = controls(args.prepared, args.out)
    else:
        if args.prepared is None or args.mapping is None:
            parser.error("binding requires --prepared and --mapping")
        result = bind_synthetic(args.prepared, args.mapping)
    Console().print(result.get("status", result.get("kind")))


if __name__ == "__main__":
    main()
