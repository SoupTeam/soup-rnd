"""Fresh-process bounded staged/packed-fused series with strict source and F1 identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from benchmarks.harness import d3_fused_trainer_protocol as protocol
from benchmarks.harness import d3_trainer_protocol as historical
from benchmarks.harness.d3_fused_trainer_gate import correctness, performance
from benchmarks.harness.d3_trainer_gate import NumericalFailureError, save
from benchmarks.harness.d3_trainer_series import gpu_peers, verify_host, verify_prepared


def source_receipt(*, synthetic=False):
    rule = protocol.read_rule()
    protocol.validate_rule(rule)
    name = "D3-FUSED-SYNTHETIC-SOURCE.json" if synthetic else protocol.RECEIPT
    receipt = json.loads((protocol.ROOT / name).read_text())
    if synthetic:
        if receipt.get("kind") != "synthetic packed-fused launcher only":
            raise ValueError("synthetic source receipt required")
    else:
        blockers = protocol.launch_blockers(rule, check_git=False)
        if blockers:
            raise ValueError("unready frozen source: " + "; ".join(blockers))
        if receipt.get("kind") != "canonical committed packed-fused trainer source":
            raise ValueError("actual canonical source receipt required")
    hashes = receipt.get("git_blob_sha256", {})
    required = {
        protocol.PROTOCOL,
        *protocol.ENGINES,
        *(p.relative_to(protocol.ROOT).as_posix() for p in (protocol.ROOT / "src").rglob("*.py")),
    }
    if not required <= hashes.keys():
        raise ValueError("incomplete executed source receipt")
    for name, want in hashes.items():
        if historical.file_hash(historical.contained(protocol.ROOT, name)) != want:
            raise ValueError("executed source changed: " + name)
    return receipt


def prepared_inputs(path, rule, source, *, synthetic=False):
    if synthetic:
        if (
            str(path) != source["prepared"]
            or historical.file_hash(path) != source["prepared_sha256"]
        ):
            raise ValueError("synthetic preparation receipt changed")
        data = json.loads(path.read_text())
        fixture = rule["required_launcher_fixture"]
        if data.get("model_geometry") != {
            key: fixture[key] for key in ("layers", "experts", "vocab", "tie_word_embeddings")
        }:
            raise ValueError("synthetic launcher refuses pretrained geometry")
        if data.get("checkpoint_files") != {"model.safetensors": fixture["weight_sha256"]}:
            raise ValueError("synthetic launcher refuses nonfixture weight identities")
        for row in data["codecs"].values():
            checkpoint_dir = Path(row["checkpoint"])
            if (
                historical.file_hash(checkpoint_dir / "model.safetensors")
                != fixture["weight_sha256"]
            ):
                raise ValueError("synthetic launcher refuses actual nonfixture weights")
            actual = json.loads((checkpoint_dir / "config.json").read_text())
            if (
                actual["num_hidden_layers"],
                actual["num_experts"],
                actual["vocab_size"],
                actual["tie_word_embeddings"],
            ) != (4, 4, 64, False):
                raise ValueError("synthetic launcher refuses actual nonfixture config")
        checked = dict(rule)
        checked["preparation"] = dict(rule["preparation"])
        checked["preparation"]["shards"] = {"sha256": source["prepared_sha256"]}
        checked["hardware"] = {"boot_id": data["boot_id"]}
        return verify_prepared(path, checked)
    data = verify_prepared(path, rule)
    if data.get("checkpoint_files") != historical.CHECKPOINT_PINS:
        raise ValueError("formal preparation is not the pinned pretrained checkpoint")
    for row in data["codecs"].values():
        for name, expected in historical.CHECKPOINT_PINS.items():
            if data["files"].get(str(Path(row["checkpoint"]) / name), {}).get("sha256") != expected:
                raise ValueError("actual original checkpoint file identity missing/changed")
    return data


def oracle_path(session, codec):
    if codec not in protocol.CODECS:
        raise ValueError("unknown oracle codec")
    return session / f"correctness-{codec}-ordinary-native-resident/result.json"


def reference_for(session, cell, source, session_info):
    path = oracle_path(session, cell["codec"])
    if cell["name"] == path.parent.name:
        return None
    record = json.loads(path.read_text())
    if (
        record["status"] != "ok"
        or record["source"] != source
        or record["session_id"] != session_info["session_id"]
        or record["F1"]["outcome"] != "ok"
        or record["protocol_sha256"] != session_info["protocol_sha256"]
        or record["formal_execution"] != session_info["formal_execution"]
        or record["cell"]["name"] != path.parent.name
    ):
        raise ValueError("invalid ordinary native oracle")
    return record["result"]


def child(args):
    import resource

    rule = protocol.read_rule()
    source = source_receipt(synthetic=args.synthetic)
    session = json.loads((args.session / "session.json").read_text())
    cells = {c["name"]: c for c in protocol.cell_plan(rule, blocks=10)}
    if args.cell not in cells or source != session["source"]:
        raise ValueError("unknown cell or source")
    cell = cells[args.cell]
    args.out.mkdir(exist_ok=False, parents=True)
    record = dict(
        status="no_verdict",
        cell=cell,
        source=source,
        pid=os.getpid(),
        session_id=session["session_id"],
        protocol_sha256=session["protocol_sha256"],
        formal_execution=not args.synthetic,
    )
    watch = None
    try:
        prepared = prepared_inputs(args.prepared, rule, source, synthetic=args.synthetic)
        host = source["host_rule"] if args.synthetic else rule
        verify_host(host, prepared)
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.get_device_capability(0) != tuple(rule["compute_capability"]):
            raise ValueError("unqualified device compute capability")
        sys.path.insert(0, str(protocol.ROOT / "benchmarks/harness"))
        from arm_validity import ArmWatch
        from d3_f1_controls import verified_counters, verified_selector

        mapping = (
            json.loads(Path(source["mapping"]).read_text())
            if args.synthetic
            else json.loads(
                historical.contained(
                    protocol.ROOT, rule["preparation"]["mapping"]["path"]
                ).read_text()
            )
        )
        if (
            args.synthetic
            and historical.file_hash(Path(source["mapping"])) != source["mapping_sha256"]
        ):
            raise ValueError("synthetic mapping changed")
        select = verified_selector(mapping)
        counters = verified_counters(mapping)
        for row in mapping["files"]:
            select(row["path"])
        reference = reference_for(args.session, cell, source, session)
        with ArmWatch(
            arm=cell["mode"],
            round=0,
            run=0,
            model_path=mapping["files"][0]["path"],
            disk_device=select,
            disk_reads=counters[0],
            other_disk_reads=counters[1],
        ) as watch:
            if cell["kind"] == "correctness":
                native = None if reference is None else reference["numerical"]
                record["result"] = correctness(
                    rule, prepared, args.out, cell["codec"], cell["mode"], cell["training"], native
                )
            else:
                if reference is None:
                    raise ValueError("performance requires native oracle")
                record["result"] = performance(rule, prepared, args.out, cell, reference)
        record["F1"] = watch.record
        if watch.record["outcome"] != "ok":
            raise ValueError("F1 unknown/void; no replacement")
        if set(gpu_peers()) - {os.getpid()}:
            raise ValueError("GPU peers changed")
        prepared_inputs(args.prepared, rule, source, synthetic=args.synthetic)
        if source_receipt(synthetic=args.synthetic) != source:
            raise ValueError("post-cell source changed")
        record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record["status"] = "ok"
    except BaseException as exc:
        record["status"] = "fail" if isinstance(exc, NumericalFailureError) else "no_verdict"
        record["traceback"] = traceback.format_exc()
    finally:
        if watch is not None and watch.record is not None:
            record["F1"] = watch.record
        save(args.out / "result.json", record)
    return 0 if record["status"] == "ok" else 1


def child_command(prepared, session, cell, *, synthetic=False):
    command = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.harness.d3_fused_trainer_series",
        "child",
        "--prepared",
        str(prepared),
        "--session",
        str(session),
        "--out",
        str(session / cell["name"]),
        "--cell",
        cell["name"],
    ]
    return command + (["--synthetic"] if synthetic else [])


def parent(args):
    rule = protocol.read_rule()
    source = source_receipt(synthetic=args.synthetic)
    data = prepared_inputs(args.prepared, rule, source, synthetic=args.synthetic)
    verify_host(source["host_rule"] if args.synthetic else rule, data)
    marker = args.prepared.parent / ".d3-packed-fused-sessions"
    marker.mkdir(exist_ok=True)
    key = (
        hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if args.synthetic
        else source["git_head"]
    )
    with (marker / (key + ".json")).open("x") as stream:
        json.dump({"output": str(args.out), "formal": not args.synthetic}, stream)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    session = dict(
        status="running",
        source=source,
        session_id=uuid.uuid4().hex,
        protocol_sha256=historical.file_hash(protocol.ROOT / protocol.PROTOCOL),
        formal_execution=not args.synthetic,
        cells=[],
        d3_accepted=False,
    )
    save(args.out / "protocol.json", rule)
    save(args.out / "session.json", session)

    def launch(cell):
        remaining = rule["session_timeout_seconds"] - (time.monotonic() - started)
        if remaining <= 0 or len(session["cells"]) >= rule["attempt_budget"]["maximum_total_cells"]:
            raise ValueError("finite session budget exhausted before spawning")
        row = dict(cell, status="started")
        session["cells"].append(row)
        save(args.out / "session.json", session)
        try:
            command = child_command(args.prepared, args.out, cell, synthetic=args.synthetic)
            row["command"] = command
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=os.pathsep.join((str(protocol.ROOT), str(protocol.ROOT / "src"))),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            with (args.out / (cell["name"] + ".log")).open("x") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=protocol.ROOT,
                    env=environment,
                    timeout=min(rule["cell_timeout_seconds"], remaining),
                    check=False,
                )
            path = args.out / cell["name"] / "result.json"
            result = json.loads(path.read_text())
            row.update(
                returncode=process.returncode,
                status=result["status"],
                result_sha256=historical.file_hash(path),
            )
            if (
                process.returncode != 0
                or result["status"] != "ok"
                or result["cell"] != cell
                or result["source"] != source
                or result["F1"]["outcome"] != "ok"
                or result["session_id"] != session["session_id"]
                or result["protocol_sha256"] != session["protocol_sha256"]
                or result["formal_execution"] != session["formal_execution"]
            ):
                raise ValueError("failed/no-verdict child: " + cell["name"])
            return result
        except BaseException:
            row.update(status="failed_or_no_verdict", traceback=traceback.format_exc())
            raise
        finally:
            save(args.out / "session.json", session)

    try:
        pilot = protocol.cell_plan(rule)
        for cell in pilot:
            if cell["kind"] == "correctness":
                launch(cell)
        session["repeat_calculations"] = {}
        for codec in protocol.CODECS:
            for training in protocol.TRAINING:
                if not args.synthetic:
                    blocks, target = [], 5
                    full = protocol.cell_plan(rule, blocks=10)
                    while len(blocks) < target:
                        block = []
                        for cell in full:
                            if (
                                cell["kind"] == "timing"
                                and cell["codec"] == codec
                                and cell["training"] == training
                                and cell["block"] == len(blocks) + 1
                            ):
                                result = launch(cell)
                                block.append(
                                    dict(
                                        mode=cell["mode"],
                                        outcome=result["F1"]["outcome"],
                                        seconds_per_input_token=result["result"][
                                            "seconds_per_input_token"
                                        ],
                                    )
                                )
                        blocks.append(block)
                        if len(blocks) == target:
                            estimate = protocol.repeat_target(blocks)
                            session["repeat_calculations"][f"{codec}/{training}"] = estimate
                            if estimate["status"] == "no_verdict_precision_budget":
                                raise ValueError("precision budget exhausted")
                            target = estimate["target"]
                for cell in pilot:
                    if (
                        cell["kind"] == "profile"
                        and cell["codec"] == codec
                        and cell["training"] == training
                    ):
                        launch(cell)
        session["status"] = (
            "synthetic_launcher_complete_pending_analysis"
            if args.synthetic
            else "complete_pending_offline_analysis_and_review"
        )
    except BaseException:
        session.update(status="incomplete", traceback=traceback.format_exc())
    finally:
        save(args.out / "session.json", session)
    return 0 if session["status"].startswith(("complete_", "synthetic_launcher_complete")) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parent", "child"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "child" and (args.session is None or args.cell is None):
        parser.error("child requires --session and --cell")
    return child(args) if args.mode == "child" else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
