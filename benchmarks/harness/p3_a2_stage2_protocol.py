"""Stage 2 receipts. Run from the fresh output directory created by the shell driver."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from p3_route_preserving_mbr_soup_bridge import MODEL, REVISION, host_preflight, sha256_file


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preflight():
    import sysconfig

    import torch

    report = host_preflight(torch, Path.cwd())
    report["python_header"] = str(Path(sysconfig.get_path("include")) / "Python.h")
    report["source_disk_free_bytes"] = shutil.disk_usage(".").free
    mem = Path("/proc/meminfo").read_text().splitlines()
    available_kib = next(line.split()[1] for line in mem if line.startswith("MemAvailable:"))
    available = int(available_kib) * 1024
    if report["cgroup_memory_max_bytes"] is not None:
        # Conservative on a fresh instance: do not assume its charged pages are reclaimable.
        available = min(available, report["cgroup_memory_max_bytes"] -
                        report["cgroup_memory_current_bytes"])
    report["available_ram_bytes"] = available
    save("results/preflight.json", report)
    if (report["visible_gpu_count"] != 1 or "4060 Ti" not in (report["gpu"] or "")
            or report["vram_bytes"] < 15 * 2**30 or not report["native_bf16"]
            or torch.cuda.get_device_capability() != (8, 9)):
        raise ValueError("expected one 16 GB RTX 4060 Ti with native BF16")
    if torch.__version__ != "2.11.0+cu130" or not Path(report["python_header"]).is_file():
        raise ValueError("expected torch 2.11.0+cu130 and system Python.h")
    if available < 48 * 2**30:
        raise ValueError("Stage 2 provisioning floor is 48 GiB available host RAM")


def cache_gate(cache):
    from b5_judge import judge, read_journal

    from soup_cli.utils.layer_shard import read_shard_index

    journal = read_journal("results/b5/journal.jsonl")
    verdict = judge([journal])
    idx = read_shard_index(str(cache))
    sig = next(r for r in journal if r.get("event") == "signature")
    passed = (
        verdict["passed"] and verdict["kind"] == "real" and verdict["steps"] == 4
        and sig["model"] == MODEL and sig["revision"] == REVISION
        and journal[-1]["event"] == "complete" and idx.expert_layout == "per_expert"
        and idx.n_layers == 27 and len(idx.expert_layers) == 26 and idx.dtype == "bfloat16"
        and idx.quant == "nf4" and idx.quant_device == "cuda" and not idx.double_quant
    )
    save("results/cache.json", {
        "status": "PASS" if passed else "FAIL", "model": sig["model"], "revision": sig["revision"],
        "source": str(Path("source").resolve()), "cache": str(cache), "b5_judge": verdict,
        "source_fingerprint": idx.source_fingerprint,
        "index_sha256": sha256_file(cache / "index.json"),
        "source_bytes": sum(p.stat().st_size for p in Path("source").glob("*.safetensors")),
        "cache_bytes": sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()),
    })
    if not passed:
        raise ValueError("cache identity, layout, or B5 journal gate failed: results/cache.json")


def config(recipe, arm, seed):
    from soup_cli.config.schema import SoupConfig

    cfg = yaml.safe_load(recipe.read_text())
    cfg["base"] = str(Path("source").resolve())
    cfg["output"] = f"quality/{arm}/{seed}"
    tcfg = cfg["training"]
    tcfg.update(seed=seed, data_seed=seed, l2l_route_preserving_mbr=arm != "control_native",
                stream_expert_gemm="batch_invariant" if arm == "mbr_invariant" else "native")
    if tcfg["l2l_route_preserving_mbr"]:
        tcfg["l2l_mbr_group_size"] = 8
    SoupConfig.model_validate(cfg)
    Path(f"config/{arm}-{seed}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("download")
    sub.add_parser("cache-gate").add_argument("cache", type=Path)
    p = sub.add_parser("config")
    p.add_argument("recipe", type=Path)
    p.add_argument("arm", choices=("control_native", "mbr_native", "mbr_invariant"))
    p.add_argument("seed", type=int)
    sub.add_parser("check-trained").add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, revision=REVISION, local_dir="source", max_workers=4,
                          allow_patterns=["*.safetensors", "*.json", "*.model", "*.py", "*.txt",
                                          "*.jinja"], token=os.environ.get("HF_TOKEN") or False)
    elif args.command == "cache-gate":
        cache_gate(args.cache)
    elif args.command == "config":
        config(args.recipe, args.arm, args.seed)
    else:
        state = json.loads((args.checkpoint / "trainer_state.json").read_text())
        adapter = args.checkpoint / "adapter_model.safetensors"
        if state["global_step"] != 40 or not adapter.is_file():
            raise ValueError("expected complete step-40 adapter")


if __name__ == "__main__":
    main()
