#!/usr/bin/env python3
"""A4 session: every block of the fixed rule, one fresh process per arm.

Planning is the default (prints the commands and writes protocol.json);
``--execute`` runs them. The order is the rule's (§4 of
benchmarks/probe-a4-l2l-ram-budget-spill.md): preflight, drive, correctness at
k = 2, 3, 4 (with the deterministic rerun only where GA is not reproducible),
cleanup, then each timing comparison in A-B-B-A blocks until the F1 repeats rule
(``arm_validity.blocks_to_add``) says stop, and finally the verdict. A child that
fails stops the session; nothing is retried except the predeclared
deterministic correctness rerun. ``--device cpu`` runs the same plan as the
SYNTHETIC / CPU PILOT, which the rule never gives a verdict on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_rule  # noqa: E402
import arm_validity  # noqa: E402
import l2l_session  # noqa: E402

HARNESS = Path(__file__).resolve().parent
PROBE = str(HARNESS / "a4_probe.py")

#: name -> (tier, k, budget of A, budget of B, where B spills). A is always the
#: same process type as B, so the ONLY difference inside a block is the budget.
COMPARISONS: Dict[str, Dict[str, Any]] = {
    "ram_k2_half": {"tier": "ram", "k": 2, "a": "1", "b": "0.5", "spill": "spill"},
    "ram_k2_min": {"tier": "ram", "k": 2, "a": "1", "b": "min", "spill": "spill"},
    "ram_k3_min": {"tier": "ram", "k": 3, "a": "1", "b": "min", "spill": "spill"},
    "ram_k4_half": {"tier": "ram", "k": 4, "a": "1", "b": "0.5", "spill": "spill"},
    "ram_k4_quarter": {"tier": "ram", "k": 4, "a": "1", "b": "0.25", "spill": "spill"},
    "ram_k4_min": {"tier": "ram", "k": 4, "a": "1", "b": "min", "spill": "spill"},
    # Step (e) of the card: weights read from the same drive the spill writes to.
    "disk_k4_same_drive": {"tier": "disk", "k": 4, "a": "1", "b": "min", "spill": "weights"},
    # Optional, needs --other-spill-dir on a second PHYSICAL drive: A spills there.
    "disk_k4_other_drive": {"tier": "disk", "k": 4, "a": "min", "b": "min",
                            "spill": "weights", "a_spill": "other"},
}
DEFAULT_COMPARISONS = tuple(name for name in COMPARISONS if name != "disk_k4_other_drive")
CORRECTNESS_KS = (2, 3, 4)


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--spill-dir", required=True,
                        help="spill directory for the RAM-tier arms (a local, non-overlay volume)")
    parser.add_argument("--weights-spill-dir", default=None,
                        help="spill directory on the SAME drive as --shards (disk-tier arms)")
    parser.add_argument("--other-spill-dir", default=None,
                        help="spill directory on a SECOND physical drive (disk_k4_other_drive)")
    parser.add_argument("--out", required=True, help="a new directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--ring", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--comparisons", default=",".join(DEFAULT_COMPARISONS))
    parser.add_argument("--correctness-ks", default=",".join(map(str, CORRECTNESS_KS)))
    parser.add_argument("--max-blocks", type=int, default=24,
                        help="cap per comparison; the repeats rule decides below it (v2: 24; "
                             "v1 ran with 12)")
    parser.add_argument("--drive-bytes", type=int, default=4 << 30)
    parser.add_argument("--host-total-bytes", type=int, default=None)
    parser.add_argument("--enospc-dir", default=None)
    parser.add_argument("--accept-network", action="store_true")
    parser.add_argument("--backend", choices=("direct", "buffered"), default="direct",
                        help="buffered: page-cache spill, for a labelled CPU pilot only")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--label", default="")
    return parser


def _common(args: argparse.Namespace) -> List[str]:
    base = ["--weights", args.weights, "--shards", args.shards, "--device", args.device,
            "--seq", str(args.seq), "--ring", str(args.ring), "--seed", "3",
            "--input-seed", "17", "--lora-r", "8", "--backend", args.backend]
    if args.device == "cuda":
        base += ["--quant", "nf4", "--buffers", "2", "--read-ahead", "2"]
    if args.host_total_bytes:
        base += ["--host-total-bytes", str(args.host_total_bytes)]
    if args.accept_network:
        base.append("--accept-network")
    if args.label:
        base += ["--label", args.label]
    return base


def _spill_dir(args: argparse.Namespace, where: str) -> str:
    if where == "spill":
        return args.spill_dir
    if where == "other":
        if not args.other_spill_dir:
            raise ValueError("disk_k4_other_drive needs --other-spill-dir on a second drive")
        return args.other_spill_dir
    if not args.weights_spill_dir:
        # No default next to --shards: if the shards directory is a mount root or a
        # symlink, its parent can be another disk, and the arms would measure a
        # different drive while being labelled "same drive".
        raise ValueError("the disk-tier comparisons need --weights-spill-dir on the shards' "
                         "physical disk")
    return args.weights_spill_dir


def placement_check(args: argparse.Namespace, names: Sequence[str]) -> Dict[str, Any]:
    """Before any child: where the shards and every spill directory really live. A
    same-drive comparison whose spill and shards are not shown on one physical disk is
    refused here (an unknown disk counts as not shown); an other-drive comparison whose
    two spill directories are not shown on two disks likewise."""
    import a4_box

    record: Dict[str, Any] = {"shards": a4_box.placement(args.shards), "refusals": []}
    for name in names:
        spec = COMPARISONS[name]
        if spec["spill"] != "weights":
            continue
        directory = _spill_dir(args, "weights")
        record["weights_spill"] = a4_box.placement(directory)
        same, why = a4_box.same_physical_disk(record["shards"], record["weights_spill"])
        record[f"{name}_same_disk"] = {"same": same, "why": why}
        if same is not True:
            record["refusals"].append(f"{name}: spill {directory} and shards {args.shards} are "
                                      f"not shown on one physical disk ({why})")
        if spec.get("a_spill") == "other":
            other = _spill_dir(args, "other")
            record["other_spill"] = a4_box.placement(other)
            same, why = a4_box.same_physical_disk(record["shards"], record["other_spill"])
            if same is not False:
                record["refusals"].append(f"{name}: --other-spill-dir is not shown on another "
                                          f"physical disk ({why})")
    return record


def fixed_plan(args: argparse.Namespace, out: Path) -> List[Dict[str, Any]]:
    """The children that run once, before the timing blocks."""
    plan: List[Dict[str, Any]] = []

    def add(name: str, arguments: List[str], **extra: Any) -> None:
        plan.append({"name": name, "output": str(out / f"{name}.json"),
                     "command": [sys.executable, PROBE, *arguments,
                                 "--out", str(out / f"{name}.json")], **extra})

    add("preflight", ["--mode", "preflight", "--spill-dir", args.spill_dir, "--weights",
                      args.weights, "--shards", args.shards]
        + (["--accept-network"] if args.accept_network else []))
    add("drive", ["--mode", "drive", "--spill-dir", args.spill_dir, "--device", args.device,
                  "--drive-bytes", str(args.drive_bytes)]
        + (["--accept-network"] if args.accept_network else []))
    for k in [int(part) for part in args.correctness_ks.split(",") if part.strip()]:
        base = [*_common(args), "--mode", "correctness", "--tier", "ram", "--k", str(k),
                "--spill-dir", args.spill_dir]
        add(f"correctness_k{k}", base)
        add(f"correctness_k{k}_det", [*base, "--deterministic"],
            fallback_for=f"correctness_k{k}")
    cleanup = [*_common(args), "--mode", "cleanup", "--tier", "ram", "--k", "4",
               "--spill-dir", args.spill_dir]
    if args.enospc_dir:
        cleanup += ["--enospc-dir", args.enospc_dir]
    add("cleanup", cleanup)
    return plan


def arm_command(args: argparse.Namespace, out: Path, name: str, block: int, position: int,
                arm: str) -> Dict[str, Any]:
    spec = COMPARISONS[name]
    budget = spec["a"] if arm == "A" else spec["b"]
    where = spec.get("a_spill", spec["spill"]) if arm == "A" else spec["spill"]
    target = out / "arms" / name / f"b{block:02d}_p{position}_{arm}.json"
    arguments = [*_common(args), "--mode", "arm", "--tier", spec["tier"], "--k", str(spec["k"]),
                 "--budget", budget, "--spill-dir", _spill_dir(args, where),
                 "--comparison", name, "--arm", arm, "--block", str(block),
                 "--position", str(position), "--steps", str(args.steps),
                 "--warmup", str(args.warmup)]
    if spec["tier"] == "disk":
        arguments += ["--watch-path", args.shards]
    return {"name": f"{name}/b{block:02d}_p{position}_{arm}", "output": str(target),
            "command": [sys.executable, PROBE, *arguments, "--out", str(target)]}


def run_child(entry: Dict[str, Any], out: Path, env: Dict[str, str], deadline: float) -> int:
    target = Path(entry["output"])
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / (entry["name"].replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        child = subprocess.Popen(entry["command"], cwd=l2l_session.ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            return child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except BaseException:
            child.kill()
            child.wait()
            raise


def needs_fallback(path: Path) -> bool:
    """The predeclared --deterministic rerun: when GA vs GA or all-pinned vs all-pinned
    is not reproducible, or the default run left no record."""
    rows = [row for row in json.loads(path.read_text(encoding="utf-8"))["records"]
            if row.get("kind") == "correctness"]
    if not rows:
        return True
    return not all(rows[0][ref]["backward"]["equal"] and rows[0][ref]["forward_equal"]
                   for ref in ("ga_vs_ga", "pinned_vs_pinned"))


def block_runs(out: Path, name: str) -> List[List[arm_validity.Run]]:
    blocks: Dict[int, Dict[int, arm_validity.Run]] = {}
    for path in sorted((out / "arms" / name).glob("b*_p*_*.json")):
        rows = [row for row in json.loads(path.read_text(encoding="utf-8"))["records"]
                if row.get("kind") == "arm"]
        if not rows or rows[0].get("skipped"):
            continue
        row = rows[0]
        blocks.setdefault(row["block"], {})[row["position"]] = arm_validity.Run(
            float(row["value"]), str(row["outcome"]))
    return [[runs[p] for p in range(4)] for _, runs in sorted(blocks.items()) if len(runs) == 4]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = args_parser()
    args = parser.parse_args(argv)
    for field in ("weights", "shards", "spill_dir", "weights_spill_dir", "other_spill_dir",
                  "out", "enospc_dir"):
        if getattr(args, field):
            setattr(args, field, os.path.abspath(getattr(args, field)))
    names = [name.strip() for name in args.comparisons.split(",") if name.strip()]
    unknown = sorted(set(names) - set(COMPARISONS))
    if unknown:
        parser.error(f"unknown comparisons: {unknown}")
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} exists; every session writes a new directory")
    try:
        placement = placement_check(args, names)
    except ValueError as exc:
        parser.error(str(exc))
    if placement["refusals"]:
        parser.error("placement: " + "; ".join(placement["refusals"]))
    fixed = fixed_plan(args, out)
    protocol = {
        "rule": a4_rule.RULE, "args": vars(args), "comparisons": {n: COMPARISONS[n] for n in names},
        "fixed": fixed, "base_commit": l2l_session.git_head(),
        "versions": l2l_session.versions(), "source_sha256": l2l_session.source_hashes(),
        "placement": placement, "planned_unix": time.time(),
    }
    if not args.execute:
        for entry in fixed:
            print(" ".join(entry["command"]))
        for name in names:
            for position, arm in enumerate(arm_validity.BLOCK_ORDER):
                print(" ".join(arm_command(args, out, name, 0, position, arm)["command"]))
        print(f"plan only: {len(fixed)} fixed children, {len(names)} comparisons x >= "
              f"{arm_validity.PILOT_BLOCKS} blocks x 4 arms; --execute runs them")
        return 0
    out.mkdir(parents=True)
    l2l_session.write_json(out / "protocol.json", protocol)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(l2l_session.ROOT / "src"), str(HARNESS)]), "HF_HUB_OFFLINE": "1"}
    deadline = time.monotonic() + args.max_hours * 3600
    status: Dict[str, Any] = {"state": "running", "children": []}
    try:
        for entry in fixed:
            if "fallback_for" in entry and not needs_fallback(
                    out / f"{entry['fallback_for']}.json"):
                continue
            code = run_child(entry, out, env, deadline)
            status["children"].append({"name": entry["name"], "returncode": code})
            print(f"{entry['name']:<28} exit {code}")
            if code != 0:
                raise RuntimeError(f"{entry['name']} exited {code}; see its log")
        for name in names:
            block = 0
            while True:
                blocks = block_runs(out, name)
                due = arm_validity.blocks_to_add(blocks)
                if due <= 0 or block >= args.max_blocks:
                    break
                for position, arm in enumerate(arm_validity.BLOCK_ORDER):
                    entry = arm_command(args, out, name, block, position, arm)
                    code = run_child(entry, out, env, deadline)
                    status["children"].append({"name": entry["name"], "returncode": code})
                    print(f"{entry['name']:<40} exit {code}")
                    if code != 0:
                        raise RuntimeError(f"{entry['name']} exited {code}; see its log")
                block += 1
        status["state"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(state="stopped", reason=f"{type(exc).__name__}: {exc}")
    l2l_session.write_json(out / "session.json", status)
    verdict = a4_rule.evaluate(str(out))
    # The same bytes ``a4_rule.py --out`` writes, on every OS, so a re-analysis compares
    # byte for byte with the session's own verdict.
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps(verdict["summary"], indent=2))
    return 0 if status["state"] == "finished" else 3


if __name__ == "__main__":
    sys.exit(main())
