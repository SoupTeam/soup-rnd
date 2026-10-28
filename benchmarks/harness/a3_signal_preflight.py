#!/usr/bin/env python3
"""A3 stage 2: the signal preflight of protocol v3 (§3.7), on a tiny CPU model.

Run on the session's box, through the session's own launch, before any model run. The stand's
own ``run_child`` and ``drive_signal`` drive three stand children (``sigint``, ``kill9``,
``kill9_next``) of a tiny 8-layer Llama through the product's L2L trainer with spill, each in
its own ``systemd-run --scope`` (``--no-scope``: plain subprocesses). The product's RAM budget
is pinned the way the CPU tests pin it (psutil's free RAM, no cgroup level), so most layers
spill on any box; nothing else of the product or of the stand is replaced. The launcher's own
SIGINT disposition is left as it came, so the children inherit what a session's children
inherit. Exit 0 when every check holds:

- sigint: one SIGINT while the child holds inside optimizer step 2 raises KeyboardInterrupt
  there (record exit 'interrupted', in a training_step, 1 step done, hold released by the
  interrupt); the spill file existed before the signal and is gone after the exit;
- kill9: SIGKILL at the same hold leaves the spill file;
- kill9_next: the next child in that directory removes it at start and trains every step.

``preflight.json`` also says how SIGINT reached each process (the handler it inherited and
the one it installed, the /proc masks at the send), so an ignored or blocked signal is told
apart from a missed phase. Usage::

    python benchmarks/harness/a3_signal_preflight.py run --work <new dir> [--no-scope]
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a3_spill_stand as stand  # noqa: E402

PREFLIGHT = str(Path(__file__).resolve())
#: Free RAM whose budget (240,000 B, what the 512 MiB reserve leaves) spills most of the tiny
#: model's 8 layers at k 4, seq 64 (as tests/test_a3_spill_stand.py).
FREE_BYTES = (512 << 20) + 240_000
LAYERS, K, SEQ, ROWS = 8, 4, 64, 10
SCOPE_MEMORY_MAX = 8 << 30
DEADLINE_S = 1800.0


def make_tiny(directory: Path) -> str:
    """A tiny 8-layer Llama checkpoint (config.json + model.safetensors), random weights."""
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=False, max_position_embeddings=128)
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    directory.mkdir(parents=True)
    save_file({name: value.contiguous() for name, value in model.state_dict().items()},
              str(directory / "model.safetensors"))
    config.save_pretrained(str(directory))
    return str(directory)


def child_main(free: int, argv: Sequence[str]) -> int:
    """One stand child with the product's budget pinned: psutil's free RAM is ``free`` and no
    cgroup level is read (the roots point at an empty directory), as the CPU tests do. The
    SIGINT disposition this process inherited is left alone: the stand's child deals with it."""
    from soup_cli.utils import l2l_spill, layer_stream

    empty = tempfile.mkdtemp(prefix="a3-no-cgroup-")
    for name in ("PROC_ROOT", "CGROUP_V2_ROOT", "CGROUP_V1_ROOT"):
        setattr(l2l_spill, name, empty)
    layer_stream.free_ram_bytes = lambda: free
    return stand.main(["child", *argv])


def child_entry(args: argparse.Namespace, work: Path, out: Path, arm: str,
                weights: str) -> Dict[str, Any]:
    """The stand's entry shape for one cleanup child of the tiny model."""
    spec = stand.spec_of(arm)
    target = out / spec["folder"] / f"{arm}.json"
    spill_dir = str(work / "spill" / spec["group"])
    marker = str(out / "markers" / f"{arm}.json")
    unit = stand.scope_unit(out, arm)
    command = [sys.executable, PREFLIGHT, "child", str(FREE_BYTES),
               "--arm", arm, "--weights", weights, "--shards", str(work / "shards"),
               "--quant", "none", "--device", "cpu", "--spill-dir", spill_dir,
               "--memory-max", str(SCOPE_MEMORY_MAX), "--k", str(K), "--seq", str(SEQ),
               "--rows", str(ROWS), "--marker", marker,
               "--hold-timeout", str(args.hold_timeout), "--out", str(target)]
    scope = [] if args.no_scope else stand._scope(args, SCOPE_MEMORY_MAX, unit)
    entry: Dict[str, Any] = {"name": arm, "output": str(target), "spill_dir": spill_dir,
                             "memory_max": SCOPE_MEMORY_MAX, "limit": "tight",
                             "scope_unit": unit, "marker": marker, "command": scope + command}
    if spec.get("signal"):
        entry["signal_name"] = spec["signal"]
    return entry


def _check(checks: List[Dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def judge(out: Path, results: Dict[str, Dict[str, Any]], *, scoped: bool
          ) -> List[Dict[str, Any]]:
    """Every check of the module docstring, from the records and the parent's observations."""
    checks: List[Dict[str, Any]] = []
    sigint = stand.read_json(out / "cleanup" / "sigint.json") or {}
    sent = results["sigint"].get("signal") or {}
    marker = sent.get("marker") or {}
    path = marker.get("spill_path")
    at = sigint.get("at_interrupt") or {}
    hold = at.get("hold") or {}
    _check(checks, "sigint: sent once, at the hold inside optimizer step 2",
           sent.get("sent_unix") is not None and marker.get("phase") == "hold"
           and marker.get("steps_done") == 1,
           {"phase": marker.get("phase"), "steps_done": marker.get("steps_done"),
            "why": sent.get("why")})
    _check(checks, "sigint: the spill file existed before the signal",
           bool(path) and sent.get("files_before_signal") == [path],
           {"spill_path": path, "before": sent.get("files_before_signal")})
    _check(checks, "sigint: KeyboardInterrupt in the hold",
           sigint.get("exit") == "interrupted" and at.get("in_step") is True
           and at.get("steps_done") == 1 and hold.get("released") == "interrupt"
           and hold.get("pass") == "backward",
           {"exit": sigint.get("exit"), "at_interrupt": at or None,
            "target_at_send": sent.get("target_at_send")})
    _check(checks, "sigint: the child exited 130 and no spill file is left",
           sent.get("exited") is True and sent.get("returncode") == 130
           and sent.get("files_after_exit") == [] and sigint.get("spill_files_after_train") == [],
           {"returncode": sent.get("returncode"), "after_exit": sent.get("files_after_exit"),
            "after_train": sigint.get("spill_files_after_train")})
    _check(checks, "sigint: the child runs Python's own SIGINT handler",
           ((sigint.get("sigint_handling") or {}).get("installed") or {}).get("handler")
           == "default_int_handler", sigint.get("sigint_handling"))
    killed = results["kill9"].get("signal") or {}
    kmarker = killed.get("marker") or {}
    kpath = kmarker.get("spill_path")
    _check(checks, "kill9: SIGKILL at the hold inside optimizer step 2 left the spill file",
           killed.get("sent_unix") is not None and kmarker.get("phase") == "hold"
           and bool(kpath) and kpath in (killed.get("files_before_signal") or [])
           and kpath in (killed.get("files_after_exit") or []) and killed.get("returncode") == -9,
           {"phase": kmarker.get("phase"), "spill_path": kpath,
            "after_exit": killed.get("files_after_exit"), "returncode": killed.get("returncode")})
    nxt = stand.read_json(out / "cleanup" / "kill9_next.json") or {}
    after = results["kill9_next"].get("spill_files_after_exit") or []
    _check(checks, "kill9_next: removed the killed run's file at start and trained every step",
           results["kill9_next"].get("returncode") == 0 and nxt.get("exit") == "normal"
           and bool(kpath) and kpath in (nxt.get("stale_removed") or [])
           and kpath not in (nxt.get("spill_files_after_train") or []) and kpath not in after
           and nxt.get("steps") == nxt.get("expected_steps"),
           {"stale_removed": nxt.get("stale_removed"), "steps": nxt.get("steps"),
            "expected_steps": nxt.get("expected_steps"), "exit": nxt.get("exit")})
    if scoped:
        for arm, rec in (("sigint", sigint), ("kill9_next", nxt)):
            cgroup = rec.get("cgroup_start") or {}
            _check(checks, f"{arm}: ran in its own systemd scope at the planned limit, swap 0",
                   cgroup.get("version") == 2 and cgroup.get("memory_max") == SCOPE_MEMORY_MAX
                   and cgroup.get("swap_max") == 0 and stand.scope_unit(out, arm)
                   in str(cgroup.get("path")), cgroup or None)
    return checks


def run_main(args: argparse.Namespace) -> int:
    work = Path(os.path.abspath(args.work))
    if work.exists():
        print(f"REFUSED: {work} exists")
        return 2
    out = work / "out"
    (out / "logs").mkdir(parents=True)
    launcher = {"pid": os.getpid(),
                "handler": stand._handler_name(signal.getsignal(signal.SIGINT)),
                "proc": stand.sigint_status()}
    weights = make_tiny(work / "tiny")
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(stand.ROOT / "src"), str(stand.HARNESS)]),
           "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}

    def no_kernel_log(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"read": False, "error": "--no-scope: no kernel log read"}

    reader = no_kernel_log if args.no_scope else stand.kernel_oom
    deadline = time.monotonic() + DEADLINE_S
    results: Dict[str, Dict[str, Any]] = {}
    for arm in stand.CLEANUP_ORDER:
        results[arm] = stand.run_child(child_entry(args, work, out, arm, weights), out, env,
                                       deadline, oom_reader=reader)
    checks = judge(out, results, scoped=not args.no_scope)
    passed = all(check["pass"] for check in checks)
    stand.write_json(work / "preflight.json", {
        "pass": passed, "checks": checks, "launcher": launcher, "children": results,
        "scope": not args.no_scope, "hold_timeout_s": args.hold_timeout,
    })
    for check in checks:
        print(f"{'ok  ' if check['pass'] else 'FAIL'} {check['check']}")
    print(f"launcher SIGINT: {launcher['handler']}, /proc {launcher['proc']}")
    print("SIGNAL PREFLIGHT PASSED" if passed else "SIGNAL PREFLIGHT FAILED")
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["child"]:
        return child_main(int(argv[1]), argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the three cleanup children of the tiny model")
    run.add_argument("--work", required=True, help="a new directory on a local disk")
    run.add_argument("--no-scope", action="store_true",
                     help="plain subprocesses, no systemd-run scope (CPU tests)")
    run.add_argument("--hold-timeout", type=float, default=60.0)
    run.add_argument("--systemd-run", default="systemd-run")
    run.add_argument("--user-scope", action="store_true")
    return run_main(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
