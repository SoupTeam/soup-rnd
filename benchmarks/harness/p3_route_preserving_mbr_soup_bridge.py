"""One-window Soup comparison: current L2L/two-read vs route-preserving MBR.

Scope: one pinned Moonlight 27-layer model, one 8x64-token window, BF16 activations,
per-expert NF4 cache, one AdamW update per measured branch. Attention and routing stay
microbatch-local; only routed expert FFN work is grouped. This is a runtime bridge diagnostic,
not a quality evaluation or a physical-SSD benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from soup_cli.utils.l2l_step import L2LStep

MODEL = "moonshotai/Moonlight-16B-A3B"
REVISION = "2a9a177f4841649f849ba941e41d2ee73c042314"
ROOT = Path("/workspace/p3-baseline")
SOURCE = ROOT / "source"
CACHE = ROOT / "cache"
DATA = Path("/workspace/p3-baseline/repo/benchmarks/protocols/b5-moonlight-20261109/data.json")
OUT = ROOT / "results/mbr_l2l_bridge"
SEED = 20261116
K = 8
SEQ = 64
MAX_STEPS = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    if result.returncode == 0 and revision:
        return revision
    marker = repo_root / "SOUP_REVISION"
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None


def tensor_digest(tensors: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode("utf-8"))
        h.update(tensors[name].detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_gradients(path: Any, tensors: Dict[str, Any]) -> None:
    """Write the LoRA gradients (name -> CPU tensor, dtype kept) for a per-parameter comparison."""
    import torch

    torch.save({n: t.detach().cpu().contiguous() for n, t in tensors.items()}, path)


def _read_int(path: Path) -> Optional[int]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def host_preflight(torch: Any, storage_path: Path) -> Dict[str, Any]:
    memory_max = _read_int(Path("/sys/fs/cgroup/memory.max"))
    memory_current = _read_int(Path("/sys/fs/cgroup/memory.current"))
    memory_swap_max = _read_int(Path("/sys/fs/cgroup/memory.swap.max"))
    memory_swap_current = _read_int(Path("/sys/fs/cgroup/memory.swap.current"))
    disk = shutil.disk_usage(storage_path)
    mount = None
    resolved = str(storage_path.resolve())
    try:
        candidates = []
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            source, target, fs_type = fields[:3]
            target = target.replace("\\040", " ")
            if resolved == target or resolved.startswith(target.rstrip("/") + "/"):
                candidates.append((len(target), source, target, fs_type))
        if candidates:
            _, source, target, fs_type = max(candidates)
            mount = {"source": source, "mountpoint": target, "filesystem": fs_type}
    except OSError:
        pass
    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device is not None else None,
        "visible_gpu_count": torch.cuda.device_count(),
        "vram_bytes": int(device.total_memory) if device is not None else None,
        "native_bf16": bool(torch.cuda.is_bf16_supported()) if device is not None else False,
        "cgroup_memory_max_bytes": memory_max,
        "cgroup_memory_current_bytes": memory_current,
        "cgroup_memory_swap_max_bytes": memory_swap_max,
        "cgroup_memory_swap_current_bytes": memory_swap_current,
        "storage_path": resolved,
        "storage_free_bytes": int(disk.free),
        "storage_mount": mount,
        "p3_ram_limit_pass": memory_max is not None and memory_max <= 32 * 2**30,
        "p3_swap_limit_pass": memory_swap_max == 0,
        "physical_nvme_confirmed": False,
        "note": (
            "Mount source is reported for inspection; this script does not claim it is a "
            "physical NVMe device."
        ),
    }


def validate_mbr_window_counts(stepper: Any, *, expected_grouped_dispatches: int) -> Dict[str, Any]:
    stats = stepper.route_preserving_mbr_stats or {}
    failures = []
    checked = []
    for layer, phases in sorted(stats.get("layers", {}).items(), key=lambda item: int(item[0])):
        init = phases.get("init_forward", {})
        if "unique_expert_count" not in init:
            continue
        recompute = phases.get("recompute_forward", {})
        backward = phases.get("backward_dX", {})
        unique = int(init["unique_expert_count"])
        row = {
            "layer": int(layer),
            "unique_experts": unique,
            "init_fetches": int(init.get("expert_fetches", -1)),
            "recompute_fetches": int(recompute.get("expert_fetches", -1)),
            "backward_dX_fetches": int(backward.get("expert_fetches", -1)),
            "init_grouped_dispatches": int(init.get("grouped_expert_dispatch_calls", -1)),
            "recompute_grouped_dispatches": int(recompute.get("grouped_expert_dispatch_calls", -1)),
            "route_id_mismatch_slots": int(phases.get("route_id_mismatch_slots", -1)),
        }
        if row["init_fetches"] != unique:
            failures.append(
                f"layer {layer}: init forward fetched {row['init_fetches']} times, "
                f"expected {unique}"
            )
        if row["recompute_fetches"] != 0:
            failures.append(f"layer {layer}: recompute fetched {row['recompute_fetches']} experts")
        if row["backward_dX_fetches"] != unique:
            failures.append(
                f"layer {layer}: backward_dX fetched {row['backward_dX_fetches']} times, "
                f"expected {unique}"
            )
        if (
            row["init_grouped_dispatches"] != expected_grouped_dispatches
            or row["recompute_grouped_dispatches"] != expected_grouped_dispatches
        ):
            failures.append(
                f"layer {layer}: expected {expected_grouped_dispatches} grouped dispatches "
                "in forward and recompute"
            )
        if row["route_id_mismatch_slots"] != 0:
            failures.append(f"layer {layer}: recompute changed route IDs")
        checked.append(row)
    if not checked:
        failures.append("no routed-expert layers were audited by route-preserving MBR")
    return {"checked_layers": checked, "failures": failures}


def validate_l2l_two_read_counts(result: Dict[str, Any]) -> Dict[str, Any]:
    """Check two-read leases and account for bytes served by the packed host hotset."""
    stats = result.get("phase_counters") or {}
    failures = []
    checked = []
    total_fetches = total_bytes = 0
    pool_stats = result.get("expert_pool_stats_after_run") or {}
    host_cache = pool_stats.get("host_cache")
    cache_delta = result.get("expert_host_cache_counter_delta") or {}
    host_cache_enabled = host_cache is not None
    for layer, phases in sorted(stats.get("layers", {}).items(), key=lambda item: int(item[0])):
        initial = phases.get("init_forward", {})
        recompute = phases.get("recompute_forward", {})
        backward = phases.get("backward_dX", {})
        if not any(int(row.get("expert_fetches", 0)) for row in (initial, recompute, backward)):
            continue
        row = {
            "layer": int(layer),
            "init_fetches": int(initial.get("expert_fetches", 0)),
            "recompute_fetches": int(recompute.get("expert_fetches", 0)),
            "backward_dX_fetches": int(backward.get("expert_fetches", 0)),
            "init_bytes": int(initial.get("expert_bytes_read", 0)),
            "recompute_bytes": int(recompute.get("expert_bytes_read", 0)),
            "backward_dX_bytes": int(backward.get("expert_bytes_read", 0)),
        }
        if row["recompute_fetches"] or row["recompute_bytes"]:
            failures.append(
                f"layer {layer}: two-read recompute recorded {row['recompute_fetches']} "
                f"fetches and {row['recompute_bytes']} bytes"
            )
        if row["init_fetches"] != row["backward_dX_fetches"]:
            failures.append(
                f"layer {layer}: initial/backward fetch counts differ "
                f"({row['init_fetches']} vs {row['backward_dX_fetches']})"
            )
        if not host_cache_enabled and row["init_bytes"] != row["backward_dX_bytes"]:
            failures.append(
                f"layer {layer}: initial/backward byte counts differ "
                f"({row['init_bytes']} vs {row['backward_dX_bytes']})"
            )
        if host_cache_enabled and row["backward_dX_bytes"] > row["init_bytes"]:
            failures.append(
                f"layer {layer}: cached backward read more bytes than the initial forward "
                f"({row['backward_dX_bytes']} vs {row['init_bytes']})"
            )
        total_fetches += sum(
            row[name] for name in ("init_fetches", "recompute_fetches", "backward_dX_fetches")
        )
        total_bytes += sum(
            row[name] for name in ("init_bytes", "recompute_bytes", "backward_dX_bytes")
        )
        checked.append(row)

    runtime = result.get("runtime_total") or {}
    expected_fetches = int(runtime.get("expert_fetches", -1))
    expected_bytes = int(runtime.get("expert_bytes_read", -1))
    if not checked:
        failures.append("no actual two-read expert leases were phase-audited")
    if total_fetches != expected_fetches:
        failures.append(
            f"phase fetch total {total_fetches} != runtime fetch total {expected_fetches}"
        )
    if total_bytes != expected_bytes:
        failures.append(f"phase byte total {total_bytes} != runtime byte total {expected_bytes}")
    if host_cache_enabled and (
        int(cache_delta.get("hits", 0)) + int(cache_delta.get("misses", 0)) != expected_fetches
    ):
        failures.append(
            "packed host-cache hit/miss total does not match expert fetches "
            f"({cache_delta.get('hits', 0)} + {cache_delta.get('misses', 0)} != "
            f"{expected_fetches})"
        )
    return {
        "checked_layers": checked,
        "phase_fetch_total": total_fetches,
        "runtime_fetch_total": expected_fetches,
        "phase_byte_total": total_bytes,
        "runtime_byte_total": expected_bytes,
        "host_cache_enabled": host_cache_enabled,
        "host_cache_counter_delta": cache_delta if host_cache_enabled else None,
        "failures": failures,
    }


class PhaseAudit:
    def __init__(self) -> None:
        self.current: Optional[tuple[int, str]] = None
        self.routes: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        self.handles: List[Any] = []
        self.capture_enabled = True

    def clear(self, *, capture_enabled: bool = True) -> None:
        self.current = None
        self.routes = {}
        self.capture_enabled = capture_enabled

    @contextlib.contextmanager
    def layer_call(self, layer: int, phase: str, token_count: int):
        previous = self.current
        self.current = (layer, phase)
        try:
            yield
        finally:
            self.current = previous

    def attach(self, layers: List[Any]) -> None:
        for layer_idx, layer in enumerate(layers):
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if experts is None or not hasattr(experts, "reader"):
                continue

            def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, li=layer_idx):
                if not self.capture_enabled or self.current is None:
                    return
                active_layer, phase = self.current
                if active_layer != li or phase not in ("init_forward", "recompute_forward"):
                    return
                ids = kwargs.get("top_k_index", args[1] if len(args) > 1 else None)
                weights = kwargs.get("top_k_weights", args[2] if len(args) > 2 else None)
                if ids is None or weights is None:
                    raise RuntimeError(
                        f"route audit could not read expert IDs/weights at layer {li}"
                    )
                self.routes.setdefault(phase, {}).setdefault(li, []).append(
                    {
                        "ids": ids.detach().long().cpu().clone(),
                        "weights": weights.detach().float().cpu().clone(),
                    }
                )

            self.handles.append(experts.register_forward_pre_hook(hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def within_branch_route_parity(audit: PhaseAudit) -> Dict[str, Any]:
    import torch

    rows = []
    failures = []
    init, recompute = (
        audit.routes.get("init_forward", {}),
        audit.routes.get("recompute_forward", {}),
    )
    for layer in sorted(set(init) | set(recompute)):
        a, b = init.get(layer, []), recompute.get(layer, [])
        if not a or not b:
            failures.append(f"layer {layer}: missing init/recompute route IDs")
            continue
        ids_a = torch.cat([r["ids"].reshape(-1, r["ids"].shape[-1]) for r in a])
        ids_b = torch.cat([r["ids"].reshape(-1, r["ids"].shape[-1]) for r in b])
        wa = torch.cat([r["weights"].reshape(-1, r["weights"].shape[-1]) for r in a])
        wb = torch.cat([r["weights"].reshape(-1, r["weights"].shape[-1]) for r in b])
        if ids_a.shape != ids_b.shape or not torch.equal(ids_a, ids_b):
            failures.append(f"layer {layer}: recompute route IDs differ")
        wdiff = (wa - wb).norm()
        wrel = float(wdiff / wa.norm().clamp_min(1e-30))
        if wrel > 1e-5:
            failures.append(f"layer {layer}: recompute route weights rel-L2={wrel:.3e}")
        rows.append(
            {
                "layer": layer,
                "calls_init": len(a),
                "calls_recompute": len(b),
                "route_weights_rel_l2": wrel,
            }
        )
    return {"layers": rows, "failures": failures}


class InstrumentedL2LStep(L2LStep):
    """Capture actual route calls for either the shipped step or route-preserving MBR."""

    def __init__(self, model: Any, runtime: Any, call: Any, audit: PhaseAudit) -> None:
        super().__init__(model, runtime, call)
        self.audit = audit

    def _layer(self, module: Any, hidden: Any, state: Any, layer_call: Any) -> Any:
        import torch

        if not self.audit.capture_enabled:
            return super()._layer(module, hidden, state, layer_call)
        li = self.layers.index(module)
        phase = "init_forward" if not torch.is_grad_enabled() else "recompute_forward"
        count = int(hidden.shape[0] * hidden.shape[1])
        with self.audit.layer_call(li, phase, count):
            return super()._layer(module, hidden, state, layer_call)

    def _route_preserving_layer_window(
        self,
        module: Any,
        hidden: Any,
        states: Any,
        calls: Any,
        group_size: int,
    ) -> Any:
        import torch

        if not self.audit.capture_enabled:
            return super()._route_preserving_layer_window(module, hidden, states, calls, group_size)
        li = self.layers.index(module)
        phase = "init_forward" if not torch.is_grad_enabled() else "recompute_forward"
        tokens = sum(int(value.shape[0] * value.shape[1]) for value in hidden)
        with self.audit.layer_call(li, phase, tokens):
            return super()._route_preserving_layer_window(module, hidden, states, calls, group_size)


def adapter_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    sq = 0.0
    elems = nonzero = 0
    grad_cpu = {}
    for name, param in params.items():
        if param.grad is None:
            continue
        grad = param.grad.detach().float().cpu().contiguous()
        grad_cpu[name] = grad
        sq += float((grad * grad).sum())
        elems += grad.numel()
        nonzero += int(torch.count_nonzero(grad))
    return {
        "norm": sq**0.5,
        "zero_fraction": 1 - nonzero / max(elems, 1),
        "gradients": grad_cpu,
        "missing_gradient_names": [name for name, param in params.items() if param.grad is None],
    }


def compare_tensors(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:

    dot = a2 = b2 = d2 = 0.0
    for name in left:
        a, b = left[name].float(), right[name].float()
        dot += float((a * b).sum())
        a2 += float((a * a).sum())
        b2 += float((b * b).sum())
        d2 += float(((a - b) ** 2).sum())
    return {
        "rel_l2_vs_left": (d2**0.5) / max(a2**0.5, 1e-30),
        "cosine": dot / max((a2 * b2) ** 0.5, 1e-30),
        "norm_left": a2**0.5,
        "norm_right": b2**0.5,
    }


def build_parser() -> argparse.ArgumentParser:
    """The bridge command line, split out of ``main`` so it can be checked without a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/p3-baseline")
    parser.add_argument("--source", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--mbr-group-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=8,
        help="number of the fixed eight micro-batches grouped into each routed-FFN call",
    )
    parser.add_argument(
        "--expert-transfer",
        choices=("sync", "packed"),
        default="sync",
        help="per-expert host-to-device transfer implementation",
    )
    parser.add_argument(
        "--expert-host-cache-mib",
        type=int,
        default=0,
        help="static packed-expert host hotset budget in MiB (requires --expert-transfer packed)",
    )
    parser.add_argument(
        "--expert-prefetch",
        action="store_true",
        help="read the next hit expert on a worker thread while the current one computes",
    )
    parser.add_argument(
        "--read-ahead",
        type=int,
        choices=tuple(range(1, 9)),
        default=1,
        help="async disk-tier staging depth (1 stages no layer ahead; 2 stages one)",
    )
    parser.add_argument(
        "--expert-gemm",
        choices=("native", "invariant", "segmented"),
        default="native",
        help="routed-expert GEMM: native cuBLAS; invariant (one row per call, plus the "
        "two-phase MBR backward so LoRA gradients add up in group-1 order; plan A2); or "
        "segmented (native cuBLAS per micro-batch segment, so every call has the L2L shape, "
        "plus the two-phase MBR backward)",
    )
    parser.add_argument(
        "--dump-grads",
        default=None,
        help="with --phase-profile-only: save the first measured step's LoRA gradients to this "
        "file (torch.save), for a per-parameter comparison between runs",
    )
    parser.add_argument("--require-p3-ram", action="store_true")
    parser.add_argument(
        "--phase-profile",
        action="store_true",
        help="after the A-B-B-A samples, run CUDA-synchronised phase-profiled steps (P72)",
    )
    parser.add_argument("--phase-profile-blocks", type=int, default=3)
    parser.add_argument(
        "--profile-kinds",
        default="l2l,mbr",
        help="with --phase-profile-only: l2l,mbr (A-B-B-A) or a single kind (A-A)",
    )
    parser.add_argument(
        "--timed-steps",
        type=int,
        default=2,
        help="with --phase-profile-only and one kind: uninstrumented timed steps (default 2)",
    )
    parser.add_argument(
        "--phase-profile-only",
        action="store_true",
        help="phase profile without the route audit (PR #72 refuses expert hooks); implies "
        "--phase-profile and skips the diagnostic gate and the route comparison",
    )
    return parser


def step_run_options(
    kind: str, *, group_size: int, expert_gemm: str, diagnostics: bool
) -> Dict[str, Any]:
    """Keyword arguments of ``L2LStep.run`` for one bridge run of ``kind`` (``l2l`` or ``mbr``)."""
    if expert_gemm not in ("native", "invariant", "segmented"):
        raise ValueError(
            f"expert gemm must be native, invariant or segmented, got {expert_gemm!r}"
        )
    mbr = kind == "mbr"
    options: Dict[str, Any] = {
        "route_preserving_mbr": mbr,
        "mbr_group_size": group_size if mbr else None,
        "collect_phase_stats": diagnostics,
    }
    if mbr and expert_gemm in ("invariant", "segmented"):
        options["mbr_split_backward"] = True
    return options


def main() -> None:
    import torch
    from b5_cold import process_read_bytes
    from b5_real_moe import _meta_construction
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils.kimi_k2 import load_stream_tokenizer
    from soup_cli.utils.l2l_step import capture_layer_call, label_tokens, model_parts, sft_head
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    args = build_parser().parse_args()
    if args.phase_profile_only:
        args.phase_profile = True
    if K % args.mbr_group_size:
        raise RuntimeError(f"STOP: mbr group size {args.mbr_group_size} must divide window K={K}")
    if args.expert_host_cache_mib < 0:
        raise RuntimeError("STOP: expert host-cache budget must be non-negative")
    if args.expert_host_cache_mib and args.expert_transfer != "packed":
        raise RuntimeError("STOP: expert host cache requires --expert-transfer packed")
    expected_grouped_dispatches = K // args.mbr_group_size
    expert_host_cache_bytes = args.expert_host_cache_mib * 2**20
    global ROOT, SOURCE, CACHE, DATA, OUT
    ROOT = Path(args.root).resolve()
    SOURCE = Path(args.source).resolve() if args.source else ROOT / "source"
    CACHE = Path(args.cache).resolve() if args.cache else ROOT / "cache"
    DATA = (
        Path(args.data).resolve()
        if args.data
        else ROOT / "repo/benchmarks/protocols/b5-moonlight-20261109/data.json"
    )
    OUT = (
        Path(args.output).resolve() if args.output else ROOT / "results/route_preserving_mbr_bridge"
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("STOP: bridge requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("STOP: this Soup path needs native BF16")
    if OUT.exists():
        raise FileExistsError(f"STOP: output directory already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    progress_path = OUT / "progress.json"
    completed_runs: List[Dict[str, Any]] = []

    def save_progress(
        phase: str, status: str = "running", extra: Optional[Dict[str, Any]] = None
    ) -> None:
        payload = {
            "status": status,
            "phase": phase,
            "completed_run_count": len(completed_runs),
            "completed_runs": completed_runs,
        }
        if extra:
            payload.update(extra)
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(progress_path)

    preflight = host_preflight(torch, CACHE if CACHE.exists() else ROOT)
    (OUT / "host_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    print("HOST PREFLIGHT", json.dumps(preflight, indent=2), flush=True)
    save_progress("host_preflight_complete", extra={"host_preflight": preflight})
    if args.require_p3_ram and not (
        preflight["p3_ram_limit_pass"] and preflight["p3_swap_limit_pass"]
    ):
        save_progress(
            "host_preflight", status="stopped_host_preflight", extra={"host_preflight": preflight}
        )
        raise RuntimeError("STOP before model load: cgroup RAM/swap does not meet P3 limits")
    save_progress("loading_model_and_cache")
    prepared = json.loads((ROOT / "results/cache.json").read_text(encoding="utf-8"))
    if prepared.get("status") != "PASS" or prepared.get("revision") != REVISION:
        raise RuntimeError("STOP: exact pinned Soup cache is not ready")
    config = AutoConfig.from_pretrained(str(SOURCE), trust_remote_code=False)
    index = read_shard_index(str(CACHE))
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with _meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(SOURCE),
            shard_dir=str(CACHE),
            index=index,
            lora_config=lora,
            device="cuda",
            dtype="bfloat16",
            buffers=2,
            pin=False,
            seed=SEED,
            quant="nf4",
            double_quant=False,
            tier="disk",
            read_ahead=args.read_ahead,
            two_read_experts=True,
            grouped_expert_routes=True,
            expert_transfer=args.expert_transfer,
            expert_host_cache_bytes=expert_host_cache_bytes,
            **({"expert_prefetch": True} if args.expert_prefetch else {}),
        )
    model.train()
    causal, owner = model_parts(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(causal, "config"):
        causal.config.use_cache = False
    if len(owner.layers) != 27:
        raise RuntimeError(f"STOP: expected 27 model layers, got {len(owner.layers)}")
    if any(p.requires_grad for p in owner.embed_tokens.parameters()):
        raise RuntimeError("STOP: bridge expects frozen token embeddings")
    if any(
        float(getattr(m, "p", 0.0) or 0.0) > 0
        for m in model.modules()
        if isinstance(m, torch.nn.Dropout)
    ):
        raise RuntimeError("STOP: first bridge requires dropout=0 for deterministic replay")

    tokenizer = load_stream_tokenizer(str(SOURCE), trust_remote_code=False)
    texts = json.loads(DATA.read_text(encoding="utf-8"))["train"]
    token_ids = tokenizer("\n\n".join(texts), add_special_tokens=True)["input_ids"]
    if len(token_ids) < K * SEQ:
        raise RuntimeError(f"STOP: fixed text produced only {len(token_ids)} tokens")
    flat = torch.tensor(token_ids[: K * SEQ], dtype=torch.long)
    micro_batches = [flat[i * SEQ : (i + 1) * SEQ].reshape(1, SEQ).cuda() for i in range(K)]
    labels = micro_batches
    input_hash = sha256_bytes(flat.numpy().tobytes())

    call = capture_layer_call(model, micro_batches[0], draws=False)
    calls = [call] * K
    audit = PhaseAudit()
    if not args.phase_profile_only:
        audit.attach(list(owner.layers))
    state = L2LState(runtime=runtime, dtype=torch.bfloat16, output_dir=str(OUT))
    l2l = (
        L2LStep(model, runtime, call)
        if args.phase_profile_only
        else InstrumentedL2LStep(model, runtime, call, audit)
    )
    state.stepper = l2l
    store = _store_for(state, l2l, micro_batches, torch.device("cuda"))
    params = {name: p for name, p in sorted(model.named_parameters()) if p.requires_grad}
    if not params:
        raise RuntimeError("STOP: no trainable LoRA parameters")
    initial = {name: p.detach().cpu().clone() for name, p in params.items()}
    initial_hash = tensor_digest(initial)
    save_progress("model_and_data_ready", extra={"input_sha256": input_hash})

    def reset_parameters() -> None:
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(initial[name].to(device=param.device, dtype=param.dtype))
        model.zero_grad(set_to_none=True)

    def one_run(
        kind: str, *, keep_artifacts: bool, collect_diagnostics: bool, phase_clock: Any = None
    ) -> Dict[str, Any]:
        stepper = l2l
        state.stepper = l2l
        audit.clear(capture_enabled=collect_diagnostics)
        reset_parameters()
        optimizer = torch.optim.AdamW(list(params.values()), lr=1e-4, weight_decay=0.01)
        optimizer.zero_grad(set_to_none=True)
        pool = getattr(runtime, "expert_pool", None)
        pool_stats_before = pool.stats() if pool is not None else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        read_before = process_read_bytes()
        started = time.perf_counter()
        profiled_wall = None
        with contextlib.ExitStack() as profiling:
            if phase_clock is not None:  # P72 diagnostic: timing wrappers only
                from p72_phase_profiler import install as install_phase_clock

                profiling.enter_context(install_phase_clock(phase_clock))
                profiling.enter_context(phase_clock.run())
                torch.cuda.synchronize()
                profiled_started = time.perf_counter()
            # Entered after the phase clock: it wraps F.linear too, and the swap must be outermost.
            from p72_invariant_expert_gemm import expert_gemm_context

            profiling.enter_context(expert_gemm_context(args.expert_gemm))
            losses = stepper.run(
                micro_batches,
                store,
                calls=calls,
                head=sft_head(labels, label_tokens(micro_batches)),
                **step_run_options(
                    kind,
                    group_size=args.mbr_group_size,
                    expert_gemm=args.expert_gemm,
                    diagnostics=collect_diagnostics,
                ),
            )
            if phase_clock is not None:
                torch.cuda.synchronize()
                profiled_wall = time.perf_counter() - profiled_started
        torch.cuda.synchronize()
        model_step_seconds = time.perf_counter() - started
        capture_started = time.perf_counter()
        grads = adapter_metrics(params)
        artifact_capture_seconds = time.perf_counter() - capture_started
        optimizer_started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        seconds = model_step_seconds + optimizer_seconds
        wall_seconds_including_capture = time.perf_counter() - started
        read_after = process_read_bytes()
        pool_stats_after = pool.stats() if pool is not None else None
        host_cache_before = (pool_stats_before or {}).get("host_cache")
        host_cache_after = (pool_stats_after or {}).get("host_cache")
        host_cache_delta = None
        if host_cache_before is not None and host_cache_after is not None:
            host_cache_delta = {
                key: int(host_cache_after[key]) - int(host_cache_before[key])
                for key in ("hits", "misses", "rejected")
            }
        loss = float(torch.stack([x.float() for x in losses]).sum().detach().cpu())
        if keep_artifacts:
            updated = {name: p.detach().float().cpu().clone() for name, p in params.items()}
            delta = {name: updated[name] - initial[name].float() for name in params}
        else:
            updated = None
            delta = None
        if collect_diagnostics:
            routes_parity = within_branch_route_parity(audit)
        else:
            routes_parity = None
        result = {
            "kind": kind,
            "loss": loss,
            "seconds": seconds,
            "wall_seconds_including_artifact_capture": wall_seconds_including_capture,
            "model_step_seconds": model_step_seconds,
            "optimizer_seconds": optimizer_seconds,
            "artifact_capture_seconds_excluded": artifact_capture_seconds,
            "tokens_per_second": (K * SEQ) / seconds,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "peak_allocated_delta_GiB": (torch.cuda.max_memory_allocated() - base_alloc) / 2**30,
            "process_read_bytes_delta": (
                read_after - read_before
                if read_before is not None and read_after is not None
                else None
            ),
            "runtime_total": {
                "expert_fetches": int(stepper.counters.expert_fetches),
                "expert_bytes_read": int(stepper.counters.expert_bytes_read),
            },
            "expert_pool_stats_after_run": pool_stats_after,
            "expert_host_cache_counter_delta": host_cache_delta,
            "phase_counters": stepper.expert_phase_stats,
            "mbr_window_counter_audit": (
                validate_mbr_window_counts(
                    stepper, expected_grouped_dispatches=expected_grouped_dispatches
                )
                if kind == "mbr" and collect_diagnostics
                else None
            ),
            "route_recompute_parity": routes_parity,
            "phase_profile": (
                {
                    "wall": profiled_wall,
                    "totals": dict(phase_clock.totals),
                    "counts": dict(phase_clock.counts),
                    "background": dict(phase_clock.background),
                    "gemm_rows": list(phase_clock.gemm_rows),
                }
                if phase_clock is not None
                else None
            ),
            "grad_norm": grads["norm"],
            "grad_zero_fraction": grads["zero_fraction"],
            "missing_gradient_names": grads["missing_gradient_names"],
            "initial_adapter_sha256": initial_hash,
            "updated_adapter_sha256": tensor_digest(updated) if updated is not None else None,
            "route_artifacts": (
                {
                    phase: {
                        str(layer): [{"ids": r["ids"], "weights": r["weights"]} for r in records]
                        for layer, records in per_layer.items()
                    }
                    for phase, per_layer in audit.routes.items()
                }
                if keep_artifacts
                else None
            ),
            "grad_artifacts": grads["gradients"] if keep_artifacts else None,
            "delta_artifacts": delta if keep_artifacts else None,
        }
        completed_runs.append(strip_artifacts(result))
        save_progress("run_complete", extra={"last_run_kind": kind})
        del optimizer
        model.zero_grad(set_to_none=True)
        return result

    def run_phase_profile(measured: List[Dict[str, Any]]) -> Dict[str, Any]:
        kinds = list(dict.fromkeys(row["kind"] for row in measured))
        order = [row["kind"] for row in measured]
        from p72_mbr_phase_profile import summarize_runs
        from p72_phase_profiler import PhaseClock

        clock = PhaseClock("cuda")
        by_kind: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in kinds}
        exact: Dict[str, bool] = {kind: True for kind in kinds}
        reference = {kind: next(r for r in measured if r["kind"] == kind) for kind in kinds}
        # A-B-B-A blocks outside the reported timing samples; serialised by CUDA syncs.
        for _ in range(args.phase_profile_blocks):
            for kind in order:
                row = one_run(
                    kind, keep_artifacts=True, collect_diagnostics=False, phase_clock=clock
                )
                by_kind[kind].append(row["phase_profile"])
                ref = reference[kind]
                exact[kind] &= row["loss"] == ref["loss"] and all(
                    torch.equal(row["grad_artifacts"][name], ref["grad_artifacts"][name])
                    for name in ref["grad_artifacts"]
                )
        phase_profile = {
            kind: summarize_runs(
                kind,
                by_kind[kind],
                uninstrumented_seconds=[
                    r["model_step_seconds"] for r in measured if r["kind"] == kind
                ],
                parity_ok=exact[kind],
            )
            for kind in kinds
        }
        (OUT / "phase_profile.json").write_text(
            json.dumps(phase_profile, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(phase_profile, indent=2), flush=True)
        return phase_profile

    try:
        if args.phase_profile_only:
            wanted = args.profile_kinds.split(",")
            if not set(wanted) <= {"l2l", "mbr"}:
                raise ValueError("--profile-kinds takes l2l and/or mbr")
            order_only = (
                ["l2l", "mbr", "mbr", "l2l"] if len(wanted) == 2 else wanted * args.timed_steps
            )
            for kind in dict.fromkeys(wanted):
                one_run(kind, keep_artifacts=False, collect_diagnostics=False)
            measured_only = [
                one_run(kind, keep_artifacts=True, collect_diagnostics=False)
                for kind in order_only
            ]
            if args.dump_grads:
                save_gradients(args.dump_grads, measured_only[0]["grad_artifacts"])
            # --phase-profile-blocks 0 is a timing-only run (the F1 blocks use it).
            profile = run_phase_profile(measured_only) if args.phase_profile_blocks else None
            (OUT / "result_phase_profile_only.json").write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(0),
                        "soup_revision": source_revision(),
                        "measured_uninstrumented": [
                            strip_artifacts(r) for r in measured_only
                        ],
                        "grad_sha256": [
                            tensor_digest(r["grad_artifacts"]) for r in measured_only
                        ],
                        "arm": {
                            "expert_transfer": args.expert_transfer,
                            "expert_host_cache_mib": args.expert_host_cache_mib,
                            "expert_prefetch": bool(args.expert_prefetch),
                        },
                        "phase_profile": profile,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return
        warm_l2l = one_run("l2l", keep_artifacts=True, collect_diagnostics=True)
        warm_mbr = one_run("mbr", keep_artifacts=True, collect_diagnostics=True)
        route_summary = compare_route_artifacts(
            warm_l2l["route_artifacts"]["init_forward"],
            warm_mbr["route_artifacts"]["init_forward"],
        )
        diagnostic_failures = []
        for name, result in (("L2L", warm_l2l), ("MBR", warm_mbr)):
            if result["route_recompute_parity"]["failures"]:
                diagnostic_failures.extend(
                    f"{name}: {failure}" for failure in result["route_recompute_parity"]["failures"]
                )
            if not torch.isfinite(torch.tensor(result["loss"])):
                diagnostic_failures.append(f"{name}: non-finite loss")
            if not result["grad_artifacts"]:
                diagnostic_failures.append(f"{name}: no adapter gradients were captured")
            elif any(
                not torch.isfinite(gradient).all().item()
                for gradient in result["grad_artifacts"].values()
            ):
                diagnostic_failures.append(f"{name}: non-finite adapter gradient")
            if result["missing_gradient_names"]:
                diagnostic_failures.append(
                    f"{name}: {len(result['missing_gradient_names'])} trainable adapter "
                    "parameters have no gradient; see run artifacts"
                )
        if warm_mbr["mbr_window_counter_audit"]["failures"]:
            diagnostic_failures.extend(
                f"MBR counter audit: {failure}"
                for failure in warm_mbr["mbr_window_counter_audit"]["failures"]
            )
        l2l_counter_audit = validate_l2l_two_read_counts(warm_l2l)
        if l2l_counter_audit["failures"]:
            diagnostic_failures.extend(
                f"L2L two-read counter audit: {failure}"
                for failure in l2l_counter_audit["failures"]
            )
        if any("tokens" not in row for row in route_summary["layers"]):
            diagnostic_failures.append("L2L and MBR route artifact shapes do not match")
        if not route_summary["layers"]:
            diagnostic_failures.append("no route layers were compared between L2L and MBR")
        torch.save(
            {
                "model": MODEL,
                "revision": REVISION,
                "input_sha256": input_hash,
                "l2l": warm_l2l["route_artifacts"],
                "mbr": warm_mbr["route_artifacts"],
                "l2l_gradients": warm_l2l["grad_artifacts"],
                "mbr_gradients": warm_mbr["grad_artifacts"],
            },
            OUT / "diagnostic_artifacts.pt",
        )
        diagnostic_gate = {
            "status": "PASS" if not diagnostic_failures else "FAIL",
            "failures": diagnostic_failures,
            "route_comparison": route_summary,
            "mbr_window_counter_audit": warm_mbr["mbr_window_counter_audit"],
            "l2l_two_read_counter_audit": l2l_counter_audit,
            "l2l_route_recompute_parity": warm_l2l["route_recompute_parity"],
            "mbr_route_recompute_parity": warm_mbr["route_recompute_parity"],
        }
        (OUT / "diagnostic_gate.json").write_text(
            json.dumps(diagnostic_gate, indent=2) + "\n", encoding="utf-8"
        )
        save_progress(
            "diagnostic_gate",
            status="running" if not diagnostic_failures else "stopped_diagnostic_gate",
            extra={"diagnostic_gate": diagnostic_gate},
        )
        if diagnostic_failures:
            raise RuntimeError(
                "STOP before A-B-B-A timing: diagnostic gate failed; see diagnostic_gate.json"
            )
        measured = [
            one_run("l2l", keep_artifacts=True, collect_diagnostics=False),
            one_run("mbr", keep_artifacts=True, collect_diagnostics=False),
            one_run("mbr", keep_artifacts=True, collect_diagnostics=False),
            one_run("l2l", keep_artifacts=True, collect_diagnostics=False),
        ]
        phase_profile = run_phase_profile(measured) if args.phase_profile else None
        diag_a, diag_b = warm_l2l, warm_mbr
        first_a, first_b = measured[0], measured[1]
        # Route capture/phase counting are separate; raw gradients are copied after each timer.
        grad_summary = compare_tensors(first_a["grad_artifacts"], first_b["grad_artifacts"])
        parameter_update_summary = compare_tensors(
            first_a["delta_artifacts"], first_b["delta_artifacts"]
        )
        aa_gradient_repeatability = compare_tensors(
            measured[0]["grad_artifacts"], measured[3]["grad_artifacts"]
        )
        bb_gradient_repeatability = compare_tensors(
            measured[1]["grad_artifacts"], measured[2]["grad_artifacts"]
        )
        route_payload = {
            "model": MODEL,
            "revision": REVISION,
            "input_sha256": input_hash,
            "routes": {
                kind: {
                    phase: {
                        layer: records
                        for layer, records in result["route_artifacts"].get(phase, {}).items()
                    }
                    for phase in ("init_forward", "recompute_forward")
                }
                for kind, result in (("l2l", diag_a), ("mbr", diag_b))
            },
        }
        torch.save(route_payload, OUT / "routes.pt")
        l2l_seconds = [row["seconds"] for row in measured if row["kind"] == "l2l"]
        mbr_seconds = [row["seconds"] for row in measured if row["kind"] == "mbr"]
        timing_summary = {
            "l2l_seconds_samples": l2l_seconds,
            "mbr_seconds_samples": mbr_seconds,
            "l2l_seconds_median": median(l2l_seconds),
            "mbr_seconds_median": median(mbr_seconds),
            "speedup_l2l_over_mbr": median(l2l_seconds) / median(mbr_seconds),
            "l2l_tokens_per_second_median": K * SEQ / median(l2l_seconds),
            "mbr_tokens_per_second_median": K * SEQ / median(mbr_seconds),
        }
        report = {
            "status": "PASS"
            if all(
                not r["route_recompute_parity"]["failures"]
                and (
                    r["mbr_window_counter_audit"] is None
                    or not r["mbr_window_counter_audit"]["failures"]
                )
                for r in (warm_l2l, warm_mbr)
            )
            else "FAIL",
            "scope": (
                "one fixed 8x64-token Soup training window, current two-read L2L vs opt-in "
                "route-preserving MBR; one update per measured arm; not quality or "
                "physical-SSD evidence"
            ),
            "model": MODEL,
            "revision": REVISION,
            "gpu": torch.cuda.get_device_name(0),
            "soup_revision": source_revision(),
            "source_file_sha256": {
                "l2l_step.py": sha256_file(
                    Path(__file__).resolve().parents[2] / "src/soup_cli/utils/l2l_step.py"
                ),
                "bridge_runner.py": sha256_file(Path(__file__).resolve()),
            },
            "host_preflight": preflight,
            "dtype": "bfloat16",
            "quant": "nf4",
            "expert_layout": "per_expert",
            "expert_transfer": args.expert_transfer,
            "expert_host_cache_budget_bytes": expert_host_cache_bytes,
            "stream_read_ahead": args.read_ahead,
            "mbr_group_size_microbatches": args.mbr_group_size,
            "mbr_grouped_dispatches_per_layer": expected_grouped_dispatches,
            "config_layers": int(config.num_hidden_layers),
            "router_tokens": K * SEQ,
            "input_sha256": input_hash,
            "initial_adapter_sha256": initial_hash,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.01, "max_grad_norm": 1.0},
            "diagnostic_runs_outside_timing": [strip_artifacts(diag_a), strip_artifacts(diag_b)],
            "l2l_two_read_counter_audit": l2l_counter_audit,
            "measured_order": ["l2l", "mbr", "mbr", "l2l"],
            "measured": [strip_artifacts(r) for r in measured],
            "timing_summary": timing_summary,
            "l2l_vs_mbr_first_pair": {
                "loss_delta_mbr_minus_l2l": first_b["loss"] - first_a["loss"],
                "gradient": grad_summary,
                "lora_parameter_update": parameter_update_summary,
                "routes": route_summary,
            },
            "gradient_repeatability": {
                "l2l_AA": aa_gradient_repeatability,
                "mbr_BB": bb_gradient_repeatability,
            },
            "route_artifact": str(OUT / "routes.pt"),
            "phase_profile": phase_profile,
            "physical_io_claim": False,
            "notes": [
                (
                    "Both arms use the current Soup L2LStep with two_read_experts=True and "
                    "grouped_expert_routes=True."
                ),
                (
                    "Only the MBR arm groups routed expert FFN work across microbatches; "
                    "attention and routing remain per microbatch."
                ),
                (
                    "The MBR group-size sweep uses the same total 8x64-token input window "
                    "in every run."
                ),
                (
                    "Group size 1 is a no-cross-microbatch MBR-path control; it still "
                    "exercises the route-preserving wrapper."
                ),
                (
                    "A RAM ballast may reserve cgroup headroom, but does not change "
                    "memory.max and is not strict P3 host qualification."
                ),
                (
                    "Runtime expert bytes and process read_bytes are logical/process "
                    "counters, not SSD-controller telemetry."
                ),
                (
                    "Packed transfer and the bounded static host hotset are enabled only "
                    "when explicitly requested; host-cache hit counters are reported per run."
                ),
                (
                    "Diagnostic route capture and phase counter collection use separate "
                    "runs outside the reported A-B-B-A timing samples."
                ),
            ],
        }
        (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        save_progress(
            "complete", status="complete", extra={"result_path": str(OUT / "result.json")}
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        audit.close()
        state.close()
        runtime.close()


def strip_artifacts(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in result.items()
        if k not in {"route_artifacts", "grad_artifacts", "delta_artifacts"}
    }


def compare_route_artifacts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    rows = []
    total_slots = changed_slots = total_tokens = changed_tokens = changed_sets = 0
    for layer in sorted(set(left) & set(right), key=int):
        a_records, b_records = left[layer], right[layer]
        a = torch.cat([r["ids"].reshape(-1, r["ids"].shape[-1]) for r in a_records])
        b = torch.cat([r["ids"].reshape(-1, r["ids"].shape[-1]) for r in b_records])
        if a.shape != b.shape:
            rows.append(
                {
                    "layer": int(layer),
                    "status": "shape mismatch",
                    "l2l": list(a.shape),
                    "mbr": list(b.shape),
                }
            )
            continue
        mismatch = a != b
        token_diff = mismatch.any(dim=-1)
        set_diff = (a.sort(dim=-1).values != b.sort(dim=-1).values).any(dim=-1)
        slots, tokens = int(a.numel()), int(a.shape[0])
        slot_diff, token_change = int(mismatch.sum()), int(token_diff.sum())
        wa = torch.cat([r["weights"].reshape(-1, r["weights"].shape[-1]) for r in a_records])
        wb = torch.cat([r["weights"].reshape(-1, r["weights"].shape[-1]) for r in b_records])
        rows.append(
            {
                "layer": int(layer),
                "tokens": tokens,
                "route_slots": slots,
                "slot_mismatch_fraction": slot_diff / max(slots, 1),
                "token_mismatch_fraction": token_change / max(tokens, 1),
                "token_expert_set_mismatch_fraction": int(set_diff.sum()) / max(tokens, 1),
                "gate_weight_rel_l2": float((wa - wb).norm() / wa.norm().clamp_min(1e-30)),
            }
        )
        total_slots += slots
        changed_slots += slot_diff
        total_tokens += tokens
        changed_tokens += token_change
        changed_sets += int(set_diff.sum())
    return {
        "layers": rows,
        "route_slots": total_slots,
        "tokens": total_tokens,
        "slot_mismatch_fraction": changed_slots / max(total_slots, 1),
        "token_mismatch_fraction": changed_tokens / max(total_tokens, 1),
        "token_expert_set_mismatch_fraction": changed_sets / max(total_tokens, 1),
    }


if __name__ == "__main__":
    main()
