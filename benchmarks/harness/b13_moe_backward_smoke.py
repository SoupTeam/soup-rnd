"""Standalone synthetic B1.3 smoke; no model downloads, pytest or imports from tests.

Run from the checkout root with its src on PYTHONPATH:
  python benchmarks/harness/b13_moe_backward_smoke.py --device cpu --output evidence.json

The negative arm changes host staging only when a real expert backward reload calls
reader.read_host inside autograd. It runs before either optimizer changes the clean
window's weights, so forward/loss equality is measured on precisely the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

try:
    from .b13_moe_fixture import (
        ALL_TARGETS,
        EMBEDDING_TARGETS,
        HEAD_SHARED_TARGETS,
        TARGETS,
        build_fixture,
        micro_batches,
        snapshot,
        trainables,
    )
except ImportError:
    from b13_moe_fixture import (
        ALL_TARGETS,
        EMBEDDING_TARGETS,
        HEAD_SHARED_TARGETS,
        TARGETS,
        build_fixture,
        micro_batches,
        snapshot,
        trainables,
    )


def source_identity() -> dict[str, Any]:
    """Identify both the imported product and dirty measured source, without mutating Git."""
    import soup_cli

    package = Path(os.path.realpath(soup_cli.__file__))

    def git(*args: str, cwd: Path = package.parent) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, check=True,
            text=True, timeout=10,
        ).stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel"))
    names = git("ls-files", "--modified", "--others", "--exclude-standard", "-z", cwd=root)
    manifest = {}
    for name in sorted(set(names.split("\0")) - {""}):
        path = root / name
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "sha": git("rev-parse", "HEAD", cwd=root), "dirty": bool(manifest),
        "dirty_sha256_manifest": manifest, "package_file": str(package),
        "harness_file": os.path.realpath(__file__),
        "harness_sha256_manifest": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(sys.modules[build_fixture.__module__].__file__))
        },
    }


def comparison(want: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    """Zero-tolerance verdict including every supported trainable, not a selected projection."""
    import torch

    rows, missing = {}, []
    for name in sorted(want.keys() | got.keys()):
        left, right = want.get(name), got.get(name)
        if left is None or right is None:
            missing.append(name)
            continue
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        diff = (left - right).abs()
        rows[name] = {
            "equal": torch.equal(left, right), "finite": finite,
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float((diff / left.abs().clamp_min(1e-30)).max()),
            "reference_nonzero": int(left.count_nonzero()),
            "actual_nonzero": int(right.count_nonzero()), "elements": right.numel(),
        }
    passed = bool(rows) and not missing and all(
        row["equal"] and row["finite"] for row in rows.values()
    )
    return {
        "verdict": "PASS" if passed else "FAIL", "exact": passed,
        "max_abs_diff": max((row["max_abs_diff"] for row in rows.values()), default=None),
        "missing": missing, "parameters": rows,
    }


class ReadProbe:
    """Observe the public reader seam and optionally corrupt ONLY backward staging."""

    def __init__(self, model: Any, pool: Any) -> None:
        from soup_cli.utils.expert_stream import streamed_experts_class

        self.pool = pool
        self.corrupt = False
        self.events: list[dict] = []
        self.originals: list[tuple] = []
        for module in model.modules():
            if isinstance(module, streamed_experts_class()):
                if module.pool is not pool:
                    raise AssertionError("MoE layers do not share the bounded expert arena")
                self._wrap(module.layer_idx, module.reader)

    def _wrap(self, layer: int, reader: Any) -> None:
        import torch

        from soup_cli.utils.expert_shard import DOWN, expert_key

        original = reader.read_host
        self.originals.append((reader, original))

        def read_host(expert: int, staging: Any) -> Any:
            backward = torch._C._current_autograd_node() is not None
            host = original(expert, staging)
            changed = backward and self.corrupt
            if changed:
                host[expert_key(expert, DOWN)].add_(0.5)
            self.events.append({
                "layer": layer, "expert": expert, "backward": backward,
                "grad_enabled": torch.is_grad_enabled(), "corrupted": changed,
                "arena_pointer": self.pool._arena.untyped_storage().data_ptr(),
            })
            return host

        reader.read_host = read_host

    def reset(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.events.clear()

    def close(self) -> None:
        for reader, original in self.originals:
            reader.read_host = original


def _read_evidence(
    events: Sequence[dict[str, Any]], before: dict[str, Any], after: dict[str, Any],
) -> dict[str, Any]:
    from collections import Counter

    backward = Counter((event["layer"], event["expert"])
                       for event in events if event["backward"])
    outside = Counter((event["layer"], event["expert"])
                      for event in events if not event["backward"])
    pointers = sorted({event["arena_pointer"] for event in events})
    return {
        "reads": len(events), "backward_reloads": sum(event["backward"] for event in events),
        "corrupted_reloads": sum(event["corrupted"] for event in events),
        "backward_layers": sorted({event["layer"] for event in events if event["backward"]}),
        "arena_pointers": pointers,
        "runtime_fetch_delta": after["expert_fetches"] - before["expert_fetches"],
        "runtime_bytes_delta": after["expert_bytes_read"] - before["expert_bytes_read"],
        "three_reads_per_visit": bool(backward) and outside == Counter(
            {key: 2 * count for key, count in backward.items()}
        ),
        "events": list(events),
    }


def _logits(model: Any, ids: Sequence[Any], *, seed: int) -> dict[str, Any]:
    import torch

    cards = [ids[0].device] if ids[0].device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cards):
        torch.manual_seed(seed)
        return {str(index): model(input_ids=batch).logits.detach().cpu().clone()
                for index, batch in enumerate(ids)}


def _rng_equal(left: Any, right: Any) -> bool:
    import torch

    return torch.equal(left.cpu, right.cpu) and len(left.cuda) == len(right.cuda) and all(
        card_a == card_b and torch.equal(state_a, state_b)
        for (card_a, state_a), (card_b, state_b) in zip(left.cuda, right.cuda)
    )


def run_case(
    root: Path, *, k: int, device: str, targets: Sequence[str] = TARGETS, dropout: float = 0.0,
) -> dict[str, Any]:
    import torch

    from soup_cli.utils import l2l_rng
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    fixture = build_fixture(root, device=device, targets=targets, dropout=dropout)
    store = probe = None
    try:
        first_ids, _ = micro_batches(fixture.config, k, first=0, device=device)
        stepper = L2LStep(
            fixture.streamed, fixture.runtime, capture_layer_call(fixture.streamed, first_ids[0]),
        )
        store = ActivationStore(
            n_layers=3, k=k, chunk_shape=(2, 6, 32), dtype=torch.float32,
            device=device, pin=device == "cuda",
        )
        reverse_visits = []
        original_get = store.get

        def get(layer: int, micro_batch: int) -> Any:
            reverse_visits.append([layer, micro_batch])
            return original_get(layer, micro_batch)

        store.get = get
        pool = fixture.runtime.expert_pool
        probe = ReadProbe(fixture.streamed, pool)
        pointer_before = pool._arena.untyped_storage().data_ptr()
        initial = snapshot(fixture.streamed)
        optimizers = [torch.optim.SGD(trainables(model).values(), lr=0.5)
                      for model in (fixture.resident, fixture.streamed)]
        windows = []
        for window in range(2):
            ids, labels = micro_batches(fixture.config, k, first=window * k, device=device)
            total = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(11 + window)
            reference_logits = _logits(fixture.resident, ids, seed=11 + window)
            reference_losses = []
            for batch, lab in zip(ids, labels):
                loss = fixture.resident(
                    input_ids=batch, labels=lab, num_items_in_batch=total,
                ).loss
                loss.backward()
                reference_losses.append(loss.detach().cpu().clone())
            reference_grads = snapshot(fixture.resident, gradients=True)
            reference_rng = l2l_rng.snapshot()

            # Corrupted pass first, without stepping: the clean pass reuses identical weights.
            probe.reset(corrupt=True)
            bad_logits = _logits(fixture.streamed, ids, seed=11 + window)
            probe.reset(corrupt=True)
            reverse_visits.clear()
            before = fixture.runtime.stats()
            torch.manual_seed(11 + window)
            bad_losses = stepper.run(ids, store, labels=labels, n_items=total)
            bad_grads = snapshot(fixture.streamed, gradients=True)
            bad_rng = l2l_rng.snapshot()
            bad_reads = _read_evidence(probe.events, before, fixture.runtime.stats())
            bad_reverse = list(reverse_visits)
            fixture.streamed.zero_grad(set_to_none=True)

            probe.reset()
            clean_logits = _logits(fixture.streamed, ids, seed=11 + window)
            probe.reset()
            reverse_visits.clear()
            before = fixture.runtime.stats()
            torch.manual_seed(11 + window)
            clean_losses = stepper.run(ids, store, labels=labels, n_items=total)
            clean_grads = snapshot(fixture.streamed, gradients=True)
            clean_rng = l2l_rng.snapshot()
            clean_reads = _read_evidence(probe.events, before, fixture.runtime.stats())
            clean_reverse = list(reverse_visits)
            expected_losses = {str(index): loss for index, loss in enumerate(reference_losses)}

            def loss_dict(losses: Sequence[Any]) -> dict[str, Any]:
                return {str(index): loss.detach().cpu() for index, loss in enumerate(losses)}

            clean = {
                "forward": comparison(reference_logits, clean_logits),
                "loss": comparison(expected_losses, loss_dict(clean_losses)),
                "gradient": comparison(reference_grads, clean_grads),
                "reader": clean_reads, "reverse_visits": clean_reverse,
                "rng_equal": _rng_equal(reference_rng, clean_rng),
            }
            bad = {
                "forward": comparison(reference_logits, bad_logits),
                "loss": comparison(expected_losses, loss_dict(bad_losses)),
                "gradient": comparison(reference_grads, bad_grads),
                "reader": bad_reads, "reverse_visits": bad_reverse,
                "rng_equal": _rng_equal(reference_rng, bad_rng),
            }
            for optimizer in optimizers:
                optimizer.step()
            state = comparison(snapshot(fixture.resident), snapshot(fixture.streamed))
            windows.append({
                "window": window, "k": k, "clean": clean, "backward_corruption": bad,
                "trained_state": state,
                "reference_losses": [float(loss) for loss in reference_losses],
                "clean_losses": [float(loss) for loss in clean_losses],
                "corrupted_losses": [float(loss) for loss in bad_losses],
            })
        pointer_after = pool._arena.untyped_storage().data_ptr()
        trained = snapshot(fixture.streamed)
        changed = [name for name in initial if not torch.equal(initial[name], trained[name])]
        expected_reverse = [[layer, mb] for layer in (2, 1, 0) for mb in range(k)]
        expected_layers = [1, 2]
        positive = all(
            all(row["clean"][key]["verdict"] == "PASS" for key in ("forward", "loss", "gradient"))
            and row["trained_state"]["verdict"] == "PASS"
            and row["clean"]["rng_equal"]
            and all(item["actual_nonzero"] > 0
                    for item in row["clean"]["gradient"]["parameters"].values())
            and row["clean"]["reader"]["backward_layers"] == expected_layers
            and row["clean"]["reader"]["backward_reloads"] > 0
            and row["clean"]["reader"]["three_reads_per_visit"]
            and row["clean"]["reader"]["reads"] == row["clean"]["reader"]["runtime_fetch_delta"]
            and row["clean"]["reader"]["runtime_bytes_delta"] > 0
            and row["clean"]["reverse_visits"] == expected_reverse
            for row in windows
        )
        negative = all(
            row["backward_corruption"]["forward"]["verdict"] == "PASS"
            and row["backward_corruption"]["loss"]["verdict"] == "PASS"
            and row["backward_corruption"]["rng_equal"]
            and row["backward_corruption"]["gradient"]["verdict"] == "FAIL"
            and row["backward_corruption"]["reader"]["corrupted_reloads"] > 0
            and row["backward_corruption"]["reader"]["three_reads_per_visit"]
            and row["backward_corruption"]["reader"]["backward_layers"] == expected_layers
            and row["backward_corruption"]["reverse_visits"] == expected_reverse
            for row in windows
        )
        arena = pool.stats()
        bounded = (
            pointer_before == pointer_after
            and arena["slot_count"] == 1
            and arena["plan"]["weight_bytes"] <= arena["weight_budget_bytes"]
            and all(row[arm]["reader"]["arena_pointers"] == [pointer_before]
                    for row in windows for arm in ("clean", "backward_corruption"))
        )
        return {
            "k": k, "windows": windows, "clean_verdict": "PASS" if positive else "FAIL",
            "targets": list(targets), "dropout": dropout, "trainable_tensors": len(initial),
            "negative_control_detected": negative, "arena_bounded": bounded,
            "arena": {**arena, "pointer_before": pointer_before, "pointer_after": pointer_after},
            "changed_trainables": changed,
            "verdict": "PASS" if positive and negative and bounded and changed else "FAIL",
        }
    finally:
        if probe is not None:
            probe.close()
        if store is not None:
            store.close()
        fixture.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets", choices=("mla", "embedding", "head-shared", "all"), default="mla",
        help="Synthetic adapter targets only; the unchanged default has 30 MLA tensors.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args(argv)
    targets = {
        "mla": TARGETS, "embedding": EMBEDDING_TARGETS,
        "head-shared": HEAD_SHARED_TARGETS, "all": ALL_TARGETS,
    }[args.targets]
    # Set before the first CUDA call. Refuse a conflicting launch instead of claiming determinism.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        parser.error("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    console = Console()
    result = {
        "source": source_identity(), "device": args.device, "dtype": "float32",
        "exactness": {"atol": 0, "rtol": 0, "deterministic_algorithms": True},
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            **{name: importlib.metadata.version(name)
               for name in ("transformers", "peft", "accelerate", "safetensors")},
        },
        "cases": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="b13-moe-smoke-") as temporary:
            for k in (1, 2, 4):
                case = run_case(
                    Path(temporary) / f"k{k}", k=k, device=args.device,
                    targets=targets, dropout=args.dropout,
                )
                result["cases"].append(case)
                console.print(
                    f"k={k}: clean={case['clean_verdict']}, "
                    f"backward corruption detected={case['negative_control_detected']}, "
                    f"arena bounded={case['arena_bounded']}"
                )
        result["verdict"] = (
            "PASS" if all(case["verdict"] == "PASS" for case in result["cases"]) else "FAIL"
        )
    except Exception as error:
        result["verdict"] = "FAIL"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        console.print_exception()
    finally:
        result["cuda_peaks_separate_from_arena"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print(f"B1.3 smoke: {result['verdict']} ({args.output})")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
