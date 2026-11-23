"""Run a Moonlight bits measurement with optional attention gradient capture."""

import argparse
import hashlib
import json
import re
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from nondet_analysis import (
    ATTENTION_PROJECTION,
    MOONLIGHT_TARGETS,
    SDPA_BACKENDS,
    SDPA_GRADS,
    projection_error,
    row_record,
    sdpa_capture_error,
    tensor_byte_view,
    window_label,
)
from rich.console import Console

_AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")


def _digest(cpu: Any) -> str:
    return hashlib.sha256(tensor_byte_view(cpu).numpy().tobytes()).hexdigest()


class _Capture:
    """Keep every projection output gradient and grad-enabled SDPA call."""

    def __init__(self, targets: tuple[str, ...]) -> None:
        self.targets = targets
        self.output_grads: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []
        self.call_zero: dict[str, Any] = {}
        self.step = -1

    def wrap_step(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.step += 1
            return original(*args, **kwargs)

        return wrapper

    def attach(self, model: Any, scope: ExitStack) -> None:
        import torch

        def hook_for(name: str) -> Callable[..., None]:
            def forward(_module: Any, _inputs: Any, output: Any) -> None:
                if not torch.is_grad_enabled() or not output.requires_grad:
                    return
                step = self.step

                def record(grad: Any) -> None:
                    cpu = grad.detach().to("cpu").contiguous()
                    self.output_grads.setdefault(name, []).append({
                        "sha256": _digest(cpu),
                        "max_abs": cpu.abs().max().item() if cpu.numel() else 0.0,
                        "numel": cpu.numel(),
                        "step": step,
                    })

                output.register_hook(record)

            return forward

        for name, module in model.named_modules():
            match = ATTENTION_PROJECTION.fullmatch(name)
            if match and match[2] in self.targets:
                scope.callback(module.register_forward_hook(hook_for(name)).remove)

    def wrap_sdpa(self, original: Callable[..., Any]) -> Callable[..., Any]:
        import torch

        def wrapper(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            if torch.is_grad_enabled():
                index = len(self.calls)
                record = {
                    "call": index, "sha_dq": None, "sha_dk": None, "sha_dv": None,
                    "q_shape": list(query.shape),
                }
                self.calls.append(record)

                def hook_for(name: str) -> Callable[[Any], None]:
                    def save(grad: Any) -> None:
                        cpu = grad.detach().to("cpu").contiguous()
                        if index == 0:
                            cpu = cpu.clone()
                            self.call_zero[name] = cpu
                        record[f"sha_{name}"] = _digest(cpu)

                    return save

                # A hook on the caller's tensor sees gradients from all its consumers.
                query, key, value = query.view_as(query), key.view_as(key), value.view_as(value)
                for name, tensor in zip(SDPA_GRADS, (query, key, value)):
                    if tensor.requires_grad:
                        tensor.register_hook(hook_for(name))
            return original(query, key, value, *args, **kwargs)

        return wrapper

    def write(self, path: Path) -> None:
        import torch

        path.mkdir(parents=True, exist_ok=True)
        for filename, data in (
            ("output_grads.json", self.output_grads), ("sdpa_calls.json", self.calls),
        ):
            (path / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        for name, tensor in self.call_zero.items():
            torch.save(tensor, path / f"sdpa_call0_{name}.pt")

    def completeness_error(self, path: Path, steps: int) -> str | None:
        if error := projection_error(self.output_grads, self.targets):
            return error
        expected_steps = set(range(steps))
        for name, occurrences in self.output_grads.items():
            if {entry["step"] for entry in occurrences} != expected_steps:
                return f"missing or unexpected projection gradient steps for {name}"
        return sdpa_capture_error(self.calls, path)


def _model_revision(source: str) -> str:
    try:
        config = json.loads((Path(source).expanduser() / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "unknown"
    return config.get("_commit_hash") or "unknown"


def _power_state() -> int | None:
    try:
        return int(_AC_ONLINE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _run(args: argparse.Namespace, capture: _Capture | None) -> dict[str, Any]:
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from benchmarks.harness import mbr_measure
    from soup_cli.utils import l2l_step, layer_stream_runtime

    harness_args = argparse.Namespace(
        source=args.source, cache=args.cache, out=args.out, arm=args.arm, k=args.k, seq=args.seq,
        steps=args.steps, lora_targets=args.lora_targets,
        deterministic=args.deterministic, device=args.device, lora_r=args.lora_r, kind="bits",
        math_attention=False, tokenizer=None, profile=False,
    )
    original_build = layer_stream_runtime.build_streamed_model
    with ExitStack() as scope:
        def build(*build_args: Any, **kwargs: Any) -> tuple[Any, Any]:
            if args.strict_determinism:
                # run_one has already enabled warn_only. Override it before the first model op.
                torch.use_deterministic_algorithms(True)
            model, runtime = original_build(*build_args, **kwargs)
            if capture is not None:
                capture.attach(model, scope)
            return model, runtime

        if args.backend != "default":
            scope.enter_context(sdpa_kernel(getattr(SDPBackend, SDPA_BACKENDS[args.backend])))
        if capture is not None or args.strict_determinism:
            scope.enter_context(patch.object(layer_stream_runtime, "build_streamed_model", build))
        if capture is not None:
            scope.enter_context(patch.object(
                l2l_step.L2LStep, "run", capture.wrap_step(l2l_step.L2LStep.run),
            ))
            scope.enter_context(patch.object(
                functional, "scaled_dot_product_attention",
                capture.wrap_sdpa(functional.scaled_dot_product_attention),
            ))
        return mbr_measure.run_one(harness_args)


def _write_rows(
    args: argparse.Namespace, harness: dict[str, Any], power_ac_online: int | None,
    model_revision: str,
) -> None:
    import torch

    from benchmarks.harness.mbr_measure import append_jsonl

    card = (
        torch.cuda.get_device_name()
        if args.device != "cpu" and torch.cuda.is_available()
        else "CPU"
    )
    harness.update(attention=args.backend, power_ac_online=power_ac_online)
    canonical = row_record(
        model="Moonlight-16B-A3B", model_revision=model_revision, card=card,
        window=window_label(args.k, args.seq), arm=args.arm, mode="bits",
        torch_version=torch.__version__, attention_backend=args.backend,
        tensor_file=harness.get("tensor_file"),
        extra={
            "tag": args.tag, "strict_determinism": args.strict_determinism,
            "harness_mode": harness["mode"],
            **{key: value for key, value in harness.items()
               if key not in {"arm", "mode", "tensor_file", "attention"}},
        },
    )
    append_jsonl(args.out / "rows.jsonl", harness)
    append_jsonl(args.out / "nondet_rows.jsonl", canonical)
    Console(soft_wrap=True, markup=False, highlight=False).print(json.dumps(canonical))


def main(argv: list[str] | None = None) -> int:
    """Run one bits window and write its rows, warnings, and optional gradients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-revision", help="Defaults to config.json _commit_hash or unknown")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lora-targets", default=",".join(MOONLIGHT_TARGETS))
    parser.add_argument("--backend", choices=("default", *SDPA_BACKENDS), default="default")
    parser.add_argument("--deterministic", action="store_true", help="Use warn-only determinism")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--strict-determinism", action="store_true", help="Raise on unsupported ops",
    )
    parser.add_argument("--warnings-file", type=Path, help="Default: <out>/warnings.txt")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args(argv)
    if args.strict_determinism:
        args.deterministic = True
    if not args.deterministic:
        parser.error("the bits harness requires --deterministic or --strict-determinism")
    if args.steps < 2 or min(args.k, args.seq, args.lora_r) < 1:
        parser.error("--steps must be at least 2; --k, --seq, and --lora-r must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must be a single filename component starting with a letter or digit")
    capture_path = args.out / "capture" / args.tag
    for path in (args.out / "nondet_rows.jsonl", capture_path):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing --out: {path} already exists")

    from benchmarks.harness.mbr_measure import parse_lora_targets

    try:
        args.lora_targets = parse_lora_targets(args.lora_targets)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.capture and len(set(args.lora_targets)) != len(MOONLIGHT_TARGETS):
        parser.error("--capture requires four distinct --lora-targets")

    args.out.mkdir(parents=True, exist_ok=True)
    warnings_file = args.warnings_file or args.out / "warnings.txt"
    warnings_file.parent.mkdir(parents=True, exist_ok=True)
    power_ac_online = _power_state()
    model_revision = args.model_revision or _model_revision(args.source)
    capture = _Capture(args.lora_targets) if args.capture else None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            harness = _run(args, capture)
        except Exception as error:
            # The harness has no row for exceptions other than OOM.
            crash = {
                "arm": args.arm, "k": args.k, "seq": args.seq, "kind": "bits",
                "mode": "deterministic", "status": "CRASH",
                "error": f"{type(error).__name__}: {error}",
            }
            _write_rows(args, crash, power_ac_online, model_revision)
            raise
        finally:
            messages = dict.fromkeys(" ".join(str(item.message).splitlines()) for item in captured)
            warnings_file.write_text("".join(msg + "\n" for msg in messages), encoding="utf-8")
            if capture is not None:
                capture.write(capture_path)
    if capture is not None and (error := capture.completeness_error(capture_path, args.steps)):
        if harness["status"] == "ok":
            harness.update(status="incomplete", error=error)
        else:
            harness["capture_error"] = error
    _write_rows(args, harness, power_ac_online, model_revision)
    return 0 if harness["status"] == "ok" else 1


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
