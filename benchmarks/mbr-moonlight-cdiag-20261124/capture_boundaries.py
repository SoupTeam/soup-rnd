"""Capture the boundaries of one Moonlight bits step (ticket 14).

Runs one arm of ``benchmarks/harness/mbr_measure.py`` in strict deterministic mode with hooks on
the layer's submodules and on every expert ``F.linear``. Writes ``boundaries/<tag>.json`` (module
digests and expert-linear row digests), ``tensors/`` (the harness loss and LoRA gradients),
``layer0.pt`` (the full layer-0 tensors behind the numeric numbers) and one canonical row.

A0 and C are two fresh processes of this script; ``compare_boundaries.py`` then finds the first
diverging boundary. Every result line names the model, card, window, arm, mode and torch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import cdiag
from rich.console import Console

_AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")
_SUBMODULE = re.compile(
    r"(?:.*\.)?layers\.(\d+)\.(?:inner\.)?"
    r"(input_layernorm|self_attn|post_attention_layernorm|mlp\.gate|mlp\.shared_experts|"
    r"mlp\.experts)$"
)
_MODEL_SOURCE = "Moonlight-16B-A3B"


def _first_tensor(args: Any, kwargs: Any) -> Any:
    for value in (*args, *kwargs.values()):
        tensors = _tensors(value)
        if tensors:
            return tensors[0]
    return None


def _tensors(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


class _Recorder:
    """Module boundaries plus the expert linears, for the first step only."""

    def __init__(self, group: int, dump_layer: int = 1) -> None:
        self.group = group
        self.dump_layer = dump_layer
        self.enabled = False
        self.modules: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.linears: list[dict[str, Any]] = []
        self.layer0: list[dict[str, Any]] = []
        self.step = -1
        self._counters: dict[tuple[int, str], int] = {}
        self._call = 0

    def wrap_step(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.step += 1
            self.enabled = self.step == 0
            try:
                return original(*args, **kwargs)
            finally:
                self.enabled = False

        return wrapper

    def attach(self, model: Any, scope: ExitStack) -> None:
        experts_class = None
        for name, module in model.named_modules():
            match = _SUBMODULE.fullmatch(name)
            if match is None:
                continue
            layer, role = int(match[1]), match[2]
            if role == "mlp.experts":
                # The MBR path refuses routed-expert module hooks, so the node is wrapped at class
                # level: every instance's forward goes through one recorder.
                experts_class = type(module)
                continue
            scope.callback(
                module.register_forward_hook(self._hook(layer, role), with_kwargs=True).remove
            )
        if experts_class is None:
            raise RuntimeError("no streamed experts module in the model")
        original = experts_class.forward

        def experts_forward(
            module: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any
        ) -> Any:
            output = original(module, hidden_states, top_k_index, top_k_weights)
            if self.enabled:
                self.record_node(module.layer_idx, hidden_states, output)
            return output

        scope.enter_context(patch.object(experts_class, "forward", experts_forward))

    def record_node(self, layer: int, hidden_states: Any, output: Any) -> None:
        """The routed-expert node: joined input rows and output rows of one call."""
        self.nodes.append(
            cdiag.node_record(
                layer=layer, call=len(self.nodes), inputs=hidden_states, outputs=output
            )
        )

    def _hook(self, layer: int, role: str) -> Callable[..., None]:
        def hook(_module: Any, args: Any, kwargs: Any, output: Any) -> None:
            if not self.enabled:
                return
            key = (layer, role)
            mb = self._counters.get(key, 0)
            self._counters[key] = mb + 1
            if mb >= self.group:  # the backward recompute replays a forward that is already here
                return
            tensors = _tensors(output)
            inputs = _first_tensor(args, kwargs)
            if not tensors or inputs is None:
                return
            self.modules.append(cdiag.module_record(
                layer=layer, role=role.rsplit(".", 1)[-1], mb=mb, inputs=inputs,
                outputs=tensors[0],
            ))

        return hook

    def wrap_linear(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
            output = original(input, weight, bias)
            if not self.enabled:
                return output
            info = _expert_call_info()
            if info is None:
                return output
            layer, expert, phase = info
            which = _projection(tuple(weight.shape))
            flat = input.reshape(-1, input.shape[-1])
            rows = output.reshape(-1, output.shape[-1])
            record = cdiag.expert_linear_record(
                layer=layer, expert=expert, which=which, phase=phase, call=self._call,
                inputs=flat, outputs=rows,
            )
            self.linears.append(record)
            if layer == self.dump_layer:
                self.layer0.append({
                    "call": self._call, "layer": layer, "expert": expert, "which": which,
                    "phase": phase,
                    "input": flat.detach().to("cpu"), "output": rows.detach().to("cpu"),
                })
            self._call += 1
            return output

        return linear

    def write(self, path: Path, tag: str) -> None:
        import torch

        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step + 1, "modules": self.modules, "nodes": self.nodes,
            "linears": self.linears,
            "module_calls": {
                f"{layer}:{role}": count
                for (layer, role), count in sorted(self._counters.items())
            },
            "expert_call_rows": _row_counts(self.linears),
        }
        (path / f"{tag}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        torch.save([dict(call) for call in self.layer0], path / f"{tag}-tensors.pt")


def _row_counts(linears: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in linears:
        if record["phase"] != "forward":
            continue
        key = f"M={record['rows']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _expert_call_info() -> tuple[int, int, str] | None:
    """``(layer, expert, phase)`` when ``linear`` was called from ``expert_mlp``."""
    frame = sys._getframe(2)
    for _ in range(8):
        if frame is None:
            return None
        if frame.f_code.co_name == "expert_mlp":
            break
        frame = frame.f_back
    else:
        return None
    caller = frame.f_back
    for _ in range(6):
        if caller is None or caller.f_code.co_name not in ("forward", "backward"):
            caller = None if caller is None else caller.f_back
            continue
        local = caller.f_locals
        context = local.get("ctx")
        owner = local.get("owner") or getattr(context, "owner", None)
        expert = local.get("expert")
        if expert is None and context is not None:
            expert = getattr(context, "expert", None)
        if owner is not None and expert is not None:
            return owner.layer_idx, int(expert), caller.f_code.co_name
        caller = caller.f_back
    return None


def _projection(shape: tuple[int, ...]) -> str:
    for name, expected in cdiag.EXPERT_PROJECTIONS.items():
        if shape == expected:
            return name
    raise RuntimeError(f"expert linear weight {shape} matches no known projection")


def _model_revision(source: str, override: str | None = None) -> str:
    """The pinned commit: an explicit ``--model-revision``, else the source config's hash.

    A source without the hash is ``unknown``; the campaign does not guess a revision from a cache
    record next to the source, because that record belongs to the cache, not to this checkout."""
    if override:
        return override
    try:
        payload = json.loads((Path(source).expanduser() / "config.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return str(payload.get("_commit_hash") or "unknown")


def _power_state() -> int | None:
    try:
        return int(_AC_ONLINE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _harness_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        source=args.source, cache=args.cache, out=args.out, arm=args.arm, k=args.k,
        seq=args.seq, steps=args.steps, lora_targets=args.lora_targets, deterministic=True,
        device=args.device, lora_r=args.lora_r, kind="bits", math_attention=args.backend == "math",
        tokenizer=None, profile=False,
    )


def _run(args: argparse.Namespace, recorder: _Recorder) -> dict[str, Any]:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from benchmarks.harness import mbr_measure
    from soup_cli.utils import l2l_step, layer_stream_runtime

    original_build = layer_stream_runtime.build_streamed_model
    with ExitStack() as scope:

        def build(*build_args: Any, **kwargs: Any) -> tuple[Any, Any]:
            # run_one has already enabled warn_only; strictness must land before the first op.
            torch.use_deterministic_algorithms(True)
            model, runtime = original_build(*build_args, **kwargs)
            recorder.attach(model, scope)
            return model, runtime

        scope.enter_context(patch.object(layer_stream_runtime, "build_streamed_model", build))
        scope.enter_context(
            patch.object(l2l_step.L2LStep, "run", recorder.wrap_step(l2l_step.L2LStep.run))
        )
        if args.backend != "default":
            backend = {"math": SDPBackend.MATH}[args.backend]
            scope.enter_context(sdpa_kernel(backend))
        scope.enter_context(
            patch.object(
                torch.nn.functional, "linear", recorder.wrap_linear(torch.nn.functional.linear)
            )
        )
        if args.expert_gemm != "native":
            from soup_cli.utils import expert_gemm

            original_context = expert_gemm.expert_gemm_context
            scope.enter_context(patch.object(
                expert_gemm, "expert_gemm_context",
                lambda mode, **kwargs: original_context(args.expert_gemm, **kwargs),
            ))
        return mbr_measure.run_one(_harness_args(args))


def _write_row(args: argparse.Namespace, harness: dict[str, Any], power: int | None) -> dict:
    import torch

    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    return {
        "model": _MODEL_SOURCE,
        "model_revision": _model_revision(args.source, getattr(args, "model_revision", None)),
        "card": card,
        "window": f"k={args.k},S={args.seq}", "arm": args.arm, "mode": "bits (deterministic)",
        "torch_version": torch.__version__, "attention_backend": args.backend,
        "expert_gemm": args.expert_gemm, "strict_determinism": True,
        "power_ac_online": power, "tag": args.tag,
        **{key: harness.get(key) for key in (
            "status", "loss", "step_seconds", "peak_allocated", "peak_reserved", "peak_nvml",
            "peak_ram", "tensor_file", "dataset_sha256", "linear_rows", "mul_rows",
            "passthrough_calls",
        )},
    }


def main(argv: list[str] | None = None) -> int:
    """Run one arm with boundary capture and write its records and row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lora-targets", default="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--backend", choices=("default", "math"), default="default")
    parser.add_argument("--expert-gemm", choices=("native", "segmented", "lt-fused"),
                        default="native")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-revision",
                        help="Defaults to config.json's _commit_hash or unknown")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must be one filename component starting with a letter or digit")
    args.lora_targets = tuple(part.strip() for part in args.lora_targets.split(",") if part.strip())
    if _power_state() != 1:
        Console(stderr=True, markup=False, highlight=False).print(
            "refusing to run: AC power is not online")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    capture_file = args.out / "boundaries" / f"{args.tag}.json"
    if capture_file.exists():
        parser.error(f"refusing to overwrite {capture_file}")

    from benchmarks.harness import l2l_mistral_stand

    l2l_mistral_stand.set_deterministic_env()
    recorder = _Recorder(group=args.k)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            harness = _run(args, recorder)
        finally:
            messages = dict.fromkeys(
                " ".join(str(item.message).splitlines()) for item in captured
            )
            (args.out / f"{args.tag}-warnings.txt").write_text(
                "".join(message + "\n" for message in messages), encoding="utf-8"
            )
            recorder.write(args.out / "boundaries", args.tag)
    row = _write_row(args, harness, _power_state())
    from benchmarks.harness.mbr_measure import append_jsonl

    append_jsonl(args.out / "rows.jsonl", row)
    Console(soft_wrap=True, markup=False, highlight=False).print(json.dumps(row))
    return 0 if harness.get("status") == "ok" else 1


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo), str(Path(__file__).resolve().parent))
    raise SystemExit(main())
