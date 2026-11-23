"""Compare two causal SDPA backward passes on Moonlight-shaped inputs."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from nondet_analysis import SDPA_BACKENDS, tensor_pair_diff
from rich.console import Console


def _run_pair(inputs: tuple[Any, ...], upstream: Any) -> dict[str, Any]:
    import torch.nn.functional as functional

    def backward() -> tuple[Any, ...]:
        query, key, value = (tensor.clone().requires_grad_() for tensor in inputs)
        output = functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        output.backward(upstream)
        return output.detach(), query.grad, key.grad, value.grad

    left, right = backward(), backward()
    return {
        name: tensor_pair_diff(first, second)
        for name, first, second in zip(("out", "dq", "dk", "dv"), left, right)
    }


def _write_report(out: Path, results: dict[int, dict[str, Any]]) -> None:
    (out / "probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| seq | backend | tensor | equal | n_diff | max_abs_diff | rel_sum_diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seq, backends in results.items():
        for backend, tensors in backends.items():
            if "error" in tensors:
                error = " ".join(tensors["error"].splitlines()).replace("|", r"\|")
                lines.append(f"| {seq} | {backend} | error: {error} | | | | |")
                continue
            for name, diff in tensors.items():
                lines.append(
                    f"| {seq} | {backend} | {name} | {diff['equal']} | {diff['n_diff']} "
                    f"| {diff['max_abs_diff']:.6g} | {diff['rel_sum_diff']:.6g} |"
                )
    (out / "probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _warning_lines(prefix: str, records: list[warnings.WarningMessage]) -> list[str]:
    return [prefix + " ".join(str(item.message).splitlines()) for item in records]


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write JSON, Markdown, and the captured warnings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="64,512", help="Comma-separated sequence lengths")
    parser.add_argument("--backends", default=",".join(SDPA_BACKENDS), help="Comma-separated names")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--qk-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warnings-file", type=Path, help="Also write warnings to this path")
    args = parser.parse_args(argv)
    try:
        seqs = [int(seq) for seq in args.seq.split(",")]
    except ValueError:
        parser.error("--seq must contain comma-separated positive integers")
    if min(*seqs, args.heads, args.qk_dim, args.v_dim) <= 0:
        parser.error("--seq, --heads, --qk-dim, and --v-dim must be positive")
    backends = [backend.strip() for backend in args.backends.split(",")]
    if unknown := set(backends) - SDPA_BACKENDS.keys():
        parser.error(f"unknown SDPA backends: {', '.join(sorted(unknown))}")

    from benchmarks.harness import l2l_mistral_stand

    # cuBLAS reads this at context creation, so it must be set before the first CUDA call.
    l2l_mistral_stand.set_deterministic_env()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    args.out.mkdir(parents=True, exist_ok=True)
    warnings_file = args.out / "warnings.txt"
    results: dict[int, dict[str, Any]] = {}
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            Console().print(
                f"torch={torch.__version__} cuda={torch.version.cuda} "
                f"gpu={torch.cuda.get_device_name()}\n"
                f"flash_sdp_enabled={torch.backends.cuda.flash_sdp_enabled()} "
                f"mem_efficient_sdp_enabled={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                f"math_sdp_enabled={torch.backends.cuda.math_sdp_enabled()} "
                f"cudnn_sdp_enabled={torch.backends.cuda.cudnn_sdp_enabled()}"
            )
            rng = torch.Generator(device="cuda").manual_seed(0)
            for seq in seqs:
                query, key, value, upstream = (
                    torch.randn(
                        (1, args.heads, seq, dim), generator=rng, device="cuda",
                        dtype=getattr(torch, args.dtype),
                    )
                    for dim in (args.qk_dim, args.qk_dim, args.v_dim, args.v_dim)
                )
                results[seq] = {}
                for backend in backends:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        try:
                            with sdpa_kernel(getattr(SDPBackend, SDPA_BACKENDS[backend])):
                                results[seq][backend] = _run_pair((query, key, value), upstream)
                        except (AttributeError, RuntimeError, NotImplementedError) as error:
                            results[seq][backend] = {"error": f"{type(error).__name__}: {error}"}
                    messages.extend(_warning_lines(f"seq={seq} backend={backend}: ", captured))
        finally:
            messages.extend(_warning_lines("global: ", outer))
            text = "".join(message + "\n" for message in messages)
            warnings_file.write_text(text, encoding="utf-8")
            if args.warnings_file is not None and args.warnings_file != warnings_file:
                args.warnings_file.parent.mkdir(parents=True, exist_ok=True)
                args.warnings_file.write_text(text, encoding="utf-8")
    _write_report(args.out, results)
    # A sequence nobody could run leaves the probe without data; one missing backend is expected.
    return 0 if all(
        any("error" not in tensors for tensors in by_backend.values())
        for by_backend in results.values()
    ) else 1


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
