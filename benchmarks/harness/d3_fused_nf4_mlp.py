"""Synthetic disk-backed MLP forward/derivative screen, not public trainer acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def run(out: Path) -> dict:
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_group_fused import fused_group_linear
    from soup_cli.utils.expert_group_fused_runtime import PackedGemm
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    torch.manual_seed(161803)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[2]
    sources = [Path(__file__), *[root / "src/soup_cli/utils" / name for name in (
        "expert_group.py", "expert_group_autograd.py", "expert_group_fused.py",
        "expert_group_fused_runtime.py", "expert_group_fused_wmma.py")]]
    report = {"kind": "synthetic packed expert MLP; no pretrained/trainer claim",
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}, "cells": [], "status": "incomplete"}

    def exact(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def native(hidden, indices, routes, dense):
        mask = functional.one_hot(indices, 4).permute(2, 1, 0)
        output = torch.zeros_like(hidden)
        for expert in range(4):
            position, token = torch.where(mask[expert])
            if not token.numel():
                continue
            first = functional.linear(hidden[token], dense[expert][0])
            gate, up = first.chunk(2, dim=-1)
            current = functional.linear(functional.silu(gate) * up, dense[expert][1])
            current = current * routes[token, position, None]
            output.index_add_(0, token, current)
        return output

    try:
        for nested in (False, True):
            folder = out / f"nested-{int(nested)}"
            folder.mkdir()
            path = folder / "experts.safetensors"
            info = write_expert_shard(
                str(path), ((torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * .02,
                             torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16) * .02)
                            for _ in range(4)),
                quantise=True, double_quant=nested, device="cuda")
            reader = ExpertShardReader(str(path), count=4, shapes=info.shapes,
                                       quant_specs=info.quant_specs, dtype="bfloat16", direct=False)
            codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
            reference_plan = plan_expert_buffer(reader, dtype="bfloat16")
            baseline = ExpertBufferPool(reader, device="cuda", dtype="bfloat16", codes=codes,
                                        weight_budget_bytes=reference_plan.weight_bytes)
            dense = []
            try:
                for expert in range(4):
                    with baseline.lease(reader, expert) as pair:
                        dense.append(tuple(t.clone() for t in pair))
            finally:
                baseline.close()
            for width in (1, 4):
                plan = plan_grouped_expert_buffer(reader, dtype="bfloat16", group_width=width,
                                                  packed_only=True)
                pool = GroupedExpertPool(reader, device="cuda", dtype="bfloat16", codes=codes,
                                         group_width=width, weight_budget_bytes=64 * 2**20,
                                         packed_only=True)
                if pool.plan != plan:
                    raise AssertionError("packed pool and preflight plans differ")
                gemm = PackedGemm(reader, device=pool.device, call_budget_bytes=64 * 2**20)
                candidate = grouped_autograd_experts_class()(
                    reader, pool, torch.nn.SiLU(), gemm=gemm, activation_budget_bytes=256 * 2**20)
                try:
                    for pattern in ("balanced", "mixed"):
                        index = torch.arange(512, device="cuda")
                        if pattern == "balanced":
                            indices = torch.stack((index % 4, (index + 1) % 4), dim=1)
                        else:
                            indices = torch.stack((torch.zeros_like(index),
                                                   torch.where(index < 400, 1, 2)), dim=1)
                        hidden = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16) * .02
                        routes = torch.rand(512, 2, device="cuda", dtype=torch.bfloat16)
                        x, y = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
                        a, b = routes.clone().requires_grad_(), routes.clone().requires_grad_()
                        expected, actual = native(x, indices, a, dense), candidate(y, indices, b)
                        probe = torch.randn_like(expected) * .01
                        expected.backward(probe)
                        actual.backward(probe)
                        row = {"nested": nested, "width": width, "pattern": pattern,
                               "forward_exact": exact(expected, actual),
                               "input_gradient_exact": exact(x.grad, y.grad),
                               "router_gradient_exact": exact(a.grad, b.grad),
                               "nonzero_gradient": bool(y.grad.abs().sum().item() > 0),
                               "gemm": gemm.stats(), "pool": pool.stats()}
                        row["pass"] = all(row[k] for k in (
                            "forward_exact", "input_gradient_exact", "router_gradient_exact",
                            "nonzero_gradient"))
                        report["cells"].append(row)
                    with pool.lease(reader, [0]) as pairs:
                        expired = pairs[0][0]
                    try:
                        fused_group_linear([hidden.detach()], [expired], backend="hybrid_wmma",
                                           call_budget_bytes=64 * 2**20)
                    except RuntimeError as exc:
                        if "lease expired" not in str(exc):
                            raise
                    else:
                        raise AssertionError("expired packed lease admitted")
                finally:
                    pool.close()
                    gemm.close()
            reader.close()
        report["status"] = ("development_mlp_pass" if len(report["cells"]) == 8
                            and all(c["pass"] for c in report["cells"]) else "fail")
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = run(parser.parse_args().out)
    Console().print(f"Status: {result['status']}; cells: {len(result['cells'])}")
    raise SystemExit(0 if result["status"] == "development_mlp_pass" else 1)
