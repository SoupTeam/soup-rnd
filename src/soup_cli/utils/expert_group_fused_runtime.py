"""Bounded packed-NF4 training adapter; CUDA/trainer qualification is separate."""

from __future__ import annotations


class PackedGemm:
    """Explicit hybrid policy, with counters separating fused and native work."""

    def __init__(self, reader, *, device, call_budget_bytes):
        import torch

        from soup_cli.utils.expert_group_gemm import require_reference_policy

        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("explicit CUDA device required for packed expert GEMM")
        if torch.cuda.get_device_capability(self.device) != (8, 6):
            raise ValueError("packed WMMA training candidate is currently limited to sm_86")
        if type(call_budget_bytes) is not int or call_budget_bytes <= 0:
            raise ValueError("positive call budget required")
        if set(reader.quant_specs) != {"gate_up", "down"}:
            raise ValueError("packed expert GEMM requires NF4 for both projections")
        require_reference_policy()
        self.call_budget_bytes = call_budget_bytes
        self.extra_call_bytes = max(
            shape[0] * shape[1] * 2 + 8 * (
                (shape[0] * shape[1] + reader.quant_specs[part].blocksize - 1)
                // reader.quant_specs[part].blocksize
            ) for part, shape in reader.shapes.items()
        )
        self.closed = False
        self.calls = self.fused_matrices = self.native_matrices = 0
        self.dx_calls = 0

    def linear(self, inputs, weights, *, transpose=False):
        from soup_cli.utils.expert_group_fused import fused_group_linear, validate_fused_group
        from soup_cli.utils.expert_group_gemm import require_reference_policy

        if self.closed:
            raise RuntimeError("packed GEMM is closed")
        require_reference_policy()
        kwargs = dict(transpose=transpose, backend="hybrid_wmma",
                      call_budget_bytes=self.call_budget_bytes)
        plan = validate_fused_group(inputs, weights, **kwargs)
        result = fused_group_linear(inputs, weights, **kwargs)
        self.calls += 1
        self.dx_calls += int(transpose)
        self.fused_matrices += plan["fused_experts"]
        self.native_matrices += plan["fallback_experts"]
        return result

    def input_gradient(self, gradient, weight):
        return self.linear([gradient.contiguous().detach()], [weight], transpose=True)[0]

    def stats(self):
        return {"backend": "packed-hybrid-wmma", "calls": self.calls,
                "dx_calls": self.dx_calls, "fused_matrices": self.fused_matrices,
                "native_matrices": self.native_matrices,
                "native_rows_max": 128, "native_rows_from": None,
                "call_budget_bytes": self.call_budget_bytes, "closed": self.closed}

    def close(self):
        self.closed = True


def packed_expert_backward(owner, hidden, grad_output, weights, autocast):
    """Recompute with packed weights; use native autograd only for SiLU/product."""
    import torch

    gate_up, down = weights
    with torch.autocast(hidden.device.type, **autocast):
        first = owner.gemm.linear([hidden.detach().contiguous()], [gate_up])[0]
        # Only the pointwise graph is retained. No reusable weight view is saved.
        with torch.enable_grad():
            first = first.detach().requires_grad_(True)
            gate, up = first.chunk(2, dim=-1)
            activated = owner.act_fn(gate) * up
        grad_mid = owner.gemm.input_gradient(grad_output, down)
        (grad_first,) = torch.autograd.grad(activated, first, grad_mid)
        return owner.gemm.input_gradient(grad_first, gate_up)
