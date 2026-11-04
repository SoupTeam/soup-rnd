"""E4 part 2, calibration rule section 4: which forward the CUDA ``gemm_4bit`` actually ran.

bitsandbytes 0.50.2 registers ``bitsandbytes::gemm_4bit`` for CUDA
(``backends/cuda/ops.py``). Inside it a heuristic picks a fused kernel
(``_gemm_4bit_kernel_impl``) or the dequantize + ``F.linear`` fallback
(``_dequant_linear_fallback``). Both names are module globals looked up at call time, so
wrapping them records the branch that really ran for every call. ``MatMul4Bit.apply`` is
wrapped too: it is entered exactly when ``matmul_4bit`` takes the grad path, which gives an
independent observation of ``needs_grad``.

The probe refuses to start if any name it needs is missing, so a bitsandbytes update cannot
silently turn it into a recorder of nothing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

FUSED = "fused"
FALLBACK = "fallback"
CUDA_OPS_MODULE = "bitsandbytes.backends.cuda.ops"
AUTOGRAD_MODULE = "bitsandbytes.autograd._functions"
KERNEL_ATTRS = {FUSED: "_gemm_4bit_kernel_impl", FALLBACK: "_dequant_linear_fallback"}


class ProbeError(RuntimeError):
    """The probe could not attach, or a call did not match the one-call-one-kernel contract."""


@dataclass(frozen=True)
class Call:
    kernel: str  # FUSED or FALLBACK
    m: int  # rows of the activation (tokens)
    n: int
    k: int


class GemmProbe:
    """Records every ``gemm_4bit`` call between ``reset()`` and ``take()``."""

    def __init__(self, ops_module: Any = None, autograd_module: Any = None,
                 grad_class_name: str = "MatMul4Bit") -> None:
        self._ops = ops_module
        self._autograd = autograd_module
        self._grad_class_name = grad_class_name
        self._saved: List[tuple] = []
        self.calls: List[Call] = []
        self.grad_path_entries = 0

    def __enter__(self) -> "GemmProbe":
        if self._ops is None:
            self._ops = importlib.import_module(CUDA_OPS_MODULE)
        if self._autograd is None:
            self._autograd = importlib.import_module(AUTOGRAD_MODULE)
        for kernel, attr in KERNEL_ATTRS.items():
            if not callable(getattr(self._ops, attr, None)):
                raise ProbeError(
                    f"{self._ops!r} has no callable {attr}; the dispatch the rule instruments "
                    "(calibration section 4) is not what this bitsandbytes ships"
                )
            self._wrap_kernel(kernel, attr)
        grad_cls = getattr(self._autograd, self._grad_class_name, None)
        if grad_cls is None or not callable(getattr(grad_cls, "apply", None)):
            raise ProbeError(f"{self._autograd!r} has no {self._grad_class_name}.apply")
        original_apply = grad_cls.apply

        def apply(*args, **kwargs):
            self.grad_path_entries += 1
            return original_apply(*args, **kwargs)

        self._saved.append((grad_cls, "apply", None if "apply" not in vars(grad_cls)
                            else vars(grad_cls)["apply"]))
        grad_cls.apply = apply
        return self

    def _wrap_kernel(self, kernel: str, attr: str) -> None:
        original = getattr(self._ops, attr)

        def wrapper(A, B, shapeB, *args, **kwargs):  # noqa: N803 - bnb's argument names
            k = A.shape[-1]
            self.calls.append(Call(kernel, A.numel() // k, int(shapeB[0]), int(k)))
            return original(A, B, shapeB, *args, **kwargs)

        self._saved.append((self._ops, attr, original))
        setattr(self._ops, attr, wrapper)

    def __exit__(self, *exc) -> bool:
        for owner, attr, original in reversed(self._saved):
            if original is None:
                delattr(owner, attr)  # the class only inherited ``apply``; inherit it again
            else:
                setattr(owner, attr, original)
        self._saved.clear()
        return False

    def reset(self) -> None:
        self.calls = []
        self.grad_path_entries = 0

    def take(self, expect_calls: Optional[int] = 1) -> Dict[str, Any]:
        """Calls since ``reset()``. With ``expect_calls`` set, any other count is an error."""
        calls, entries = self.calls, self.grad_path_entries
        self.reset()
        if expect_calls is not None and len(calls) != expect_calls:
            raise ProbeError(
                f"expected {expect_calls} gemm_4bit call(s), the probe saw {len(calls)}"
            )
        return {
            "kernels": [c.kernel for c in calls],
            "shapes": [(c.m, c.n, c.k) for c in calls],
            "needs_grad_observed": entries > 0,
        }


def expected_needs_grad(x_requires_grad: bool, bias_requires_grad: bool,
                        grad_enabled: bool) -> bool:
    """``matmul_4bit``'s own condition (``autograd/_functions.py:453``).

    The rule's wording ("input or bias requires grad") leaves out the grad-mode switch; it is
    part of the condition, so a no-grad context makes ``needs_grad`` False even for an input
    that requires grad.
    """
    return grad_enabled and (x_requires_grad or bias_requires_grad)
