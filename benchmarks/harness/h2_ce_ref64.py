"""fp64-reference cross entropy (CE64), recording hooks and the one-ulp perturbation for attempt018.

CE64 replaces torch.nn.functional.cross_entropy (the same attribute the CE-only arm rebinds) in the
`ref64` arm. Its backward returns the fp32 rounding of the exact fp64 gradient, so the existing
`.float()` backward then rounds to bf16 exactly as in the other arms. Protocol:
benchmarks/h2/CE_FP64_REFERENCE_ADDENDUM.txt.
The experiment compares arms with an fp64 CE reference propagated through the bf16 backward,
NOT with the true full-model gradient. Nothing here changes a threshold, the policy, the workset
or the fixture.
"""

from __future__ import annotations

import hashlib
from typing import Any

REAL_VOCAB = 128256
IGNORE_INDEX = -100
CHUNK_ROWS = 32
GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
SEEDS_PER_LEVEL = 3


class Ce64RefusedError(ValueError):
    """CE64 received an argument outside its fail-closed specification."""


class State:
    """Call counter, integrity record and the optional one-ulp perturbation (Rp)."""

    def __init__(self, vocab: int, real: bool, perturbation: dict | None = None) -> None:
        self.vocab = vocab
        self.real = real
        self.perturbation = perturbation
        self.calls = 0
        self.integrity: list[dict] = []
        self.refusals: list[str] = []
        self.flips: list[dict] = []


_ACTIVE: dict[str, State | None] = {"state": None}


def sha256_tensor(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def perturbation_seed(level_index: int, seed_index: int) -> int:
    """Written formula: seed = 1000 * level_index + seed_index (levels 0..5, seeds 0..2)."""
    if not (0 <= level_index < len(GRID) and 0 <= seed_index < SEEDS_PER_LEVEL):
        raise ValueError("Perturbation index outside the grid")
    return 1000 * level_index + seed_index


def flip_one_ulp(grad_bf16: Any, rows: Any, level: float, seed: int) -> tuple[Any, dict]:
    """Flip round(level * active) (at least one) random bf16 elements of the rows by one ulp.

    Active = all elements of the given rows (the phi denominator). Positions are drawn among the
    active elements with a normal, non-maximal bf16 magnitude (so a one-ulp step stays finite,
    non-zero and keeps its sign). Direction is random per element.
    The step is an integer step on the bf16 bit pattern's magnitude bits.
    """
    import torch

    if grad_bf16.dtype != torch.bfloat16 or grad_bf16.ndim != 2:
        raise ValueError("Perturbation needs a 2-D bf16 gradient")
    out = grad_bf16.clone()
    selected = out[rows]
    total = selected.numel()
    bits = selected.reshape(-1).view(torch.int16).clone()
    magnitude = bits & 0x7FFF
    eligible = ((magnitude >= 0x0080) & (magnitude <= 0x7F7E)).nonzero()[:, 0]
    count = max(1, round(level * total))
    if eligible.numel() < count:
        raise ValueError("Fewer eligible elements than requested flips")
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(eligible.numel(), generator=generator)[:count].to(bits.device)
    direction = (torch.randint(0, 2, (count,), generator=generator) * 2 - 1).to(bits.device)
    index = eligible[chosen]
    sign_negative = bits[index] < 0
    # value up = +1 on the magnitude of a positive number, -1 on a negative one; down: reverse.
    delta = torch.where(sign_negative, -direction, direction).to(torch.int16)
    bits[index] = (bits[index] & -32768) | (magnitude[index] + delta)
    out[rows] = bits.view(torch.bfloat16).reshape(selected.shape)
    return out, {
        "positions_sha256": sha256_tensor(index.cpu()) + ":" + sha256_tensor(direction.cpu()),
        "eligible_total": total,
        "eligible_usable": int(eligible.numel()),
        "flipped": count,
        "level": level,
        "seed": seed,
    }


def _check_arguments(state: State, input: Any, target: Any, kwargs: dict) -> None:
    import torch

    problems = []
    allowed_defaults = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "label_smoothing": 0.0,
    }
    for name, value in kwargs.items():
        if name == "ignore_index":
            if value != IGNORE_INDEX:
                problems.append("ignore_index must be -100")
        elif name == "reduction":
            if value != "mean":
                problems.append("reduction must be 'mean'")
        elif name in allowed_defaults:
            if value != allowed_defaults[name]:
                problems.append(f"{name} must be at its default")
        else:
            problems.append(f"unexpected argument {name}")
    if not isinstance(input, torch.Tensor) or input.dtype != torch.float32 or input.ndim != 2:
        problems.append("input must be a 2-D fp32 tensor")
    elif state.real and (input.shape[1] != REAL_VOCAB or input.device.type != "cuda"):
        problems.append("real configuration requires V = 128256 on CUDA")
    elif input.shape[1] != state.vocab:
        problems.append("input vocabulary differs from the configured one")
    if not isinstance(target, torch.Tensor) or target.dtype != torch.int64 or target.ndim != 1:
        problems.append("target must be a 1-D int64 tensor")
    elif (
        isinstance(input, torch.Tensor)
        and input.ndim == 2
        and (target.shape[0] != input.shape[0] or target.device != input.device)
    ):
        problems.append("target must match the input rows and device")
    if problems:
        state.refusals.append("; ".join(problems))
        raise Ce64RefusedError("CE64 refused: " + "; ".join(problems))


_FUNCTION: dict[str, Any] = {
    "cls": None
}  # built lazily: importing this module must not import torch


def _build_function() -> Any:
    import torch

    class Ce64(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input: Any, target: Any, state: State) -> Any:
            with torch.autocast(device_type=input.device.type, enabled=False):
                before = {
                    "input": sha256_tensor(input),
                    "input_version": input._version,
                    "target": sha256_tensor(target),
                    "target_version": target._version,
                }
                mask = target != IGNORE_INDEX
                count = int(mask.sum())
                if count == 0:
                    raise Ce64RefusedError("CE64 refused: no non-ignored target")
                lse = torch.empty(input.shape[0], dtype=torch.float64, device=input.device)
                picked = torch.zeros(input.shape[0], dtype=torch.float64, device=input.device)
                for start in range(0, input.shape[0], CHUNK_ROWS):
                    block = input[start : start + CHUNK_ROWS].double()
                    lse[start : start + CHUNK_ROWS] = torch.logsumexp(block, dim=-1)
                    index = target[start : start + CHUNK_ROWS].clamp(min=0)
                    picked[start : start + CHUNK_ROWS] = block.gather(1, index[:, None])[:, 0]
                loss = (((lse - picked) * mask).sum() / count).to(torch.float32)
            ctx.save_for_backward(input, target, lse)
            ctx.count, ctx.state, ctx.before = count, state, before
            return loss

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            input, target, lse = ctx.saved_tensors
            state, count = ctx.state, ctx.count
            with torch.autocast(device_type=input.device.type, enabled=False):
                scale = grad_output.double()
                grad = torch.empty_like(input)
                for start in range(0, input.shape[0], CHUNK_ROWS):
                    block = input[start : start + CHUNK_ROWS].double()
                    probs = torch.exp(block - lse[start : start + CHUNK_ROWS, None])
                    chunk_target = target[start : start + CHUNK_ROWS]
                    index = chunk_target.clamp(min=0)
                    probs.scatter_add_(
                        1,
                        index[:, None],
                        torch.where(chunk_target[:, None] != IGNORE_INDEX, -1.0, 0.0).to(
                            torch.float64
                        ),
                    )
                    probs = probs * (chunk_target != IGNORE_INDEX)[:, None] / count * scale
                    grad[start : start + CHUNK_ROWS] = probs.to(torch.float32)
                if state.perturbation is not None:
                    rows = (target != IGNORE_INDEX).nonzero()[:, 0]
                    bf16, info = flip_one_ulp(
                        grad.to(torch.bfloat16),
                        rows,
                        state.perturbation["level"],
                        state.perturbation["seed"],
                    )
                    state.flips.append(info)
                    grad = bf16.to(torch.float32)
                after = {
                    "input": sha256_tensor(input),
                    "input_version": input._version,
                    "target": sha256_tensor(target),
                    "target_version": target._version,
                }
            ok = after == ctx.before
            state.integrity.append({"ok": ok, "rows": int(input.shape[0])})
            if not ok:
                raise Ce64RefusedError(
                    "CE64 refused: input or target changed during the evaluation"
                )
            return grad, None, None

    return Ce64


def make_cross_entropy(state: State) -> Any:
    """The callable that replaces torch.nn.functional.cross_entropy."""
    if _FUNCTION["cls"] is None:
        _FUNCTION["cls"] = _build_function()
    function = _FUNCTION["cls"]

    def cross_entropy(
        input: Any,
        target: Any,
        weight: Any = None,
        size_average: Any = None,
        ignore_index: int = IGNORE_INDEX,
        reduce: Any = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> Any:
        state.calls += 1
        _check_arguments(
            state,
            input,
            target,
            dict(
                weight=weight,
                size_average=size_average,
                ignore_index=ignore_index,
                reduce=reduce,
                reduction=reduction,
                label_smoothing=label_smoothing,
            ),
        )
        return function.apply(input, target, state)

    cross_entropy.__name__ = "ce64_cross_entropy"
    cross_entropy.__module__ = "h2_ce_ref64"
    cross_entropy.ce64_state = state
    return cross_entropy


def install(vocab: int = REAL_VOCAB, real: bool = True, perturbation: dict | None = None) -> Any:
    """Rebind torch.nn.functional.cross_entropy once per process; returns the installed wrapper."""
    import torch

    if _ACTIVE["state"] is not None:
        raise ValueError("CE64 is already installed in this process")
    state = State(vocab, real, perturbation)
    wrapper = make_cross_entropy(state)
    _ACTIVE["state"] = state
    torch.nn.functional.cross_entropy = wrapper
    return wrapper


def set_perturbation(wrapper: Any, level_index: int, seed_index: int) -> dict:
    config = {
        "level_index": level_index,
        "seed_index": seed_index,
        "level": GRID[level_index],
        "seed": perturbation_seed(level_index, seed_index),
    }
    wrapper.ce64_state.perturbation = config
    return config


class LmHeadHooks:
    """Recording only: SHA-256 of the lm_head bf16 output and a clone of its gradient per key."""

    def __init__(self, model: Any, keep_gradients: bool = True) -> None:
        import torch

        heads = [
            m for name, m in model.named_modules() if name == "lm_head" or name.endswith(".lm_head")
        ]
        if len(heads) != 1:
            raise ValueError("Exactly one lm_head module required")
        self.key: str | None = None
        self.sha: dict[str, str] = {}
        self.dlogits: dict[str, Any] = {}
        self.calls: dict[str, int] = {}
        self.keep_gradients = keep_gradients
        self.handle = heads[0].register_forward_hook(self._forward)
        self._torch = torch

    def begin(self, key: str) -> None:
        if key in self.calls:
            raise ValueError("Hook key reused: " + key)
        self.key = key
        self.calls[key] = 0

    def _forward(self, module: Any, inputs: tuple, output: Any) -> None:
        key = self.key
        if key is None:
            return None
        self.calls[key] += 1
        if self.calls[key] != 1:
            raise ValueError("More than one lm_head forward for " + key)
        self.sha[key] = sha256_tensor(output)
        if output.requires_grad and self.keep_gradients:

            def store(grad: Any) -> None:
                self.dlogits[key] = grad.detach().clone().cpu()
                return None  # never alters the gradient

            output.register_hook(store)
        return None

    def end(self) -> None:
        self.key = None

    def remove(self) -> None:
        self.handle.remove()


def peak_memory_start() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes() -> int | None:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
