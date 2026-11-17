"""Complete-model frozen-base training fixture; not a public trainer API.

Reuse the audited whole-model assembly and unchanged B1 storage interfaces.
The only trainable parameters are an explicitly declared post-embedding adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from d3_full_model import open_resident_model, open_streamed_model

MODES = ("native", "sequential", "grouped-resident", "grouped",
         "triton-resident", "triton")


class SingleGroupView:
    """Adapt the original one-slot B1 pool without increasing its allocation."""

    def __init__(self, pool: Any):
        self.pool = pool
        self.device, self.dtype = pool.device, pool.dtype
        self._shapes = pool._shapes
        self.group_width = 1

    def validate_reader(self, reader: Any) -> None:
        self.pool.validate_reader(reader)

    @contextmanager
    def lease(self, reader: Any, experts: list[int]):
        if len(experts) != 1:
            raise ValueError("the sequential control leases one expert at a time")
        with self.pool.lease(reader, experts[0]) as pair:
            yield (pair,)


def input_adapter_class() -> Any:
    import torch

    class InputAdapter(torch.nn.Module):
        """Residual B(A(h)) with rank four and scale one, after frozen embedding."""

        def __init__(self, base, *, seed: int, rank: int):
            super().__init__()
            self.base = base
            generator = torch.Generator().manual_seed(seed)
            width = base.weight.shape[1]
            self.a = torch.nn.Parameter((torch.randn(rank, width, generator=generator)
                                         * 0.01).to(base.weight))
            self.b = torch.nn.Parameter((torch.randn(width, rank, generator=generator)
                                         * 0.01).to(base.weight))

        def forward(self, ids):
            hidden = self.base(ids)
            return hidden + torch.nn.functional.linear(
                torch.nn.functional.linear(hidden, self.a), self.b,
            )

    return InputAdapter


class TrainingRun:
    def __init__(self, owner: Any, *, mode: str, seed: int, rank: int, lr: float,
                 retain_derivatives: bool):
        import torch

        self.owner, self.mode = owner, mode
        self.closed = self.active = False
        self.hooks: list = []
        self.captured: dict = {}
        self.routes: dict = {}
        self.retain_derivatives = retain_derivatives
        try:
            self.adapter = input_adapter_class()(
                owner.model.model.embed_tokens, seed=seed, rank=rank,
            )
            owner.model.model.embed_tokens = self.adapter
            self.parameters = {"a": self.adapter.a, "b": self.adapter.b}
            self.initial = {name: value.detach().cpu().clone()
                            for name, value in self.parameters.items()}
            self.optimizer = torch.optim.SGD(self.parameters.values(), lr=lr,
                                              momentum=0, foreach=False)
            for index, layer in enumerate(owner.model.model.layers):
                self.hooks.append(layer.mlp.experts.register_forward_pre_hook(
                    self._capture(index)
                ))
        except BaseException:
            self.close()
            raise

    def _capture(self, index):
        def hook(_module, args):
            hidden, ids, weights = args
            self.routes[index] = ids.detach()
            if self.retain_derivatives:
                for name, tensor in (("input", hidden), ("router", weights)):
                    if not tensor.requires_grad:
                        raise ValueError("every expert layer must remain connected to the adapter")
                    tensor.retain_grad()
                    self.captured[f"{index}/{name}"] = tensor
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def reset(self) -> None:
        import torch

        if self.closed or self.active:
            raise RuntimeError("cannot reset a closed or active training fixture")
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in self.parameters.items():
                value.copy_(self.initial[name])
        self.captured.clear()
        self.routes.clear()

    def step(self, ids: Any) -> tuple[Any, Any]:
        """One full forward, next-token loss, backward and optimizer update.

        No CPU tensor capture or evidence serialization belongs to the timer.
        Returns detached logits/loss; parameter gradients survive until next step.
        """
        import torch

        if self.closed or self.active:
            raise RuntimeError("closed or reentrant training fixture")
        if not torch.is_grad_enabled() or ids.ndim != 2 or ids.shape[1] < 2:
            raise ValueError("training requires autograd and at least two input tokens")
        self.active = True
        try:
            self.captured.clear()
            self.routes.clear()
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.owner.model(input_ids=ids, use_cache=False, logits_to_keep=0).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
            )
            if not torch.isfinite(loss):
                raise ValueError("training loss is nonfinite")
            loss.backward()
            for parameter in self.parameters.values():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError("adapter derivative is absent or nonfinite")
            self.optimizer.step()
            return logits.detach(), loss.detach()
        finally:
            self.active = False

    def frame(self, outputs: tuple[Any, Any]) -> dict:
        """Independent CPU tensor evidence, outside the timed optimizer step."""
        import torch

        if self.closed or self.active:
            raise RuntimeError("capture requires a live completed step")
        tensors = {"logits": outputs[0], "loss": outputs[1]}
        for name, parameter in self.parameters.items():
            tensors[f"adapter/{name}"] = parameter
            tensors[f"gradient/{name}"] = parameter.grad
        for name, tensor in self.captured.items():
            tensors[f"derivative/{name}"] = tensor.grad
        if any(value is None or not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("training evidence contains absent/nonfinite tensors")
        return {name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}

    def check_dispatch(self) -> dict:
        if len(self.routes) != len(self.owner.model.model.layers):
            raise ValueError("not every decoder layer executed its routed experts")
        if self.mode != "native":
            for layer in self.owner.model.model.layers:
                expert = layer.mlp.experts
                if not expert.forward_trace or not expert.backward_trace:
                    raise ValueError("every decoder layer needs forward and backward dispatch")
            for index, layer in enumerate(self.owner.model.model.layers):
                active = sorted(self.routes[index].unique().cpu().tolist())
                expert = layer.mlp.experts
                if ([row["expert"] for row in expert.forward_trace] != active
                        or sorted(row["expert"] for row in expert.backward_trace) != active
                        or any(row["status"] != "ok" for row in expert.backward_trace)):
                    raise ValueError("expert dispatch/derivative coverage differs from routing")
        return {"layers": len(self.routes),
                "routes": {str(key): value.cpu().tolist() for key, value in self.routes.items()},
                "gemm": self.owner.gemm.stats() if self.owner.gemm is not None else None,
                "pool": self.owner.pool.stats() if self.owner.pool is not None else None}

    def close(self) -> None:
        if self.active:
            raise RuntimeError("cannot close while an optimizer step is active")
        if self.closed:
            return
        self.closed = True
        for hook in self.hooks:
            hook.remove()
        self.captured.clear()
        self.owner.close()


def open_training_model(folder: Path, *, mode: str, device: str, library: Path | None,
                        retain_derivatives: bool, seed: int = 25, rank: int = 4,
                        lr: float = 0.5, native_rows_max: int = 128,
                        native_rows_from: int | None = None,
                        weight_budget_bytes: int = 64 * 2**20,
                        gemm_call_budget_bytes: int = 64 * 2**20,
                        activation_budget_bytes: int = 256 * 2**20) -> TrainingRun:
    """Reuse existing assembly; never relax its public forward-only wrapper."""
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class

    if mode not in MODES:
        raise ValueError("unknown training comparison mode")
    resident = mode in ("native", "grouped-resident", "triton-resident")
    if resident:
        owner = open_resident_model(folder, device=device, library=library,
                                    backend="native" if mode == "native" else "grouped-compute",
                                    gemm_call_budget_bytes=gemm_call_budget_bytes,
                                    native_olmoe_rows_max=native_rows_max,
                                    native_olmoe_rows_from=native_rows_from)
    else:
        owner = open_streamed_model(folder, device=device, library=library,
                                    backend="sequential" if mode == "sequential"
                                    else "grouped-compute", weight_budget_bytes=weight_budget_bytes,
                                    gemm_call_budget_bytes=gemm_call_budget_bytes,
                                    activation_budget_bytes=activation_budget_bytes,
                                    native_olmoe_rows_max=native_rows_max,
                                    native_olmoe_rows_from=native_rows_from)
    try:
        if mode != "native":
            for index, layer in enumerate(owner.model.model.layers):
                prior = layer.mlp.experts
                pool = SingleGroupView(owner.pool) if mode == "sequential" else prior.pool
                layer.mlp.experts = grouped_autograd_experts_class()(
                    owner.readers[index], pool, prior.act_fn, gemm=owner.gemm,
                    activation_budget_bytes=activation_budget_bytes,
                    activation_backend="triton" if mode.startswith("triton") else "native",
                )
        return TrainingRun(owner, mode=mode, seed=seed, rank=rank, lr=lr,
                           retain_derivatives=retain_derivatives)
    except BaseException:
        owner.close()
        raise
