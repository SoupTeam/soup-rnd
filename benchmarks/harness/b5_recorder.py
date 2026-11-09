"""B5 ticket 02: writes the run journal that ``b5_judge.py`` reads.

All three B5 harnesses (small real MoE, Qwen3.5-35B-A3B, synthetic 671B) call the same
recorder, so the judge sees one format. The event names are those of the B6 journal; the
fields the rule adds are listed in ``benchmarks/b5-adapter-judge-rule-20261109.md``.

Call order in a training loop::

    recorder = AdapterRecorder(path)
    recorder.signature("real", model_id, revision=...)      # or "synthetic", generator=...
    params = adapter_params(model)
    recorder.start(params)                                    # before the first step
    for step in range(n):
        ...loss.backward()
        recorder.gradients(params)                            # before clipping and the step
        optimizer.step()
        recorder.step(step, float(loss), params)
    model.save_pretrained(adapter_dir)
    ...load the adapter into a fresh model...
    recorder.reload(adapter_params(fresh_model))

The journal is written with ``allow_nan=False``: a NaN or inf stops the run at that record
with a ``ValueError``. An existing journal is never overwritten.

Benchmark harness code, not shipped. torch is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def adapter_params(model: Any) -> Dict[str, Any]:
    """Trainable parameters by name, the streaming wrapper's ``.inner.`` segment removed."""
    named = {name.replace(".inner.", "."): param
             for name, param in model.named_parameters() if param.requires_grad}
    return dict(sorted(named.items()))


def adapter_digest(params: Mapping[str, Any]) -> str:
    """SHA-256 over the names and the float32 values of the parameters, in name order."""
    digest = hashlib.sha256()
    for name in sorted(params):
        digest.update(name.encode("utf-8"))
        digest.update(params[name].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AdapterRecorder:
    def __init__(self, path: Any) -> None:
        # "x" refuses an existing file: a journal is evidence and is never overwritten.
        self._path = Path(path)
        with self._path.open("x", encoding="utf-8"):
            pass
        self._initial: Optional[Dict[str, Any]] = None
        self._initial_digest: Optional[str] = None
        self._pending: Optional[Dict[str, Any]] = None

    def event(self, name: str, /, **values: Any) -> None:
        row = {"event": name, "unix": time.time(), **values}
        line = json.dumps(row, allow_nan=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def signature(self, kind: str, model: str, **source: Any) -> None:
        self.event("signature", kind=kind, model=model, **source)

    def start(self, params: Mapping[str, Any]) -> None:
        import torch

        self._initial = {name: param.detach().float().cpu().clone()
                         for name, param in params.items()}
        self._initial_digest = adapter_digest(params)
        total = math.sqrt(float(sum(torch.sum(value * value) for value in self._initial.values())))
        self.event("adapter_initial", sha256=self._initial_digest, l2_norm=total,
                   tensors=len(params), elements=sum(v.numel() for v in self._initial.values()))
        self._before = self._initial_digest

    def gradients(self, params: Mapping[str, Any]) -> None:
        """Gradient statistics of the adapter. Call after ``backward`` and before clipping."""
        import torch

        squares = torch.zeros((), dtype=torch.float32)
        with_grad = nonfinite = 0
        for param in params.values():
            if param.grad is None:
                continue
            with_grad += 1
            grad = param.grad.detach().float().cpu()
            if not bool(torch.isfinite(grad).all()):
                nonfinite += 1
            squares += torch.sum(grad * grad)
        self._pending = {"gradient_norm": float(squares.sqrt()), "gradient_tensors": with_grad,
                         "gradient_nonfinite_tensors": nonfinite,
                         "trainable_tensors": len(params)}

    def step(self, step: int, loss: float, params: Mapping[str, Any], **extra: Any) -> None:
        """One optimizer step. Call after ``optimizer.step()``."""
        import torch

        if self._initial is None or self._pending is None:
            raise RuntimeError("call start() before the loop and gradients() before each step")
        after = adapter_digest(params)
        squares = torch.zeros((), dtype=torch.float32)
        for name, param in params.items():
            delta = param.detach().float().cpu() - self._initial[name]
            squares += torch.sum(delta * delta)
        self.event("step", step=step, loss=loss, **self._pending,
                   adapter_sha256_before=self._before, adapter_sha256_after=after,
                   distance_from_initial=float(squares.sqrt()), **extra)
        self._before = after
        self._pending = None

    def reload(self, params: Mapping[str, Any]) -> None:
        """The adapter read back from disk into a fresh model."""
        self.event("reload", parameter_sha256=adapter_digest(params))
