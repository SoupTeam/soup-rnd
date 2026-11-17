"""Packed resident controls using the same explicit fused runtime as streaming."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import fields, replace

from benchmarks.harness import d3_trainer_gate as original


def make_config(rule, prepared, out, codec, mode, training):
    from benchmarks.harness.d3_fused_trainer_protocol import CODECS, MODES
    from soup_cli.config.schema import SoupConfig

    if codec not in CODECS or mode not in MODES:
        raise ValueError("unknown fused qualification cell")
    fused = mode.startswith("fused")
    mapped = mode.replace("fused", "grouped", 1) if fused else mode
    config = original.make_config(rule, prepared, out, codec, mapped, training)
    if not fused:
        return config
    payload = config.model_dump()
    payload["training"]["stream_experts"].update(backend="fused_nf4", native_rows_from=None)
    with original.public_workspace(prepared, out):
        return SoupConfig.model_validate(payload)


class PackedResidentPool:
    """Correctness-only original packed weights; never a streamed memory arm."""

    packed_only = True

    def __init__(self, reader, source):
        import torch

        self.device, self.dtype = source.device, source.dtype
        self.group_width = source.group_width
        self.path, self.count = reader.path, reader.count
        self._shapes = dict(reader.shapes)
        self.weights = []
        self._active = self._closed = False
        self._generation = 0
        self._lock = threading.Lock()
        try:
            with torch.no_grad():
                for expert in range(reader.count):
                    with source.lease(reader, [expert]) as pairs:
                        # Clone original packed bytes/statistics, not restored dense weights.
                        self.weights.append(
                            tuple(
                                replace(
                                    weight,
                                    **{
                                        field.name: getattr(weight, field.name).clone()
                                        for field in fields(weight)
                                        if isinstance(getattr(weight, field.name), torch.Tensor)
                                    },
                                )
                                for weight in pairs[0]
                            )
                        )
        except BaseException:
            self.close()
            raise

    def validate_reader(self, reader):
        if self._closed or (reader.path, reader.count, dict(reader.shapes)) != (
            self.path,
            self.count,
            self._shapes,
        ):
            raise ValueError("packed resident reader changed or closed")

    @contextmanager
    def lease(self, reader, experts, *, trace=None):
        import torch

        self.validate_reader(reader)
        if (
            not experts
            or len(experts) > self.group_width
            or len(set(experts)) != len(experts)
            or any(type(index) is not int or not 0 <= index < self.count for index in experts)
        ):
            raise ValueError("invalid packed resident expert group")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("packed resident lease already active")
        self._active = True
        self._generation += 1
        generation = self._generation

        def owner_check():
            if self._closed or not self._active or self._generation != generation:
                raise RuntimeError("packed resident lease expired")

        try:
            pairs = []
            for expert in experts:
                # A separate tensor object per lease retains its own generation
                # guard even when a later lease selects the same resident storage.
                pair = tuple(
                    replace(weight, packed=weight.packed.view_as(weight.packed))
                    for weight in self.weights[expert]
                )
                for weight in pair:
                    weight.packed._soup_stream_owner_check = owner_check
                pairs.append(pair)
            if trace is not None:
                trace.extend({"expert": index, "logical_bytes_read": 0} for index in experts)
            yield tuple(pairs)
        finally:
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            finally:
                self._active = False
                self._lock.release()

    def close(self):
        if self._active:
            raise RuntimeError("cannot close an active packed resident lease")
        self.weights.clear()
        self._closed = True


def restore_resident(wrapper, row, mode, device):
    if mode != "fused-resident":
        return original.restore_resident_experts(wrapper, row, mode, device)
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    runtime = wrapper._stream_runtime
    resources = runtime.expert_resources
    result = []
    for index, reader in resources.readers.items():
        pool = PackedResidentPool(reader, resources.pool)
        resources._stack.callback(pool.close)
        decoder_owner(wrapper.model).layers[index].mlp.experts.pool = pool
        result.append(pool)
    return result


def assemble(rule, prepared, out, codec, mode, training, *, device="cuda"):
    return original.assemble(
        rule,
        prepared,
        out,
        codec,
        mode,
        training,
        device=device,
        config_factory=make_config,
        resident_factory=restore_resident,
    )


def correctness(*args, **kwargs):
    return original.correctness(*args, **kwargs, assemble_fn=assemble)


def performance(*args, **kwargs):
    from benchmarks.harness.d3_trainer_series import performance as run

    return run(*args, **kwargs, assemble_fn=assemble)
