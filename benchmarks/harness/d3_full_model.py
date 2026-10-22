"""D3 whole-model forward assembly; no trainer, downloads or formal verdict.

Reuses the native OLMoE attention/router and B1 readers/pools. A single shared
streaming pool spans every layer. Preparation and CPU controls are not a trained
resident CUDA gate; that gate requires its separately frozen execution protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(folder: Path, name: str) -> Path:
    target = os.path.realpath(os.path.join(folder, name))
    if (os.path.isabs(name)
            or os.path.commonpath([os.path.realpath(folder), target]) != os.path.realpath(folder)):
        raise ValueError("whole-model bundle path escapes its directory")
    return Path(target)


def tensor_hash(value: Any) -> str:
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def prepare_bundle(
    config: Any, nonexpert: dict, layer_weights: Any, output: Path, *,
    codec: str, quant_device: str, provenance: dict,
) -> dict:
    """Pack a complete checkpoint through the existing B1.1 expert codec.

    ``layer_weights`` yields ordered (layer, gate/up-down pairs) without retaining
    every dense expert layer in host RAM. Preserve partial output on any error.
    """
    from safetensors.torch import save_file

    from soup_cli.utils.expert_shard import write_expert_shard
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    if codec not in ("dense", "nf4", "nf4-dq"):
        raise ValueError("unsupported whole-model codec")
    if any(".mlp.experts." in key for key in nonexpert):
        raise ValueError("routed expert tensors cannot enter the nonexpert checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(output)
    save_file({key: value.detach().cpu().contiguous().clone()
               for key, value in nonexpert.items()}, str(output / "nonexpert.safetensors"))
    record = {"scope": "preparation only; no formal measurements", "codec": codec,
              "provenance": provenance, "layers": [], "complete": False}
    try:
        for layer, pairs in layer_weights:
            if type(layer) is not int or layer != len(record["layers"]):
                raise ValueError("all expert layers must be supplied once, in order")
            name = f"experts_{layer:03d}.safetensors"
            info = write_expert_shard(
                str(output / name), pairs, quantise=codec != "dense",
                double_quant=codec == "nf4-dq", device=quant_device,
            )
            if info.count != config.num_experts:
                raise ValueError("expert count differs from the native model")
            codes = {key: value.tolist() for key, value in zip(
                (NF4_CODE_KEY, NF4_NESTED_CODE_KEY), info.codes,
            ) if value is not None}
            record["layers"].append({
                "layer": layer, "file": name, "count": info.count,
                "shapes": dict(info.shapes),
                "quant_specs": {key: value.to_json() for key, value in info.quant_specs.items()},
                "dtype": str(nonexpert["model.embed_tokens.weight"].dtype).removeprefix("torch."),
                "codes": codes, "sha256": file_hash(output / name),
            })
            _write_json(output / "bundle.json", record)
        if len(record["layers"]) != config.num_hidden_layers:
            raise ValueError("incomplete model: an expert layer is missing")
        record["files"] = {name: file_hash(output / name)
                           for name in ("config.json", "nonexpert.safetensors")}
        record["complete"] = True
        return record
    finally:
        _write_json(output / "bundle.json", record)


def validate_bundle(folder: Path) -> dict:
    record = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if record.get("complete") is not True:
        raise ValueError("whole-model bundle preparation did not complete")
    if record.get("codec") not in ("dense", "nf4", "nf4-dq"):
        raise ValueError("unsupported whole-model codec")
    if set(record.get("files", {})) != {"config.json", "nonexpert.safetensors"}:
        raise ValueError("nonexpert/config identities are missing")
    for name, digest in record["files"].items():
        if file_hash(contained(folder, name)) != digest:
            raise ValueError(f"whole-model file hash differs: {name}")
    for index, row in enumerate(record["layers"]):
        if row["layer"] != index or row["file"] != f"experts_{index:03d}.safetensors":
            raise ValueError("whole-model layers are incomplete or reordered")
        if file_hash(contained(folder, row["file"])) != row["sha256"]:
            raise ValueError("whole-model expert file hash differs")
    return record


def _shell(config: Any, state: dict, *, device: Any, dtype: Any) -> Any:
    import torch
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeForCausalLM,
        OlmoeRotaryEmbedding,
    )

    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.use_cache = False
    with torch.device("meta"):
        model = OlmoeForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp.experts = torch.nn.Identity()
    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError("nonexpert state must cover exactly the native model skeleton")
    # Do not leave meta RoPE buffers or lower their explicit FP32 arithmetic.
    model.model.rotary_emb = OlmoeRotaryEmbedding(config).to(device=device)
    values = {key: value.to(device=device, dtype=dtype) for key, value in state.items()}
    model.load_state_dict(values, strict=True, assign=True)
    return model.requires_grad_(False).eval()


class WholeModel:
    """Own readers, one shared weight pool and an optional shared GEMM backend."""

    def __init__(self, stack: ExitStack, model: Any, readers: list, pool: Any, gemm: Any):
        self._stack = stack
        self._closed = False
        self.model, self.readers, self.pool, self.gemm = model, readers, pool, gemm

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        self._closed = True
        self._stack.close()
        self.model = None
        if hasattr(self, "resident_experts"):
            self.resident_experts = []

    def forward(self, input_ids: Any) -> Any:
        import torch

        if self._closed:
            raise RuntimeError("whole-model owner is closed")
        if torch.is_grad_enabled():
            raise RuntimeError("D3 whole-model harness is forward-only; use no_grad")
        if self.model.training:
            raise RuntimeError("whole-model validation requires eval mode")
        return self.model(input_ids=input_ids, use_cache=False, logits_to_keep=0).logits

    def traces(self) -> list:
        return [list(layer.mlp.experts.last_trace) for layer in self.model.model.layers]


def open_streamed_model(
    folder: Path, *, device: str, backend: str, library: Path | None = None,
    weight_budget_bytes: int = 64 * 2**20, activation_budget_bytes: int = 256 * 2**20,
    gemm_call_budget_bytes: int = 64 * 2**20, native_olmoe_rows_max: int = 16,
) -> WholeModel:
    """Assemble all native model blocks, replacing only their routed experts.

    ``grouped-restore-control`` is for CPU/development integration only. It is
    never a substitute for the actual ``grouped-compute`` CUDA candidate.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import OlmoeConfig

    from soup_cli.utils.expert_group import GroupedExpertPool, grouped_experts_class
    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.layer_shard import NF4WeightSpec

    if backend not in ("sequential", "grouped-restore-control", "grouped-compute"):
        raise ValueError("unknown whole-model expert backend")
    target = torch.device(device)
    if backend == "grouped-compute" and (target.type != "cuda" or library is None):
        raise ValueError("the grouped compute backend needs native BF16 CUDA and its library")
    record = validate_bundle(folder)
    config = OlmoeConfig.from_pretrained(folder, local_files_only=True)
    if not record["layers"] or len(record["layers"]) != config.num_hidden_layers:
        raise ValueError("bundle layer count differs from model config")
    stack = ExitStack()
    try:
        readers = []
        for row in record["layers"]:
            reader = ExpertShardReader(
                str(contained(folder, row["file"])), count=row["count"],
                shapes={key: tuple(value) for key, value in row["shapes"].items()},
                quant_specs={key: NF4WeightSpec.from_json(value)
                             for key, value in row["quant_specs"].items()},
                dtype=row["dtype"], direct=False,
            )
            stack.callback(reader.close)
            readers.append(reader)
            if reader.count != config.num_experts:
                raise ValueError("bundle expert count differs from model config")
        dtype_name = record["layers"][0]["dtype"]
        if any(row["dtype"] != dtype_name or row["codes"] != record["layers"][0]["codes"]
               for row in record["layers"]):
            raise ValueError("shared pool requires the same dtype and code tables in every layer")
        dtype = getattr(torch, dtype_name)
        codes = {key: torch.tensor(value, dtype=torch.float32)
                 for key, value in record["layers"][0]["codes"].items()}
        if backend == "sequential":
            pool = ExpertBufferPool(readers[0], device=device, dtype=dtype_name,
                                    weight_budget_bytes=weight_budget_bytes, codes=codes)
            factory = streamed_experts_class()
        else:
            pool = GroupedExpertPool(readers[0], device=device, dtype=dtype_name, group_width=4,
                                    weight_budget_bytes=weight_budget_bytes, codes=codes)
            factory = grouped_experts_class()
        stack.callback(pool.close)
        for reader in readers:
            pool.validate_reader(reader)
        gemm = None
        if backend == "grouped-compute":
            gemm = GroupedGemm(library, call_budget_bytes=gemm_call_budget_bytes,
                               native_olmoe_rows_max=native_olmoe_rows_max)
            stack.callback(gemm.close)
            factory = grouped_compute_experts_class()
        state = load_file(str(contained(folder, "nonexpert.safetensors")))
        model = _shell(config, state, device=target, dtype=dtype)
        from transformers.activations import ACT2FN

        for index, layer in enumerate(model.model.layers):
            kwargs = {"gemm": gemm, "activation_budget_bytes": activation_budget_bytes}
            layer.mlp.experts = factory(
                readers[index], pool, ACT2FN[config.hidden_act], index,
                **(kwargs if gemm is not None else {}),
            )
        model.requires_grad_(False).eval()
        return WholeModel(stack, model, readers, pool, gemm)
    except BaseException:
        stack.close()
        raise


def open_resident_model(
    folder: Path, *, device: str, backend: str = "native", library: Path | None = None,
    gemm_call_budget_bytes: int = 64 * 2**20, native_olmoe_rows_max: int = 16,
) -> WholeModel:
    """Independent full resident oracle; deliberately NOT the bounded candidate.

    NF4 is restored through the audited B1.2 bitsandbytes oracle, rather than the
    grouped Triton decoder. The grouped resident control uses the exact same
    GroupedComputeExperts class as its streamed counterpart.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from soup_cli.utils.expert_shard import ExpertShardReader, expert_key
    from soup_cli.utils.layer_shard import NF4WeightSpec

    if backend not in ("native", "grouped-compute"):
        raise ValueError("unknown resident oracle backend")
    if backend == "grouped-compute" and (torch.device(device).type != "cuda" or library is None):
        raise ValueError("resident grouped compute requires native BF16 CUDA and its library")
    record = validate_bundle(folder)
    config = OlmoeConfig.from_pretrained(folder, local_files_only=True)
    if not record["layers"] or len(record["layers"]) != config.num_hidden_layers:
        raise ValueError("bundle layer count differs from model config")
    stack = ExitStack()
    try:
        dtype = getattr(torch, record["layers"][0]["dtype"])
        state = load_file(str(contained(folder, "nonexpert.safetensors")))
        model = _shell(config, state, device=device, dtype=dtype)
        readers, residents = [], []
        gemm = None
        if backend == "grouped-compute":
            gemm = GroupedGemm(library, call_budget_bytes=gemm_call_budget_bytes,
                               native_olmoe_rows_max=native_olmoe_rows_max)
            stack.callback(gemm.close)
        # Loaded lazily because this is a benchmark oracle, not a public runtime.
        from d3_grouped_restore_check import ResidentControlPool, _baseline

        for index, row in enumerate(record["layers"]):
            reader = ExpertShardReader(
                str(contained(folder, row["file"])), count=row["count"],
                shapes={key: tuple(value) for key, value in row["shapes"].items()},
                quant_specs={key: NF4WeightSpec.from_json(value)
                             for key, value in row["quant_specs"].items()},
                dtype=row["dtype"], direct=False,
            )
            stack.callback(reader.close)
            readers.append(reader)
            if reader.count != config.num_experts or row["dtype"] != str(dtype).split(".")[-1]:
                raise ValueError("resident expert count/dtype differs from the model")
            with torch.device("meta"):
                experts = OlmoeExperts(model.config)
            experts.to(dtype=dtype).to_empty(device=device).requires_grad_(False).eval()
            codes = {key: torch.tensor(value, dtype=torch.float32)
                     for key, value in row["codes"].items()}
            with torch.no_grad():
                if reader.quant_specs:
                    _baseline()._nf4_reference(experts, reader, codes, device)
                else:
                    with safe_open(reader.path, framework="pt", device="cpu") as handle:
                        for expert in range(reader.count):
                            for part, target in (("gate_up", experts.gate_up_proj),
                                                 ("down", experts.down_proj)):
                                target[expert].copy_(handle.get_tensor(expert_key(expert, part)))
            residents.append(experts)
            if gemm is None:
                experts.last_trace = []

                def routes(module, args, layer=index):
                    indices = args[1]
                    module.last_trace = [
                        {"layer": layer, "expert": expert,
                         "token_routes": int((indices == expert).sum().item()),
                         "logical_bytes_read": 0, "status": "routed"}
                        for expert in sorted(indices.unique().tolist())
                    ]

                hook = experts.register_forward_pre_hook(routes)
                stack.callback(hook.remove)
                model.model.layers[index].mlp.experts = experts
            else:
                pool = ResidentControlPool(experts, reader, width=4)
                model.model.layers[index].mlp.experts = grouped_compute_experts_class()(
                    reader, pool, experts.act_fn, index, gemm=gemm,
                    activation_budget_bytes=256 * 2**20,
                )
        run = WholeModel(stack, model.requires_grad_(False).eval(), readers, None, gemm)
        run.resident_experts = residents
        return run
    except BaseException:
        stack.close()
        raise
