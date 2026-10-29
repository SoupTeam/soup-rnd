"""Actual public-wrapper preparation and trainer qualification building blocks.

Preparation never executes a pretrained model. Formal parent/child execution
is gated by the separate prospective rule and canonical committed source packet.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.harness.d3_trainer_protocol import (
    CODECS,
    CORRECTNESS_MODES,
    file_hash,
)


class NumericalFailureError(ValueError):
    """A numerical failure, distinct from an unavailable instrument or capacity."""


def save(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_record(value: Any) -> dict:
    import torch

    data = value.detach().cpu().contiguous()
    raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
    if not torch.isfinite(data).all().item():
        raise NumericalFailureError("nonfinite numerical evidence")
    return {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def require_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise NumericalFailureError(f"exact numerical comparison failed: {label}")


@contextmanager
def public_workspace(prepared: dict, output: Path) -> Any:
    """Honor the public schema's containment policy in the prepared workspace."""
    root = os.path.realpath(prepared["workspace"])
    if root == os.path.dirname(root):
        raise ValueError("filesystem root is not an acceptable prepared workspace")
    for path in (
        output,
        prepared["rows_path"],
        *(
            row[key]
            for row in prepared["codecs"].values()
            for key in ("checkpoint", "dataset", "initial_adapters")
        ),
    ):
        if os.path.commonpath((root, os.path.realpath(path))) != root:
            raise ValueError("public trainer path escapes the prepared workspace")
    previous = os.getcwd()
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(previous)


def make_config(rule: dict, prepared: dict, out: Path, codec: str, mode: str, training: str) -> Any:
    from soup_cli.config.schema import SoupConfig

    if codec not in CODECS or mode not in CORRECTNESS_MODES or training not in ("ordinary", "l2l"):
        raise ValueError("unsupported protocol cell")
    row = prepared["codecs"][codec]
    backend = "sequential" if mode == "native-resident" else mode.removesuffix("-resident")
    payload = dict(
        base=row["checkpoint"],
        output=str(out),
        data={
            "train": prepared["rows_path"],
            "format": "pre_tokenized",
            "tokenized_path": row["dataset"],
            "max_length": rule["tokens"],
            "train_on_responses_only": False,
            "val_split": 0,
        },
        training={
            "backend": "transformers",
            "stream_layers": True,
            "stream_source": "ram",
            "batch_size": rule["batch"],
            "epochs": 1,
            "gradient_accumulation_steps": rule["gradient_accumulation_steps"],
            "seed": rule["adapter_seed"],
            "data_seed": rule["data_seed"],
            "lr": rule["optimizer"]["learning_rate"],
            "optimizer": "sgd",
            "scheduler": "constant",
            "weight_decay": 0,
            "warmup_ratio": 0,
            "max_grad_norm": rule["optimizer"]["max_grad_norm"],
            "save_steps": 1000000,
            "logging_steps": 1,
            "rewind_log": False,
            "packing": False,
            "l2l": training == "l2l",
            "moe_aux_loss_coeff": 0,
            "quantization": "none" if codec == "dense" else "4bit",
            "bnb_4bit_use_double_quant": codec == "nf4-dq",
            "lora": {
                "r": rule["adapter_rank"],
                "alpha": rule["adapter_alpha"],
                "dropout": rule["adapter_dropout"],
                "target_modules": rule["entry_point"]["adapter_targets"],
            },
            "stream_experts": {
                "backend": backend,
                "group_width": 1 if backend == "sequential" else 4,
                "weight_budget_bytes": rule["weight_budget_bytes"],
                "activation_budget_bytes": rule["activation_budget_bytes"],
                "call_budget_bytes": rule["call_budget_bytes"],
                "native_rows_max": rule["native_rows_max"],
                "native_rows_from": rule["native_rows_from"],
                "cublas_library": prepared["library"],
            },
        },
    )
    with public_workspace(prepared, out):
        return SoupConfig(**payload)


def restore_resident_experts(wrapper: Any, row: dict, mode: str, device: str) -> list:
    """Independent B1 reconstruction; reuse exact compute for paired controls."""
    import torch
    from safetensors import safe_open
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    from benchmarks.harness.b1_expert_forward_gate import _nf4_reference
    from benchmarks.harness.d3_grouped_restore_check import ResidentControlPool
    from soup_cli.utils.expert_shard import expert_key
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY, extras_shard_path
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    runtime = wrapper._stream_runtime
    wrapper.model.config._experts_implementation = "eager"
    layers = decoder_owner(wrapper.model).layers
    codes = {}
    with safe_open(extras_shard_path(row["shards"]), framework="pt", device="cpu") as handle:
        for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY):
            if key in handle.keys():
                codes[key] = handle.get_tensor(key)
    residents = []
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    for index, reader in runtime.expert_resources.readers.items():
        with torch.device("meta"):
            native = OlmoeExperts(wrapper.model.config)
        native.to(dtype=dtype).to_empty(device=device).requires_grad_(False)
        if reader.quant_specs:
            _nf4_reference(native, reader, codes, device)
        else:
            with torch.no_grad(), safe_open(reader.path, framework="pt", device="cpu") as handle:
                for expert in range(reader.count):
                    value = handle.get_tensor(expert_key(expert, "gate_up"))
                    native.gate_up_proj[expert].copy_(value)
                    native.down_proj[expert].copy_(handle.get_tensor(expert_key(expert, "down")))
        residents.append(native)
        if mode == "native-resident":
            layers[index].mlp.experts = native
        else:
            candidate = layers[index].mlp.experts
            candidate.pool = ResidentControlPool(native, reader, candidate.pool.group_width)
    return residents


def assemble(
    rule: dict,
    prepared: dict,
    out: Path,
    codec: str,
    mode: str,
    training: str,
    *,
    device: str = "cuda",
) -> Any:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    from soup_cli.trainer.sft import SFTTrainerWrapper

    config = make_config(rule, prepared, out, codec, mode, training)
    wrapper = SFTTrainerWrapper(config, device=device, report_to="none")
    try:
        with public_workspace(prepared, out):
            wrapper.setup({"train": [], "val": []})
        initial = load_file(prepared["codecs"][codec]["initial_adapters"])
        current = get_peft_model_state_dict(wrapper.model)
        if set(initial) != set(current) or any(
            initial[key].dtype != current[key].dtype for key in initial
        ):
            raise NumericalFailureError("initial adapter names/dtypes differ from preparation")
        result = set_peft_model_state_dict(wrapper.model, initial)
        if result.unexpected_keys:
            raise NumericalFailureError("unexpected initial adapter keys")
        require_exact(
            {
                key: tensor_record(value)
                for key, value in get_peft_model_state_dict(wrapper.model).items()
            },
            {key: tensor_record(value) for key, value in initial.items()},
            "initial factors",
        )
        if mode.endswith("-resident"):
            wrapper._d3_resident_references = restore_resident_experts(
                wrapper,
                prepared["codecs"][codec],
                mode,
                device,
            )
        return wrapper
    except BaseException:
        wrapper._close_stream_runtime()
        raise


def capture_class() -> Any:
    import torch
    from transformers import TrainerCallback

    from soup_cli.utils.layer_stream_runtime import decoder_owner

    class Capture(TrainerCallback):
        def __init__(self, wrapper, *, oracle_dir=None):
            self.wrapper = wrapper
            self.frames, self.hooks, self.windows = [], [], []
            self.current, self.losses, self.gradients = {}, [], {}
            self.raw_parameter_gradients = {}
            self.active = False
            self.original_get = wrapper.trainer.get_batch_samples
            self.causal = wrapper.model.get_base_model()
            self.original_loss = self.causal.loss_function
            self.oracle_dir = oracle_dir
            self.raw = {}

            def batches(*args, **kwargs):
                samples, count = self.original_get(*args, **kwargs)
                self.windows.append(
                    {
                        "samples": [
                            {
                                key: tensor_record(value)
                                for key, value in row.items()
                                if isinstance(value, torch.Tensor)
                            }
                            for row in samples
                        ],
                        "num_items": int(count) if count is not None else None,
                    }
                )
                return samples, count

            def loss(*args, **kwargs):
                value = self.original_loss(*args, **kwargs)
                logits = kwargs["logits"] if "logits" in kwargs else args[0]
                self.losses.append({"logits": tensor_record(logits), "loss": tensor_record(value)})
                if self.oracle_dir is not None:
                    name = f"logits-{len(self.frames)}-{len(self.losses)}"
                    self.raw[name] = logits.detach().cpu().contiguous().clone()
                    self.raw[name + "-loss"] = value.detach().cpu().contiguous().clone().reshape(1)
                return value

            wrapper.trainer.get_batch_samples = batches
            self.causal.loss_function = loss
            for index, layer in enumerate(decoder_owner(wrapper.model).layers):
                self.hooks.append(layer.mlp.experts.register_forward_pre_hook(self._hook(index)))
                self.hooks.append(layer.mlp.experts.register_forward_hook(self._output_hook(index)))
            for name, parameter in wrapper.model.named_parameters():
                if parameter.requires_grad:

                    def capture_parameter(value, key=name):
                        self.raw_parameter_gradients[key] = tensor_record(value.grad)
                        if self.oracle_dir is not None:
                            self.raw["raw_gradient/" + key] = (
                                value.grad.detach().cpu().contiguous().clone()
                            )

                    self.hooks.append(
                        parameter.register_post_accumulate_grad_hook(
                            capture_parameter,
                        )
                    )

        def _hook(self, index):
            def hook(_module, args):
                hidden, ids, weights = args
                if not self.active:
                    return
                identity = f"{index}:{tensor_record(hidden)['sha256']}"
                self.current.setdefault(identity, {}).update({
                    "input": tensor_record(hidden),
                    "routes": tensor_record(ids),
                    "weights": tensor_record(weights),
                })
                if self.oracle_dir is not None:
                    for key, value in (("input", hidden), ("routes", ids), ("weights", weights)):
                        self.raw[f"expert_forward/{identity}/{key}"] = (
                            value.detach().cpu().contiguous().clone()
                        )
                if not torch.is_grad_enabled() or not hidden.requires_grad:
                    return

                def gradient(kind):
                    def collect(value):
                        self.gradients.setdefault(identity, {})[kind] = tensor_record(value)
                        if self.oracle_dir is not None:
                            self.raw[f"expert_gradient/{identity}/{kind}"] = (
                                value.detach().cpu().contiguous().clone()
                            )

                    return collect

                hidden.register_hook(gradient("input"))
                if weights.requires_grad:
                    weights.register_hook(gradient("router"))

            return hook

        def _output_hook(self, index):
            def hook(_module, args, output):
                hidden = args[0]
                if not self.active:
                    return
                identity = f"{index}:{tensor_record(hidden)['sha256']}"
                self.current[identity]["output"] = tensor_record(output)
                if self.oracle_dir is not None:
                    self.raw[f"expert_forward/{identity}/output"] = (
                        output.detach().cpu().contiguous().clone()
                    )

            return hook

        def on_step_begin(self, args, state, control, **kwargs):
            self.active = True
            self.current, self.losses, self.gradients = {}, [], {}
            self.raw_parameter_gradients = {}

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = {
                name: tensor_record(param.grad)
                for name, param in model.named_parameters()
                if param.requires_grad and param.grad is not None
            }
            parameters = {
                name: param for name, param in model.named_parameters() if param.requires_grad
            }
            if set(gradients) != set(parameters) or not any(
                param.grad.count_nonzero().item() for param in parameters.values()
            ):
                raise NumericalFailureError("trainable gradients missing or ineffective")
            count = len(decoder_owner(model).layers)
            covered = {int(name.split(":", 1)[0]) for name in self.gradients}
            if covered != set(range(count)) or any(
                set(value) != {"input", "router"} for value in self.gradients.values()
            ):
                raise NumericalFailureError("incomplete per-layer input/router derivatives")
            self.pending = {
                "forward": self.losses,
                "routes": dict(sorted(self.current.items())),
                "expert_gradients": dict(sorted(self.gradients.items())),
                "raw_parameter_gradients": self.raw_parameter_gradients,
                "parameter_gradients": gradients,
            }
            self.active = False
            if self.oracle_dir is not None:
                for name, param in parameters.items():
                    self.raw[f"gradient/{name}"] = param.grad.detach().cpu().contiguous().clone()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.pending["parameters"] = {
                name: tensor_record(param)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            self.frames.append(self.pending)
            if self.oracle_dir is not None:
                from safetensors.torch import save_file

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.raw[f"parameter/{name}"] = param.detach().cpu().contiguous().clone()
                save_file(self.raw, str(self.oracle_dir / f"step-{state.global_step}.safetensors"))
                self.raw = {}

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self.causal.loss_function = self.original_loss
            model.eval()
            ids = torch.tensor(
                [self.wrapper.trainer.train_dataset[0]["input_ids"]],
                # Untied streamed embeddings keep a meta placeholder parameter;
                # it describes storage, not the execution device for real IDs.
                device=self.wrapper._stream_runtime.device,
            )
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.device.type == "cuda"),
            ):
                self.eval_logits = tensor_record(model(input_ids=ids).logits.float())

        def close(self):
            self.causal.loss_function = self.original_loss
            self.wrapper.trainer.get_batch_samples = self.original_get
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            self.wrapper = None

    return Capture


def one_replay(
    rule: dict,
    prepared: dict,
    out: Path,
    codec: str,
    mode: str,
    training: str,
    *,
    negative: bool = False,
    raw_oracle: bool = False,
    device: str = "cuda",
) -> dict:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    out.mkdir(parents=True, exist_ok=False)
    wrapper = None
    capture = None
    try:
        wrapper = assemble(rule, prepared, out / "adapter", codec, mode, training, device=device)
        initial = {
            name: tensor_record(value)
            for name, value in get_peft_model_state_dict(wrapper.model).items()
        }
        if negative:
            candidates = sorted(
                (name, param)
                for name, param in wrapper.model.named_parameters()
                if "v_proj" in name and "lora_B" in name and param.requires_grad
            )
            if not candidates:
                raise ValueError("no active v_proj lora_B for the predeclared negative")
            with torch.no_grad():
                candidates[0][1][:, 0].add_(0.5)
        raw = out / "oracle"
        if raw_oracle:
            raw.mkdir()
        capture = capture_class()(wrapper, oracle_dir=raw if raw_oracle else None)
        wrapper.trainer.add_callback(capture)
        with public_workspace(prepared, out):
            result = wrapper.train()
        final = {
            name: tensor_record(value)
            for name, value in get_peft_model_state_dict(wrapper.model).items()
        }
        if result["total_steps"] != rule["optimizer_steps"] or initial == final:
            raise NumericalFailureError("missing real optimizer updates")
        artifact = out / "adapter/adapter_model.safetensors"
        saved = {name: tensor_record(value) for name, value in load_file(str(artifact)).items()}
        require_exact(saved, final, "public saved adapter")
        frames, windows = capture.frames, capture.windows
        record = {
            "steps": frames,
            "windows": windows,
            "final": final,
            "evaluation": capture.eval_logits,
            "runtime": wrapper._stream_runtime.stats(),
            "artifact": str(artifact),
            "artifact_sha256": file_hash(artifact),
            "attention": wrapper.model.config._attn_implementation,
        }
        if mode not in ("native-resident", "sequential") and device.startswith("cuda"):
            if record["runtime"]["expert_gemm"]["grouped_matrices"] <= 0:
                raise ValueError("required grouped path was inactive")
        save(out / "record.json", record)
        return record
    finally:
        if capture is not None:
            capture.close()
        if wrapper is not None:
            wrapper._close_stream_runtime()
        del capture, wrapper
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def numerical_view(record: dict) -> dict:
    return {key: record[key] for key in ("steps", "windows", "final", "evaluation", "attention")}


def correctness(
    rule: dict,
    prepared: dict,
    out: Path,
    codec: str,
    mode: str,
    training: str,
    oracle: dict | None,
    *,
    device: str = "cuda",
) -> dict:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    positive = one_replay(
        rule,
        prepared,
        out / "positive",
        codec,
        mode,
        training,
        raw_oracle=mode == "native-resident" and training == "ordinary",
        device=device,
    )
    again = one_replay(rule, prepared, out / "A-A", codec, mode, training, device=device)
    require_exact(numerical_view(again), numerical_view(positive), "same-process A/A")
    negative = one_replay(
        rule, prepared, out / "negative", codec, mode, training, negative=True, device=device
    )
    if negative["evaluation"] == positive["evaluation"] or all(
        a["parameter_gradients"] == b["parameter_gradients"]
        for a, b in zip(negative["steps"], positive["steps"], strict=True)
    ):
        raise ValueError("predeclared negative control is ineffective")
    restored = one_replay(rule, prepared, out / "restored", codec, mode, training, device=device)
    require_exact(numerical_view(restored), numerical_view(positive), "restored replay")
    if oracle is not None:
        require_exact(numerical_view(positive), numerical_view(oracle), "native ordinary GA oracle")
    wrapper = assemble(rule, prepared, out / "reload", codec, mode, training, device=device)
    try:
        state = load_file(positive["artifact"])
        result = set_peft_model_state_dict(wrapper.model, state)
        if result.unexpected_keys:
            raise NumericalFailureError("unexpected saved adapter keys")
        require_exact(
            {
                name: tensor_record(value)
                for name, value in get_peft_model_state_dict(wrapper.model).items()
            },
            positive["final"],
            "reload factor dtypes/bytes",
        )
        model = wrapper.model.eval()
        ids = torch.tensor([wrapper.trainer.train_dataset[0]["input_ids"]], device=device)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
        ):
            logits = tensor_record(model(input_ids=ids).logits.float())
        require_exact(logits, positive["evaluation"], "reloaded logits")
        reload_evidence = {
            "evaluation": logits,
            "factors": {key: tensor_record(value) for key, value in
                        get_peft_model_state_dict(wrapper.model).items()},
        }
    finally:
        wrapper._close_stream_runtime()
    return {
        "numerical": positive,
        "A_A": True,
        "negative_effective": True,
        "restored": True,
        "save_reload": True,
        "reload": reload_evidence,
    }


def software() -> dict:
    import torch

    names = ("torch", "triton", "transformers", "peft", "trl", "bitsandbytes", "safetensors")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in names},
        "cuda": torch.version.cuda,
    }


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "series", "child"):
        entry = sub.add_parser(name)
        entry.add_argument("--prepared", type=Path, required=True)
        entry.add_argument("--out", type=Path, required=True)
        if name == "child":
            entry.add_argument("--session", type=Path, required=True)
            entry.add_argument("--cell", required=True)
            entry.add_argument("--oracle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from benchmarks.harness.d3_trainer_prepare import prepare

        prepare(args.prepared, args.out)
        return 0
    from benchmarks.harness.d3_trainer_series import child, parent

    try:
        return child(args) if args.command == "child" else parent(args)
    except (OSError, ValueError) as exc:
        Console().print(f"[red]D3 public trainer execution refused:[/] {exc}")
        return 1


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())
