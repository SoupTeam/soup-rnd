"""C5 ticket 02: swap the codec of a MoE's expert weights only, NF4 or Q2_0.

Spec: .scratch/c5-q2-0-experts/issues/02-expert-only-swap.md. Codec:
``q2_0_codec.py`` (ticket 01).

The swap is fake-quant (docs/adr/0001-c5-fake-quant-training.md): each expert
tensor is quantized, restored and written back in the model's dtype. Router,
attention, embeddings, norms and shared experts are not touched.

An expert tensor is a parameter whose dotted name has a component exactly
``experts`` (``block_sparse_moe.experts.gate_up_proj`` in granitemoe,
``mlp.experts.down_proj`` in olmoe). ``shared_experts`` and ``shared_expert``
do not match. A second rule checks the first: the parameters held directly by
modules whose class name ends in ``Experts`` (``OlmoeExperts``,
``GraniteMoeExperts`` in transformers 5). The arm fails unless both rules find
the same non-empty set.

    python benchmarks/harness/c5_expert_swap.py --fixture granite --out-dir DIR
    python benchmarks/harness/c5_expert_swap.py --model allenai/OLMoE-1B-7B-0924 \
        --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda --out-dir DIR

Each run writes ``nf4.json`` and ``q2_0.json`` and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import sys
import time
from typing import TYPE_CHECKING, Any

from rich.console import Console

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import q2_0_codec as codec  # noqa: E402

if TYPE_CHECKING:
    import torch

console = Console()


def granite_fixture(
    seed: int = 0, layers: int = 2, intermediate_size: int = 128, vocab_size: int = 256
) -> torch.nn.Module:
    """A tiny random-init granitemoe in bf16. SYNTHETIC: its weights mean nothing."""
    import torch
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    config = GraniteMoeConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    torch.manual_seed(seed)
    return GraniteMoeForCausalLM(config).to(torch.bfloat16).eval()


def expert_tensor_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, _ in model.named_parameters()
        if "experts" in name.split(".")
    ]


def expert_module_tensor_names(model: torch.nn.Module) -> list[str]:
    """The second rule: parameters of modules whose class name ends in ``Experts``."""
    return [
        f"{prefix}.{name}"
        for prefix, module in model.named_modules()
        if type(module).__name__.endswith("Experts")
        for name, _ in module.named_parameters(recurse=False)
    ]


def _mse_from_sums(sq_err: float, sq_ref: float, n: int) -> dict[str, float]:
    """Same definitions as ``codec.reconstruction_error``, from summed squares."""
    return {"mse": sq_err / n, "rel_mse": sq_err / sq_ref if sq_ref > 0 else 0.0}


def _slice_errors(
    weight: torch.Tensor, codec_name: str
) -> tuple[dict[str, float], int]:
    """Swap one expert slice in place; its summed squared errors and its size."""
    import torch

    original = weight.to(torch.float32, copy=True)
    restored = codec.dequantize(codec.quantize(weight, codec_name)).to(original.device)
    weight.copy_(restored.to(weight.dtype))
    sums = {
        "error": (restored - original).pow(2).sum().item(),
        "error_in_model": (weight.to(torch.float32) - original).pow(2).sum().item(),
        "weight": original.pow(2).sum().item(),
    }
    if codec_name == "q2_0":
        ref = codec.dequantize(codec.quantize(original, "q2_0_absmax")).to(original.device)
        sums["reference_q2_0_absmax"] = (ref - original).pow(2).sum().item()
    return sums, original.numel()


def _errors(sums: dict[str, float], n: int) -> dict[str, dict[str, float]]:
    return {
        key: _mse_from_sums(value, sums["weight"], n)
        for key, value in sums.items() if key != "weight"
    }


def _swap_tensor(param: torch.nn.Parameter, codec_name: str) -> dict[str, Any]:
    """Quantize and restore ``param`` in place, one expert slice at a time.

    A fused 3D tensor [experts, out, in] is quantized per expert, so blocks run
    along rows of one expert; with rows a multiple of 64 that is byte-for-byte
    the same as flattening the whole tensor. Errors are against the weight as
    loaded: ``error`` is the codec's float32 restore, ``error_in_model`` is the
    restore cast back to the model's dtype, which is what training sees.
    ``reference_q2_0_absmax`` (Q2_0 arm only) is llama.cpp's ``d = absmax`` on
    the same slices, for comparison only; it never reaches the model. The same
    three errors are kept per expert in ``per_expert``. A fused ``gate_up_proj``
    holds gate and up together, so its errors cover both.
    """
    import torch

    per_expert = []
    total: dict[str, float] = {}
    with torch.no_grad():
        slices = param.data if param.dim() == 3 else param.data[None]
        for weight in slices:
            sums, n = _slice_errors(weight, codec_name)
            per_expert.append(_errors(sums, n))
            for key, value in sums.items():
                total[key] = total.get(key, 0.0) + value

    row_size = int(param.shape[-1])
    return {
        "shape": list(param.shape),
        "dtype": str(param.dtype),
        "row_size": row_size,
        "rows_multiple_of_64": row_size % codec.BLOCK == 0,
        **_errors(total, param.numel()),
        "per_expert": per_expert,
    }


def swap_experts(
    model: torch.nn.Module, codec_name: str, names: list[str]
) -> dict[str, dict[str, Any]]:
    """Swap the codec of ``names`` in place; one record entry per tensor."""
    params = dict(model.named_parameters())
    return {name: _swap_tensor(params[name], codec_name) for name in names}


def _sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's dtype, shape and raw bytes."""
    import torch

    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
    digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def fingerprints(model: torch.nn.Module) -> dict[str, str]:
    """SHA-256 of every parameter and buffer, tied ones under each of their names."""
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    return {name: _sha256(tensor) for name, tensor in tensors.items()}


def run_arm(model: torch.nn.Module, codec_name: str) -> dict[str, Any]:
    """Swap the experts of ``model`` to ``codec_name`` in place; the arm's record."""
    experts = expert_tensor_names(model)
    by_module = expert_module_tensor_names(model)
    before = fingerprints(model)
    tensors = swap_experts(model, codec_name, experts)
    after = fingerprints(model)
    changed = [name for name in after if after[name] != before.get(name)]
    quantized = list(tensors)
    return {
        "codec": codec_name,
        "experts": experts,
        "quantized": quantized,
        "changed": changed,
        "tensors": tensors,
        "rows_not_multiple_of_64": [
            name for name, entry in tensors.items() if not entry["rows_multiple_of_64"]
        ],
        "non_expert_fingerprints": {
            name: digest for name, digest in after.items() if name not in experts
        },
        "checks": {
            "expert_rules_agree": bool(experts) and sorted(experts) == sorted(by_module),
            "every_expert_changed": set(experts) <= set(changed),
            "non_expert_untouched": set(changed) <= set(experts),
        },
    }


def non_expert_mismatch(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Names whose non-expert fingerprint differs between two arm records,
    or that only one of them has."""
    a, b = left["non_expert_fingerprints"], right["non_expert_fingerprints"]
    return sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))


def harness_sha256() -> str:
    """SHA-256 of this file, so a record names the code that produced it."""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    """A fresh bf16 model for one arm, and what the record says about it."""
    import torch

    if args.fixture:
        model = granite_fixture(seed=args.seed)
        info = {
            "id": "fixture:granite",
            "revision": None,
            "synthetic": True,
            "note": (
                "SYNTHETIC: a tiny random-init granitemoe built from a config, "
                f"seed {args.seed}. Its weights and errors say nothing about a real model."
            ),
        }
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        info = {
            "id": args.model,
            "revision": getattr(model.config, "_commit_hash", None) or args.revision,
            "revision_requested": args.revision,
            "synthetic": False,
            "note": "",
        }
    config = model.config
    info["model_type"] = config.model_type
    info["n_experts"] = getattr(config, "num_experts", None) or getattr(
        config, "num_local_experts", None
    )
    info["dtype"] = str(next(model.parameters()).dtype)
    return model.to(args.device), info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="model id or local path")
    source.add_argument(
        "--fixture", choices=("granite",), help="tiny random-init model, labelled synthetic"
    )
    parser.add_argument("--revision", default=None, help="commit to load; required with --model")
    parser.add_argument("--device", default="cpu", help="where the swap runs, e.g. cpu or cuda")
    parser.add_argument("--seed", type=int, default=0, help="fixture seed")
    parser.add_argument("--out-dir", required=True, help="writes <codec>.json per arm")
    args = parser.parse_args(argv)
    if args.model and not args.revision:
        parser.error("--model needs --revision: the record must name the exact weights")
    if args.fixture and args.revision:
        parser.error("--revision applies to --model only")

    common = {
        "versions": _versions(),
        "harness": {"sha256": harness_sha256(), "codec_sha256": codec.codec_sha256()},
        "device": args.device,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    records = {}
    for codec_name in ("nf4", "q2_0"):
        model, info = _load(args)
        console.print(f"{codec_name}: swapping experts of {info['id']} on {args.device}")
        if info["synthetic"]:
            console.print(f"  {info['note']}")
        record = run_arm(model, codec_name)
        records[codec_name] = {"model": info, **common, **record}
        del model

    mismatch = non_expert_mismatch(records["nf4"], records["q2_0"])
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for codec_name, record in records.items():
        record["non_expert_mismatch"] = mismatch
        record["checks"]["non_expert_match_other_arm"] = not mismatch
        path = out_dir / f"{codec_name}.json"
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        tensors = record["tensors"].values()
        worst = max((entry["error"]["rel_mse"] for entry in tensors), default=0.0)
        failed = [check for check, ok in record["checks"].items() if not ok]
        console.print(
            f"  {codec_name}: {len(record['quantized'])} expert tensors, "
            f"worst rel_mse {worst:.4f}, "
            f"rows not a multiple of 64: {len(record['rows_not_multiple_of_64'])}, "
            f"failed checks: {failed or 'none'} -> {path}"
        )
    ok = all(all(record["checks"].values()) for record in records.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
