"""B1.4 ticket 08: per_expert streaming against a resident model on a real checkpoint.

Takes a native DeepSeek-V3 checkpoint directory and compares, on CPU, the per_expert
streamed model with a resident eager model of the same checkpoint and the same LoRA
adapters: logits, loss and every LoRA gradient of one forward and backward pass.

Pass rule, fixed before any real run (B1.3 contract, ``docs/expert-forward.md``):
  float32   bitwise equality (equal-length rows),
  bfloat16  ``atol=1e-6, rtol=1e-5``.
The streamed pass runs first and its results go to host memory; its cache and model are
released before the resident model loads, so peak RAM is the larger of the two, not their sum.

Run from the checkout root with its src on PYTHONPATH:
  python benchmarks/harness/b14_real_moe_correctness.py --weights DIR --work DIR --output out.json
"""

from __future__ import annotations

import gc
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import resource
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None  # type: ignore[assignment]

try:
    from . import b13_moe_fixture as fixture
except ImportError:  # run as a script
    import b13_moe_fixture as fixture  # type: ignore[no-redef]

#: bf16 tolerance of the B1.3 trainer contract.
BF16_ATOL = 1e-6
BF16_RTOL = 1e-5


#: Definition of a real small MoE (ticket 08). Both numbers are fixed before any real run.
MAX_SAFETENSORS_BYTES = 20_000_000_000
#: Expert std must differ from ``initializer_range`` by at least this fraction.
MIN_STD_DEVIATION = 0.05


_EXPERT_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)


def _expert_keys_complete(config: Dict[str, Any], weight_names: Iterable[str]) -> bool:
    """Every MoE layer has ids 0..n-1 with all three projections, and no other layer has any."""
    found: Dict[int, Dict[int, set]] = {}
    for name in weight_names:
        match = _EXPERT_KEY.match(name)
        if match:
            layer, expert = int(match.group(1)), int(match.group(2))
            if layer >= config["num_hidden_layers"]:
                continue  # multi-token-prediction layers are dropped by the sharder
            found.setdefault(layer, {}).setdefault(expert, set()).add(match.group(3))
    moe_layers = range(config["first_k_dense_replace"], config["num_hidden_layers"])
    want = {expert: {"gate", "up", "down"} for expert in range(config["n_routed_experts"])}
    return set(found) == set(moe_layers) and all(found[layer] == want for layer in moe_layers)


def check_definition(
    config: Dict[str, Any],
    weight_names: Iterable[str],
    safetensors_bytes: int,
    expert_std: float,
) -> List[str]:
    """Return the ids of the definition points the checkpoint fails; empty means accepted.

    ``config`` is the parsed ``config.json``, ``weight_names`` the keys of
    ``model.safetensors.index.json``, ``safetensors_bytes`` the sum of the ``.safetensors``
    file sizes, ``expert_std`` the std of a sample of routed expert weights.

    Points 1, 3, 4, 5 and 6 are checked here. Point 2 of the ticket definition (the config loads
    without ``trust_remote_code`` and without edits) is checked by the run itself, which loads the
    checkpoint that way; the ids therefore skip "2".
    """
    failed: List[str] = []
    if config.get("model_type") != "deepseek_v3" or not isinstance(
        config.get("routed_scaling_factor", 1.0), float
    ):
        failed.append("1-config")
    if not _expert_keys_complete(config, weight_names):
        failed.append("3-expert-keys")
    if safetensors_bytes > MAX_SAFETENSORS_BYTES:
        failed.append("4-size")
    if (
        config["num_hidden_layers"] - config["first_k_dense_replace"] < 2
        or config.get("n_shared_experts", 0) < 1
        or config["num_experts_per_tok"] >= config["n_routed_experts"]
    ):
        failed.append("5-moe-shape")
    init_range = config.get("initializer_range", 0.02)
    if abs(expert_std - init_range) / init_range < MIN_STD_DEVIATION:
        failed.append("6-trained-weights")
    return failed


def lora_targets(config: Any) -> Sequence[str]:
    """MLA projections; a config without ``q_lora_rank`` has a single ``q_proj``."""
    if getattr(config, "q_lora_rank", None) is None:
        return tuple(name for name in fixture.TARGETS if name not in ("q_a_proj", "q_b_proj")) + (
            "q_proj",
        )
    return fixture.TARGETS


def default_batches(config: Any, *, count: int = 1, rows: int = 2, seq: int = 6) -> List[Any]:
    """Seeded random token ids with the first tokens of row 0 masked in the labels."""
    batches = []
    for index in range(count):
        batch = fixture.moe_batch(config, seed=index, rows=rows, seq=seq)
        batch["labels"][0, : 1 + index % 3] = -100
        batches.append(batch)
    return batches


def _peak_rss_bytes() -> int:
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    import psutil

    return int(psutil.Process().memory_info().peak_wset)


def _track_routing(model: Any) -> tuple[Dict[int, set], List[Any]]:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    hits: Dict[int, set] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV3TopkRouter):
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            seen = hits.setdefault(layer, set())

            def hook(_module: Any, _args: Any, output: Any, seen: set = seen) -> None:
                seen.update(output[2].flatten().tolist())

            handles.append(module.register_forward_hook(hook))
    return hits, handles


def _double_one_hit_expert(base: Any, hits: Dict[int, set]) -> None:
    import torch

    layer = min(hits)
    with torch.no_grad():
        base.model.layers[layer].mlp.experts.gate_up_proj[min(hits[layer])].mul_(2.0)


def _run_pass(model: Any, batches: Sequence[Any]) -> Dict[str, Any]:
    """One optimizer-free pass over ``batches``: summed loss backward, per-batch logits."""
    model.train()
    model.zero_grad(set_to_none=True)
    logits, losses = [], []
    for batch in batches:
        out = model(**batch)
        out.loss.backward()
        logits.append(out.logits.detach().clone())
        losses.append(out.loss.detach().clone())
    grads = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in fixture.trainables(model).items()
    }
    return {"logits": logits, "losses": losses, "grads": grads}


def _same(got: Any, want: Any, dtype: str) -> bool:
    import torch

    if got is None or want is None:
        return False
    if dtype == "float32":
        return bool(torch.equal(got, want))
    return bool(
        torch.allclose(got.float(), want.float(), atol=BF16_ATOL, rtol=BF16_RTOL)
    )


def _compare(streamed: Dict[str, Any], resident: Dict[str, Any], dtype: str) -> Dict[str, Any]:
    import torch

    logits_ok = all(_same(a, b, dtype) for a, b in zip(streamed["logits"], resident["logits"]))
    loss_ok = all(_same(a, b, dtype) for a, b in zip(streamed["losses"], resident["losses"]))
    names = sorted(set(streamed["grads"]) | set(resident["grads"]))
    mismatches = [
        name for name in names
        if not _same(streamed["grads"].get(name), resident["grads"].get(name), dtype)
    ]
    grads = [g for g in streamed["grads"].values() if g is not None]
    bitwise = (
        logits_ok and loss_ok and not mismatches
        and all(torch.equal(a, b) for a, b in zip(streamed["logits"], resident["logits"]))
        and all(
            torch.equal(streamed["grads"][n], resident["grads"][n]) for n in names
        )
    )
    return {
        "logits_equal": logits_ok,
        "loss_equal": loss_ok,
        "grad_mismatches": mismatches,
        "grads_compared": len(names),
        "grads_nonzero": sum(1 for g in grads if bool(g.abs().max() > 0)),
        "grads_missing": [n for n, g in streamed["grads"].items() if g is None],
        "bitwise": bitwise,
    }


def run_correctness(
    weights: Path,
    work_dir: Path,
    *,
    dtype: str = "float32",
    batches: Optional[Sequence[Any]] = None,
    seed: int = 71,
    keep_cache: bool = False,
    perturb_resident_expert: bool = False,
) -> Dict[str, Any]:
    """Compare the per_expert streamed model with a resident eager model; return evidence.

    ``perturb_resident_expert`` doubles one routed expert, chosen from the experts the router
    sent tokens to, in the resident model only. It is the negative control: the verdict must
    become FAIL, which shows that the comparison sees a wrong expert.
    """
    import torch

    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    if dtype not in ("float32", "bfloat16"):
        raise ValueError(f"dtype must be float32 or bfloat16, got {dtype!r}")
    weights, work_dir = Path(weights), Path(work_dir)
    config = load_stream_config(str(weights), trust_remote_code=False)
    config.use_cache = False
    batches = list(batches) if batches is not None else default_batches(config)
    shards = work_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    lora = fixture.moe_lora_config(targets=lora_targets(config))

    index = shard_checkpoint(
        str(weights), str(shards), dtype=dtype, arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device="cpu", dtype=dtype, buffers=2, pin=False, seed=seed, quant="none",
    )
    try:
        fixture.randomize_adapters(streamed)
        adapters = fixture.snapshot(streamed)
        hits, handles = _track_routing(streamed)
        try:
            streamed_pass = _run_pass(streamed, batches)
        finally:
            for handle in handles:
                handle.remove()
    finally:
        runtime.close()
    del streamed
    gc.collect()
    if not keep_cache:
        shutil.rmtree(shards, ignore_errors=True)
    streamed_peak = _peak_rss_bytes()

    base = fixture.resident_base(weights, dtype=dtype)
    resident = fixture.resident_adapter(base, lora)
    params = fixture.trainables(resident)
    if params.keys() != adapters.keys():
        raise AssertionError("streamed and resident models have different trainable parameters")
    with torch.no_grad():
        for name, param in params.items():
            if param.dtype != adapters[name].dtype:
                raise AssertionError(f"adapter dtype differs for {name}")
            param.copy_(adapters[name])
    if perturb_resident_expert:
        _double_one_hit_expert(base, hits)
    resident_pass = _run_pass(resident, batches)

    result = _compare(streamed_pass, resident_pass, dtype)
    passed = (
        result["logits_equal"] and result["loss_equal"] and not result["grad_mismatches"]
        and not result["grads_missing"] and result["grads_nonzero"] > 0
    )
    control_detected = None
    if not perturb_resident_expert:
        # Negative control: corrupt one resident expert after the comparison and compare again.
        _double_one_hit_expert(base, hits)
        with torch.no_grad():
            corrupted = resident(**batches[0]).logits
        control_detected = not _same(streamed_pass["logits"][0], corrupted, dtype)
    result.update(
        control_detected=control_detected,
        verdict="PASS" if passed and control_detected is not False else "FAIL",
        dtype=dtype,
        experts_hit_by_layer={layer: len(seen) for layer, seen in sorted(hits.items())},
        peak_rss_bytes_after_streamed=streamed_peak,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    return result
