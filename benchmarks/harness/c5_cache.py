"""C5 ticket 05: shard-cache size and expert restore cost, experts in NF4 or Q2_0.

Spec: .scratch/c5-q2-0-experts/issues/05-cache-size-and-restore-cost.md. Rule
(binding, in Russian): benchmarks/gate-c5-q2-0-experts.md, section "Размер кеша
и восстановление весов (тикет 05)". Codec: ``q2_0_codec.py`` (ticket 01).

The shipped sharder (``soup_cli.utils.layer_shard.shard_checkpoint``) runs
unchanged and writes the bf16 cache. It stacks OLMoE's per-expert weights into
``mlp.experts.gate_up_proj [E, 2I, H]`` and ``mlp.experts.down_proj [E, H, I]``
and never quantizes them: its NF4 path covers unstacked decoder keys only. So
this harness copies that cache once per codec and replaces those two tensors of
each layer file with their packed form. Every other file and tensor is copied
as is. Each expert is quantized on its own, as in ticket 04, so the packed
cache restores to exactly the weights the ticket-04 arms trained over.

A packed tensor keeps its key for the codes and adds sidecars, named like the
shipped NF4 sidecars:

- Q2_0: the key holds ``qs`` uint8 [E, blocks, 16], ``::d`` holds fp16 [E, blocks].
- NF4: the key holds the packed nibbles uint8 [E, n/2], ``::absmax`` uint8
  [E, blocks], ``::nested_absmax`` fp32 [E, blocks/256] and ``::nested_offset``
  fp32 [E]. The two code tables go into ``extras.safetensors`` under the
  shipped sharder's keys, as its NF4 path stores them.

Each layer file's metadata names the codec and the shape of every packed
tensor. The copied ``index.json`` is the shipped one and does not describe the
packed experts: Soup's runtime has no packed-expert path and cannot load these
caches. They exist to be measured.

    python benchmarks/harness/c5_cache.py build --model allenai/OLMoE-1B-7B-0924 \\
        --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 --device cuda \\
        --cache-root CACHES --out RESULTS/cache.json
    python benchmarks/harness/c5_cache.py compare --cache-root CACHES \\
        --label "RTX 3090, real" --device cuda --out RESULTS/restore.json
    python benchmarks/harness/c5_cache.py build --fixture olmoe --cache-root CACHES --out X.json

``build`` writes ``bf16/``, ``nf4/`` and ``q2_0/`` under ``CACHES`` and a record
with the bytes of every file, per layer and in total, and its checks: the
non-expert tensors match the shipped cache in both variants, and on OLMoE the
restored experts match ticket 04's ``expert_digest``. ``compare`` runs the
restore timing in A-B-B-A blocks (A = NF4, B = Q2_0) by ``benchmarks/METHOD.md``:
each run is ``restore`` in a fresh process under ``arm_validity.ArmWatch`` with
``suspend`` and ``power`` required, and the block count comes from
``blocks_to_add``. A run loads one layer's packed experts into VRAM and times
restoring all 64 experts (both tensors, so gate, up and down) to bf16, and one
expert alone; its values are the medians of its timed steps. The repeats rule
reads the layer value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console

from soup_cli.utils.layer_shard import (
    ABSMAX_SUFFIX,
    NESTED_ABSMAX_SUFFIX,
    NESTED_OFFSET_SUFFIX,
    NF4_CODE_KEY,
    NF4_NESTED_CODE_KEY,
    extras_shard_path,
    layer_shard_path,
    read_shard_index,
    shard_checkpoint,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arm_validity  # noqa: E402
import c5_expert_swap as swap  # noqa: E402
import compare_shard_caches as shard_files  # noqa: E402
import q2_0_codec as codec  # noqa: E402

if TYPE_CHECKING:
    import torch

console = Console()

CODEC_NAMES = ("nf4", "q2_0")
#: Arm A is the NF4 control, arm B the Q2_0 candidate.
ARMS = {"A": "nf4", "B": "q2_0"}
#: The rule requires ``suspend`` and ``power``; the disk takes no part in a restore.
REQUIRED_CHECKS = ("suspend", "power")
#: Blocks run before the comparison gives up on void runs.
DEFAULT_MAX_BLOCKS = 40
#: The run value the repeats rule reads; ``expert_s`` is reported from the same runs.
RUN_VALUE = "layer_s"
DEFAULT_WARMUP = 5
DEFAULT_STEPS = 30
#: ``expert_digest`` of the ticket-04 runs on OLMoE (every seed of each codec), from
#: benchmarks/results/c5-train/runs/*/run.json. A real build must restore to these.
TICKET_04_EXPERT_DIGESTS = {
    "nf4": "84ae958591eff2f996bc041532aba416e517458e9ef330172392e8b2f4b9fb5c",
    "q2_0": "090f4fbd1d860c4052150ac81a8be767bc03bdcc4a879bbd8a3e8da7b2f33645",
}
EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
SCALE_SUFFIX = "::d"
METADATA_CODEC = "c5_codec"
METADATA_LAYOUT = "c5_layout"


def olmoe_fixture(seed: int = 0, layers: int = 2) -> torch.nn.Module:
    """A tiny random-init OLMoE in bf16. SYNTHETIC: its weights mean nothing."""
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=None,
    )
    torch.manual_seed(seed)
    return OlmoeForCausalLM(config).to(torch.bfloat16).eval()


def _pack_q2_0(packs: list[codec.Packed]):
    import torch

    tensors = {
        "": torch.stack([p.tensors["qs"] for p in packs]),
        SCALE_SUFFIX: torch.stack([p.tensors["d"] for p in packs]),
    }
    return tensors, {}, None


def _pack_nf4(packs: list[codec.Packed]):
    import torch

    first = packs[0].state
    for p in packs[1:]:
        if not (torch.equal(p.state.code, first.code)
                and torch.equal(p.state.state2.code, first.state2.code)):
            raise ValueError("NF4 code tables differ between experts")
    tensors = {
        "": torch.stack([p.tensors["data"].reshape(-1) for p in packs]),
        ABSMAX_SUFFIX: torch.stack([p.tensors["absmax"] for p in packs]),
        NESTED_ABSMAX_SUFFIX: torch.stack([p.tensors["absmax2"] for p in packs]),
        NESTED_OFFSET_SUFFIX: torch.stack([p.tensors["offset"].reshape(()) for p in packs]),
    }
    layout = {"blocksize": first.blocksize, "nested_blocksize": first.state2.blocksize}
    return tensors, layout, (first.code.cpu(), first.state2.code.cpu())


def pack_experts(weight: torch.Tensor, codec_name: str, device: str = "cpu"):
    """Quantize a stacked expert tensor [E, out, in] one expert at a time on ``device``.

    Returns the stored tensors keyed by suffix (``""`` holds the codes), the
    layout for the file's metadata, and the NF4 code tables (None for Q2_0).
    The tensors are on the CPU.
    """
    if codec_name not in CODEC_NAMES:
        raise ValueError(f"unknown codec {codec_name!r}; expected one of {CODEC_NAMES}")
    if codec_name == "nf4" and weight[0].numel() % codec.BLOCK:
        raise ValueError("NF4 experts must hold a multiple of 64 weights each")
    packs = [codec.quantize(expert.to(device), codec_name) for expert in weight]
    pack = _pack_q2_0 if codec_name == "q2_0" else _pack_nf4
    tensors, layout, tables = pack(packs)
    layout = {"shape": list(weight.shape), **layout}
    return {s: t.cpu().contiguous() for s, t in tensors.items()}, layout, tables


def _layer_names(base_dir: str) -> dict[str, int]:
    index = read_shard_index(base_dir)
    return {
        os.path.basename(layer_shard_path(base_dir, idx)): idx for idx in range(index.n_layers)
    }


def _pack_layer_file(source: str, target: str, codec_name: str, device: str):
    """Write ``source`` to ``target`` with its expert tensors packed; the NF4 code tables."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    blob: dict[str, Any] = {}
    layout: dict[str, Any] = {}
    tables = None
    with safe_open(source, framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key not in EXPERT_KEYS:
                blob[key] = tensor
                continue
            sidecars, layout[key], tables = pack_experts(tensor, codec_name, device)
            for suffix, value in sidecars.items():
                blob[key + suffix] = value
    missing = sorted(set(EXPERT_KEYS) - set(layout))
    if missing:
        raise ValueError(f"{source} has no {', '.join(missing)}")
    save_file(blob, target, metadata={METADATA_CODEC: codec_name,
                                      METADATA_LAYOUT: json.dumps(layout)})
    return tables


def write_variant(base_dir: str, out_dir: str, codec_name: str, device: str = "cpu") -> None:
    """Copy the shipped cache in ``base_dir`` to ``out_dir`` with packed experts.

    Layer files get packed experts, NF4 adds its code tables to the extras file,
    and every other file is copied byte for byte.
    """
    from safetensors.torch import load_file, save_file

    os.makedirs(out_dir, exist_ok=True)
    layers = _layer_names(base_dir)
    tables = None
    for name in sorted(layers, key=layers.get):
        layer_tables = _pack_layer_file(
            os.path.join(base_dir, name), os.path.join(out_dir, name), codec_name, device
        )
        if tables is not None and not all(a.equal(b) for a, b in zip(tables, layer_tables)):
            raise ValueError(f"NF4 code tables differ in {name}")
        tables = layer_tables
    extras = os.path.basename(extras_shard_path(base_dir))
    for name in sorted(set(os.listdir(base_dir)) - set(layers)):
        source, target = os.path.join(base_dir, name), os.path.join(out_dir, name)
        if name == extras and tables is not None:
            blob = load_file(source)
            blob[NF4_CODE_KEY], blob[NF4_NESTED_CODE_KEY] = tables
            save_file(blob, target)
        elif os.path.isfile(source):
            shutil.copyfile(source, target)


@dataclass
class LayerPack:
    """One layer's packed experts on a device, ready to restore."""

    codec: str
    tensors: dict[str, Any]
    layout: dict[str, dict[str, Any]]
    tables: tuple[Any, Any] | None = None


def load_layer(cache_dir: str, layer: int, device: str = "cpu") -> LayerPack:
    """The packed expert tensors of decoder layer ``layer``, moved to ``device``."""
    from safetensors import safe_open

    with safe_open(layer_shard_path(cache_dir, layer), framework="pt") as handle:
        metadata = handle.metadata() or {}
        codec_name = metadata[METADATA_CODEC]
        layout = json.loads(metadata[METADATA_LAYOUT])
        tensors = {
            key: handle.get_tensor(key).to(device)
            for key in handle.keys()
            if key.split("::", 1)[0] in EXPERT_KEYS
        }
    tables = None
    if codec_name == "nf4":
        with safe_open(extras_shard_path(cache_dir), framework="pt") as handle:
            tables = (handle.get_tensor(NF4_CODE_KEY).to(device),
                      handle.get_tensor(NF4_NESTED_CODE_KEY).to(device))
    return LayerPack(codec_name, tensors, layout, tables)


def _restore_q2_0(pack: LayerPack, key: str, experts: slice) -> torch.Tensor:
    """``(q - 1) * d`` in float32 for every expert at once, then bf16, as the codec does."""
    import torch

    qs = pack.tensors[key][experts]
    d = pack.tensors[key + SCALE_SUFFIX][experts]
    count, blocks = d.shape
    codes = (qs[..., None] >> codec._shifts(qs.device)) & 3
    values = (codes.view(count, blocks, codec.BLOCK).to(torch.float32) - 1) * d.to(
        torch.float32
    )[..., None]
    shape = pack.layout[key]["shape"][1:]
    per_expert = 1
    for dim in shape:
        per_expert *= dim
    return values.view(count, -1)[:, :per_expert].reshape(count, *shape).to(torch.bfloat16)


def _restore_nf4(pack: LayerPack, key: str, experts: slice) -> torch.Tensor:
    """bitsandbytes' own two steps for every expert at once: the absmax, then the nibbles.

    The nested absmax of each expert is restored with its own offset. When an
    expert's absmax count is a multiple of the nested block, the experts'
    blocks line up and one call covers them all; otherwise it is one call per
    expert.
    """
    import bitsandbytes.functional as bnb
    import torch

    layout = pack.layout[key]
    code, nested_code = pack.tables
    absmax = pack.tensors[key + ABSMAX_SUFFIX][experts]
    nested = pack.tensors[key + NESTED_ABSMAX_SUFFIX][experts]
    offset = pack.tensors[key + NESTED_OFFSET_SUFFIX][experts]
    count, blocks = absmax.shape
    nested_blocksize = layout["nested_blocksize"]
    if blocks % nested_blocksize == 0:
        scales = bnb.dequantize_blockwise(
            absmax.reshape(-1), absmax=nested.reshape(-1), code=nested_code,
            blocksize=nested_blocksize,
        ).view(count, blocks)
    else:
        scales = torch.stack([
            bnb.dequantize_blockwise(
                absmax[e], absmax=nested[e], code=nested_code, blocksize=nested_blocksize
            )
            for e in range(count)
        ])
    scales = (scales + offset[:, None]).to(torch.float32).reshape(-1)
    state = bnb.QuantState(
        absmax=scales, shape=torch.Size([count, *layout["shape"][1:]]), code=code,
        blocksize=layout["blocksize"], quant_type="nf4", dtype=torch.bfloat16,
    )
    return bnb.dequantize_4bit(pack.tensors[key][experts].reshape(-1, 1), quant_state=state)


def restore_layer(pack: LayerPack, experts: slice = slice(None)) -> dict[str, torch.Tensor]:
    """Restore the packed experts ``experts`` of ``pack`` to bf16, on its device."""
    restore = {"q2_0": _restore_q2_0, "nf4": _restore_nf4}[pack.codec]
    return {key: restore(pack, key, experts) for key in EXPERT_KEYS}


def _tensor_bytes(path: str) -> dict[str, int]:
    """Data bytes of each tensor in a safetensors file, from its header."""
    _, header = shard_files.read_header(path)
    header.pop("__metadata__", None)
    return {name: end - start for name, (start, end) in
            ((name, spec["data_offsets"]) for name, spec in header.items())}


def cache_sizes(cache_dir: str) -> dict[str, Any]:
    """What a cache costs on disk and in a RAM store, read from its files.

    ``files`` and ``ssd_bytes`` are file sizes from ``os.stat``. ``ram_store_bytes``
    is the data bytes of every tensor, which is what a RAM tier holding the whole
    cache keeps; safetensors headers and ``index.json`` are not in it. Per layer:
    the file size and the data bytes of its expert tensors and their sidecars.
    """
    files = {name: os.stat(os.path.join(cache_dir, name)).st_size
             for name in sorted(os.listdir(cache_dir))}
    ram = 0
    for name in files:
        if name.endswith(".safetensors"):
            ram += sum(_tensor_bytes(os.path.join(cache_dir, name)).values())
    layers = []
    for name, idx in sorted(_layer_names(cache_dir).items(), key=lambda item: item[1]):
        tensors = _tensor_bytes(os.path.join(cache_dir, name))
        layers.append({
            "layer": idx,
            "file": name,
            "file_bytes": files[name],
            "expert_bytes": sum(
                nbytes for key, nbytes in tensors.items()
                if key.split("::", 1)[0] in EXPERT_KEYS
            ),
        })
    return {"files": files, "ssd_bytes": sum(files.values()), "ram_store_bytes": ram,
            "layers": layers}


def _is_expert_key(key: str) -> bool:
    """An expert tensor, one of its sidecars, or an NF4 code table."""
    return key.split("::", 1)[0] in EXPERT_KEYS or key.startswith("__nf4_")


def _non_expert_digests(cache_dir: str) -> dict[str, str]:
    """SHA-256 of every non-expert tensor as ``file:key``, and of every other file."""
    out: dict[str, str] = {}
    for name in sorted(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, name)
        if not name.endswith(".safetensors"):
            out[name] = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
            continue
        for key, spec in shard_files.tensor_digests(path).items():
            if not _is_expert_key(key):
                out[f"{name}:{key}"] = f"{spec['dtype']}{spec['shape']}{spec['sha256']}"
    return out


def non_expert_mismatch(left_dir: str, right_dir: str) -> list[str]:
    """Non-expert tensors (``file:key``) and other files that differ between two caches,
    or that only one of them has."""
    left, right = _non_expert_digests(left_dir), _non_expert_digests(right_dir)
    return sorted(name for name in left.keys() | right.keys() if left.get(name) != right.get(name))


def expert_digest(cache_dir: str, device: str = "cpu") -> str:
    """Ticket 04's ``expert_digest`` of the weights this cache restores to.

    The SHA-256 of the JSON of every restored expert tensor's fingerprint
    (``c5_expert_swap._sha256``) under its model parameter name.
    """
    fingerprints = {}
    for name, idx in _layer_names(cache_dir).items():
        restored = restore_layer(load_layer(cache_dir, idx, device))
        for key, tensor in restored.items():
            fingerprints[f"model.layers.{idx}.{key}"] = swap._sha256(tensor)
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()


def build_caches(weights_dir: str, cache_root: str, device: str = "cpu") -> dict[str, Any]:
    """Shard ``weights_dir`` with the shipped sharder, then write both packed variants."""

    base = os.path.join(cache_root, "bf16")
    shard_checkpoint(weights_dir, base, dtype="bfloat16")
    for codec_name in CODEC_NAMES:
        write_variant(base, os.path.join(cache_root, codec_name), codec_name, device)
    variants = {v: cache_sizes(os.path.join(cache_root, v)) for v in ("bf16", *CODEC_NAMES)}
    mismatch = []
    for codec_name in CODEC_NAMES:
        folder = os.path.join(cache_root, codec_name)
        variants[codec_name]["expert_digest"] = expert_digest(folder, device)
        mismatch += [f"{codec_name} {name}" for name in non_expert_mismatch(base, folder)]
    return {
        "variants": variants,
        "non_expert_mismatch": mismatch,
        "checks": {"non_expert_identical": not mismatch},
    }


def make_watch(label: str, cache_dir: str):
    """The ArmWatch factory of a restore comparison, with the rule's required checks."""

    def watch(*, arm: str, round: int, run: int) -> arm_validity.ArmWatch:
        return arm_validity.ArmWatch(
            arm=arm, round=round, run=run, label=label, required=REQUIRED_CHECKS,
            model_path=cache_dir,
        )

    return watch


def _as_runs(blocks: list[list[dict[str, Any]]]) -> list[list[arm_validity.Run]]:
    return [[arm_validity.Run(run["value"][RUN_VALUE], run["outcome"]) for run in block]
            for block in blocks]


def _summary(blocks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Per arm and per value: mean, median, min and max over the runs of valid blocks."""
    import statistics

    valid = [block for block in blocks if all(r["outcome"] != arm_validity.VOID for r in block)]
    out: dict[str, Any] = {}
    for arm in ARMS:
        out[arm] = {}
        for name in (RUN_VALUE, "expert_s"):
            values = [r["value"][name] for block in valid for r in block if r["arm"] == arm]
            out[arm][name] = {
                "n": len(values), "mean": statistics.fmean(values),
                "median": statistics.median(values), "min": min(values), "max": max(values),
            } if values else None
    return out


def run_comparison(runner, watch, max_blocks: int = DEFAULT_MAX_BLOCKS) -> dict[str, Any]:
    """A-B-B-A blocks of restore runs until ``arm_validity.blocks_to_add`` asks for none.

    ``runner(arm, round, run)`` runs one arm in a fresh process and returns its
    medians, ``layer_s`` and ``expert_s``. ``watch(arm=, round=, run=)`` is the
    arm's ArmWatch. A block with a void run is kept and replaced. After
    ``max_blocks`` blocks the comparison stops and is marked incomplete.
    """
    blocks: list[list[dict[str, Any]]] = []
    while arm_validity.blocks_to_add(_as_runs(blocks)) > 0 and len(blocks) < max_blocks:
        block = []
        for position, arm in enumerate(arm_validity.block_order(1)[0]):
            round, run = 2 * len(blocks) + 1 + position // 2, 4 * len(blocks) + position + 1
            with watch(arm=arm, round=round, run=run) as watched:
                value = runner(arm, round, run)
            record = watched.record
            block.append({"arm": arm, "codec": ARMS[arm], "round": round, "run": run,
                          "value": value, "outcome": str(record["outcome"]), "record": record})
        blocks.append(block)
    runs = _as_runs(blocks)
    complete = arm_validity.blocks_to_add(runs) == 0
    repeats = arm_validity.repeats(runs) if complete else None
    return {
        "arms": ARMS,
        "run_value": RUN_VALUE,
        "required_checks": list(REQUIRED_CHECKS),
        "complete": complete,
        "valid_blocks": len(arm_validity.valid_blocks(runs)),
        "repeats": None if repeats is None else {
            "sigma": repeats.sigma, "mean_a": repeats.mean_a,
            "blocks_needed": repeats.blocks_needed,
        },
        "summary": _summary(blocks),
        "blocks": blocks,
    }


def harness_sha256() -> str:
    """SHA-256 of this file, so a record names the code that produced it."""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def _provenance(device: str) -> dict[str, Any]:
    import c5_train

    return {
        "versions": c5_train._versions(),
        "commit": c5_train._commit(),
        "gpu": c5_train.gpu_info(device),
        "harness": {"sha256": harness_sha256(), "codec_sha256": codec.codec_sha256()},
        "command": [sys.executable, *sys.argv],
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def plain_files(snapshot: str, out_dir: str) -> str:
    """Real files for the symlinks of a Hugging Face snapshot folder.

    The shipped sharder skips symlinked shards, and the Hub cache stores every
    file as a link into ``blobs/``. Each file is hard-linked (copied when the
    folders sit on different file systems), so nothing is downloaded twice.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(snapshot)):
        source = os.path.realpath(os.path.join(snapshot, name))
        target = os.path.join(out_dir, name)
        if os.path.lexists(target):
            os.remove(target)
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    return out_dir


def _weights(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """The checkpoint folder to shard, and what the record says about it."""
    if args.fixture:
        weights = os.path.join(args.cache_root, "weights")
        olmoe_fixture(seed=0).save_pretrained(weights)
        return weights, {
            "id": "fixture:olmoe", "revision": None, "synthetic": True,
            "note": "SYNTHETIC: a tiny random-init OLMoE built from a config, seed 0. "
                    "Its sizes and times say nothing about a real model.",
        }
    from huggingface_hub import snapshot_download

    weights = snapshot_download(
        args.model, revision=args.revision, allow_patterns=["*.safetensors", "*.json"]
    )
    weights = plain_files(weights, os.path.join(args.cache_root, "weights"))
    return weights, {"id": args.model, "revision": args.revision, "synthetic": False,
                     "note": ""}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> int:
    record = {"synthetic": bool(args.fixture), **_provenance(args.device), "device": args.device}
    weights, record["model"] = _weights(args)
    start = time.perf_counter()
    record.update(build_caches(weights, args.cache_root, device=args.device))
    record["build_s"] = time.perf_counter() - start
    expected = {} if args.fixture else TICKET_04_EXPERT_DIGESTS
    for codec_name, digest in expected.items():
        record["checks"][f"{codec_name}_expert_digest_matches_ticket_04"] = (
            record["variants"][codec_name]["expert_digest"] == digest
        )
    _write_json(args.out, record)
    for variant, sizes in record["variants"].items():
        console.print(f"{variant}: {sizes['ssd_bytes']:,} bytes on disk, "
                      f"{sizes['ram_store_bytes']:,} bytes of tensors")
    failed = [name for name, ok in record["checks"].items() if not ok]
    console.print(f"failed checks: {failed or 'none'} -> {args.out}")
    return 1 if failed else 0


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def time_restore(pack: LayerPack, device: str, experts: slice) -> float:
    """Seconds for one restore of ``experts`` of ``pack``, the GPU work included."""
    _sync(device)
    start = time.perf_counter()
    restored = restore_layer(pack, experts)
    _sync(device)
    elapsed = time.perf_counter() - start
    del restored
    return elapsed


def cmd_restore(args: argparse.Namespace) -> int:
    """One run in its own process: load one layer's packed experts, time its restores."""
    import statistics

    pack = load_layer(args.cache_dir, args.layer, device=args.device)
    _sync(args.device)
    steps: dict[str, list[float]] = {"layer_s": [], "expert_s": []}
    for step in range(args.warmup + args.steps):
        layer = time_restore(pack, args.device, slice(None))
        expert = time_restore(pack, args.device, slice(args.expert, args.expert + 1))
        if step >= args.warmup:
            steps["layer_s"].append(layer)
            steps["expert_s"].append(expert)
    result = {
        "codec": pack.codec, "layer": args.layer, "expert": args.expert,
        "warmup": args.warmup, "timed_steps": args.steps,
        "layer_s": statistics.median(steps["layer_s"]),
        "expert_s": statistics.median(steps["expert_s"]),
        "steps": steps,
        "packed_bytes": sum(t.numel() * t.element_size() for t in pack.tensors.values()),
    }
    if args.device.startswith("cuda"):
        import torch

        result["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    _write_json(args.out, result)
    return 0


class RestoreRun:
    """A runner that starts ``c5_cache.py restore`` in a fresh process for each run."""

    def __init__(self, args: argparse.Namespace, runs_dir: str) -> None:
        self.args = args
        self.runs_dir = runs_dir

    def __call__(self, arm: str, round: int, run: int) -> dict[str, float]:
        out = os.path.join(self.runs_dir, f"run{run:03d}-{arm}-{ARMS[arm]}.json")
        command = [
            sys.executable, __file__, "restore",
            "--cache-dir", os.path.join(self.args.cache_root, ARMS[arm]),
            "--layer", str(self.args.layer), "--expert", str(self.args.expert),
            "--device", self.args.device, "--warmup", str(self.args.warmup),
            "--steps", str(self.args.steps), "--out", out,
        ]  # fmt: skip
        console.print(f"[dim]run {run} (arm {arm} {ARMS[arm]}, round {round})[/]")
        subprocess.run(command, check=True)
        result = json.loads(pathlib.Path(out).read_text(encoding="utf-8"))
        return {"layer_s": result["layer_s"], "expert_s": result["expert_s"]}


def cmd_compare(args: argparse.Namespace) -> int:
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    record = {"synthetic": args.synthetic, "label": args.label, **_provenance(args.device),
              "settings": {"layer": args.layer, "expert": args.expert, "warmup": args.warmup,
                           "steps": args.steps, "device": args.device,
                           "cache_root": args.cache_root, "max_blocks": args.max_blocks}}
    record.update(run_comparison(RestoreRun(args, runs_dir),
                                 make_watch(args.label, args.cache_root),
                                 max_blocks=args.max_blocks))
    _write_json(args.out, record)
    for arm, values in record["summary"].items():
        if values[RUN_VALUE] is not None:
            console.print(f"arm {arm} ({ARMS[arm]}): layer median of runs "
                          f"{values[RUN_VALUE]['median'] * 1e3:.3f} ms, expert "
                          f"{values['expert_s']['median'] * 1e3:.4f} ms")
    console.print(f"blocks: {len(record['blocks'])}, valid: {record['valid_blocks']}, "
                  f"complete: {record['complete']} -> {args.out}")
    return 0 if record["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="shard the model and write both packed caches")
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model id")
    source.add_argument("--fixture", choices=("olmoe",), help="tiny random-init OLMoE")
    build.add_argument("--revision", help="commit to download; required with --model")
    build.add_argument("--cache-root", required=True, help="writes bf16/, nf4/ and q2_0/ here")
    build.add_argument("--device", default="cpu", help="where experts are quantized")
    build.add_argument("--out", required=True, help="the size record (JSON)")

    for name, helptext in (("restore", "one timed run in this process"),
                           ("compare", "A-B-B-A blocks of restore runs")):
        cmd = sub.add_parser(name, help=helptext)
        if name == "restore":
            cmd.add_argument("--cache-dir", required=True, help="one packed cache")
        else:
            cmd.add_argument("--cache-root", required=True, help="holds nf4/ and q2_0/")
            cmd.add_argument("--label", required=True, help='e.g. "RTX 3090, real"')
            cmd.add_argument("--synthetic", action="store_true", help="label a fixture run")
            cmd.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
        cmd.add_argument("--layer", type=int, default=0, help="decoder layer to restore")
        cmd.add_argument("--expert", type=int, default=0, help="expert timed on its own")
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
        cmd.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        cmd.add_argument("--out", required=True, help="JSON to write")

    args = parser.parse_args(argv)
    if args.command == "build" and args.model and not args.revision:
        parser.error("--model needs --revision: the record must name the exact weights")
    return {"build": cmd_build, "restore": cmd_restore, "compare": cmd_compare}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
