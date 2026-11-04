"""G5: full shard-cache bytes of all-NF4, all-Q2_0 and per-expert mixed maps.

Rule: benchmarks/gate-g5-mixed-precision-experts.md. No cache is written here:
every number is computed from tensor shapes with the safetensors file layout
(8-byte header length, JSON header padded with spaces to a multiple of 8, then
the tensor data). For all-NF4 and all-Q2_0 the layer files are C5's format
(``c5_cache.py``) and the computed sizes equal the bytes C5 measured on OLMoE
(``benchmarks/results/c5-cache/cache.json``); ``--cache-json`` checks that.

A mixed layer file, as G5 defines it (computed only, no writer exists):

- non-expert tensors as in C5, in the model dtype;
- per stacked expert tensor ``K`` [E, out, in]: the NF4 experts packed as C5's
  NF4 under ``K``, ``K::absmax``, ``K::nested_absmax``, ``K::nested_offset``,
  and the Q2_0 experts packed as C5's Q2_0 under ``K::q2_0`` and ``K::q2_0::d``;
- metadata: C5's ``c5_codec`` (``mixed``) and ``c5_layout``, plus the codec map
  ``g5_codec_map``, one character per expert (``n`` NF4, ``q`` Q2_0). An
  expert's row in its group is its rank among the experts of its codec, so
  the map is the whole index.

The NF4 code tables go into the extras file whenever any expert is NF4, as in
C5. Files outside the layers come from C5's measurement where given
(``measured``), otherwise from the size model (``computed``).

    python benchmarks/harness/g5_bytes.py --model allenai/OLMoE-1B-7B-0924 \\
        --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \\
        --cache-json benchmarks/results/c5-cache/cache.json --out OUT/bytes.json
    python benchmarks/harness/g5_bytes.py --fixture granite --out OUT/bytes.json

The CLI builds the model on the meta device, so it reads only the config.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    import torch

console = Console()

#: Weights per quantization block (Q2_0 and NF4) and absmax values per NF4 nested block.
BLOCK = 64
NESTED_BLOCK = 256
#: safetensors writes tensors largest dtype first, then by name; this is its dtype order.
DTYPE_ORDER = ("BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16", "I32",
               "U32", "F32", "F64", "I64", "U64")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8}
#: C5's NF4 code tables in the extras file: 16 and 256 fp32 values.
NF4_TABLES = {"__nf4_code": ("F32", (16,)), "__nf4_nested_code": ("F32", (256,))}
CODE_LETTERS = {"nf4": "n", "q2_0": "q"}
SCALE_SUFFIXES = ("::absmax", "::nested_absmax", "::nested_offset", "::d")
_LAYER_PREFIX = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(.+)$")

Entry = tuple[str, tuple[int, ...]]


def nbytes(entry: Entry) -> int:
    dtype, shape = entry
    return DTYPE_BYTES[dtype] * math.prod(shape)


def safetensors_size(
    entries: dict[str, Entry], metadata: dict[str, str] | None = None
) -> dict[str, int]:
    """Bytes of a safetensors file holding ``entries``: header (with its length prefix) and data."""
    order = sorted(entries.items(), key=lambda item: (-DTYPE_ORDER.index(item[1][0]), item[0]))
    header: dict[str, Any] = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, entry in order:
        header[name] = {"dtype": entry[0], "shape": list(entry[1]),
                        "data_offsets": [offset, offset + nbytes(entry)]}
        offset += nbytes(entry)
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    padded = len(text) + (-len(text)) % 8
    return {"header": 8 + padded, "data": offset, "file": 8 + padded + offset}


def packed_entries(codec_name: str, key: str, shape: Sequence[int], count: int,
                   prefix: str = "") -> dict[str, Entry]:
    """C5's packed form of ``count`` experts of a stacked tensor ``shape`` [E, out, in]."""
    n = math.prod(shape[1:])
    blocks = -(-n // BLOCK)
    base = key + prefix
    if codec_name == "q2_0":
        return {base: ("U8", (count, blocks, 16)), base + "::d": ("F16", (count, blocks))}
    if codec_name == "nf4":
        if n % BLOCK:
            raise ValueError(f"{key}: NF4 experts must hold a multiple of {BLOCK} weights each")
        return {
            base: ("U8", (count, n // 2)),
            base + "::absmax": ("U8", (count, blocks)),
            base + "::nested_absmax": ("F32", (count, -(-blocks // NESTED_BLOCK))),
            base + "::nested_offset": ("F32", (count,)),
        }
    raise ValueError(f"unknown codec {codec_name!r}")


def layer_kind(codes: Sequence[str]) -> str:
    """``nf4`` or ``q2_0`` when every expert has that codec, else ``mixed``."""
    unknown = set(codes) - set(CODE_LETTERS)
    if unknown:
        raise ValueError(f"unknown codecs in a map: {sorted(unknown)}")
    kinds = set(codes)
    return kinds.pop() if len(kinds) == 1 else "mixed"


def layer_entries(
    tensors: dict[str, Entry], expert_keys: Sequence[str], codes: Sequence[str]
) -> tuple[dict[str, Entry], dict[str, str]]:
    """The tensors and metadata of one layer file with its experts packed by ``codes``.

    ``tensors`` holds every tensor of the layer as stored unpacked; the expert
    keys among them are stacked [E, out, in] with E == len(codes).
    """
    kind = layer_kind(codes)
    count = {name: list(codes).count(name) for name in CODE_LETTERS}
    entries = {key: entry for key, entry in tensors.items() if key not in expert_keys}
    layout: dict[str, dict[str, Any]] = {}
    for key in sorted(expert_keys):
        shape = tensors[key][1]
        if shape[0] != len(codes):
            raise ValueError(f"{key} has {shape[0]} experts, the map {len(codes)}")
        layout[key] = {"shape": list(shape)}
        if count["nf4"]:
            layout[key].update(blocksize=BLOCK, nested_blocksize=NESTED_BLOCK)
            entries.update(packed_entries("nf4", key, shape, count["nf4"]))
        if count["q2_0"]:
            prefix = "::q2_0" if kind == "mixed" else ""
            entries.update(packed_entries("q2_0", key, shape, count["q2_0"], prefix))
    metadata = {"c5_codec": kind, "c5_layout": json.dumps(layout)}
    if kind == "mixed":
        metadata["g5_codec_map"] = "".join(CODE_LETTERS[code] for code in codes)
    return entries, metadata


def layer_file(
    tensors: dict[str, Entry], expert_keys: Sequence[str], codes: Sequence[str]
) -> dict[str, Any]:
    """Bytes of one layer file with its experts packed by ``codes`` (one codec per expert)."""
    entries, metadata = layer_entries(tensors, expert_keys, codes)
    kind = metadata["c5_codec"]
    codec_map = metadata.get("g5_codec_map", "")
    size = safetensors_size(entries, metadata)
    expert_names = [name for name in entries if name.split("::", 1)[0] in expert_keys]
    scales = sum(nbytes(entries[name]) for name in expert_names
                 if name.endswith(SCALE_SUFFIXES))
    payload = sum(nbytes(entries[name]) for name in expert_names) - scales
    return {
        "codec": kind,
        "file": size["file"],
        "header": size["header"],
        "payload": payload,
        "scales": scales,
        "non_expert": size["data"] - payload - scales,
        # the bytes the map adds to the header: "g5_codec_map":"nq..", and the comma
        "codec_map": len(json.dumps({"g5_codec_map": codec_map}, separators=(",", ":"))) - 1
        if codec_map else 0,
    }


def inventory(model: torch.nn.Module, expert_names: Sequence[str]) -> dict[str, Any]:
    """The tensors a shard cache stores, from a model's parameters, in bf16.

    ``layers[i]`` maps the key inside layer file i (``mlp.experts.down_proj``)
    to its entry; ``rest`` holds the parameters outside the decoder layers.
    ``expert_keys`` are the layer keys of ``expert_names`` (C5's expert rule).
    """
    layers: dict[int, dict[str, Entry]] = {}
    rest: dict[str, Entry] = {}
    expert_keys: set[str] = set()
    for name, param in model.named_parameters():
        dtype = "BF16" if param.dtype.is_floating_point else str(param.dtype)
        entry = (dtype, tuple(int(d) for d in param.shape))
        match = _LAYER_PREFIX.match(name)
        if match is None:
            rest[name] = entry
            continue
        layers.setdefault(int(match.group(1)), {})[match.group(2)] = entry
        if name in expert_names:
            expert_keys.add(match.group(2))
    return {"layers": [layers[i] for i in sorted(layers)], "rest": rest,
            "expert_keys": sorted(expert_keys)}


def _rest_files(
    inv: dict[str, Any], any_nf4: bool, cache: dict[str, Any] | None
) -> dict[str, Any]:
    """Everything but the layer files: C5's measured files, or one computed file."""
    if cache is not None:
        variant = cache["variants"]["nf4" if any_nf4 else "q2_0"]
        layer_files = {layer["file"] for layer in variant["layers"]}
        files = {name: size for name, size in variant["files"].items() if name not in layer_files}
        return {"source": "measured", "c5_variant": "nf4" if any_nf4 else "q2_0",
                "files": files, "bytes": sum(files.values())}
    entries = dict(inv["rest"])
    if any_nf4:
        entries.update(NF4_TABLES)
    size = safetensors_size(entries)
    return {"source": "computed", "c5_variant": None,
            "files": {"rest.safetensors": size["file"]}, "bytes": size["file"],
            "note": "one file of the non-layer parameters; the sharder's split of them "
                    "and its index.json are not modelled"}


def arm_bytes(
    inv: dict[str, Any], codes: Sequence[Sequence[str]], cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full cache bytes of one arm whose layer i stores its experts by ``codes[i]``."""
    if len(codes) != len(inv["layers"]):
        raise ValueError(f"the map has {len(codes)} layers, the model {len(inv['layers'])}")
    layers = [layer_file(tensors, inv["expert_keys"], layer_codes)
              for tensors, layer_codes in zip(inv["layers"], codes)]
    any_nf4 = any("nf4" in layer_codes for layer_codes in codes)
    rest = _rest_files(inv, any_nf4, cache)
    expert_weights = sum(
        math.prod(tensors[key][1]) for tensors in inv["layers"] for key in inv["expert_keys"]
    )
    total = {key: sum(layer[key] for layer in layers)
             for key in ("file", "header", "payload", "scales", "non_expert", "codec_map")}
    return {
        "q2_0_experts": sum(list(layer_codes).count("q2_0") for layer_codes in codes),
        "experts": sum(len(layer_codes) for layer_codes in codes),
        "layers": layers,
        "rest": rest,
        "layer_files_bytes": total["file"],
        "expert_payload_bytes": total["payload"],
        "expert_scale_bytes": total["scales"],
        "codec_map_bytes": total["codec_map"],
        "layer_header_bytes": total["header"],
        "non_expert_layer_bytes": total["non_expert"],
        "ssd_bytes": total["file"] + rest["bytes"],
        "expert_bits_per_weight": 8 * (total["payload"] + total["scales"]) / expert_weights,
    }


def uniform_codes(inv: dict[str, Any], codec_name: str) -> list[list[str]]:
    """Every expert of every layer in ``codec_name``."""
    n_experts = inv["layers"][0][inv["expert_keys"][0]][1][0]
    return [[codec_name] * n_experts for _ in inv["layers"]]


def check_against_c5(arms: dict[str, Any], cache: dict[str, Any]) -> dict[str, bool]:
    """Computed all-NF4 and all-Q2_0 layer files and totals against C5's measured bytes."""
    checks = {}
    for arm, variant in (("all_nf4", "nf4"), ("all_q2_0", "q2_0")):
        measured = cache["variants"][variant]
        checks[f"{arm}_layer_files_match_c5"] = [
            layer["file"] for layer in arms[arm]["layers"]
        ] == [layer["file_bytes"] for layer in measured["layers"]]
        checks[f"{arm}_ssd_bytes_match_c5"] = arms[arm]["ssd_bytes"] == measured["ssd_bytes"]
    return checks


def accounting(
    inv: dict[str, Any], maps: dict[str, dict[str, Any]], cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bytes of all-NF4, all-Q2_0 and every map in ``maps`` (arm name to its map record).

    Each arm says where its ``ssd_bytes`` come from: ``measured`` (C5's
    cache.json, which the computed number must equal) or ``computed``.
    """
    arms = {"all_nf4": arm_bytes(inv, uniform_codes(inv, "nf4"), cache),
            "all_q2_0": arm_bytes(inv, uniform_codes(inv, "q2_0"), cache)}
    for arm, codec_map in maps.items():
        arms[arm] = {**arm_bytes(inv, codec_map["codes"], cache),
                     "map_sha256": codec_map["map_sha256"]}
    checks = check_against_c5(arms, cache) if cache is not None else {}
    for arm, entry in arms.items():
        uniform = arm in ("all_nf4", "all_q2_0")
        entry["ssd_bytes_source"] = (
            "measured by C5 (cache.json), equal to the computed size"
            if uniform and checks.get(f"{arm}_ssd_bytes_match_c5") else "computed"
        )
    record: dict[str, Any] = {
        "kind": "g5_bytes",
        "source": "layer files computed from tensor shapes; files outside the layers "
                  + ("measured by C5 (cache.json)" if cache else "computed"),
        "arms": arms,
    }
    budgets: dict[Any, set[int]] = {}
    for arm, codec_map in maps.items():
        budgets.setdefault(codec_map.get("fraction_q2_0"), set()).add(arms[arm]["ssd_bytes"])
    record["budgets"] = {str(fraction): sorted(sizes) for fraction, sizes in budgets.items()}
    checks = {**checks, "mixed_arms_of_one_fraction_equal_bytes": all(
        len(sizes) == 1 for sizes in budgets.values())}
    if cache is not None:
        record["c5_bf16_ssd_bytes"] = cache["variants"]["bf16"]["ssd_bytes"]
    record["checks"] = checks
    return record


def load_inventory(args: argparse.Namespace) -> dict[str, Any]:
    """The model's tensors without loading weights: meta device for a real model."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import c5_expert_swap as swap
    import c5_train

    if args.fixture:
        tokenizer = c5_train.byte_tokenizer()
        model = swap.granite_fixture(seed=0, vocab_size=len(tokenizer))
    else:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(args.model, revision=args.revision)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    return inventory(model, swap.expert_tensor_names(model))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="model id or local path; only its config is read")
    source.add_argument("--fixture", choices=("granite",), help="C5's tiny granitemoe")
    parser.add_argument("--revision", help="commit of the model; required with --model")
    parser.add_argument("--map", action="append", default=[],
                        help="a maps/<arm>.json from g5_select.py; repeatable")
    parser.add_argument("--cache-json", help="C5's cache.json, for the measured files")
    parser.add_argument("--out", required=True, help="where bytes.json is written")
    args = parser.parse_args(argv)
    if args.model and not args.revision:
        parser.error("--model needs --revision: the record must name the exact weights")
    inv = load_inventory(args)
    maps = {}
    for path in args.map:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        maps[record["arm"]] = record
    cache = None
    if args.cache_json:
        cache = json.loads(Path(args.cache_json).read_text(encoding="utf-8"))
    record = accounting(inv, maps, cache)
    # the fixture's bytes say nothing about OLMoE; a verdict refuses to mix them with real ones
    record["synthetic"] = bool(args.fixture)
    record["model"] = {"id": f"fixture:{args.fixture}" if args.fixture else args.model,
                       "revision": None if args.fixture else args.revision}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    label = "SYNTHETIC " if args.fixture else ""
    for arm, entry in record["arms"].items():
        console.print(f"{label}{arm}: {entry['ssd_bytes']:,} B on disk, experts "
                      f"{entry['expert_bits_per_weight']:.3f} bits per weight, "
                      f"codec map {entry['codec_map_bytes']} B")
    failed = [name for name, ok in record.get("checks", {}).items() if not ok]
    if failed:
        console.print(f"[red]computed bytes differ from C5's measurement[/]: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
