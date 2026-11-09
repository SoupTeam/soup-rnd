"""B5 ticket 05: a synthetic MoE store of random NF4 experts, in the format of the real cache.

The store has the files and the ``index.json`` that ``shard_checkpoint(..., expert_layout=
"per_expert", quant="nf4")`` writes for a DeepSeek-V3-family model: ``layer_NNN.safetensors``,
``experts_NNN.safetensors``, ``extras.safetensors``, the two large-weight files and the index.
The real loader (``read_shard_index``, ``ExpertShardReader``, ``build_streamed_model``) reads it
with no change. Quantised weights are random packed nibbles plus random block statistics, drawn
directly, so a DeepSeek-V3 shape (hundreds of GB) is written in the time the disk needs and no
bf16 source checkpoint exists. The weights are random: they say nothing about the quality or the
speed of a real model, and every table and name here carries ``SIGNATURE``.

Rules fixed before the first run (ticket 05):

* The plan (:func:`plan_store`) computes every file's bytes from the shape before anything is
  written. It must match the bytes on disk within ``SIZE_TOLERANCE`` of the actual size. The
  tensor bytes are exact; the plan only estimates each safetensors header from the tensor names,
  dtypes and shapes in the order the writer lists them, and writes the three small files into a
  temporary folder to measure them. 1 percent is far above the few bytes a header can differ by
  and far below any real error in a shape formula.
* The writer refuses when the free bytes at the destination are below ``DISK_MARGIN`` times the
  plan plus the largest file (the temporary copy of one file during its atomic write).
* The destination must not exist or must be empty, and its name must contain ``synthetic``.
  ``index.json`` is written last and atomically. A failure removes what this call wrote, so no
  index ever describes a store that is not complete.
* The same seed writes the same bytes: every tensor is drawn from a generator seeded by the
  seed, the file name and the tensor name, so the result does not depend on write order.

Benchmark harness code, not shipped in a public ``soup`` command: the writer only makes random
weights for tests and for the cold-disk run of ticket 06, and a user has no use for it. Heavy
imports sit inside the functions.

    python benchmarks/harness/b5_synth_store.py plan  --shape deepseek-v3 --out /data/synthetic-v3
    python benchmarks/harness/b5_synth_store.py write --shape deepseek-v3 --out /data/synthetic-v3
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()

SIGNATURE = "synthetic"
WRITER = "b5_synth_store/1"
MARKER_NAME = "SYNTHETIC.json"
#: Plan against actual bytes, as a fraction of the actual size (see the module rules).
SIZE_TOLERANCE = 0.01
#: Free space needed, as a multiple of the plan, before the largest file is added.
DISK_MARGIN = 1.05
DTYPES = ("float32", "bfloat16", "float16")

#: Draw scales, the same as ``b14_synthetic_checkpoint`` so activations stay finite.
DENSE_STD = 0.02
ROUTER_STD = 0.05
BIAS_STD = 0.3
#: A real block of 64 N(0, std) weights has a maximum near 2.4 std. Block maxima are drawn in
#: [ABSMAX_LOW, ABSMAX_HIGH] std. Under double quantisation the same range is made from an
#: offset in the middle and a nested scale of one tenth of the range.
ABSMAX_LOW = 2.0 * DENSE_STD
ABSMAX_HIGH = 3.0 * DENSE_STD
NF4_BLOCK = 64
NESTED_BLOCK = 256
FP32_KEY = "mlp.gate.e_score_correction_bias"
ARCH = "deepseek_v3"
INDEX_FORMAT_VERSION = 5


class StoreRefusedError(RuntimeError):
    """The destination is too small or not acceptable: a missing precondition, not a result."""


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """Everything that fixes the tensors of a DeepSeek-V3-family MoE.

    The routed MoE layers are the last ``moe_layers`` layers, as ``first_k_dense_replace`` makes
    them. ``expert_size`` is ``moe_intermediate_size``, the inner size of one routed expert.
    """

    layers: int
    moe_layers: int
    experts: int
    hidden: int
    expert_size: int
    dense_size: int
    vocab: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    heads: int
    shared_experts: int = 1
    experts_per_token: int = 8
    n_group: int = 1
    topk_group: int = 1
    max_positions: int = 4096

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"shape field {field.name} must be a positive int, got {value!r}")
        if self.moe_layers > self.layers:
            raise ValueError(f"moe_layers {self.moe_layers} exceeds layers {self.layers}")
        if self.experts > 4096:
            raise ValueError(f"the loader accepts at most 4096 experts, got {self.experts}")
        if self.experts_per_token > self.experts:
            raise ValueError("experts_per_token exceeds experts")
        if self.experts % self.n_group or self.topk_group > self.n_group:
            raise ValueError("experts must divide into n_group groups and topk_group <= n_group")

    @property
    def first_dense(self) -> int:
        return self.layers - self.moe_layers

    def is_moe(self, layer: int) -> bool:
        return layer >= self.first_dense

    def config(self) -> Any:
        """The ``DeepseekV3Config`` of this shape."""
        from transformers import DeepseekV3Config

        return DeepseekV3Config(
            vocab_size=self.vocab, hidden_size=self.hidden, intermediate_size=self.dense_size,
            moe_intermediate_size=self.expert_size, num_hidden_layers=self.layers,
            first_k_dense_replace=self.first_dense, n_routed_experts=self.experts,
            n_shared_experts=self.shared_experts, num_experts_per_tok=self.experts_per_token,
            n_group=self.n_group, topk_group=self.topk_group, q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank, qk_rope_head_dim=self.qk_rope_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim, v_head_dim=self.v_head_dim,
            num_attention_heads=self.heads, num_key_value_heads=self.heads,
            max_position_embeddings=self.max_positions, use_cache=False,
        )


#: A shape a CPU trains on in seconds: one dense layer and two MoE layers of eight experts.
SMALL = Shape(
    layers=3, moe_layers=2, experts=8, hidden=64, expert_size=64, dense_size=128, vocab=64,
    q_lora_rank=32, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    heads=4, shared_experts=1, experts_per_token=2, n_group=4, topk_group=2, max_positions=128,
)
#: DeepSeek-V3 (671B): 61 layers, 58 of them MoE with 256 routed experts each.
DEEPSEEK_V3 = Shape(
    layers=61, moe_layers=58, experts=256, hidden=7168, expert_size=2048, dense_size=18432,
    vocab=129280, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64,
    qk_nope_head_dim=128, v_head_dim=128, heads=128, shared_experts=1, experts_per_token=8,
    n_group=8, topk_group=4, max_positions=163840,
)
PRESETS: Dict[str, Shape] = {"small": SMALL, "deepseek-v3": DEEPSEEK_V3}


# --------------------------------------------------------------------------
# the entries of every file: one description for the plan and for the writer
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Entry:
    name: str
    dtype: str  # "uint8", "float32" or the run dtype
    dims: Tuple[int, ...]
    fill: str

    @property
    def numel(self) -> int:
        return math.prod(self.dims)


_ITEMSIZE = {"uint8": 1, "float32": 4, "bfloat16": 2, "float16": 2}
_SAFETENSORS_DTYPE = {"uint8": "U8", "float32": "F32", "bfloat16": "BF16", "float16": "F16"}


def _nf4_entries(
    key: str, dims: Tuple[int, ...], double_quant: bool
) -> List[_Entry]:
    """The stored tensors of one NF4 weight: packed nibbles, then the block statistics."""
    weights = math.prod(dims)
    blocks = -(-weights // NF4_BLOCK)
    out = [_Entry(key, "uint8", ((weights + 1) // 2, 1), "packed")]
    if double_quant:
        out += [
            _Entry(key + "::absmax", "uint8", (blocks,), "absmax_code"),
            _Entry(key + "::nested_absmax", "float32", (-(-blocks // NESTED_BLOCK),),
                   "nested_absmax"),
            _Entry(key + "::nested_offset", "float32", (), "nested_offset"),
        ]
    else:
        out.append(_Entry(key + "::absmax", "float32", (blocks,), "absmax"))
    return out


def _layer_parts(shape: Shape, layer: int) -> List[Tuple[str, Tuple[int, ...], str]]:
    """(short key, dense dims, role) of one decoder layer; role is nf4, norm, router or bias."""
    hidden, heads = shape.hidden, shape.heads
    qk = shape.qk_nope_head_dim + shape.qk_rope_head_dim
    parts: List[Tuple[str, Tuple[int, ...], str]] = [
        ("input_layernorm.weight", (hidden,), "norm"),
        ("post_attention_layernorm.weight", (hidden,), "norm"),
        ("self_attn.q_a_proj.weight", (shape.q_lora_rank, hidden), "nf4"),
        ("self_attn.q_a_layernorm.weight", (shape.q_lora_rank,), "norm"),
        ("self_attn.q_b_proj.weight", (heads * qk, shape.q_lora_rank), "nf4"),
        ("self_attn.kv_a_proj_with_mqa.weight",
         (shape.kv_lora_rank + shape.qk_rope_head_dim, hidden), "nf4"),
        ("self_attn.kv_a_layernorm.weight", (shape.kv_lora_rank,), "norm"),
        ("self_attn.kv_b_proj.weight",
         (heads * (shape.qk_nope_head_dim + shape.v_head_dim), shape.kv_lora_rank), "nf4"),
        ("self_attn.o_proj.weight", (hidden, heads * shape.v_head_dim), "nf4"),
    ]
    if shape.is_moe(layer):
        shared = shape.expert_size * shape.shared_experts
        parts += [
            ("mlp.gate.weight", (shape.experts, hidden), "router"),
            (FP32_KEY, (shape.experts,), "bias"),
            ("mlp.shared_experts.gate_proj.weight", (shared, hidden), "nf4"),
            ("mlp.shared_experts.up_proj.weight", (shared, hidden), "nf4"),
            ("mlp.shared_experts.down_proj.weight", (hidden, shared), "nf4"),
        ]
    else:
        parts += [
            ("mlp.gate_proj.weight", (shape.dense_size, hidden), "nf4"),
            ("mlp.up_proj.weight", (shape.dense_size, hidden), "nf4"),
            ("mlp.down_proj.weight", (hidden, shape.dense_size), "nf4"),
        ]
    return parts


def _layer_entries(shape: Shape, layer: int, dtype: str, double_quant: bool) -> List[_Entry]:
    fills = {"norm": "ones", "router": "router", "bias": "bias"}
    out: List[_Entry] = []
    for key, dims, role in _layer_parts(shape, layer):
        if role == "nf4":
            out += _nf4_entries(key, dims, double_quant)
        else:
            out.append(_Entry(key, "float32" if role == "bias" else dtype, dims, fills[role]))
    return out


def _expert_dims(shape: Shape) -> Dict[str, Tuple[int, int]]:
    return {"gate_up": (2 * shape.expert_size, shape.hidden),
            "down": (shape.hidden, shape.expert_size)}


def _expert_entries(shape: Shape, double_quant: bool) -> List[_Entry]:
    out: List[_Entry] = []
    for expert in range(shape.experts):
        for part, dims in _expert_dims(shape).items():
            out += _nf4_entries(f"{expert}.{part}", dims, double_quant)
    return out


def _extras_entries(shape: Shape, dtype: str, double_quant: bool) -> List[_Entry]:
    out = [_Entry("model.norm.weight", dtype, (shape.hidden,), "ones"),
           _Entry("__nf4_code", "float32", (16,), "code")]
    if double_quant:
        out.append(_Entry("__nf4_nested_code", "float32", (256,), "nested_code"))
    return out


def _large_entries(shape: Shape, dtype: str) -> Dict[str, _Entry]:
    return {
        "model.embed_tokens.weight":
            _Entry("model.embed_tokens.weight", dtype, (shape.vocab, shape.hidden), "normal"),
        "lm_head.weight":
            _Entry("lm_head.weight", dtype, (shape.vocab, shape.hidden), "normal"),
    }


def _files(shape: Shape, dtype: str, double_quant: bool) -> Dict[str, List[_Entry]]:
    """Every safetensors file of the store, by file name, in the order the writer writes them."""
    from soup_cli.utils.expert_shard import expert_shard_name
    from soup_cli.utils.layer_shard import large_shard_path, layer_shard_path

    files: Dict[str, List[_Entry]] = {}
    for layer in range(shape.layers):
        files[os.path.basename(layer_shard_path("", layer))] = _layer_entries(
            shape, layer, dtype, double_quant)
        if shape.is_moe(layer):
            files[expert_shard_name(layer)] = _expert_entries(shape, double_quant)
    for key, entry in _large_entries(shape, dtype).items():
        files[os.path.basename(large_shard_path("", key))] = [entry]
    files["extras.safetensors"] = _extras_entries(shape, dtype, double_quant)
    return files


def _header_bytes(entries: Sequence[_Entry]) -> int:
    """Size of the safetensors header (with its 8-byte length) for ``entries`` in this order."""
    body: Dict[str, Any] = {}
    cursor = 0
    for entry in entries:
        size = entry.numel * _ITEMSIZE[entry.dtype]
        body[entry.name] = {"dtype": _SAFETENSORS_DTYPE[entry.dtype], "shape": list(entry.dims),
                            "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw = len(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return 8 + raw + (-raw) % 8


def _data_bytes(entries: Sequence[_Entry]) -> int:
    return sum(entry.numel * _ITEMSIZE[entry.dtype] for entry in entries)


# --------------------------------------------------------------------------
# index and the small files
# --------------------------------------------------------------------------
def _spec(dims: Tuple[int, ...], dtype: str, double_quant: bool) -> Any:
    from soup_cli.utils.layer_shard import NF4WeightSpec

    return NF4WeightSpec(
        shape=tuple(dims), dtype=dtype, blocksize=NF4_BLOCK, quant_type="nf4",
        nested=double_quant, nested_blocksize=NESTED_BLOCK if double_quant else 0)


def total_params(shape: Shape) -> int:
    """Parameters of the dense weights, as ``shard_checkpoint`` counts ``total_params``."""
    count = sum(math.prod(dims) for layer in range(shape.layers)
                for _key, dims, _role in _layer_parts(shape, layer))
    experts = sum(math.prod(dims) for dims in _expert_dims(shape).values())
    count += shape.moe_layers * shape.experts * experts
    count += shape.hidden + 2 * shape.vocab * shape.hidden  # final norm, embedding, head
    return count


def describe(shape: Shape, seed: int, dtype: str, double_quant: bool) -> str:
    """The generator string of the journal signature: writer, shape, seed, dtype."""
    fields = json.dumps(dataclasses.asdict(shape), sort_keys=True, separators=(",", ":"))
    return f"{WRITER} seed={seed} dtype={dtype} double_quant={double_quant} shape={fields}"


def _build_index(shape: Shape, seed: int, dtype: str, double_quant: bool) -> Any:
    from soup_cli import __version__
    from soup_cli.utils.layer_shard import ShardIndex

    layer_keys: set = set()
    specs: Dict[str, Any] = {}
    for layer in range(shape.layers):
        layer_keys.update(entry.name for entry in _layer_entries(shape, layer, dtype, double_quant))
        for key, dims, role in _layer_parts(shape, layer):
            if role == "nf4":
                specs.setdefault(key, _spec(dims, dtype, double_quant))
    large = _large_entries(shape, dtype)
    fingerprint = hashlib.sha256(describe(shape, seed, dtype, double_quant).encode()).hexdigest()
    return ShardIndex(
        n_layers=shape.layers, layer_keys=tuple(sorted(layer_keys)),
        extra_keys=tuple(sorted(
            entry.name for entry in _extras_entries(shape, dtype, double_quant))),
        dtype=dtype, total_params=total_params(shape), arch=ARCH, soup_version=__version__,
        large_keys=tuple(sorted(large)), format_version=INDEX_FORMAT_VERSION,
        source_fingerprint=fingerprint, source_files=(), quant="nf4", double_quant=double_quant,
        # No device quantised these bytes. The runtime does not read this field; a rebuild from
        # a real source compares it with "cpu" or "cuda" and never matches a synthetic store.
        quant_device=SIGNATURE, quant_specs=specs, stacked_experts=True,
        fp32_keys=(FP32_KEY,), expert_layout="per_expert", expert_layout_version=1,
        expert_layers=tuple(range(shape.first_dense, shape.layers)), expert_count=shape.experts,
        expert_shapes=_expert_dims(shape),
        expert_quant_specs={part: _spec(dims, dtype, double_quant)
                            for part, dims in _expert_dims(shape).items()},
    )


def _write_small_files(
    out: Path, shape: Shape, seed: int, dtype: str, double_quant: bool, *, with_index: bool
) -> List[Path]:
    """``config.json``, the signature marker and (last) ``index.json``."""
    from soup_cli.utils.layer_shard import _atomic_write_index

    written: List[Path] = []
    marker = out / MARKER_NAME
    marker.write_text(json.dumps({
        "signature": SIGNATURE,
        "note": "random weights, no information about the quality of a real model",
        "generator": describe(shape, seed, dtype, double_quant), "seed": seed, "dtype": dtype,
        "double_quant": double_quant, "shape": dataclasses.asdict(shape),
    }, indent=2, sort_keys=True), encoding="utf-8")
    written.append(marker)
    shape.config().save_pretrained(str(out))
    written.append(out / "config.json")
    if with_index:
        _atomic_write_index(_build_index(shape, seed, dtype, double_quant), str(out))
        written.append(out / "index.json")
    return written


def _small_files_bytes(shape: Shape, seed: int, dtype: str, double_quant: bool) -> int:
    with tempfile.TemporaryDirectory() as folder:
        written = _write_small_files(Path(folder), shape, seed, dtype, double_quant,
                                     with_index=True)
        return sum(path.stat().st_size for path in written)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StorePlan:
    """What the store will take, computed from the shape alone."""

    signature: str
    shape: Shape
    dtype: str
    double_quant: bool
    seed: int
    files: Dict[str, int]  # safetensors file name -> bytes, header included
    small_files_bytes: int
    total_bytes: int
    largest_file_bytes: int
    parameters: int
    #: Host RAM the writer needs for its largest file: tensors plus the serialised copy.
    peak_ram_bytes: int

    def required_free_bytes(self) -> int:
        return int(self.total_bytes * DISK_MARGIN) + self.largest_file_bytes


def plan_store(
    shape: Shape, *, dtype: str = "float32", double_quant: bool = False, seed: int = 0
) -> StorePlan:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    files = {name: _header_bytes(entries) + _data_bytes(entries)
             for name, entries in _files(shape, dtype, double_quant).items()}
    small = _small_files_bytes(shape, seed, dtype, double_quant)
    largest = max(files.values())
    return StorePlan(
        signature=SIGNATURE, shape=shape, dtype=dtype, double_quant=double_quant, seed=seed,
        files=files, small_files_bytes=small, total_bytes=sum(files.values()) + small,
        largest_file_bytes=largest, parameters=total_params(shape), peak_ram_bytes=2 * largest,
    )


def store_bytes(out: Path) -> int:
    """Bytes the store takes on disk (every file under ``out``)."""
    return sum(item.stat().st_size for item in out.rglob("*") if item.is_file())


def read_marker(store: Path) -> Dict[str, Any]:
    """The signature marker of a synthetic store. A ValueError says why a folder is not one."""
    path = store / MARKER_NAME
    if not path.is_file():
        raise ValueError(f"{store} has no {MARKER_NAME} marker: not a synthetic store")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("signature") != SIGNATURE or not marker.get("generator"):
        raise ValueError(f"{path} is not a {SIGNATURE} store marker")
    return marker


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def _generator(seed: int, file_name: str, tensor: str) -> Any:
    import torch

    digest = hashlib.sha256(f"{seed}|{file_name}|{tensor}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % 2**63)


def _draw(entry: _Entry, gen: Any, tables: Dict[str, Any]) -> Any:
    import torch

    dtype = getattr(torch, entry.dtype)
    fill = entry.fill
    if fill in ("packed", "absmax_code"):
        return torch.randint(0, 256, entry.dims, dtype=torch.uint8, generator=gen)
    if fill == "absmax":
        return (ABSMAX_LOW + torch.rand(entry.dims, generator=gen, dtype=torch.float32)
                * (ABSMAX_HIGH - ABSMAX_LOW))
    if fill == "nested_absmax":
        # Block maxima of the nested codes span a tenth of the absmax range.
        return (ABSMAX_HIGH - ABSMAX_LOW) / 10 * (
            0.8 + 0.4 * torch.rand(entry.dims, generator=gen, dtype=torch.float32))
    if fill == "nested_offset":
        return torch.tensor((ABSMAX_LOW + ABSMAX_HIGH) / 2, dtype=torch.float32)
    if fill == "ones":
        return torch.ones(entry.dims, dtype=dtype)
    if fill == "code":
        return tables["code"]
    if fill == "nested_code":
        return tables["nested_code"]
    std = {"router": ROUTER_STD, "bias": BIAS_STD}.get(fill, DENSE_STD)
    return (torch.randn(entry.dims, generator=gen, dtype=torch.float32) * std).to(dtype)


def _code_tables(double_quant: bool) -> Dict[str, Any]:
    """The NF4 code table and the nested one, as bitsandbytes makes them (every weight shares)."""
    import torch

    from soup_cli.utils.layer_shard import _quantize_nf4

    probe = torch.linspace(-1, 1, NF4_BLOCK * NESTED_BLOCK)
    _sidecars, _unused_spec, code, nested = _quantize_nf4(
        probe, double_quant=double_quant, device="cpu")
    return {"code": code.to(torch.float32), "nested_code": None if nested is None
            else nested.to(torch.float32)}


def _existing_ancestor(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def write_store(
    out: Path,
    shape: Shape,
    *,
    seed: int = 0,
    dtype: str = "float32",
    double_quant: bool = False,
    free_bytes: Optional[int] = None,
    on_file: Optional[Callable[[str, int, int], None]] = None,
) -> StorePlan:
    """Write the store under ``out`` and return the plan it followed.

    ``free_bytes`` replaces the measured free space (tests). ``on_file(name, done, total)`` is
    called after each safetensors file.
    """
    from soup_cli.utils.layer_shard import _atomic_save, _atomic_write_index

    out = Path(out)
    if SIGNATURE not in out.name.lower():
        raise StoreRefusedError(f"the store folder name must contain {SIGNATURE!r}: {out.name}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise StoreRefusedError(f"{out} exists and is not an empty folder")
    plan = plan_store(shape, dtype=dtype, double_quant=double_quant, seed=seed)
    free = (shutil.disk_usage(_existing_ancestor(out)).free if free_bytes is None
            else free_bytes)
    if free < plan.required_free_bytes():
        raise StoreRefusedError(
            f"the {SIGNATURE} store needs {plan.required_free_bytes()} free bytes "
            f"({plan.total_bytes} planned x {DISK_MARGIN} + the largest file "
            f"{plan.largest_file_bytes}), {free} are free at {_existing_ancestor(out)}")
    created = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    tables = _code_tables(double_quant)
    files = _files(shape, dtype, double_quant)
    try:
        written += _write_small_files(out, shape, seed, dtype, double_quant, with_index=False)
        for done, (name, entries) in enumerate(files.items(), start=1):
            blob = {entry.name: _draw(entry, _generator(seed, name, entry.name), tables)
                    for entry in entries}
            path = out / name
            written.append(path)
            _atomic_save(blob, str(path))
            del blob
            if on_file is not None:
                on_file(name, done, len(files))
        written.append(out / "index.json")
        _atomic_write_index(_build_index(shape, seed, dtype, double_quant), str(out))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        for stale in out.glob(".soup.*.tmp"):
            stale.unlink(missing_ok=True)
        if created:
            shutil.rmtree(out, ignore_errors=True)
        raise
    return plan


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _plan_table(plan: StorePlan) -> Table:
    table = Table(title=f"{SIGNATURE} MoE store plan (random weights, no model quality)")
    table.add_column(f"{SIGNATURE} quantity")
    table.add_column("value", justify="right")
    shape = plan.shape
    rows = [
        ("layers / MoE layers", f"{shape.layers} / {shape.moe_layers}"),
        ("routed experts per MoE layer", str(shape.experts)),
        ("hidden / expert size", f"{shape.hidden} / {shape.expert_size}"),
        ("parameters", f"{plan.parameters:,}"),
        ("planned store bytes", f"{plan.total_bytes:,}"),
        ("largest file bytes", f"{plan.largest_file_bytes:,}"),
        ("peak writer RAM bytes", f"{plan.peak_ram_bytes:,}"),
        ("free bytes required", f"{plan.required_free_bytes():,}"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    return table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("plan", "write"))
    parser.add_argument("--shape", choices=sorted(PRESETS), default="small")
    for name in ("layers", "moe-layers", "experts", "hidden", "expert-size"):
        parser.add_argument("--" + name, type=int, help="override the preset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in
                 ("layers", "moe_layers", "experts", "hidden", "expert_size")
                 if getattr(args, name) is not None}
    shape = dataclasses.replace(PRESETS[args.shape], **overrides)
    plan = plan_store(shape, dtype=args.dtype, double_quant=args.double_quant, seed=args.seed)
    console.print(_plan_table(plan))
    if args.command == "plan":
        return 0
    try:
        write_store(args.out, shape, seed=args.seed, dtype=args.dtype,
                    double_quant=args.double_quant,
                    on_file=lambda name, done, total: console.print(f"{done}/{total} {name}"))
    except StoreRefusedError as error:
        console.print(f"[red]{SIGNATURE} store refused:[/] {escape(str(error))}")
        return 3
    console.print(f"{SIGNATURE} store written to {args.out}: {store_bytes(args.out):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
