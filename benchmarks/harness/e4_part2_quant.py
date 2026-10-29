"""E4 part 2, main rule section 2: Mac-side NF4 quantization of the 3,136 units and checks U1-U3.

    python benchmarks/harness/e4_part2_quant.py plan   --source SRC --out OUT
    python benchmarks/harness/e4_part2_quant.py run    --source SRC --out OUT --part1-rows ROWS
                                                       --manifest MANIFEST

``plan`` prints the disk needed against the disk free and exits 1 if it does not fit.
``run`` quantizes (DQ off, DQ on), writes codes and written-back bf16 checkpoints under OUT,
verifies U1-U3 from the files on disk, and writes one manifest (the only file meant for the repo).
Outputs: ``OUT/codes-dq{off,on}/layer-XX.safetensors`` (packed codes + the library's own packed
QuantState tensors, keys ``<unit>`` and ``<unit>::<qs key>``), ``OUT/writeback-dq{off,on}/``
(a loadable checkpoint: per-layer shards, index, config, tokenizer files), ``OUT/SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import e4_part2_common as common

NF4_BLOCK = 64
DQ_SETTINGS = (("off", False), ("on", True))
KIND_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
#: Files copied next to the written-back shards so ``from_pretrained(dir)`` works.
SIDECAR_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json",
)
EXPECTED_UNITS = 3136
#: One metadata key only. The safetensors writer takes ``metadata`` as a hash map and writes the
#: keys in an order that changes from process to process, so a header with two keys is not
#: reproducible. The double-quant setting is in the directory name and in the manifest.
SAFETENSORS_METADATA = {"format": "pt"}


class OutputSetError(RuntimeError):
    """The files under OUT are not exactly the expected outputs (stray, missing, or listed
    wrongly in SHA256SUMS)."""


@dataclass(frozen=True)
class Unit:
    key: str  # checkpoint key, also the unit's name everywhere
    layer: int
    kind: str  # q_proj ... down_proj
    expert: Optional[int]


def enumerate_units(num_layers: int = 16, num_experts: int = 64) -> List[Unit]:
    """16 x (64 x 3 + 4) units in a fixed order: layer, attention q k v o, then experts."""
    units: List[Unit] = []
    for layer in range(num_layers):
        for kind in _ATTN:
            units.append(Unit(f"model.layers.{layer}.self_attn.{kind}.weight", layer, kind, None))
        for expert in range(num_experts):
            for kind in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
                units.append(Unit(key, layer, kind, expert))
    return units


def fused_view(model, unit: Unit, intermediate: int):
    """The unit's tensor inside the loaded transformers model (a view, no copy).

    gate_up_proj[e, :I] is gate, gate_up_proj[e, I:] is up, down_proj[e] is down; the
    qwen2_moe conversion has no transpose (main rule section 2).
    """
    layer = model.model.layers[unit.layer]
    if unit.expert is None:
        return getattr(layer.self_attn, unit.kind).weight.data
    experts = layer.mlp.experts
    if unit.kind == "gate_proj":
        return experts.gate_up_proj.data[unit.expert, :intermediate, :]
    if unit.kind == "up_proj":
        return experts.gate_up_proj.data[unit.expert, intermediate:, :]
    return experts.down_proj.data[unit.expert]


# --------------------------------------------------------------------------
# which bitsandbytes implementation ran each op (main rule section 2)
# --------------------------------------------------------------------------

BNB_OPS = ("quantize_4bit", "quantize_blockwise", "dequantize_4bit", "dequantize_blockwise")


class _CountingLib:
    """Proxy for the ctypes library handle of ``bitsandbytes.backends.cpu.ops``."""

    def __init__(self, lib, counter: Counter) -> None:
        self._lib = lib
        self._counter = counter

    def __getattr__(self, name):
        target = getattr(self._lib, name)
        if not name.startswith("c") or not callable(target):
            return target

        def counted(*args, **kwargs):
            self._counter[name] += 1
            return target(*args, **kwargs)

        return counted


class BackendProbe:
    """Context manager: records, per bnb op, calls, registered kernels, native vs fallback.

    Native calls are counted through a proxy on the CPU backend's library handle; calls
    that fell back to the pure-torch code are counted by wrapping its compute function; the
    registered kernels come from the dispatcher's own table.
    """

    def __enter__(self):
        import bitsandbytes.backends.cpu.ops as cpu_ops
        import bitsandbytes.backends.default.ops as default_ops
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        self._cpu_ops, self._default_ops = cpu_ops, default_ops
        self.native: Counter = Counter()
        self.fallback: Counter = Counter()
        self.calls: Counter = Counter()
        self._orig_lib = cpu_ops.lib
        cpu_ops.lib = _CountingLib(self._orig_lib, self.native)
        self._orig_compute = default_ops._dequantize_4bit_compute
        probe = self

        def wrapped(*args, **kwargs):
            probe.fallback["dequantize_4bit (pure-torch _dequantize_4bit_compute)"] += 1
            return probe._orig_compute(*args, **kwargs)

        default_ops._dequantize_4bit_compute = wrapped

        class _Counter(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func)
                if name.startswith("bitsandbytes."):
                    probe.calls[name] += 1
                return func(*args, **(kwargs or {}))

        self._mode = _Counter()
        self._mode.__enter__()
        self._torch = torch
        return self

    def __exit__(self, *exc):
        self._mode.__exit__(*exc)
        self._cpu_ops.lib = self._orig_lib
        self._default_ops._dequantize_4bit_compute = self._orig_compute
        return False

    def report(self) -> Dict[str, Any]:
        import bitsandbytes
        from bitsandbytes.cextension import ErrorHandlerMockBNBNativeLibrary, lib

        torch = self._torch
        native_loaded = not isinstance(lib, ErrorHandlerMockBNBNativeLibrary)
        ops: Dict[str, Any] = {}
        for op in BNB_OPS:
            qualified = f"bitsandbytes::{op}"
            ops[op] = {
                "calls": {k: v for k, v in self.calls.items()
                          if k.startswith(f"bitsandbytes.{op}.")},
                "cpu_kernel_registered": bool(
                    torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "CPU")
                ),
                "dispatch_table": torch._C._dispatch_dump(qualified),
            }
        return {
            "bitsandbytes": bitsandbytes.__version__,
            "native_library_loaded": native_loaded,
            "has_avx512": bool(getattr(self._cpu_ops, "_has_avx512", False)),
            "native_functions_called": dict(self.native),
            "pure_torch_fallback_calls": dict(self.fallback),
            "ops": ops,
            "reading": (
                "quantize_4bit has no CPU kernel: the registered 'default' pure-torch kernel "
                "runs. quantize_blockwise / dequantize_blockwise / dequantize_4bit run the "
                "native CPU kernel when native_functions_called lists them and "
                "pure_torch_fallback_calls is empty."
            ),
        }


# --------------------------------------------------------------------------
# the codec
# --------------------------------------------------------------------------


def quantize_unit(tensor, double_quant: bool):
    """(packed codes, QuantState, written-back bf16 weight) for one contiguous unit."""
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    unit = tensor.contiguous()
    packed, state = quantize_4bit(
        unit, blocksize=NF4_BLOCK, compress_statistics=double_quant, quant_type="nf4"
    )
    back = dequantize_4bit(packed, state)
    return packed, state, back


def quant_state_tensors(state) -> Dict[str, Any]:
    """The library's own packed serialization of a QuantState (tensors only)."""
    return state.as_dict(packed=True)


def _tensors_hash(tensors: Dict[str, Any]) -> str:
    digest = common.hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode() + b"=" + common.sha256_tensor(tensors[name]).encode() + b";")
    return digest.hexdigest()


def bitwise_equal(a, b) -> bool:
    """Bit-identical tensors (NaN-safe): same dtype, shape and bytes."""
    import torch

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(width), b.contiguous().view(width))


def block_metrics(ref, back) -> Dict[str, float]:
    """Part 1's frac_over and worst for a (reference, dequantized) pair; same arithmetic."""
    import e4_double_quant_prevalence as part1
    import torch

    r = part1._blocks(ref.float().reshape(-1), part1.NF4_BLOCK)
    d = part1._blocks(back.float().reshape(-1), part1.NF4_BLOCK)
    err = (d - r).abs().amax(dim=1)
    absmax = r.abs().amax(dim=1)
    zero = absmax == 0
    over = torch.where(zero, err > 0, err > part1.BOUND * absmax)
    ratio = torch.where(zero, torch.zeros_like(err), err / absmax.clamp_min(1e-38))
    return {"frac_over": over.float().mean().item(), "worst": ratio.max().item()}


# --------------------------------------------------------------------------
# disk
# --------------------------------------------------------------------------


def _unit_numel(source: Path, units: List[Unit]) -> int:
    """Quantized parameter count, read from the safetensors headers (no tensor is loaded)."""
    from safetensors import safe_open

    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    total, handles = 0, {}
    for unit in units:
        fname = index[unit.key]
        if fname not in handles:
            handles[fname] = safe_open(str(source / fname), framework="pt")
        shape = handles[fname].get_slice(unit.key).get_shape()
        total += shape[0] * shape[1]
    return total


def disk_plan(source: Path, out: Path, units: List[Unit], safety: float = 1.2) -> Dict[str, Any]:
    """Disk needed for all outputs against what is free where OUT will live."""
    source_bytes = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    weights = sum(p.stat().st_size for p in source.glob("*.safetensors"))
    # packed codes: half a byte per quantized weight; DQ-off stats 4 B / 64 weights is the
    # upper bound (the on case is smaller); a bf16 written-back copy of the whole checkpoint
    # per setting.
    quantized_params = _unit_numel(source, units)
    codes = quantized_params // 2 + quantized_params // NF4_BLOCK * 4
    needed = 2 * codes + 2 * weights
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    return {
        "source_bytes_already_on_disk": source_bytes,
        "quantized_params": quantized_params,
        "codes_bytes_per_setting_upper_bound": codes,
        "writeback_bytes_per_setting": weights,
        "needed_new_bytes": needed,
        "needed_with_safety_factor": int(needed * safety),
        "safety_factor": safety,
        "free_bytes": free,
        "fits": int(needed * safety) < free,
    }


# --------------------------------------------------------------------------
# quantization pass
# --------------------------------------------------------------------------


def _load_model(source: Path):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)


def _shard_name(group: str) -> str:
    return f"model-{group}.safetensors"


def writeback_groups(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Checkpoint keys by write-back shard group: ``layer-NN`` per layer, ``other`` for the rest."""
    groups: Dict[str, List[str]] = {}
    for key in weight_map:
        match = re.match(r"^model\.layers\.(\d+)\.", key)
        group = f"layer-{int(match.group(1)):02d}" if match else "other"
        groups.setdefault(group, []).append(key)
    return groups


def expected_output_names(source: Path, units: List[Unit]) -> List[str]:
    """Every file ``run`` writes under OUT (SHA256SUMS aside), as sorted relative paths."""
    weight_map = json.loads((Path(source) / "model.safetensors.index.json").read_text())[
        "weight_map"]
    layers = sorted({u.layer for u in units})
    names = []
    for tag, _ in DQ_SETTINGS:
        names += [f"codes-dq{tag}/layer-{layer:02d}.safetensors" for layer in layers]
        names += [f"writeback-dq{tag}/{_shard_name(group)}" for group in writeback_groups(
            weight_map)]
        names.append(f"writeback-dq{tag}/model.safetensors.index.json")
        names += [f"writeback-dq{tag}/{name}" for name in SIDECAR_FILES
                  if (Path(source) / name).exists()]
    return sorted(names)


def quantize_pass(source: Path, out: Path, tag: str, double_quant: bool,
                  units: List[Unit], log) -> Dict[str, Any]:
    """Quantize every unit of one DQ setting; write codes, write-back shards; return rows."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.time()
    model = _load_model(source)
    intermediate = model.config.intermediate_size
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map: Dict[str, str] = index["weight_map"]
    handles: Dict[str, Any] = {}

    def source_tensor(key: str):
        name = weight_map[key]
        if name not in handles:
            handles[name] = safe_open(str(source / name), framework="pt")
        return handles[name].get_tensor(key)

    codes_dir = out / f"codes-dq{tag}"
    wb_dir = out / f"writeback-dq{tag}"
    codes_dir.mkdir(parents=True, exist_ok=True)
    wb_dir.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, Any]] = {}
    mapping_mismatch: List[str] = []
    layers = sorted({u.layer for u in units})
    for layer in layers:
        layer_tensors: Dict[str, Any] = {}
        for unit in (u for u in units if u.layer == layer):
            view = fused_view(model, unit, intermediate)
            if not bitwise_equal(view, source_tensor(unit.key)):
                mapping_mismatch.append(unit.key)  # fused slice must equal the checkpoint tensor
            packed, state, back = quantize_unit(view, double_quant)
            qs = quant_state_tensors(state)
            layer_tensors[unit.key] = packed.contiguous()
            for name, tensor in qs.items():
                layer_tensors[f"{unit.key}::{name}"] = tensor.contiguous()
            view.copy_(back)  # write back into the model, in place
            rows[unit.key] = {
                "codes_sha256": common.sha256_tensor(packed),
                "quant_state_sha256": _tensors_hash(qs),
                "writeback_sha256": common.sha256_tensor(back),
            }
        save_file(layer_tensors, str(codes_dir / f"layer-{layer:02d}.safetensors"),
                  metadata=dict(SAFETENSORS_METADATA))
        log(f"  dq {tag}: layer {layer} quantized ({time.time() - started:.0f}s)")
        del layer_tensors

    # write-back shards from the modified model: views into model memory, one file per layer
    new_map: Dict[str, str] = {}
    total = 0
    groups = writeback_groups(weight_map)
    state_view = {u.key: fused_view(model, u, intermediate) for u in units}
    plain = dict(model.named_parameters())
    for name, buf in model.named_buffers():
        plain.setdefault(name, buf)
    for group, keys in sorted(groups.items()):
        tensors = {}
        for key in keys:
            if key in state_view:
                tensors[key] = state_view[key].contiguous()
            else:
                tensor = plain[key].data if key in plain else source_tensor(key)
                tensors[key] = tensor.contiguous()
        fname = _shard_name(group)
        save_file(tensors, str(wb_dir / fname), metadata=dict(SAFETENSORS_METADATA))
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        for key in keys:
            new_map[key] = fname
        del tensors
    (wb_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2,
                   sort_keys=True) + "\n"
    )
    for name in SIDECAR_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, wb_dir / name)
    del model
    return {"rows": rows, "mapping_mismatch": mapping_mismatch,
            "seconds": round(time.time() - started, 1)}


# --------------------------------------------------------------------------
# verification from the files on disk: U1, U2, U3
# --------------------------------------------------------------------------


def _read_all(directory: Path) -> Iterator[tuple]:
    from safetensors import safe_open

    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                yield path.name, key, handle


def verify(source: Path, out: Path, units: List[Unit], part1_rows: Optional[Dict[str, dict]],
           log) -> Dict[str, Any]:
    """U1-U3 read back from disk. Never trusts what the quantization pass held in memory."""
    from bitsandbytes.functional import QuantState
    from safetensors import safe_open

    source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    unit_keys = {u.key for u in units}
    results: Dict[str, Any] = {}
    for tag, double_quant in DQ_SETTINGS:
        codes_dir, wb_dir = out / f"codes-dq{tag}", out / f"writeback-dq{tag}"
        wb_map = json.loads((wb_dir / "model.safetensors.index.json").read_text())["weight_map"]
        src_handles: Dict[str, Any] = {}
        wb_handles: Dict[str, Any] = {}
        code_handles: Dict[Path, Any] = {}

        def get(handles, root, fname, key):
            if fname not in handles:
                handles[fname] = safe_open(str(root / fname), framework="pt")
            return handles[fname].get_tensor(key)

        u1_bad, u2_bad, reload_bad = [], [], []
        u2_missing = 0
        metrics: Dict[str, dict] = {}
        for unit in units:
            ref = get(src_handles, source, source_index[unit.key], unit.key)
            fresh_packed, fresh_state, _ = quantize_unit(ref, double_quant)
            fresh_qs = quant_state_tensors(fresh_state)
            cfile = codes_dir / f"layer-{unit.layer:02d}.safetensors"
            if cfile not in code_handles:
                code_handles[cfile] = safe_open(str(cfile), framework="pt")
            handle = code_handles[cfile]
            saved_packed = handle.get_tensor(unit.key)
            saved_qs = {k[len(unit.key) + 2:]: handle.get_tensor(k)
                        for k in handle.keys() if k.startswith(unit.key + "::")}
            same = bitwise_equal(fresh_packed, saved_packed) and set(fresh_qs) == set(saved_qs)
            same = same and all(bitwise_equal(fresh_qs[k], saved_qs[k]) for k in fresh_qs)
            if not same:
                u1_bad.append(unit.key)
            back = get(wb_handles, wb_dir, wb_map[unit.key], unit.key)
            # the stored codes must decode to the stored written-back weight
            from bitsandbytes.functional import dequantize_4bit

            state = QuantState.from_dict(saved_qs, device="cpu")
            if not bitwise_equal(dequantize_4bit(saved_packed, state), back):
                reload_bad.append(unit.key)
            got = block_metrics(ref, back)
            metrics[unit.key] = got
            if part1_rows is not None:
                row = part1_rows.get(unit.key)
                if row is None:
                    u2_missing += 1
                else:
                    want = row["dq_on" if double_quant else "dq_off"]
                    if got["frac_over"] != want["frac_over"] or got["worst"] != want["worst"]:
                        u2_bad.append({"key": unit.key, "got": got,
                                       "part1": {k: want[k] for k in ("frac_over", "worst")}})
        # U3: every source key, bytes of the written-back file against the source
        changed, unchanged_bad, missing, extra = 0, [], [], []
        for key, fname in source_index.items():
            if key not in wb_map:
                missing.append(key)
                continue
            a = get(src_handles, source, fname, key)
            b = get(wb_handles, wb_dir, wb_map[key], key)
            if bitwise_equal(a, b):
                if key in unit_keys:
                    pass  # an unchanged unit is counted below as "not changed"
            else:
                if key in unit_keys:
                    changed += 1
                else:
                    unchanged_bad.append(key)
        extra = sorted(set(wb_map) - set(source_index))
        results[tag] = {
            "U1": {"pass": not u1_bad, "mismatched_units": u1_bad[:20], "count": len(u1_bad)},
            "codes_decode_to_writeback": {"pass": not reload_bad, "count": len(reload_bad)},
            "U2": {
                "pass": part1_rows is not None and not u2_bad and not u2_missing,
                "compared": part1_rows is not None, "mismatches": u2_bad[:20],
                "mismatch_count": len(u2_bad), "units_missing_from_part1": u2_missing,
            },
            "U3": {
                "pass": changed == len(units) and not unchanged_bad and not missing and not extra,
                "units_changed": changed, "expected_units": len(units),
                "non_unit_tensors_differing": unchanged_bad[:20],
                "keys_missing_from_writeback": missing[:20], "keys_extra": extra[:20],
            },
            "unit_metrics": metrics,
        }
        log(f"  dq {tag}: U1 {results[tag]['U1']['pass']}  U2 {results[tag]['U2']['pass']}  "
            f"U3 {results[tag]['U3']['pass']}")
    return results


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def hash_outputs(out: Path, log, expected: List[str]) -> Dict[str, Dict[str, Any]]:
    """Hash exactly the expected output files. A stray or missing file raises
    ``OutputSetError``; nothing outside ``expected`` is ever a manifest entry."""
    on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                     if p.is_file() and p.name != "SHA256SUMS")
    stray = sorted(set(on_disk) - set(expected))
    missing = sorted(set(expected) - set(on_disk))
    if stray or missing:
        raise OutputSetError(
            f"output set is not the expected one: {len(stray)} stray file(s) {stray[:10]}, "
            f"{len(missing)} missing file(s) {missing[:10]}")
    files: Dict[str, Dict[str, Any]] = {}
    for rel in sorted(expected):
        path = out / rel
        files[rel] = {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}
    (out / "SHA256SUMS").write_text(
        "".join(f"{v['sha256']}  {k}\n" for k, v in files.items())
    )
    check_sums_file(out, files)
    log(f"hashed {len(files)} files")
    return files


def check_sums_file(out: Path, files: Dict[str, Dict[str, Any]]) -> None:
    """SHA256SUMS must list exactly the manifest's files with the manifest's hashes."""
    listed = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name in listed:
            raise OutputSetError(f"SHA256SUMS lists {name} twice")
        listed[name] = digest
    want = {k: v["sha256"] for k, v in files.items()}
    if listed != want:
        extra = sorted(set(listed) - set(want))
        absent = sorted(set(want) - set(listed))
        wrong = sorted(k for k in set(listed) & set(want) if listed[k] != want[k])
        raise OutputSetError(f"SHA256SUMS differs from the manifest: extra {extra[:10]}, "
                             f"missing {absent[:10]}, wrong hash {wrong[:10]}")


def read_part1_rows(path: Path) -> Dict[str, dict]:
    return {row["key"]: row for row in common.read_jsonl(path)}


def run(source: Path, out: Path, manifest: Path, part1_rows_path: Optional[Path],
        num_layers: int = 16, num_experts: int = 64, log=print) -> int:
    import platform

    import torch

    torch.set_num_threads(4)  # part 1's stamp: torch_threads 4
    units = enumerate_units(num_layers, num_experts)
    plan = disk_plan(source, out, units)
    log(f"disk: need {plan['needed_with_safety_factor'] / 1e9:.1f} GB "
        f"(incl. x{plan['safety_factor']}), free {plan['free_bytes'] / 1e9:.1f} GB")
    if not plan["fits"]:
        log("STOP: outputs do not fit on the disk")
        return 1
    if num_layers == 16 and num_experts == 64 and len(units) != EXPECTED_UNITS:
        raise SystemExit(f"unit count {len(units)} != {EXPECTED_UNITS}")
    part1 = read_part1_rows(part1_rows_path) if part1_rows_path else None
    started = time.time()
    source_files = {p.name: {"sha256": common.sha256_file(p), "bytes": p.stat().st_size}
                    for p in sorted(source.iterdir()) if p.is_file()}
    log(f"hashed {len(source_files)} source files")
    passes, backends = {}, {}
    for tag, double_quant in DQ_SETTINGS:
        with BackendProbe() as probe:
            passes[tag] = quantize_pass(source, out, tag, double_quant, units, log)
        backends[f"quantize_pass_dq_{tag}"] = probe.report()
    with BackendProbe() as probe:
        checks = verify(source, out, units, part1, log)
    backends["verify"] = probe.report()
    files = hash_outputs(out, log, expected_output_names(source, units))
    unit_rows = {
        tag: {k: dict(v, **{m: checks[tag]["unit_metrics"][k][m] for m in ("frac_over", "worst")})
              for k, v in passes[tag]["rows"].items()}
        for tag, _ in DQ_SETTINGS
    }
    for tag, _ in DQ_SETTINGS:
        checks[tag].pop("unit_metrics")
    body = {
        "what": "E4 part 2 main rule section 2: Mac-side NF4 quantization, checks U1-U3",
        "rule_version": "e4-rule-part2 v4.2",
        "model": {"repo": common.MODEL_REPO, "revision": common.MODEL_REV},
        "unit_count": len(units),
        "kind_counts": dict(Counter(u.kind for u in units)),
        "box": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "library_versions": common.library_versions(),
        },
        "provenance": common.provenance(),
        "disk": plan,
        "source_files": source_files,
        "seconds": {t: p["seconds"] for t, p in passes.items()} | {
            "total": round(time.time() - started, 1)},
        "peak_rss_bytes_all_children_excluded": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "bnb_cpu_backends": backends,
        "fused_slice_equals_checkpoint_tensor": {
            t: {"mismatches": p["mapping_mismatch"], "pass": not p["mapping_mismatch"]}
            for t, p in passes.items()},
        "checks": checks,
        "files": files,
        "units": unit_rows,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    all_pass = all(checks[t][c]["pass"] for t, _ in DQ_SETTINGS for c in ("U1", "U2", "U3"))
    log(f"manifest: {manifest}  U1-U3 {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        if name == "run":
            p.add_argument("--manifest", required=True, type=Path)
            p.add_argument("--part1-rows", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        plan = disk_plan(args.source, args.out, enumerate_units())
        print(json.dumps(plan, indent=1))
        return 0 if plan["fits"] else 1
    try:
        return run(args.source, args.out, args.manifest, args.part1_rows)
    except OutputSetError as exc:
        print(f"STOP: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
