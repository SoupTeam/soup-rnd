#!/usr/bin/env python3
"""A3 criterion 1b diagnosis, not a repeat of the spill/cost/cleanup gate.

``session`` prints fresh-child commands; ``--execute`` runs them sequentially. Both the
session and each child refuse an existing output directory. Product and old harness
imports come ONLY from ``--baseline-path`` at PINNED_COMMIT, never this script's checkout.
CUDA requires the historical local Mistral fixture; CPU/quant-none permits a tiny local
streamed fixture, but is explicitly a rehearsal and cannot establish a GPU verdict.

Private artifact contract (schema 1): ``arms/<arm>/arm.json`` retains the pinned recorder's
logs/rates/grads and ``tensors/step_0000.pt`` etc. ``traces.json`` is an array of metadata;
each row's ``file`` is a single independently owned CPU tensor saved by torch.save. The
stable key includes window, actual fetched microbatch/row, phase, stage/layer, occurrence.
It is NOT permission to align keys without matching metadata. Phases distinguish shape
capture, ordinary grad/nograd forward, recomputation and optional boundary adjoints.
Hidden tensors contain real tokens only; head tensors contain ALL vocabulary entries at
shifted nonignored causal targets only. CE numerators are independently computed on CPU
from these logits (chunked FP32 CE, documented separately from the actual model loss).
There is no all-layer tensor dictionary or retained GPU tensor in observer state.

The padding intervention calls the pinned _window_tensors on the ACTUAL fetched window,
after HF computed its original N. It drops the original attention mask/position override,
as the real L2L path does. Thus native-to-padded is padding PLUS mask-policy intervention
when the native mask is explicit. Padded-to-L2L is a residual execution-path difference,
not a demonstrated schedule cause. No automatic causal conclusion is made here.

Examples::

    python a3_1b_diagnose.py session --baseline-path /work/baseline-ec795ef1 \
        --weights /work/mistral-7b --shards /work/f1-nf4 --out /work/diagnosis --execute
    python a3_1b_diagnose.py session --baseline-path C:/Temp/baseline-ec795ef1 \
        --weights C:/Temp/tiny --device cpu --quant none --seq 32 --out C:/Temp/diag --execute

This program does not rent, SSH, delete a VM, change the product, or loosen tolerances.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

PINNED_COMMIT = "ec795ef175c222b60ff91b9ba44f7f2bfad55dd5"
MODEL_ID = "unsloth/mistral-7b-instruct-v0.3"
MODEL_REVISION = "60b99dcbc7f853b7cd18472a607e8d045b96f03e"
HISTORICAL_WEIGHTS = {
    "model-00001-of-00003.safetensors":
        "ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19",
    "model-00002-of-00003.safetensors":
        "8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0",
    "model-00003-of-00003.safetensors":
        "905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca",
    "config.json": "9262d15f830e511429ccd175a71680e51c656ef3a4aa7460f3e9596508d32487",
}
ARMS = {
    "ga_3step": {"l2l": False, "trace": False, "pad": False, "steps": 3},
    "ga_repeat_3step": {"l2l": False, "trace": False, "pad": False, "steps": 3},
    "l2l_3step": {"l2l": True, "trace": False, "pad": False, "steps": 3},
    "ga_floor_3step": {"l2l": False, "trace": False, "pad": False, "steps": 3,
                       "sdpa": "EFFICIENT_ATTENTION"},
    "ga_native_1step": {"l2l": False, "trace": False, "pad": False, "steps": 1},
    "ga_native_trace_1step": {"l2l": False, "trace": True, "pad": False, "steps": 1},
    "ga_padded_1step": {"l2l": False, "trace": False, "pad": True, "steps": 1},
    "ga_padded_trace_1step": {"l2l": False, "trace": True, "pad": True, "steps": 1},
    "l2l_trace_1step": {"l2l": True, "trace": True, "pad": False, "steps": 1},
}


def _say(message: str) -> None:
    from rich.console import Console

    Console().print(message, markup=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path: Path, value: Any) -> None:
    """Replace only this attempt's metadata; tensor/output directories are exclusive."""
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(256 << 10)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as handle:
        while (count := handle.readinto(buffer)):
            digest.update(view[:count])
    return digest.hexdigest()


def _inside(path: str | Path, root: str | Path) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.normcase(os.path.commonpath((path, root))) == os.path.normcase(root)
    except ValueError:
        return False


def _git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *arguments],
        input=data, capture_output=True, check=False, timeout=90,
    )
    if result.returncode:
        raise RuntimeError(f"baseline git {' '.join(arguments)}: "
                           f"{result.stderr.decode(errors='replace')}")
    return result.stdout


def _python_source_hashes(path: Path) -> tuple[str, str]:
    """Raw and CRLF-only-normalized source hashes, with bounded read scratch."""
    raw, normalized = hashlib.sha256(), hashlib.sha256()
    pending = b""
    with path.open("rb") as handle:
        while block := handle.read(64 << 10):
            raw.update(block)
            joined = pending + block
            pending = b"\r" if joined.endswith(b"\r") else b""
            if pending:
                joined = joined[:-1]
            normalized.update(joined.replace(b"\r\n", b"\n"))
    normalized.update(pending)
    return raw.hexdigest(), normalized.hexdigest()


def baseline_facts(path: str) -> dict[str, Any]:
    """Attest Python bytes before importing anything from the baseline checkout.

    Windows checkout CRLF is allowed ONLY for Python source, and both the actual bytes
    and LF-normalized bytes are retained. No tensor/model/binary bytes are normalized.
    """
    root = Path(os.path.realpath(path))
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"baseline HEAD {head!r} is not {PINNED_COMMIT}")
    entries = _git(root, "ls-tree", "-r", "-z", "--name-only", head,
                   "--", "src", "benchmarks/harness").split(b"\0")
    names = sorted(entry.decode("utf-8") for entry in entries if entry.endswith(b".py"))
    if not names:
        raise RuntimeError("baseline contains no Python source")
    raw_hashes, normalized_hashes, git_hashes = {}, {}, {}
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
               "cat-file", "--batch"]
    with tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=errors)
        expired = threading.Event()

        def expire() -> None:
            if child.poll() is None:
                expired.set()
                try:
                    child.kill()
                except ProcessLookupError:
                    pass

        watchdog = threading.Timer(90, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            if child.stdin is None or child.stdout is None:
                raise RuntimeError("git cat-file has no batch pipes")
            for name in names:
                # One request at a time: never buffer the complete source checkout.
                child.stdin.write(f"{head}:{name}\n".encode("utf-8"))
                child.stdin.flush()
                header = child.stdout.readline(4096).split()
                if len(header) != 3 or header[1] != b"blob":
                    raise RuntimeError(f"cannot attest baseline blob {name}: {header!r}")
                remaining, digest = int(header[2]), hashlib.sha256()
                while remaining:
                    block = child.stdout.read(min(remaining, 64 << 10))
                    if not block:
                        raise RuntimeError(f"git cat-file truncated baseline blob {name}")
                    digest.update(block)
                    remaining -= len(block)
                if child.stdout.read(1) != b"\n":
                    raise RuntimeError(f"git cat-file malformed blob terminator: {name}")
                target = root / name
                if not target.is_file() or not _inside(target, root):
                    raise RuntimeError(f"missing/escaped baseline source {name}")
                raw, normalized = _python_source_hashes(target)
                original = digest.hexdigest()
                if raw != original and normalized != original:
                    raise RuntimeError(f"baseline source differs from pinned blob: {name}")
                raw_hashes[name], normalized_hashes[name], git_hashes[name] = (
                    raw, normalized, original)
            child.stdin.close()
            if child.wait(timeout=90):
                errors.seek(0)
                raise RuntimeError("baseline git cat-file failed: "
                                   + errors.read(16 << 10).decode(errors="replace"))
        except Exception as exc:
            if expired.is_set():
                raise TimeoutError("baseline git cat-file exceeded 90 seconds") from exc
            raise
        finally:
            watchdog.cancel()
            watchdog.join()
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            if child.stdin is not None:
                child.stdin.close()
            if child.stdout is not None:
                child.stdout.close()
    for folder in (root / "src", root / "benchmarks" / "harness"):
        for target in folder.rglob("*.py"):
            relative = os.path.relpath(target, root).replace(os.sep, "/")
            if relative not in raw_hashes:
                raise RuntimeError(f"untracked Python source in pinned import root: {relative}")
    return {"path": str(root), "commit": head, "source_sha256": git_hashes,
            "working_raw_source_sha256": raw_hashes,
            "normalized_source_sha256": normalized_hashes,
            "git_blob_sha256": git_hashes, "python_files": len(names)}


def _import_baseline(facts: dict[str, Any]) -> Any:
    root = Path(facts["path"])
    own_root = Path(os.path.realpath(__file__)).parents[2]
    for name in sys.modules:
        if name == "soup_cli" or name.startswith("soup_cli.") or name == "a3_spill_stand":
            raise RuntimeError(f"refusing pre-imported product/harness module {name}")
    excluded = {os.path.normcase(os.path.realpath(path)) for path in
                (own_root, own_root / "src", own_root / "benchmarks" / "harness")}
    rest = [entry for entry in sys.path if entry
            and os.path.normcase(os.path.realpath(entry)) not in excluded]
    sys.path[:] = [str(root / "benchmarks" / "harness"), str(root / "src"),
                   str(root), *rest]
    importlib.invalidate_caches()
    stand = importlib.import_module("a3_spill_stand")
    if not _inside(stand.__file__, root):
        raise RuntimeError("a3_spill_stand did not come from the pinned checkout")
    return stand


def _tensor_digest(cpu: Any) -> dict[str, Any]:
    """Raw bytes, with bounded scratch (no full tensor .tobytes() allocation)."""
    import torch

    cpu = cpu.detach().contiguous()
    if cpu.device.type != "cpu":
        raise RuntimeError("digest expects an already copied CPU tensor")
    raw = cpu.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(raw)).hexdigest()
    finite, nonzero = True, False
    for chunk in cpu.reshape(-1).split(1 << 20):
        finite = finite and bool(torch.isfinite(chunk).all())
        nonzero = nonzero or bool(chunk.count_nonzero())
    return {"sha256": digest, "dtype": str(cpu.dtype).removeprefix("torch."),
            "shape": list(cpu.shape), "finite": finite, "nonzero": nonzero}


def _cpu(tensor: Any) -> Any:
    return tensor.detach().to("cpu", copy=True, non_blocking=False).contiguous()


def _save_tensor(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def _fixture_facts(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.weights)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise RuntimeError("--weights must be an existing local checkpoint directory")
    paths = sorted({*root.glob("*.safetensors"), *root.glob("*.safetensors.index.json"),
                    root / "config.json"})
    if not any(path.suffix == ".safetensors" for path in paths):
        raise RuntimeError("local fixture contains no safetensors weights")
    hashes = {path.name: _sha(path) for path in paths}
    matched = all(hashes.get(name) == digest for name, digest in HISTORICAL_WEIGHTS.items())
    if args.device == "cuda" and not matched:
        raise RuntimeError("CUDA fixture hashes differ from historical session 3")
    return {"path": str(root), "files_sha256": hashes,
            "historical_match": matched, "expected_files_sha256": HISTORICAL_WEIGHTS,
            "historical_model_id": MODEL_ID, "historical_revision": MODEL_REVISION,
            "config": json.loads((root / "config.json").read_text(encoding="utf-8"))}


def _dependency_facts() -> dict[str, Any]:
    """Package versions plus hashes of imported code/native binaries, not only labels."""
    versions, files = {}, {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy",
                 "safetensors", "psutil", "rich"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    critical = {
        "torch", "torch._C", "torch.nn.functional", "torch.utils.checkpoint",
        "torch.amp.autocast_mode", "torch.autograd", "torch.autograd.function",
        "transformers", "transformers.trainer", "transformers.training_args",
        "transformers.modeling_utils", "transformers.loss.loss_utils",
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral", "transformers.modeling_attn_mask_utils",
        "transformers.integrations.sdpa_attention", "transformers.masking_utils",
        "accelerate", "accelerate.accelerator", "accelerate.utils.operations",
        "peft", "peft.peft_model", "peft.tuners.lora.layer", "peft.tuners.lora.bnb",
        "bitsandbytes", "bitsandbytes.functional", "bitsandbytes.cextension",
        "bitsandbytes.backends.cuda.ops", "bitsandbytes.nn.modules",
        "bitsandbytes.autograd._functions", "safetensors", "safetensors.torch", "numpy",
    }
    for name in sorted(critical):
        module = sys.modules.get(name)
        source = vars(module).get("__file__") if module is not None else None
        if not isinstance(source, str):
            package = sys.modules.get(name.split(".", 1)[0])
            package_file = vars(package).get("__file__") if package is not None else None
            if not isinstance(package_file, str):
                continue
            stem = Path(package_file).parent.joinpath(*name.split(".")[1:])
            candidates = (stem.with_suffix(".py"), stem / "__init__.py")
            source = next((str(path) for path in candidates if path.is_file()), None)
        if source is not None:
            path = Path(source)
            if path.is_file():
                files[name] = {"path": str(path), "sha256": _sha(path),
                               "bytes": path.stat().st_size, "loaded": module is not None}
    native = {}
    for package in ("torch", "bitsandbytes"):
        module = sys.modules.get(package)
        source = vars(module).get("__file__") if module is not None else None
        if not isinstance(source, str):
            continue
        root = Path(source).parent
        candidates = root.rglob("*") if package == "bitsandbytes" else (root / "lib").glob("*")
        for path in sorted(candidates):
            if path.is_file() and (".so" in path.name or path.suffix.lower() in (".pyd", ".dll")):
                native[str(path)] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    return {"versions": versions, "loaded_source_binary_files": files,
            "package_native_binaries": native, "python": sys.version,
            "executable": sys.executable, "platform": platform.platform()}


class InputLedger:
    """Observe actual HF windows and original N, without changing data or sampler."""

    def __init__(self, record: dict[str, Any], directory: Path, rows: Any, stand: Any):
        self.record, self.directory, self.rows, self.stand = record, directory, rows, stand
        self.window = -1
        self.row_hashes = record["rows"]["row_sha256"]
        if len(set(self.row_hashes)) != len(self.row_hashes):
            raise RuntimeError("ambiguous duplicate fixture rows")
        self.record.update(windows=[], row_order=[], training_step_losses=[], applied_rates=[],
                           per_microbatch_losses=[], dispatch=[], layer_calls=[])

    def flush(self) -> None:
        _json(self.directory / "arm.json", self.record)

    def _row_id(self, inputs: dict[str, Any]) -> int:
        ids, labels = _cpu(inputs["input_ids"]).reshape(-1), _cpu(inputs["labels"]).reshape(-1)
        digest = hashlib.sha256()
        digest.update(memoryview(ids.numpy()))
        digest.update(b"|")
        digest.update(memoryview(labels.numpy()))
        try:
            return self.row_hashes.index(digest.hexdigest())
        except ValueError as exc:
            raise RuntimeError("actual fetched sample does not match an original row") from exc

    def fetched(self, samples: list[Any], n_items: Any) -> None:
        self.window += 1
        row_ids = [self._row_id(sample) for sample in samples]
        shifted = [int((sample["labels"][..., 1:] != -100).sum().item()) for sample in samples]
        unshifted = [int((sample["labels"] != -100).sum().item()) for sample in samples]
        total = int(n_items.item()) if hasattr(n_items, "item") else int(n_items)
        definition = ("causal_supervised_tokens" if total == sum(shifted) else
                      "label_tokens" if total == sum(unshifted) else "other_original")
        self.record["row_order"].extend(row_ids)
        self.record["windows"].append({
            "window": self.window, "row_ids": row_ids,
            "row_sha256": [self.row_hashes[index] for index in row_ids],
            "valid_lengths": [int(sample["input_ids"].shape[-1]) for sample in samples],
            "supervised_tokens": shifted, "label_tokens": unshifted,
            "n_items": total, "n_items_definition": definition,
            "padded_lengths": [], "effective_inputs": [],
        })
        self.flush()

    def match_ids(self, ids: Any) -> int:
        import torch

        data = _cpu(ids)
        if data.ndim != 2 or data.shape[0] != 1:
            raise RuntimeError("diagnostic requires actual one-row microbatches")
        window = self.record["windows"][self.window]
        matches = []
        for mb, row_id in enumerate(window["row_ids"]):
            original = self.rows[row_id]["input_ids"]
            if data.shape[1] >= original.numel() and torch.equal(data[0, :original.numel()],
                                                               original):
                matches.append(mb)
        if len(matches) != 1:
            raise RuntimeError(f"cannot identify actual input row uniquely: {matches}")
        return matches[0]

    def context(self, mb: int, phase: str) -> dict[str, Any]:
        window = self.record["windows"][self.window]
        return {"window": self.window, "micro_batch": mb, "row_id": window["row_ids"][mb],
                "phase": phase, "valid_length": window["valid_lengths"][mb]}

    def effective(self, samples: Sequence[dict[str, Any]], *, mask_policy: str) -> None:
        window = self.record["windows"][self.window]
        if window["effective_inputs"]:
            raise RuntimeError("effective window was recorded twice")
        for mb, sample in enumerate(samples):
            tensors = {name: _cpu(sample[name]) for name in
                       ("input_ids", "labels", "attention_mask", "position_ids")
                       if sample.get(name) is not None}
            name = f"inputs/w{self.window:04d}_mb{mb:04d}.pt"
            _save_tensor(self.directory / name, tensors)
            facts = {key: _tensor_digest(value) for key, value in tensors.items()}
            item = {"row_id": window["row_ids"][mb], "micro_batch": mb, "file": name,
                    "mask_policy": mask_policy, "position_policy": "implicit_causal_positions",
                    "input_shape": facts["input_ids"]["shape"],
                    "label_shape": facts["labels"]["shape"],
                    "input_dtype": facts["input_ids"]["dtype"],
                    "label_dtype": facts["labels"]["dtype"], "tensors": facts}
            for key in ("input_ids", "labels", "attention_mask", "position_ids"):
                item[key + "_sha256"] = facts.get(key, {}).get("sha256")
            window["effective_inputs"].append(item)
            window["padded_lengths"].append(int(tensors["input_ids"].shape[-1]))
            del tensors
        self.flush()


@contextlib.contextmanager
def _patch(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    """Restore the exact preexisting instance/class attribute shape on every exit."""
    missing = object()
    previous_local = vars(owner).get(name, missing)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if previous_local is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, previous_local)


class Observer:
    """Hooks never replace outputs; wrappers call their original exactly once.

    Row provenance is explicit, not an invocation counter. Embedding IDs identify the
    actual fetched input row. Boundary output storage identities carry only integer
    provenance through detach views. ActivationStore.put/get supply their EXPLICIT mb
    argument before each L2L _layer call. No tensor or graph is held in provenance state.
    """

    def __init__(self, ledger: InputLedger, model: Any, l2l: bool):
        from soup_cli.utils.l2l_step import model_parts

        self.ledger, self.model, self.l2l = ledger, model, l2l
        self.causal, self.owner = model_parts(model)
        self.local = threading.local()
        self.local.context = None
        self.local.modules = []
        self.provenance: dict[tuple[Any, ...], int] = {}
        self.occurrences: dict[tuple[Any, ...], int] = {}
        self.adjoint_seen: set[tuple[Any, ...]] = set()
        self.handles: list[Any] = []
        self.traces: list[dict[str, Any]] = []
        self.call_seen: set[tuple[Any, ...]] = set()
        self.stack = contextlib.ExitStack()
        self.dispatch_file: Any = None
        self.ledger.record["observer"] = {
            "enabled": True, "copy": "detach; blocking CPU copy=True; contiguous; disk",
            "tensor_retention": "integer storage provenance only; no live tensor references",
            "ce_sum_method": "CPU FP32 cross_entropy reduction=sum, chunks of 16 targets",
            "phases": ["capture", "nogradforward", "gradforward", "recompute", "adjoint"],
            "checkpoint_early_stop": "unchanged; recompute output may not be reached",
        }

    @staticmethod
    def token(tensor: Any) -> tuple[Any, ...]:
        return (str(tensor.device), int(tensor.data_ptr()), int(tensor.storage_offset()),
                tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))

    def bind(self, tensor: Any, mb: int) -> None:
        self.provenance[self.token(tensor)] = mb

    def lookup(self, tensor: Any) -> int:
        try:
            return self.provenance[self.token(tensor)]
        except KeyError as exc:
            raise RuntimeError("observer cannot identify boundary input provenance") from exc

    @contextlib.contextmanager
    def scope(self, context: dict[str, Any]) -> Iterator[None]:
        previous = getattr(self.local, "context", None)
        self.local.context = context
        try:
            yield
        finally:
            self.local.context = previous

    def state(self) -> dict[str, Any]:
        import torch

        device = self.ledger.record["device"]
        return {"grad_enabled": bool(torch.is_grad_enabled()),
                "autocast_enabled": bool(torch.is_autocast_enabled(device)),
                "autocast_dtype": str(torch.get_autocast_dtype(device)).removeprefix("torch.")}

    def save(self, cpu: Any, context: dict[str, Any], stage: str, layer: int | None,
             *, slicing: str, positions: list[int], boundary: str | None = None,
             state: dict[str, Any] | None = None, source_phase: str | None = None) -> None:
        identity = (context["window"], context["micro_batch"], context["phase"], stage, layer,
                    boundary, slicing)
        occurrence = self.occurrences.get(identity, 0)
        self.occurrences[identity] = occurrence + 1
        part = stage if layer is None else f"{stage}.{layer:02d}"
        key = (f"w{context['window']:04d}/mb{context['micro_batch']:04d}"
               f"/row{context['row_id']:04d}/{context['phase']}/{part}"
               f"/{boundary + '/' if boundary else ''}occ{occurrence:04d}")
        filename = f"trace/{len(self.traces):06d}.pt"
        digest = _tensor_digest(cpu)
        _save_tensor(self.ledger.directory / filename, cpu)
        self.traces.append({**context, **(state or self.state()), **digest, "key": key,
                            "stage": stage, "layer": layer, "occurrence": occurrence,
                            "slicing": slicing, "valid_positions": positions, "file": filename,
                            "boundary": boundary, "source_phase": source_phase,
                            "arm": self.ledger.record["arm"]})
        _json(self.ledger.directory / "traces.json", self.traces)

    def cropped(self, tensor: Any, context: dict[str, Any], stage: str,
                layer: int | None, *, boundary: str | None = None,
                source_phase: str | None = None) -> None:
        import torch

        state = self.state()
        length = context["valid_length"]
        if stage == "head" and boundary != "input":
            labels = self.ledger.rows[context["row_id"]]["labels"]
            positions = (labels[1:] != -100).nonzero(as_tuple=False).reshape(-1).tolist()
            selector = torch.tensor(positions, dtype=torch.long, device=tensor.device)
            cpu = _cpu(tensor.detach()[0, :length - 1, :].index_select(0, selector))
            slicing = "causal_targets"
        else:
            positions = list(range(length))
            cpu = _cpu(tensor.detach()[:, :length, :])
            slicing = "valid_tokens"
        self.save(cpu, context, stage, layer, slicing=slicing, positions=positions,
                  boundary=boundary, state=state, source_phase=source_phase)
        if stage == "head" and boundary is None and context["phase"] != "capture":
            targets = self.ledger.rows[context["row_id"]]["labels"][1:]
            targets = targets[targets != -100]
            numerator = torch.zeros((), dtype=torch.float32)
            for start in range(0, cpu.shape[0], 16):
                numerator += torch.nn.functional.cross_entropy(
                    cpu[start:start + 16].float(), targets[start:start + 16], reduction="sum")
            self.save(numerator, context, "ce_sum", None, slicing="ce_sum", positions=positions,
                      state=state)
        del cpu

    def adjoint(self, tensor: Any, context: dict[str, Any], stage: str,
                layer: int | None, boundary: str) -> None:
        if not tensor.requires_grad or context["phase"] == "capture":
            return
        identity = (self.token(tensor), context["window"], context["micro_batch"], stage,
                    layer, boundary)
        if identity in self.adjoint_seen:
            return
        self.adjoint_seen.add(identity)
        metadata = {**context, "phase": "adjoint"}
        source_phase = context["phase"]

        def observe(gradient: Any) -> None:
            self.cropped(gradient, metadata, stage, layer, boundary=boundary,
                         source_phase=source_phase)
            return None

        self.handles.append(tensor.register_hook(observe))

    def stage(self, module: Any, stage: str, layer: int | None = None) -> None:
        import torch

        contexts: list[dict[str, Any]] = []

        def before(_module: Any, arguments: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            inputs = arguments[0] if arguments else kwargs.get("hidden_states")
            if inputs is None:
                inputs = kwargs.get("input_ids")
            current = getattr(self.local, "context", None)
            mb = self.ledger.match_ids(inputs) if stage == "embed" else self.lookup(inputs)
            if current is not None and current.get("phase") in ("capture", "recompute"):
                phase = current["phase"]
            elif stage == "decoder" and current is not None:
                phase = current["phase"]
            else:
                phase = "gradforward" if torch.is_grad_enabled() else "nogradforward"
            context = self.ledger.context(mb, phase)
            contexts.append(context)
            if stage != "embed":
                self.adjoint(inputs, context, stage, layer, "input")
            if stage == "decoder" and layer == 0:
                self.layer_call(arguments[1:], kwargs, context)
            if stage == "head":
                self.local.head_context = context
            return None

        def after(_module: Any, _arguments: tuple[Any, ...], output: Any) -> None:
            if not contexts:
                return None
            context = contexts.pop()
            if output is None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            self.bind(hidden, context["micro_batch"])
            self.cropped(hidden, context, stage, layer)
            self.adjoint(hidden, context, stage, layer, "output")
            return None

        self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(module.register_forward_hook(after, always_call=True))

    def layer_call(self, arguments: Any, kwargs: Any, context: dict[str, Any]) -> None:
        identity = (context["window"], context["micro_batch"], context["phase"])
        if identity in self.call_seen:
            return
        self.call_seen.add(identity)
        tensors: dict[str, Any] = {}

        def describe(value: Any, name: str) -> Any:
            if hasattr(value, "detach") and hasattr(value, "shape"):
                cpu = _cpu(value)
                tensors[name] = cpu
                return {"tensor": name, **_tensor_digest(cpu)}
            if isinstance(value, dict):
                return {str(key): describe(item, f"{name}.{key}") for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [describe(item, f"{name}.{index}") for index, item in enumerate(value)]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

        metadata = {"args": describe(arguments, "args"), "kwargs": describe(kwargs, "kwargs")}
        name = f"calls/{len(self.ledger.record['layer_calls']):06d}.pt"
        _save_tensor(self.ledger.directory / name, tensors)
        self.ledger.record["layer_calls"].append({**context, **self.state(), **metadata,
                                                   "file": name})
        tensors.clear()

    def install(self) -> "Observer":
        from soup_cli.trainer import l2l_trainer
        from soup_cli.utils import l2l_activations, l2l_step, layer_stream_runtime

        self.dispatch_file = (self.ledger.directory / "dispatch.jsonl").open(
            "x", encoding="utf-8", newline="\n")
        original_capture = l2l_step.capture_layer_call

        @functools.wraps(original_capture)
        def capture(model: Any, ids: Any) -> Any:
            mb = self.ledger.match_ids(ids)
            with self.scope(self.ledger.context(mb, "capture")):
                return original_capture(model, ids)

        self.stack.enter_context(_patch(l2l_step, "capture_layer_call", capture))
        original_run = l2l_step.L2LStep.run

        @functools.wraps(original_run)
        def run(stepper: Any, micro_batches: Sequence[Any], store: Any, **kwargs: Any) -> Any:
            for mb, ids in enumerate(micro_batches):
                if self.ledger.match_ids(ids) != mb:
                    raise RuntimeError("L2L actual input order differs from fetched window")
                self.bind(ids, mb)
            return original_run(stepper, micro_batches, store, **kwargs)

        self.stack.enter_context(_patch(l2l_step.L2LStep, "run", run))
        original_put, original_get = (l2l_activations.ActivationStore.put,
                                       l2l_activations.ActivationStore.get)

        @functools.wraps(original_put)
        def put(store: Any, layer: int, mb: int, tensor: Any) -> Any:
            result = original_put(store, layer, mb, tensor)
            self.bind(tensor, mb)
            self.local.pending_layer = (mb, "nogradforward")
            return result

        @functools.wraps(original_get)
        def get(store: Any, layer: int, mb: int) -> Any:
            result = original_get(store, layer, mb)
            self.bind(result, mb)
            self.local.pending_layer = (mb, "recompute")
            return result

        self.stack.enter_context(_patch(l2l_activations.ActivationStore, "put", put))
        self.stack.enter_context(_patch(l2l_activations.ActivationStore, "get", get))
        original_layer = l2l_step.L2LStep._layer

        @functools.wraps(original_layer)
        def layer_call(stepper: Any, module: Any, hidden: Any, state: Any) -> Any:
            pending = getattr(self.local, "pending_layer", None)
            if pending is None or self.lookup(hidden) != pending[0]:
                raise RuntimeError("L2L layer input has no explicit store microbatch context")
            self.local.pending_layer = None
            with self.scope(self.ledger.context(*pending)):
                return original_layer(stepper, module, hidden, state)

        self.stack.enter_context(_patch(l2l_step.L2LStep, "_layer", layer_call))
        # Each checkpoint captures this INVOCATION's scalar row/window context, not
        # caller-thread TLS. CUDA autograd can execute recomputation on another thread.
        for index, layer in enumerate(self.owner.layers):
            original_forward, original_body = layer.forward, layer._body

            def forward(*args: Any, _original: Any = original_forward,
                        _body: Any = original_body, _layer: Any = layer,
                        **kwargs: Any) -> Any:
                current = getattr(self.local, "context", None)
                if current is None:
                    raise RuntimeError("decoder forward has no actual input-row context")
                captured = dict(current)
                forward_active = True

                def invocation_body(*body_args: Any, **body_kwargs: Any) -> Any:
                    phase = captured["phase"] if forward_active else "recompute"
                    with self.scope({**captured, "phase": phase}):
                        self.bind(body_args[0], captured["micro_batch"])
                        return _body(*body_args, **body_kwargs)

                try:
                    # The unchanged forward passes this exact callable to checkpoint.
                    # It captures metadata only; no input/output tensor is retained here.
                    with _patch(_layer, "_body", invocation_body):
                        return _original(*args, **kwargs)
                finally:
                    forward_active = False

            self.stack.enter_context(_patch(layer, "forward", forward))
            self.stage(layer.inner, "decoder", index)
        self.stage(self.owner.embed_tokens, "embed")
        self.stage(self.owner.norm, "norm")
        self.stage(self.causal.lm_head, "head")
        original_loss = self.causal.loss_function
        saved_loss_attributes = {
            name: self.causal.__dict__[name] for name in ("loss_function", "_loss_function")
            if name in self.causal.__dict__
        }

        @functools.wraps(original_loss)
        def loss(*args: Any, **kwargs: Any) -> Any:
            result = original_loss(*args, **kwargs)
            context = getattr(self.local, "head_context", None)
            if context is None:
                raise RuntimeError("model loss has no explicit head input-row context")
            window = self.ledger.record["windows"][context["window"]]
            n_items = kwargs.get("num_items_in_batch")
            n_items = int(n_items.item()) if hasattr(n_items, "item") else n_items
            value = float(result.detach().cpu())
            logits = kwargs.get("logits", args[0] if args else None)
            self.ledger.record["per_microbatch_losses"].append({
                **context, **self.state(), "loss": value,
                "supervised_tokens": window["supervised_tokens"][context["micro_batch"]],
                "label_tokens": window["label_tokens"][context["micro_batch"]],
                "n_items": n_items, "n_items_definition": window["n_items_definition"],
                "loss_times_denominator": value * n_items if n_items is not None else None,
                "logits_dtype": str(getattr(logits, "dtype", None)).removeprefix("torch."),
            })
            self.ledger.flush()
            return result

        self.causal.loss_function = loss

        def restore_loss() -> None:
            for name in ("loss_function", "_loss_function"):
                self.causal.__dict__.pop(name, None)
            self.causal.__dict__.update(saved_loss_attributes)

        self.stack.callback(restore_loss)
        original_decide = layer_stream_runtime._can_use_checkpoint_visible_nf4_gemm
        for name, module in self.model.named_modules():
            if type(module).__name__ != "Linear4bit":
                continue

            def push(_module: Any, _args: Any, _name: str = name) -> None:
                modules = getattr(self.local, "modules", None)
                if modules is None:
                    modules = []
                    self.local.modules = modules
                modules.append(_name)
                return None

            def pop(_module: Any, _args: Any, _output: Any) -> None:
                self.local.modules.pop()
                return None

            self.handles.append(module.register_forward_pre_hook(push))
            self.handles.append(module.register_forward_hook(pop, always_call=True))

        @functools.wraps(original_decide)
        def decide(inputs: Any, quant_state: Any) -> bool:
            result = original_decide(inputs, quant_state)
            context = getattr(self.local, "context", None)
            modules = getattr(self.local, "modules", None)
            if context is None or not modules:
                raise RuntimeError("NF4 dispatch lacks module/actual microbatch context")
            width = int(inputs.shape[-1])
            item = {**context, **self.state(), "module": modules[-1],
                    "M": int(inputs.numel() // width), "N": int(quant_state.shape[0]),
                    "K": width, "input_dtype": str(inputs.dtype).removeprefix("torch."),
                    "weight_dtype": str(quant_state.dtype).removeprefix("torch."),
                    "result": bool(result),
                    "path": "checkpoint_visible_custom" if result else "dense_fallback"}
            self.ledger.record["dispatch"].append(item)
            self.dispatch_file.write(json.dumps(_jsonable(item), allow_nan=False) + "\n")
            self.dispatch_file.flush()
            return result

        self.stack.enter_context(_patch(layer_stream_runtime,
                                        "_can_use_checkpoint_visible_nf4_gemm", decide))
        # _l2l_step_for imports capture_layer_call at call time from l2l_step above.
        if l2l_trainer._window_tensors is None:
            raise RuntimeError("missing pinned L2L padding helper")
        return self

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.stack.close()
        if self.dispatch_file is not None:
            self.dispatch_file.close()
        _json(self.ledger.directory / "traces.json", self.traces)
        self.provenance.clear()


@contextlib.contextmanager
def _training_instrumentation(stand: Any, ledger: InputLedger, spec: dict[str, Any],
                              observer: Observer | None) -> Iterator[None]:
    """Keep pinned build/train_recorded/trainer_class/recorder and real L2L path."""
    from soup_cli.trainer import l2l_trainer

    factory, recorder = stand.trainer_class, stand.recorder
    original_window = l2l_trainer._window_tensors

    @functools.wraps(original_window)
    def window_tensors(window: list[Any], device: Any, **kwargs: Any) -> Any:
        result = original_window(window, device, **kwargs)
        ids, labels = result
        ledger.effective([{"input_ids": item, "labels": target}
                          for item, target in zip(ids, labels)], mask_policy="none_causal_default")
        return result

    def trainer_class(l2l: bool, hooks: Any) -> type:
        parent = factory(l2l, hooks)

        class DiagnosticTrainer(parent):
            def get_batch_samples(self, epoch_iterator, num_batches, device):
                samples, n_items = super().get_batch_samples(epoch_iterator, num_batches, device)
                ledger.fetched(samples, n_items)
                if spec["pad"]:
                    ids, labels = l2l_trainer._window_tensors(
                        samples, self.args.device, neftune=bool(self.args.neftune_noise_alpha),
                        has_dropout=l2l_trainer._has_dropout(self.model))
                    # Exactly L2L's effective mask policy: it forwards only ids and labels.
                    samples = [{"input_ids": item, "labels": target}
                               for item, target in zip(ids, labels)]
                elif not l2l:
                    ledger.effective(samples, mask_policy="explicit_original_attention_mask")
                return samples, n_items

            def training_step(self, model, inputs, num_items_in_batch=None):
                mb = ledger.match_ids(inputs["input_ids"])
                context = ledger.context(mb, "gradforward")
                scope = observer.scope(context) if observer else contextlib.nullcontext()
                with scope:
                    result = super().training_step(model, inputs, num_items_in_batch)
                ledger.record["training_step_losses"].append(
                    {**context, "loss": float(result.detach().cpu()),
                     "n_items": int(num_items_in_batch.item())
                     if hasattr(num_items_in_batch, "item") else num_items_in_batch})
                return result

        return DiagnosticTrainer

    def make_recorder(*args: Any, **kwargs: Any) -> Any:
        callback = recorder(*args, **kwargs)
        original = callback.on_pre_optimizer_step

        def before_step(*arguments: Any, **keywords: Any) -> Any:
            optimizer = keywords.get("optimizer")
            ledger.record["applied_rates"].append(float(optimizer.param_groups[0]["lr"]))
            return original(*arguments, **keywords)

        callback.on_pre_optimizer_step = before_step
        return callback

    with _patch(stand, "trainer_class", trainer_class), _patch(stand, "recorder", make_recorder), \
            _patch(l2l_trainer, "_window_tensors", window_tensors):
        yield


def _manifest(directory: Path, record: dict[str, Any]) -> None:
    artifacts = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in ("arm.json", "manifest.json") \
                and not path.name.endswith(".writing"):
            relative = os.path.relpath(path, directory).replace(os.sep, "/")
            artifacts[relative] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    record["artifacts"] = artifacts
    _json(directory / "arm.json", record)
    _json(directory / "manifest.json", {"schema_version": 1, "arm": record["arm"],
                                       "phase_complete": record["phase_complete"],
                                       "exit": record["exit"],
                                       "files": {**artifacts, "arm.json": {
                                           "sha256": _sha(directory / "arm.json"),
                                           "bytes": (directory / "arm.json").stat().st_size}}})


def child_main(args: argparse.Namespace) -> int:
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    spec = ARMS[args.arm]
    record: dict[str, Any] = {
        "schema_version": 1, "arm": args.arm, "kind": "criterion_1b_diagnosis",
        "l2l": spec["l2l"], "traced": spec["trace"], "padded_ga": spec["pad"],
        "device": args.device, "quant": args.quant, "k": args.k, "seq": args.seq,
        "argv": [sys.executable, *sys.argv], "pid": os.getpid(),
        "started_unix": time.time(), "exit": "starting", "phase_complete": False,
        "baseline_commit": PINNED_COMMIT, "expected_steps": spec["steps"],
        "diagnostic_source": {"path": os.path.realpath(__file__),
                              "sha256": _sha(Path(__file__))},
        "seeds": {"model": 3, "nonzero_lora_B": 23, "training": 1, "data": 1,
                  "synthetic_tokens": 5},
        "adapter_config": {"r": 8, "alpha": 16, "targets": ["q_proj", "k_proj", "v_proj",
                                                                "o_proj"], "dropout": 0.0},
        "rehearsal": args.device == "cpu", "causal_verdict": "NO CAUSAL VERDICT",
        "observer": {"enabled": False},
    }
    _json(directory / "arm.json", record)
    _json(directory / "traces.json", [])
    built, observer, sampler, stand = None, None, None, None
    code = 1
    try:
        facts = baseline_facts(args.baseline_path)
        record.update(baseline_source=facts, source_sha256=facts["source_sha256"])
        # Set before importing torch or any baseline modules that can import it.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        stand = _import_baseline(facts)
        stand.base.set_deterministic_env()
        import l2l_schedule
        import stream_probe
        import torch

        if args.device == "cuda" and not stream_probe.cuda_available():
            raise RuntimeError("CUDA unavailable; no synthetic replacement is permitted")
        stand.base.enable_deterministic()
        requested_sdpa = spec.get("sdpa", "MATH")
        # EFFICIENT_ATTENTION has no CPU kernel. This is a real MATH training rehearsal,
        # explicitly not a measurement of the GPU noise floor.
        sdpa = requested_sdpa if args.device == "cuda" else "MATH"
        record.update(sdpa=sdpa, requested_sdpa=requested_sdpa,
                      floor_available=args.device == "cuda",
                      deterministic=stand.deterministic_facts(sdpa),
                      fixture=_fixture_facts(args), env=stand.base.environment_facts())
        if args.device == "cuda":
            record["env"]["gpu_facts"] = stream_probe.gpu_facts(args.device)
        _json(directory / "arm.json", record)
        sampler = stand.MemorySampler(pinned=stand.host_allocator if args.device == "cuda"
                                      else (lambda: None))
        with sampler:
            built = stand.build(args)
            model, config = built["model"], built["config"]
            record.update(dtype=built["dtype"], n_layers=int(config.num_hidden_layers),
                          hidden_size=int(config.hidden_size), build_seconds=built["build_seconds"],
                          shard_dir=built["shard_dir"],
                          cgroup_after_build=sampler.sample("build_end")["cgroup"])
            shard_root = Path(built["shard_dir"])
            record["shard_cache"] = {
                "path": str(shard_root),
                "files": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
                          for path in sorted(shard_root.iterdir()) if path.is_file()
                          and (path.suffix == ".safetensors" or path.name == "index.json")},
                "index": json.loads((shard_root / "index.json").read_text(encoding="utf-8")),
            }
            if args.device == "cuda" and (record["n_layers"], record["hidden_size"]) != (32, 4096):
                raise RuntimeError("CUDA fixture is not historical Mistral-7B")
            l2l_schedule.make_non_vacuous(model, seed=23)
            record["initial_adapter_hashes"] = {
                name: _tensor_digest(_cpu(parameter)) for name, parameter in
                sorted(l2l_schedule.lora_parameters(model).items())}
            if not all(item["nonzero"] for name, item in record["initial_adapter_hashes"].items()
                       if "lora_B" in name):
                raise RuntimeError("nonzero-B initialization did not take effect")
            rows = stand.make_rows("correctness", int(config.vocab_size), k=args.k, seq=args.seq,
                                   rows=args.rows)
            record["rows"] = stand.rows_fingerprint(rows)
            ledger = InputLedger(record, directory, rows, stand)
            record["dependencies"] = _dependency_facts()
            ledger.flush()
            if spec["trace"]:
                observer = Observer(ledger, model, spec["l2l"])
                observer.install()
            options = {"bf16": built["dtype"] == "bfloat16", "use_cpu": args.device == "cpu"}
            if spec["steps"] == 1:
                options.update(max_steps=1, learning_rate=0.0)
            with tempfile.TemporaryDirectory(prefix="a3-1b-trainer-") as work:
                with _training_instrumentation(stand, ledger, spec, observer):
                    stand.train_recorded(
                        record, model, built["runtime"], rows, arm=args.arm, l2l=spec["l2l"],
                        sdpa=sdpa, k=args.k, seq=args.seq, dtype=getattr(torch, built["dtype"]),
                        device=args.device, work_dir=Path(work), spill_dir=None,
                        shard_dir=built["shard_dir"], seed=1, record_grads=True,
                        tensor_dir=directory / "tensors", sampler=sampler,
                        trainer_options=options)
            if record["exit"] != "normal" or record["steps"] != spec["steps"]:
                raise RuntimeError("training did not complete the specified optimizer steps")
            if (record.get("spill_plan") or {}).get("spill_layers", 0):
                raise RuntimeError("diagnostic intended no spill, but product spilled activations")
            record["phase_complete"] = True
            code = 0
    except BaseException as exc:
        record.update(exit="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        cleanup_errors = []
        if observer is not None:
            try:
                observer.close()
            except BaseException as exc:
                cleanup_errors.append(f"observer: {type(exc).__name__}: {exc}")
        if sampler is not None:
            sampler.stop()
            record["memory"] = {"labeled": sampler.labeled(), "failed_readings": sampler.failed}
        if built is not None:
            try:
                built["runtime"].close()
            except BaseException as exc:
                cleanup_errors.append(f"runtime: {type(exc).__name__}: {exc}")
        record.update(ended_unix=time.time(), cleanup_errors=cleanup_errors)
        if "dependencies" not in record:
            try:
                record["dependencies"] = _dependency_facts()
            except Exception as exc:
                cleanup_errors.append(f"dependency hashes: {type(exc).__name__}: {exc}")
                record["dependency_hash_traceback"] = traceback.format_exc()
        if stand is not None:
            root = Path(record["baseline_source"]["path"])
            record["imported_baseline_modules"] = {
                name: source for name, module in sorted(tuple(sys.modules.items()))
                if module is not None and isinstance(source := vars(module).get("__file__"), str)
                and _inside(source, root)}
            harness_names = {Path(path).stem for path in record["source_sha256"]
                             if path.startswith("benchmarks/harness/")}
            for name, module in tuple(sys.modules.items()):
                if name == "soup_cli" or name.startswith("soup_cli.") or name in harness_names:
                    source = vars(module).get("__file__") if module is not None else None
                    if source and not _inside(source, root):
                        cleanup_errors.append(f"unpinned product/harness import: {name}: {source}")
        if cleanup_errors:
            record.update(exit="error", phase_complete=False)
            code = 1
        _manifest(directory, record)
    _say(f"{args.arm}: {record['exit']}; steps={record.get('steps')}; "
         f"phase_complete={record['phase_complete']}")
    return code


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-path", required=True, help="clean pinned baseline checkout")
    parser.add_argument("--weights", required=True, help="local checkpoint; never downloaded")
    parser.add_argument("--shards", default=None, help="shared streamed-layer cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--quant", choices=("nf4", "none"), default="nf4")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--out", required=True, help="new directory; overwrite is refused")


def _command(args: argparse.Namespace, arm: str, directory: Path) -> list[str]:
    return [sys.executable, os.path.realpath(__file__), "child", "--arm", arm,
            "--baseline-path", args.baseline_path, "--weights", args.weights,
            "--shards", args.shards, "--device", args.device, "--quant", args.quant,
            "--k", str(args.k), "--seq", str(args.seq), "--rows", str(args.rows),
            "--out", str(directory)]


def session_main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"REFUSED: {out} exists; attempts are never overwritten")
    facts = baseline_facts(args.baseline_path)
    plan = [{"arm": arm, "command": _command(args, arm, out / "arms" / arm)}
            for arm in args.arms]
    for item in plan:
        _say(subprocess.list2cmdline(item["command"]))
    if not args.execute:
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    session = {"schema_version": 1, "state": "running", "started_unix": time.time(),
               "baseline_source": facts, "diagnostic_source": {"path": os.path.realpath(__file__),
                                                              "sha256": _sha(Path(__file__))},
               "args": vars(args), "plan": plan, "children": [],
               "rehearsal": args.device == "cpu", "causal_verdict": "NO CAUSAL VERDICT"}
    _json(out / "session.json", session)
    deadline = time.monotonic() + args.max_hours * 3600
    code = 0
    try:
        for entry in plan:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("diagnostic session wall-time deadline reached")
            environment = os.environ.copy()
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            child = {"arm": entry["arm"], "started_unix": time.time(),
                     "command": entry["command"], "log": f"logs/{entry['arm']}.log"}
            session["children"].append(child)
            _json(out / "session.json", session)
            with (out / child["log"]).open("xb") as log:
                result = subprocess.run(entry["command"], cwd=args.baseline_path,
                                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=remaining, check=False)
            child.update(returncode=result.returncode, ended_unix=time.time())
            manifest = out / "arms" / entry["arm"] / "manifest.json"
            if manifest.is_file():
                child["manifest_sha256"] = _sha(manifest)
            else:
                child["missing_manifest"] = True
            _json(out / "session.json", session)
            if result.returncode or not manifest.is_file():
                raise RuntimeError(f"{entry['arm']} failed; preserving completed and partial arms")
        session["state"] = "finished"
    except BaseException as exc:
        session.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped",
                       error=f"{type(exc).__name__}: {exc}")
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        session["ended_unix"] = time.time()
        _json(out / "session.json", session)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    session = subparsers.add_parser("session", help="print commands, or execute fresh subprocesses")
    _common(session)
    session.add_argument("--execute", action="store_true")
    session.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    session.add_argument("--max-hours", type=float, default=5.0,
                         help="child work only; does not delete a VM")
    child = subparsers.add_parser("child", help="one arm; normally launched by session")
    _common(child)
    child.add_argument("--arm", required=True, choices=tuple(ARMS))
    args = parser.parse_args(argv)
    for field in ("baseline_path", "weights", "out", "shards"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, os.path.realpath(value))
    if args.k < 1 or args.seq < 8 or args.rows < 1:
        parser.error("k and rows must be positive; seq must be at least 8")
    if args.device == "cpu" and args.quant != "none":
        parser.error("CPU rehearsal requires --quant none")
    if args.device == "cuda" and (args.quant, args.k, args.seq, args.rows) != ("nf4", 4, 512, 10):
        parser.error("CUDA diagnosis fixes historical --quant nf4 --k 4 --seq 512 --rows 10")
    if (args.rows + args.k - 1) // args.k != 3:
        parser.error("original fixture must produce exactly three optimizer windows")
    if args.shards is None:
        args.shards = os.path.realpath(str(Path(args.out) / "shards"))
    if args.command == "session" and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be finite and positive")
    if args.command == "session" and len(args.arms) != len(set(args.arms)):
        parser.error("an arm may run only once in an attempt")
    try:
        return session_main(args) if args.command == "session" else child_main(args)
    except (OSError, RuntimeError, ValueError) as exc:
        _say(f"REFUSED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
