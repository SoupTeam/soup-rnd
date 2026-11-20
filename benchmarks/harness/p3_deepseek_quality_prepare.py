"""Explicitly acknowledged, immutable profile downloads and NF4 preparation stages.

No provider operations, gate, optimizer step or quality evaluation are performed here.
The acknowledgement is an operator assertion, not proof of external human approval.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

from benchmarks.harness import p3_deepseek_quality_model as model_api
from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

REQUIRED_METADATA = frozenset({
    "config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json",
})
OPTIONAL_METADATA = frozenset({
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "vocab.json", "merges.txt",
    "generation_config.json",
})
_MOONLIGHT_METADATA = frozenset({
    "config.json", "model.safetensors.index.json", "generation_config.json",
    "tokenizer_config.json", "tiktoken.model",
})
WEIGHTS = get_profile("deepseek-v3").weight_files
MIN_FREE_BYTES = get_profile("deepseek-v3").min_free_bytes
MIN_RAM_BYTES = get_profile("deepseek-v3").min_ram_bytes
MAX_METADATA_BYTES = 32 * 2**20
PARTIAL_SUFFIX = ".partial"
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (10, 30)


class TruncatedPayloadError(ValueError):
    """The connection ended before the verified byte count: a transient transport failure."""


_TRANSIENT_ERRORS = (
    urllib.error.URLError, TimeoutError, socket.timeout, http.client.IncompleteRead,
    ConnectionError, TruncatedPayloadError,
)
_sleep = time.sleep


def safe_child(root: Path, name: str) -> Path:
    return model_api._child(root, name)


def _value(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, dict) else getattr(record, name, None)


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def hub_inventory(
    info: Any, *, profile_id: str = "deepseek-v3",
) -> dict[str, dict[str, Any]]:
    """Admit complete pinned Hub metadata before any payload is requested."""
    profile = get_profile(profile_id)
    weights = profile.weight_files
    required = REQUIRED_METADATA if profile_id == "deepseek-v3" else _MOONLIGHT_METADATA
    optional = OPTIONAL_METADATA if profile_id == "deepseek-v3" else frozenset()
    if _value(info, "sha") != profile.revision:
        raise ValueError("Hub commit differs from the pinned revision")
    siblings = _value(info, "siblings")
    if not isinstance(siblings, (list, tuple)) or not siblings:
        raise ValueError("Hub siblings metadata is missing or malformed")
    rows = {}
    all_weights = set()
    for sibling in siblings:
        name = _value(sibling, "rfilename")
        safe_child(Path("."), name)
        if name in rows:
            raise ValueError("duplicate Hub file metadata")
        rows[name] = sibling
        if name.endswith(".safetensors"):
            all_weights.add(name)
    if all_weights != weights or not required.issubset(rows):
        raise ValueError("Hub inventory requires all pinned profile shards and metadata")
    records = {}
    for name in sorted(weights | required | (optional & rows.keys())):
        row = rows[name]
        size, lfs = _value(row, "size"), _value(row, "lfs")
        if type(size) is not int or size <= 0:
            raise ValueError(f"missing or malformed Hub byte size: {name}")
        if name not in weights and size > MAX_METADATA_BYTES:
            raise ValueError(f"small metadata exceeds bounded hashing size: {name}")
        hub_sha256 = None
        if lfs is not None:
            hub_sha256 = _value(lfs, "sha256")
            if not _hex(hub_sha256, 64) or _value(lfs, "size") != size:
                raise ValueError(f"missing/malformed LFS hash or size: {name}")
            etag = hub_sha256
        else:
            etag = _value(row, "blob_id")
            if name in weights or not _hex(etag, 40):
                raise ValueError(f"missing verified Hub content hash: {name}")
        records[name] = {"bytes": size, "hub_etag": etag, "hub_sha256": hub_sha256}
    expected_bytes = model_api.SOURCE_BYTES if profile_id == "deepseek-v3" else profile.weight_bytes
    if sum(records[name]["bytes"] for name in weights) != expected_bytes:
        raise ValueError("Hub weight byte total differs from the pinned profile")
    pinned = (model_api.source_inventory() if profile_id == "deepseek-v3"
              else model_api.source_inventory(profile_id))
    if records != pinned:
        raise ValueError("Hub inventory differs from the reviewed source pin")
    return records


def verify_head(
    metadata: Any, descriptor: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> None:
    etag = _value(metadata, "etag")
    if isinstance(etag, str):
        etag = etag.strip('"')
    location = _value(metadata, "location")
    if (_value(metadata, "commit_hash") != get_profile(profile_id).revision
            or etag != descriptor["hub_etag"]
            or type(_value(metadata, "size")) is not int
            or _value(metadata, "size") != descriptor["bytes"]
            or not isinstance(location, str)
            or urllib.parse.urlsplit(location).scheme != "https"):
        raise ValueError("Hub HEAD commit, ETag, byte size or HTTPS location is unverified")


def local_descriptor(
    path: Path, remote: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != remote["bytes"]:
        raise ValueError(f"downloaded file size/type differs: {path.name}")
    if path.name.endswith(".safetensors"):
        if not _hex(remote.get("hub_sha256"), 64):
            raise ValueError("weight content identity requires a verified Hub LFS SHA256")
        digest = remote["hub_sha256"]
        origin = "hub_lfs_content_identity_not_locally_hashed"
    else:
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("metadata hashing bound exceeded")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if remote["hub_sha256"] is not None:
            expected = digest
        else:
            expected = hashlib.sha1(
                b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
        if expected != remote["hub_etag"]:
            raise ValueError(f"local metadata differs from Hub content identity: {path.name}")
        origin = "locally_measured_metadata"
    return {**remote, "sha256": digest, "sha256_origin": origin,
            "hub_commit": get_profile(profile_id).revision}


def source_receipt(
    records: dict[str, dict[str, Any]], *, profile_id: str = "deepseek-v3",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    if profile_id == "moonlight-16b-a3b" and config is None:
        raise ValueError("Moonlight source receipt requires the original source config")
    auxiliary = model_api._source_aux_metadata(config or {}, profile_id=profile_id)
    return {
        "schema": "p3-deepseek-quality-checkpoint-identity-v1", "profile_id": profile_id,
        "model_id": profile.model_id, "revision": profile.revision,
        "source_files": records, **auxiliary,
        "inventory_sha256": profile.source_inventory_sha256,
        "source_verification": "hub_commit_etag_and_size_plus_cache_source_fingerprint",
        "weight_content_sha256_measured_this_session": False,
        "weight_hash_scope": "Hub LFS content identities; local payload byte sizes checked. "
                             "No claim of a fresh local SHA256 pass over the source checkpoint.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _inside(root: Path, child: Path) -> bool:
    try:
        parent, candidate = os.path.realpath(root), os.path.realpath(child)
        return os.path.commonpath((parent, candidate)) == parent
    except ValueError:
        return False


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def inspect_nvme(
    target: Path, *, proc: Path = Path("/proc"), sysfs: Path = Path("/sys"),
    device_number: str | None = None,
) -> dict[str, Any]:
    """Bind the exact target mount to a direct, nonrotating PCIe NVMe namespace.

    Device mapper/RAID and hidden sysfs are refused, rather than guessed from a name,
    unrelated NVMe device, offer description, rotational bit or throughput alone.
    """
    target = Path(os.path.realpath(target))
    existing = target
    while not existing.exists():
        existing = existing.parent
    observed = existing.stat().st_dev
    device_number = device_number or f"{os.major(observed)}:{os.minor(observed)}"
    candidates = []
    for line in (proc / "self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        point = Path(_unescape_mount(fields[4]))
        if _inside(point, target):
            candidates.append((len(str(point)), fields, filesystem, str(point)))
    if not candidates:
        raise ValueError("cannot identify target mount from Linux mountinfo")
    longest = max(row[0] for row in candidates)
    matches = [row for row in candidates if row[0] == longest]
    if len(matches) != 1:
        raise ValueError("stacked target mounts are ambiguous")
    _, fields, filesystem, mountpoint = matches[0]
    if filesystem[0] not in {"ext4", "xfs"} or fields[2] != device_number:
        raise ValueError("target is overlay, network, unsupported or not the observed block mount")
    if "rw" not in fields[5].split(","):
        raise ValueError("target mount is not writable")
    block = Path(os.path.realpath(sysfs / "dev/block" / device_number))
    if not block.is_dir() or not _inside(sysfs / "devices", block):
        raise ValueError("target block-device sysfs provenance is unavailable")
    if (block / "dev").read_text(encoding="ascii").strip() != device_number:
        raise ValueError("sysfs block device differs from target mount")
    namespace = block.parent if (block / "partition").is_file() else block
    controller = namespace.parent
    subsystem = Path(os.path.realpath(controller / "subsystem")).name
    transport = (controller / "transport").read_text(encoding="ascii").strip()
    rotational = (namespace / "queue/rotational").read_text(encoding="ascii").strip()
    if (subsystem != "nvme" or transport != "pcie" or rotational != "0"
            or "/virtual/" in str(block).replace("\\", "/")):
        raise ValueError("target lacks direct local nonrotating PCIe NVMe provenance")
    return {"path": str(target), "mountpoint": mountpoint, "filesystem": filesystem[0],
            "major_minor": device_number, "mount_source": _unescape_mount(filesystem[1]),
            "sysfs_block": str(block), "sysfs_controller": str(controller),
            "transport": transport, "rotational": int(rotational), "subsystem": subsystem,
            "provenance": "target st_dev + mountinfo major:minor + sysfs controller transport"}


def _require_visible_cgroup_limits(
    proc: Path, memberships: str, v2_root: Path | None, v1_root: Path | None,
) -> list[tuple[Path, Path, int]]:
    """Refuse a hidden or malformed controller instead of assuming host RAM is usable."""
    mounts = []
    visible = []
    mountinfo = proc / "self/mountinfo"
    if mountinfo.is_file():
        for line in mountinfo.read_text(encoding="ascii").splitlines():
            left, right = line.split(" - ", 1)
            fields, filesystem = left.split(), right.split()
            mounts.append((fields, filesystem))
    memory_in_v1 = any("memory" in line.split(":", 2)[1].split(",")
                       for line in memberships.splitlines())
    for line in memberships.splitlines():
        identifier, controllers, member = line.split(":", 2)
        version = 2 if identifier == "0" and not controllers else 1
        if version == 2 and memory_in_v1:
            continue
        if version == 1 and "memory" not in controllers.split(","):
            continue
        if ".." in member.split("/") or not member.startswith("/"):
            raise ValueError("cgroup membership path is unsafe")
        explicit = v2_root if version == 2 else v1_root
        marker = "memory.max" if version == 2 else "memory.limit_in_bytes"
        candidates = []
        if explicit is not None:
            candidates.append((explicit, member.lstrip("/")))
        else:
            for fields, filesystem in mounts:
                if (version == 2 and filesystem[0] != "cgroup2") or (
                    version == 1 and (filesystem[0] != "cgroup"
                                      or "memory" not in filesystem[2].split(","))
                ):
                    continue
                root = _unescape_mount(fields[3]).rstrip("/")
                if root and member != root and not member.startswith(root + "/"):
                    continue
                point = Path(_unescape_mount(fields[4]))
                candidates.append((point, member[len(root):].lstrip("/")))
        found = False
        for root, relative in candidates:
            root = Path(os.path.realpath(root))
            directory = safe_child(root, relative) if relative else root
            path = directory / marker
            if not path.is_file():
                continue
            raw = path.read_text(encoding="ascii").strip()
            if raw != "max" and not raw.isdigit():
                raise ValueError("cgroup memory capacity is malformed")
            found = True
            visible.append((directory, root, version))
        if not found:
            raise ValueError("own cgroup memory limit is hidden or unavailable")
    return visible


def inspect_ram(
    *, proc: Path = Path("/proc"), v2_root: Path | None = None, v1_root: Path | None = None,
) -> dict[str, Any]:
    from soup_cli.utils.l2l_spill import cgroup_levels

    text = (proc / "meminfo").read_text(encoding="ascii")
    match = re.search(r"^MemTotal:\s+(\d+) kB$", text, re.MULTILINE)
    if match is None:
        raise ValueError("physical host RAM cannot be established")
    memberships = (proc / "self/cgroup").read_text(encoding="ascii")
    if not memberships.strip():
        raise ValueError("cgroup membership cannot be established")
    _require_visible_cgroup_limits(proc, memberships, v2_root, v1_root)
    levels, failures = cgroup_levels(
        proc_root=str(proc), v2_root=str(v2_root) if v2_root is not None else None,
        v1_root=str(v1_root) if v1_root is not None else None,
    )
    if failures:
        raise ValueError(f"cgroup RAM bounds are unreadable: {failures}")
    host = int(match[1]) * 1024
    return {"host_bytes": host, "effective_bytes": min([host, *(level.limit for level in levels)]),
            "cgroups": [asdict(level) for level in levels],
            "capacity_scope": "physical RAM bounded by visible cgroup limits, not swap"}


def require_capacity(
    free_bytes: int, memory: dict[str, Any], *, reused_bytes: int = 0,
    profile_id: str = "deepseek-v3",
) -> None:
    profile = get_profile(profile_id)
    if free_bytes + reused_bytes < profile.min_free_bytes:
        minimum = "1.4TB" if profile_id == "deepseek-v3" else "150GB"
        raise ValueError(f"preparation requires at least {minimum} free on admitted local NVMe")
    if memory["effective_bytes"] < profile.min_ram_bytes:
        raise ValueError(
            f"preparation requires at least {profile.min_ram_bytes // 2**30}GiB effective RAM")


def admit_machine(
    target: Path, *, download: bool = False, reused_bytes: int = 0,
    profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    if sys.platform != "linux":
        raise ValueError("production preparation requires Linux physical NVMe provenance")
    if not Path("/data").is_dir() or not _inside(Path("/data"), target):
        raise ValueError("source/cache target must be contained by the mounted /data directory")
    storage, memory = inspect_nvme(target), inspect_ram()
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if download:
        require_capacity(free, memory, reused_bytes=reused_bytes, profile_id=profile_id)
    elif memory["effective_bytes"] < profile.min_ram_bytes:
        raise ValueError(
            f"conversion requires at least {profile.min_ram_bytes // 2**30}GiB effective RAM")
    return {"profile_id": profile_id, "storage": storage, "memory": memory,
            "free_bytes_before": free, "reused_bytes_credit": reused_bytes,
            "minimum_download_free_bytes": profile.min_free_bytes,
            "minimum_effective_ram_bytes": profile.min_ram_bytes}


def _source_path(raw: str, *, profile_id: str = "deepseek-v3") -> Path:
    path = Path(raw)
    revision = get_profile(profile_id).revision
    if path.name != revision or Path(os.path.realpath(path)).name != revision:
        raise ValueError("source directory basename must be the pinned revision")
    return Path(os.path.realpath(path))


def _separate(*paths: Path) -> None:
    for number, first in enumerate(paths):
        for second in paths[number + 1:]:
            if _inside(first, second) or _inside(second, first):
                raise ValueError("source, cache and receipt output must be separate directories")


def _begin(args: argparse.Namespace, command: str) -> tuple[Path, dict[str, Any], float]:
    if not args.approved_pilot:
        raise ValueError("--approved-pilot operator acknowledgement is required")
    profile = get_profile(getattr(args, "profile_id", "deepseek-v3"))
    raw_output = Path(args.output)
    if raw_output.exists():
        raise FileExistsError(raw_output)
    destinations = [raw_output]
    if getattr(args, "source", None):
        destinations.append(Path(args.source))
    if command == "convert" and getattr(args, "cache", None):
        destinations.append(Path(args.cache))
    _separate(*destinations)
    output = Path(os.path.realpath(args.output))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    return output, {
        "schema": "p3-deepseek-quality-prepare-v1", "command": command, "status": "FAILED",
        "profile_id": profile.profile_id, "model_id": profile.model_id,
        "revision": profile.revision,
        "started_unix": time.time(), "operator_assertion": {
            "approved_pilot_acknowledged": True, "external_human_approval_verified": False,
            "scope": "operator assertion only; not proof of human approval or a spending grant",
        },
    }, started


def _finish(output: Path, manifest: dict[str, Any], started: float) -> None:
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["finished_unix"] = time.time()
    write_json(output / "manifest.json", manifest)


def _failure(manifest: dict[str, Any], error: BaseException) -> None:
    manifest["status"] = "FAILED"
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    """Close redirect bodies without draining them; require HTTPS at every hop."""

    def http_error_302(self, request, response, code, message, headers):
        try:
            location = headers.get("Location")
            destination = urllib.parse.urljoin(request.full_url, location or "")
            count = getattr(request, "_pilot_redirects", 0)
            if (not location or urllib.parse.urlsplit(destination).scheme != "https"
                    or count >= 5):
                raise ValueError("download redirect must stay HTTPS within five hops")
            following = urllib.request.Request(destination, headers=dict(request.header_items()))
            following._pilot_redirects = count + 1
        finally:
            response.close()
        return self.parent.open(following, timeout=request.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _open_https(request):
    if urllib.parse.urlsplit(request.full_url).scheme != "https":
        raise ValueError("download must start on HTTPS")
    return urllib.request.build_opener(_HttpsRedirects()).open(request, timeout=120)


def _stream_download(url: str, path: Path, descriptor: dict[str, Any]) -> None:
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with _open_https(request) as response, path.open("xb") as stream:
        if response.status != 200 or urllib.parse.urlsplit(response.url).scheme != "https":
            raise ValueError("download response must be a complete HTTPS 200 payload")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdigit() or int(length) != descriptor["bytes"]):
            raise ValueError("download response byte length differs from verified Hub metadata")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("download requires identity content encoding")
        remaining = descriptor["bytes"]
        while remaining:
            chunk = response.read(min(4 * 2**20, remaining))
            if not chunk:
                raise TruncatedPayloadError("downloaded payload is truncated")
            stream.write(chunk)
            remaining -= len(chunk)
        if response.read(1):
            raise ValueError("downloaded payload exceeds verified Hub byte size")
        stream.flush()
        os.fsync(stream.fileno())


def _check_metadata(source: Path, *, profile_id: str = "deepseek-v3") -> None:
    profile = get_profile(profile_id)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(
        config.get(name) != value for name, value in profile.shape.items()
    ):
        name = profile.model_id.rsplit("/", 1)[-1]
        raise ValueError(f"downloaded config differs from the production {name} shape")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or set(weight_map.values()) != profile.weight_files:
        raise ValueError("downloaded checkpoint index does not cover all pinned profile shards")
    if profile_id == "moonlight-16b-a3b":
        model_api._validate_moonlight_index(config, weight_map)
        model_api._source_aux_metadata(config, profile_id=profile_id)
    names = ("tokenizer.json", "tokenizer_config.json") if profile_id == "deepseek-v3" else (
        "generation_config.json", "tokenizer_config.json")
    for name in names:
        value = json.loads((source / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value:
            raise ValueError(f"downloaded metadata must be a nonempty JSON object: {name}")


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_transient(error: BaseException) -> bool:
    """Transport failures worth a fresh signed URL; never interruptions or integrity errors."""
    return not isinstance(error, InterruptedError) and isinstance(error, _TRANSIENT_ERRORS)


def _download_file(
    source: Path, name: str, descriptor: dict[str, Any], head: Any, *,
    profile_id: str = "deepseek-v3",
) -> int:
    """Write one file atomically via ``<name>.partial``; return the attempt count used."""
    final, partial = safe_child(source, name), safe_child(source, f"{name}{PARTIAL_SUFFIX}")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        if partial.is_symlink() or partial.exists():
            partial.unlink()
        try:
            metadata = head(name)
            verify_head(metadata, descriptor, profile_id=profile_id)
            _stream_download(metadata.location, partial, descriptor)
        except Exception as error:
            if not _is_transient(error) or attempt == DOWNLOAD_ATTEMPTS:
                raise
            _sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
            continue
        if final.is_symlink() or final.exists():
            raise FileExistsError(final)
        os.replace(partial, final)
        _fsync_directory(source)
        return attempt
    raise AssertionError("unreachable: the final attempt returns or raises")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def _scan_resume(
    source: Path, raw_source: str, pin: dict[str, dict[str, Any]],
) -> tuple[list[Path], list[str]]:
    """Read-only structural admission of an interrupted source directory."""
    if Path(raw_source).is_symlink() or source.is_symlink() or not source.is_dir():
        raise ValueError("--resume requires an existing real source directory")
    if os.path.lexists(source / "checkpoint-identity.json"):
        raise ValueError("--resume refused: source already contains checkpoint-identity.json")
    partials, finals = [], []
    for entry in sorted(os.scandir(source), key=lambda item: item.name):
        status = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise ValueError(f"--resume refused: not a regular file: {entry.name}")
        base = entry.name.removesuffix(PARTIAL_SUFFIX)
        if base not in pin:
            raise ValueError(f"--resume refused: unexpected file in source: {entry.name}")
        if base == entry.name:
            if status.st_size != pin[base]["bytes"]:
                raise ValueError(f"--resume refused: complete file has the wrong size: {base}")
            finals.append(base)
        else:
            partials.append(Path(entry.path))
    return partials, finals


def _verify_reused(
    source: Path, name: str, remote: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    path = safe_child(source, name)
    record = local_descriptor(path, remote, profile_id=profile_id)
    if name in get_profile(profile_id).weight_files:
        if _sha256_file(path) != remote["hub_sha256"]:
            raise ValueError(f"--resume refused: local SHA256 differs from Hub LFS hash: {name}")
        record = {**record, "sha256_origin": "locally_hashed_on_resume"}
    return {**record, "resumed": True}


def download(args: argparse.Namespace) -> dict[str, Any]:
    output, manifest, started = _begin(args, "download")
    profile_id = manifest["profile_id"]
    profile = get_profile(profile_id)
    weights = profile.weight_files
    profile_kwargs = {} if profile_id == "deepseek-v3" else {"profile_id": profile_id}
    resume = bool(getattr(args, "resume", False))
    try:
        source = _source_path(args.source, profile_id=profile_id)
        _separate(source, output)
        reused: dict[str, dict[str, Any]] = {}
        partials: list[Path] = []
        finals: list[str] = []
        pin: dict[str, dict[str, Any]] = {}
        credit = 0
        if resume:
            pin = model_api.source_inventory(**profile_kwargs)
            partials, finals = _scan_resume(source, args.source, pin)
            credit = sum(pin[name]["bytes"] for name in finals)
        elif source.exists():
            raise FileExistsError(source)
        manifest["source_path"] = str(source)
        manifest["resumed"] = resume
        manifest["machine"] = admit_machine(
            source, download=True, reused_bytes=credit, **profile_kwargs)
        if resume:
            # Any verification failure below refuses the whole run, so the provisional
            # capacity credit is never spent on an unverified file.
            for partial in partials:
                partial.unlink()
            for name in finals:
                reused[name] = _verify_reused(source, name, pin[name], **profile_kwargs)
            manifest["reused_files"] = sorted(reused)
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

        info = HfApi(token=False).model_info(
            profile.model_id, revision=profile.revision, files_metadata=True)
        inventory = hub_inventory(info, profile_id=profile_id)
        for name, descriptor in inventory.items():
            url = hf_hub_url(profile.model_id, name, revision=profile.revision)
            metadata = get_hf_file_metadata(url, token=False)
            verify_head(metadata, descriptor, profile_id=profile_id)
        write_json(output / "hub-inventory.json", {
            "profile_id": profile_id, "model_id": profile.model_id, "revision": profile.revision,
            "files": inventory, "all_head_commit_etag_size_verified": True,
        })
        if not resume:
            source.mkdir(parents=True, exist_ok=False)

        def head(file_name: str) -> Any:
            return get_hf_file_metadata(
                hf_hub_url(profile.model_id, file_name, revision=profile.revision),
                token=False)

        records = {}
        metadata_names = sorted(set(inventory) - weights)
        for name in [*metadata_names, *sorted(weights)]:
            if name in weights and not records.keys() & weights:
                _check_metadata(source, profile_id=profile_id)
            if name in reused:
                records[name] = reused[name]
                file_receipt = {**records[name], "attempts": 0}
            else:
                attempts = _download_file(source, name, inventory[name], head, **profile_kwargs)
                records[name] = local_descriptor(
                    safe_child(source, name), inventory[name], **profile_kwargs)
                file_receipt = {**records[name], "attempts": attempts}
            write_json(output / f"{name}.receipt.json", {"profile_id": profile_id, **file_receipt})
        receipt = source_receipt({
            name: {key: value for key, value in record.items() if key != "resumed"}
            | ({"sha256_origin": "hub_lfs_content_identity_not_locally_hashed"}
               if name in weights else {})
            for name, record in records.items()
        }, profile_id=profile_id,
            config=json.loads((source / "config.json").read_text(encoding="utf-8")))
        write_json(source / "checkpoint-identity.json", receipt)
        identity = admit_source(source, **profile_kwargs)
        manifest.update(
            status="PASS", source=identity,
            checkpoint_identity_sha256=model_api.sha256(source / "checkpoint-identity.json"),
        )
    except BaseException as error:
        _failure(manifest, error)
        raise
    finally:
        _finish(output, manifest, started)
    return manifest


def admit_source(source: Path, *, profile_id: str = "deepseek-v3") -> dict[str, Any]:
    from soup_cli.utils.layer_shard import (
        checkpoint_source_components,
        fingerprint_source_files,
        source_identity_includes_config,
    )

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    identity = model_api._source_identity(source, config, profile_id=profile_id)
    components = tuple((name, row["bytes"], row["mtime_ns"])
                       for name, row in identity["files"].items())
    components = checkpoint_source_components(
        str(source), components,
        include_config=source_identity_includes_config(str(source), "deepseek_v3"),
    )
    identity["cache_source_fingerprint"] = fingerprint_source_files(components)
    model_api.require_production_source(identity, profile_id=profile_id)
    return identity


def memory_observation(*args: Any, **kwargs: Any) -> Any:
    from benchmarks.harness.p3_fp8_smoke import MemoryObservation

    return MemoryObservation(*args, **kwargs)


def cgroup_observation(*, proc: Path = Path("/proc")) -> dict[str, Any]:
    text = (proc / "self/cgroup").read_text(encoding="ascii")
    visible = _require_visible_cgroup_limits(proc, text, None, None)
    peaks, unavailable = {}, []
    for directory, root, version in visible:
        filename = "memory.peak" if version == 2 else "memory.max_usage_in_bytes"
        while True:
            path = directory / filename
            try:
                raw = path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                unavailable.append({"path": str(path), "reason": "peak file not exposed"})
            else:
                if not raw.isdigit():
                    raise ValueError(f"cgroup peak is malformed: {path}")
                peaks[str(path)] = {"path": str(path), "bytes": int(raw)}
            if directory == root:
                break
            directory = directory.parent
    return {
        "status": "MEASURED" if peaks else "UNAVAILABLE",
        "observed_hierarchy_peaks": list(peaks.values()), "unavailable": unavailable,
        "peak_scope": "visible cgroup lifetime, not reset or added to RSS/CUDA",
        "unavailable_policy": "absent kernel counters are explicit; read/parse errors refuse",
    }


def require_memory_evidence(value: dict[str, Any], *, cuda: bool = True) -> None:
    required = ["process_rss_sampled_peak_bytes"]
    if cuda:
        required += ["cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes"]
    if (value.get("sampling_errors") != []
            or any(type(value.get(key)) is not int or value[key] <= 0 for key in required)):
        raise ValueError("required RSS/CUDA memory observations are incomplete or errored")


def require_ampere_cuda(device: str) -> dict[str, Any]:
    from benchmarks.harness.p3_deepseek_quality_attention import require_ampere_cuda as require

    return require(device)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    output, manifest, started = _begin(args, "convert")
    profile_id = manifest["profile_id"]
    profile = get_profile(profile_id)
    profile_kwargs = {} if profile_id == "deepseek-v3" else {"profile_id": profile_id}
    session = observation = None
    previous_threads = None
    try:
        source = _source_path(args.source, profile_id=profile_id)
        cache = Path(os.path.realpath(args.cache))
        _separate(source, cache, output)
        if cache.exists():
            raise FileExistsError(cache)
        if cache.parent.is_symlink() or not cache.parent.is_dir():
            raise ValueError(
                "cache parent directory must already exist on the admitted /data volume")
        manifest.update(source_path=str(source), cache_path=str(cache), shard_threads=8,
                        quant="nf4", double_quant=False, expert_layout="per_expert",
                        compute_dtype="bfloat16", device="cuda", seed=23)
        manifest["source_machine"] = admit_machine(source, **profile_kwargs)
        manifest["cache_machine"] = admit_machine(cache, **profile_kwargs)
        if source.stat().st_dev != cache.parent.stat().st_dev:
            raise ValueError("source and cache must occupy the same admitted /data NVMe")
        manifest["source_before"] = admit_source(source, **profile_kwargs)
        manifest["gpu"] = require_ampere_cuda("cuda")
        from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

        budget = new_expert_cache_budget(
            [str(source / name) for name in sorted(profile.weight_files)])
        free = shutil.disk_usage(cache.parent).free
        manifest.update(cache_budget_bytes=budget, cache_free_bytes_before=free)
        if free < budget:
            raise ValueError("insufficient free NVMe space for the production NF4 cache budget")
        import torch

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(8)
        observation = memory_observation("cuda", scope=(
            "Full NF4 noDQ per-expert preparation, native session, seed23 initial adapter "
            "and session cleanup. Whole process/allocator peaks, not activation attribution."))
        with observation:
            try:
                session = model_api.open_session(
                    source, cache, device="cuda", dtype="bfloat16", prepare=True,
                    **profile_kwargs)
                model_api.require_production_source(session.metadata["source"], **profile_kwargs)
                manifest["session"] = session.metadata
                manifest["initial_adapter"] = model_api.write_initial_adapter(
                    session.model, output / "initial-adapter.safetensors", seed=23,
                    **profile_kwargs)
                manifest["initial_adapter"]["path"] = str(output / "initial-adapter.safetensors")
            finally:
                active_error = sys.exc_info()[1]
                if session is not None:
                    try:
                        session.close()
                    except BaseException as cleanup_error:
                        manifest["cleanup_error"] = {
                            "type": type(cleanup_error).__name__, "message": str(cleanup_error),
                        }
                        if active_error is None:
                            raise
        manifest["status"] = "PASS"
    except BaseException as error:
        _failure(manifest, error)
        raise
    finally:
        if previous_threads is not None:
            try:
                torch.set_num_threads(previous_threads)
            except BaseException as cleanup_error:
                manifest["status"] = "FAILED"
                manifest["thread_restore_error"] = str(cleanup_error)
        manifest["session_closed"] = None if session is None else bool(session.closed)
        manifest["resources"] = {
            "memory": {} if observation is None else observation.result,
            "resources_are_not_summed": True,
        }
        try:
            manifest["resources"]["cgroup"] = cgroup_observation()
            if manifest["status"] == "PASS":
                require_memory_evidence(manifest["resources"]["memory"])
        except (OSError, ValueError) as error:
            manifest["resources"]["evidence_error"] = str(error)
            manifest["status"] = "FAILED"
        _finish(output, manifest, started)
    if manifest["status"] != "PASS":
        raise RuntimeError("conversion cleanup or resource evidence failed; see manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "convert"):
        child = commands.add_parser(command)
        child.add_argument("--profile", dest="profile_id", default="deepseek-v3",
                           choices=("deepseek-v3", "moonlight-16b-a3b"))
        child.add_argument("--source", required=True,
                           help="source directory with pinned-revision basename")
        child.add_argument("--output", required=True, help="fresh immutable receipt directory")
        child.add_argument("--approved-pilot", action="store_true",
                           help="operator asserts pilot approval; not external approval proof")
        if command == "convert":
            child.add_argument("--cache", required=True, help="fresh per-expert NF4 cache")
        else:
            child.add_argument(
                "--resume", action="store_true",
                help="continue an interrupted download in the existing source directory; every "
                     "complete file is re-verified and .partial files are discarded")
    return parser


def _terminate(number: int, _frame: Any) -> None:
    raise InterruptedError(f"preparation interrupted by signal {number}")


def main() -> int:
    args = build_parser().parse_args()
    console = Console(width=100)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        result = download(args) if args.command == "download" else convert(args)
    except (Exception, KeyboardInterrupt) as error:
        console.print(f"Preparation refused/failed: {error}", markup=False)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    console.print({"status": result["status"], "command": result["command"],
                   "wall_seconds": result["wall_seconds"], "output": args.output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
