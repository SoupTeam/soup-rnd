"""Deduplicated G3 evidence export and bounded, safe local readback; no GPU imports."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

from rich.console import Console

MAX_FILES = 1000
MAX_TOTAL = 6 * 1024**3
MAX_MANIFEST = 1024**2


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name: str) -> None:
    if (not name or Path(name).is_absolute() or '..' in Path(name).parts
            or '\\' in name or name in ['.', '']):
        raise ValueError('Unsafe logical evidence name')


def pack(root: Path, archive: Path) -> dict:
    root_real, archive_real = os.path.realpath(root), os.path.realpath(archive)
    if os.path.commonpath([root_real, archive_real]) == root_real:
        raise ValueError('Archive must be outside the captured session tree')
    rows, originals, stamps = {}, {}, {}
    total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink evidence refused')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('Only regular files may be exported')
        name = path.relative_to(root).as_posix()
        safe_name(name)
        stat = path.stat()
        total += stat.st_size
        if len(rows) >= MAX_FILES or total > MAX_TOTAL:
            raise ValueError('Evidence exceeds predeclared export bounds')
        digest = file_sha(path)
        rows[name] = {'sha256': digest, 'bytes': stat.st_size}
        originals.setdefault(digest, path)
        stamps[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if not rows:
        raise ValueError('No evidence files')
    raw = (json.dumps({'schema': 'g3-export-v1', 'files': rows}, indent=2) + '\n').encode()
    if len(raw) > MAX_MANIFEST:
        raise ValueError('Manifest too large')
    with archive.open('xb') as target, tarfile.open(fileobj=target, mode='w:gz') as tar:
        item = tarfile.TarInfo('manifest.json')
        item.size, item.mode = len(raw), 0o600
        tar.addfile(item, io.BytesIO(raw))
        for digest, path in originals.items():
            stat = path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != stamps[path]:
                raise ValueError('Evidence changed while exporting')
            item = tarfile.TarInfo('blobs/' + digest)
            item.size, item.mode = stat.st_size, 0o600
            with path.open('rb') as stream:
                tar.addfile(item, stream)
            if file_sha(path) != digest:
                raise ValueError('Evidence changed during archive read')
    return {'schema': 'g3-export-v1', 'archive_sha256': file_sha(archive),
            'archive_bytes': archive.stat().st_size, 'logical_files': len(rows),
            'logical_bytes': total, 'unique_blobs': len(originals)}


def verify(archive: Path, expected_sha: str, output: Path) -> dict:
    if not re.fullmatch('[a-f0-9]{64}', expected_sha) or file_sha(archive) != expected_sha:
        raise ValueError('Archive SHA differs')
    output.mkdir(exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        first = tar.next()
        if (not first or first.name != 'manifest.json' or not first.isfile()
                or first.size > MAX_MANIFEST):
            raise ValueError('Manifest must be the first bounded regular member')
        raw = tar.extractfile(first).read()
        manifest = json.loads(raw)
        if manifest.get('schema') != 'g3-export-v1':
            raise ValueError('Unknown export schema')
        rows = manifest['files']
        if not isinstance(rows, dict) or not 0 < len(rows) <= MAX_FILES:
            raise ValueError('Invalid evidence count')
        sizes = {}
        total = 0
        for name, row in rows.items():
            safe_name(name)
            if (not re.fullmatch('[a-f0-9]{64}', row['sha256'])
                    or type(row['bytes']) is not int or row['bytes'] < 0):
                raise ValueError('Invalid manifest entry')
            total += row['bytes']
            if row['sha256'] in sizes and sizes[row['sha256']] != row['bytes']:
                raise ValueError('Contradictory blob sizes')
            sizes[row['sha256']] = row['bytes']
        if total > MAX_TOTAL:
            raise ValueError('Expanded evidence exceeds bound')
        blobs = output / 'blobs'
        blobs.mkdir()
        seen = set()
        for item in tar:
            # tarfile iterates from its first member even after next(); skip precisely once.
            if item is first:
                continue
            if (not item.isfile() or not re.fullmatch('blobs/[a-f0-9]{64}', item.name)
                    or item.name[6:] not in sizes or item.name in seen
                    or item.size != sizes[item.name[6:]]):
                raise ValueError('Unexpected, linked, duplicated or oversized archive member')
            seen.add(item.name)
            path = blobs / item.name[6:]
            with path.open('xb') as target:
                shutil.copyfileobj(tar.extractfile(item), target, length=1024**2)
            if file_sha(path) != item.name[6:]:
                raise ValueError('Member SHA differs')
        if seen != {'blobs/' + digest for digest in sizes}:
            raise ValueError('Archive members incomplete')
        expanded = output / 'files'
        expanded.mkdir()
        for name, row in rows.items():
            path = expanded / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with (blobs / row['sha256']).open('rb') as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024**2)
            if file_sha(path) != row['sha256']:
                raise ValueError('Expanded file SHA differs')
        (output / 'manifest.json').write_bytes(raw)
    value = {'status': 'ARCHIVE_EVERY_MEMBER_SAFE_EXPANSION_SHA_READBACK_PASS',
             'archive_sha256': expected_sha, 'logical_files': len(rows),
             'unique_blobs': len(sizes), 'logical_bytes': total,
             'readable_without_server': True}
    (output / 'verification.json').write_text(json.dumps(value, indent=2) + '\n')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    pack_args = modes.add_parser('pack')
    pack_args.add_argument('--root', type=Path, required=True)
    pack_args.add_argument('--archive', type=Path, required=True)
    verify_args = modes.add_parser('verify')
    verify_args.add_argument('--archive', type=Path, required=True)
    verify_args.add_argument('--expected-sha', required=True)
    verify_args.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'pack':
        receipt = args.archive.with_name(args.archive.name + '.receipt.json')
        if receipt.exists():
            parser.error('Archive receipt exists; choose a new export path')
        value = pack(args.root, args.archive)
        raw = (json.dumps(value, indent=2) + '\n').encode()
        with receipt.open('xb') as target:
            target.write(raw)
        if receipt.read_bytes() != raw:
            raise OSError('Export receipt readback differs')
        Console().print('Archive and machine-readable receipt saved: ' + str(receipt))
    else:
        value = verify(args.archive, args.expected_sha, args.out)
        Console().print(value['status'])


if __name__ == '__main__':
    main()
