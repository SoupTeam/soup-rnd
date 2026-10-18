"""Verify and restore the original D1 evidence with Python 3.10+ standard library."""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New, empty destination')
    args = parser.parse_args()
    here = Path(__file__).absolute().parent
    manifest = json.loads((here / 'manifest.json').read_text(encoding='utf-8'))
    destination = Path(os.path.realpath(args.output))
    if destination.exists():
        parser.error('Destination must not exist; existing files are never overwritten')
    records = manifest['files']
    for record in records:
        target = os.path.realpath(destination / record['path'])
        if os.path.commonpath([str(destination), target]) != str(destination):
            raise ValueError(f"Unsafe destination path: {record['path']}")

    with tempfile.TemporaryDirectory(prefix='d1-restore-') as temporary:
        joined = Path(temporary) / manifest['archive']
        whole = hashlib.sha256()
        with joined.open('wb') as output:
            for part in manifest['parts']:
                source = here / part['path']
                digest = hashlib.sha256()
                count = 0
                with source.open('rb') as stream:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        whole.update(chunk)
                        count += len(chunk)
                if count != part['bytes'] or digest.hexdigest() != part['sha256']:
                    raise ValueError(f"Corrupt part: {part['path']}")
        if joined.stat().st_size != manifest['archive_bytes']:
            raise ValueError('Incorrect archive length')
        if whole.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Incorrect whole archive SHA-256')
        with zipfile.ZipFile(joined) as package:
            names = package.namelist()
            if len(names) != len(records) or set(names) != {r['path'] for r in records}:
                raise ValueError('Archive inventory mismatch')
            # Verify every record before writing any destination file.
            for record in records:
                data = package.read(record['path'])
                header = f'blob {len(data)}\0'.encode()
                if (len(data) != record['bytes']
                        or hashlib.sha256(data).hexdigest() != record['sha256']
                        or hashlib.sha1(header + data).hexdigest() != record['git_blob']):
                    raise ValueError(f"Corrupt original Git blob: {record['path']}")
            destination.mkdir(parents=True, exist_ok=False)
            for record in records:
                target = destination / record['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(package.read(record['path']))
                target.chmod(int(record['mode'], 8) & 0o777)
    print(f"Verified and restored {len(records)} original files to {destination}")


if __name__ == '__main__':
    main()
