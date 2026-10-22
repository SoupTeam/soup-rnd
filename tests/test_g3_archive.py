"""Mutation tests for local-only G3 preservation, with no GPU or network."""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'benchmarks/harness'))
import g3_archive as archive  # noqa: E402


def test_unique_bytes_and_every_logical_file_roundtrip(tmp_path):
    root = tmp_path/'session'
    root.mkdir()
    (root/'readback1.bin').write_bytes(bytes(range(256))*100)
    (root/'readback2.bin').write_bytes((root/'readback1.bin').read_bytes())
    (root/'raw.json').write_text('{"status":"NO_VERDICT"}\n')
    packed = tmp_path/'evidence.tar.gz'
    receipt = archive.pack(root, packed)
    assert receipt['logical_files'] == 3 and receipt['unique_blobs'] == 2
    result = archive.verify(packed, receipt['archive_sha256'], tmp_path/'verified')
    assert result['readable_without_server']
    for path in root.iterdir():
        assert (tmp_path/'verified/files'/path.name).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        archive.pack(root, packed)
    with pytest.raises(FileExistsError):
        archive.verify(packed, receipt['archive_sha256'], tmp_path/'verified')


@pytest.mark.parametrize('attack', ['traversal', 'absolute', 'symlink', 'missing', 'digest',
                                   'duplicate', 'extra', 'oversize'])
def test_untrusted_archives_refused(tmp_path, attack):
    content = b'content'
    sha = hashlib.sha256(content).hexdigest()
    name = '../escape' if attack == 'traversal' else '/escape' if attack == 'absolute' else 'raw'
    manifest = {'schema': 'g3-export-v1', 'files': {name: {'sha256': sha, 'bytes': len(content)}}}
    if attack == 'oversize':
        manifest['files'][name]['bytes'] = archive.MAX_TOTAL+1
    path = tmp_path/'malicious.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        raw = json.dumps(manifest).encode()
        item = tarfile.TarInfo('manifest.json')
        item.size = len(raw)
        tar.addfile(item, io.BytesIO(raw))
        if attack != 'missing':
            item = tarfile.TarInfo('blobs/'+sha)
            item.size = len(content)
            if attack == 'symlink':
                item.type = tarfile.SYMTYPE
                item.linkname = '/etc/passwd'
                item.size = 0
            payload = b'changed' if attack == 'digest' else content
            tar.addfile(item, io.BytesIO(payload))
            if attack == 'duplicate':
                tar.addfile(item, io.BytesIO(payload))
            if attack == 'extra':
                extra = tarfile.TarInfo('unexpected')
                tar.addfile(extra)
    with pytest.raises(ValueError):
        archive.verify(path, archive.file_sha(path), tmp_path/'verified')
    assert not (tmp_path/'escape').exists()


def test_changed_archive_sha_refused_before_extracting(tmp_path):
    data = tmp_path/'archive'
    data.write_bytes(b'not a tar')
    with pytest.raises(ValueError, match='SHA'):
        archive.verify(data, '0'*64, tmp_path/'verified')
    assert not (tmp_path/'verified').exists()


def test_pack_rejects_links_and_output_inside_session(tmp_path):
    root = tmp_path/'session'
    root.mkdir()
    with pytest.raises(ValueError):
        archive.pack(root, root/'archive.tar.gz')
    (root/'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='Symlink'):
        archive.pack(root, tmp_path/'archive.tar.gz')


def test_cli_receipt_is_machine_readable_and_readback_verified(tmp_path):
    import subprocess
    root = tmp_path/'session'
    root.mkdir()
    (root/'unicode.json').write_text('{"note":"проверка"}\n')
    path = tmp_path/'evidence.tar.gz'
    script = Path(archive.__file__)
    subprocess.run([sys.executable, str(script), 'pack', '--root', str(root),
                    '--archive', str(path)], check=True, capture_output=True, timeout=10)
    receipt = json.loads(path.with_name(path.name+'.receipt.json').read_bytes())
    assert receipt['archive_sha256'] == archive.file_sha(path)
    subprocess.run([sys.executable, str(script), 'verify', '--archive', str(path),
                    '--expected-sha', receipt['archive_sha256'], '--out', str(tmp_path/'local')],
                   check=True, capture_output=True, timeout=10)
    assert (tmp_path/'local/files/unicode.json').read_bytes() == (root/'unicode.json').read_bytes()
