"""Exercise the notebook's dataset setup using tiny local archives, without network."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.error
import zipfile

import pytest

from src.data_prep import PARENTS


@pytest.fixture
def setup_functions():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'EfficientNet_B0_DataSEC.ipynb').read_text())
    source = ''.join(next(c for c in notebook['cells'] if c['id'] == 'effnet-dataset-setup')['source'])
    tree = ast.parse(source)
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.If):
            break  # Do not execute the actual user-selected download/setup.
        declarations.append(node)
    namespace = {}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), '<notebook-setup>', 'exec'), namespace)
    return namespace


def archive(tmp_path, entries):
    source = tmp_path / 'source.zip'
    with zipfile.ZipFile(source, 'w') as z:
        for name in entries:
            z.writestr(name, b'synthetic path-detection fixture')
    return source.as_uri(), hashlib.md5(source.read_bytes()).hexdigest()


def test_download_extract_and_reuse(setup_functions, tmp_path):
    url, md5 = archive(tmp_path, [f'DATASEC/{label}/sample.wav' for label in PARENTS])
    extracted = setup_functions['download_and_extract'](tmp_path / 'download', url, md5)
    assert len(list(extracted.rglob('*.wav'))) == 22
    assert setup_functions['find_datasec_roots']([extracted]) == [extracted / 'DATASEC']
    # A successful rerun must not download again (the source is now absent).
    (tmp_path / 'source.zip').unlink()
    assert setup_functions['download_and_extract'](tmp_path / 'download', url, md5) == extracted


def test_checksum_failure_does_not_extract(setup_functions, tmp_path):
    url, _ = archive(tmp_path, ['DATASEC/Bells/a.wav'])
    with pytest.raises(ValueError, match='checksum mismatch'):
        setup_functions['download_and_extract'](tmp_path / 'download', url, '0' * 32)
    assert not (tmp_path / 'download/extracted').exists()


def test_unsafe_archive_path_rejected(setup_functions, tmp_path):
    url, md5 = archive(tmp_path, ['../../outside.wav'])
    with pytest.raises(ValueError, match='Unsafe path'):
        setup_functions['download_and_extract'](tmp_path / 'download', url, md5)
    assert not (tmp_path / 'outside.wav').exists()


def test_network_error_is_actionable(setup_functions, tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError('offline')
    monkeypatch.setattr('urllib.request.urlopen', offline)
    with pytest.raises(RuntimeError, match='Enable Internet in Kaggle'):
        setup_functions['download_and_extract'](tmp_path / 'download')


def test_ambiguous_inputs_return_both_candidates(setup_functions, tmp_path):
    for copy in ('first', 'second'):
        for label in PARENTS:
            folder = tmp_path / copy / 'DATASEC' / label
            folder.mkdir(parents=True)
            (folder / 'a.WAV').write_bytes(b'fixture')
    assert setup_functions['find_datasec_roots']([tmp_path]) == [
        tmp_path / 'first/DATASEC', tmp_path / 'second/DATASEC']


def test_nested_subclasses_preserve_parent_root(setup_functions, tmp_path):
    for label in PARENTS:
        folder = tmp_path / 'DATASEC' / label
        if label == 'Propeller aircrafts':
            folder /= 'Helicopters'
        folder.mkdir(parents=True)
        (folder / 'a.wav').write_bytes(b'fixture')
    assert setup_functions['find_datasec_roots']([tmp_path]) == [tmp_path / 'DATASEC']


def test_missing_or_incomplete_dataset_not_selected(setup_functions, tmp_path):
    assert setup_functions['find_datasec_roots']([tmp_path / 'missing']) == []
    (tmp_path / 'Bells').mkdir()
    (tmp_path / 'Bells/a.wav').write_bytes(b'fixture')
    assert setup_functions['find_datasec_roots']([tmp_path]) == []
