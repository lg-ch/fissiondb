"""Native anchor ingestion, persistence, filtering and concurrent visibility."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mangrove.anchors import AnchorIndex
from mangrove.anchors import _lib, AnchorBatchError
from mangrove.metatypes import FloatSpec


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    directory = tmp_path_factory.mktemp('anchor_frozen')
    rng = np.random.default_rng(19)
    vectors = rng.normal(size=(5120, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    vectors = vectors.astype(np.float16)
    base = directory / 'base.f16bin'
    base.write_bytes(struct.pack('<II', *vectors.shape) + vectors.tobytes())
    index = directory / 'index'
    index.mkdir()
    subprocess.run([str(ROOT / 'mangrove-engine'), 'abuild', str(base), str(index), '32',
                    '--m', '2', '--eps', '999', '--tqbits', '1', '--seed', '52'],
                   check=True, capture_output=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    return index, base, vectors.astype(np.float32)


def open_live(frozen, path, **kwargs):
    return AnchorIndex(frozen[0], frozen[1], live_dir=path, **kwargs)


def test_insert_visible_and_reopen(frozen, tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(20, 128)).astype(np.float32)
    before = hashlib.sha256((frozen[0] / 'blocks.bin').read_bytes()).digest()
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=32, rerank=300) as query:
            ids = index.insert_batch(vectors, [{'lang': 'fr', 'rank': i} for i in range(20)])
            assert list(ids) == list(range(5120, 5140))
            for doc_id, vector in zip(ids, vectors):
                found, scores, _ = query.search(vector, top_k=10)
                assert found[0] == doc_id
                assert scores[0] == pytest.approx(1, abs=2e-6)
                assert len(set(found)) == len(found)
            assert index.count == 5140
    with open_live(frozen, tmp_path, int8=False) as index:
        with index.context(nprobe=1, rerank=20) as query:
            for i, vector in enumerate(vectors):
                found, _, _ = query.search(vector, where={'rank': i})
                assert list(found) == [5120 + i]
    assert before == hashlib.sha256((frozen[0] / 'blocks.bin').read_bytes()).digest()


def test_metadata_operators_replacement_and_intersection(frozen, tmp_path):
    with open_live(frozen, tmp_path, float_specs={'price': FloatSpec(2)}) as index:
        index.set_metadata(0, {'lang': 'fr', 'rank': 12, 'active': True, 'price': 1.25})
        index.set_metadata(1, {'lang': 'en', 'rank': 99})
        new = index.insert(frozen[2][2], {'lang': ['fr', 'it'], 'rank': 30})
        with index.context(nprobe=1, rerank=20) as query:
            def search(where, allowed=None):
                return set(query.search(frozen[2][0], where=where, allowed_ids=allowed)[0])
            assert search({'lang': 'fr'}) == {0, new}
            assert search({'lang': ['fr', 'en'], 'rank': ('range', 10, 40)}) == {0, new}
            assert search({'active': ('exists',)}) == {0}
            assert search({'active': True, 'price': ('range', 1.2, 1.3)}) == {0}
            assert search({'lang': ('re', '^(en|it)$')}) == {1, new}
            assert search({'lang': ('re', '^zz$')}) == set()
            assert search({'lang': []}) == set()
            assert search({'lang': 'fr'}, [new, new, 1, 123456]) == {new}
            assert search({}, []) == set()
            index.set_metadata(new, {'lang': 'de'})
            assert search({'lang': 'fr'}) == {0}
            assert search({'rank': ('exists',)}) == {0, 1}
            index.set_metadata(0, {})
            assert search({'lang': 'fr'}) == set()
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=1, rerank=20) as query:
            assert set(query.search(frozen[2][0], where={'lang': ('exists',)})[0]) == {1, new}


@pytest.mark.parametrize('threads', [1, 2])
def test_prefilter_precedes_candidate_caps(frozen, tmp_path, threads):
    # Cardinality >4096 forces the ANN path, including its prefix heap.
    with open_live(frozen, tmp_path) as index:
        for i in range(4100):
            index.set_metadata(i, {'eligible': True})
        with index.context(nprobe=32, rerank=300, threads=threads) as query:
            result, _, _ = query.search(frozen[2][5000], where={'eligible': True})
            assert len(result) == 10 and np.all(result < 4100)
            allowed, _, _ = query.search(frozen[2][5000], allowed_ids=range(4100))
            np.testing.assert_array_equal(result, allowed)
            exact = frozen[2][:4100].copy()
            exact /= np.linalg.norm(exact, axis=1)[:, None]
            truth = np.argsort(-(exact @ frozen[2][5000]))[:10]
            assert len(set(truth) & set(result)) >= 9
            new = index.insert(frozen[2][5000], {'new': True})
            live_only, _, stats = query.search(frozen[2][5000], allowed_ids=range(5120, 9217))
            assert list(live_only) == [new]
            assert stats['bytes'] > 0


def test_sparse_filter_exact_scores_and_batches(frozen):
    with AnchorIndex(frozen[0], frozen[1]) as index:
        with index.context(nprobe=1, rerank=10) as query:
            allowed = np.arange(0, 400, 3)
            found, scores, stats = query.search(frozen[2][3], allowed_ids=allowed)
            rows = frozen[2][allowed].copy()
            rows /= np.linalg.norm(rows, axis=1)[:, None]
            q = frozen[2][3] / np.linalg.norm(frozen[2][3])
            exact = rows @ q
            order = np.argsort(-exact)[:10]
            np.testing.assert_array_equal(found, allowed[order])
            np.testing.assert_allclose(scores, exact[order], atol=2e-6)
            assert stats['entries'] == len(allowed)


@pytest.mark.parametrize('tail', [b'bad', b'\0' * 79])
def test_torn_tail_repaired(frozen, tmp_path, tail):
    with open_live(frozen, tmp_path) as index:
        index.insert(frozen[2][0], {'lang': 'fr'})
    log = tmp_path / 'live.log'
    good = log.read_bytes()
    log.write_bytes(good + tail)
    with open_live(frozen, tmp_path) as index:
        assert index.count == 5121
        assert index.insert(frozen[2][1]) == 5121
    assert log.read_bytes().startswith(good)


def test_partial_payload_and_full_corruption(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        index.insert(frozen[2][0])
        index.insert(frozen[2][1])
    log = tmp_path / 'live.log'
    good = log.read_bytes()
    log.write_bytes(good[:-100])
    with open_live(frozen, tmp_path) as index:
        assert index.count == 5121
    corrupt = bytearray(log.read_bytes())
    corrupt[64 + 80 + 30] ^= 1
    log.write_bytes(corrupt)
    with pytest.raises(OSError):
        open_live(frozen, tmp_path)
    assert log.read_bytes() == corrupt


def test_exclusive_lock_and_fingerprint(frozen, tmp_path):
    with open_live(frozen, tmp_path):
        with pytest.raises(OSError):
            open_live(frozen, tmp_path)
    copy = tmp_path / 'other'
    shutil.copytree(frozen[0], copy)
    anchors = copy / 'anchors.bin'
    data = bytearray(anchors.read_bytes())
    data[0] ^= 1
    anchors.write_bytes(data)
    with pytest.raises(OSError):
        AnchorIndex(copy, frozen[1], live_dir=tmp_path)


def test_acknowledged_insert_survives_process_exit(frozen, tmp_path):
    script = '''
import os, sys, numpy as np
from mangrove.anchors import AnchorIndex
index = AnchorIndex(sys.argv[1], sys.argv[2], live_dir=sys.argv[3])
v = np.fromfile(sys.argv[2], np.float16, offset=8, count=128).astype(np.float32)
assert index.insert(v, {'durable': True}) == 5120
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(frozen[0]), str(frozen[1]), str(tmp_path)],
                   check=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'scripts')})
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=1, rerank=10) as query:
            assert list(query.search(frozen[2][0], where={'durable': True})[0]) == [5120]


def test_validation_before_commit_and_close(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        for vector in [np.zeros(128), np.ones(127), np.full(128, np.nan)]:
            with pytest.raises(ValueError):
                index.insert(vector)
        with pytest.raises(ValueError):
            index.insert_batch([frozen[2][0], np.zeros(128)])
        with pytest.raises(ValueError):
            index.insert(frozen[2][0], {'bad.field': 'x'})
        with pytest.raises(ValueError):
            index.insert(frozen[2][0], {'lang': 'x\0y'})
        with pytest.raises(ValueError):
            index.set_metadata(99999, {})
        assert index.count == 5120
        query = index.context(nprobe=2, rerank=20)
        with pytest.raises(ValueError):
            query.search(frozen[2][0], allowed_ids=[-1])
    with pytest.raises(RuntimeError):
        query.search(frozen[2][0])


def test_queries_concurrent_with_ingestion(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        contexts = [index.context(nprobe=32, rerank=100) for _ in range(2)]
        def writer():
            for i in range(40):
                assert index.insert(frozen[2][i], {'live': True}) == 5120 + i
        def reader(query):
            for _ in range(40):
                ids, _, _ = query.search(frozen[2][0], where={'live': True})
                assert all(5120 <= i < 5160 for i in ids)
                ids, _, _ = query.search(frozen[2][0])
                assert len(set(ids)) == 10
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(writer)] + [pool.submit(reader, q) for q in contexts]
            for future in futures:
                future.result(timeout=60)
        assert index.count == 5160


def test_batch_reports_acknowledged_prefix(frozen, tmp_path, monkeypatch):
    native_insert = _lib.anchor_index_insert
    calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        return -1 if calls == 2 else native_insert(*args)
    with open_live(frozen, tmp_path) as index:
        monkeypatch.setattr(_lib, 'anchor_index_insert', fail_second)
        with pytest.raises(AnchorBatchError) as error:
            index.insert_batch(frozen[2][:3])
        assert error.value.committed_ids == [5120]
        assert index.count == 5121
        monkeypatch.setattr(_lib, 'anchor_index_count', lambda handle: 0)
        with pytest.raises(OSError):
            _ = index.count
        with pytest.raises(OSError):
            index.set_metadata(0, {})
