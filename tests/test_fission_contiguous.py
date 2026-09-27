"""Gate physical cell layout, code read count, ranking and migration recovery."""
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np
import pytest
from fissiondb import AnchorIndex
from fission_layout_fixture import assert_contiguous, fragment_as_v1, read_layout
from test_fission_concurrency import instrumented_library  # shared test-only native barrier

ROOT = Path(__file__).resolve().parents[1]


def reopen(root):
    return AnchorIndex(root, root/'base.f16bin', residual_dir=root/'residual',
                       live_dir=root/'live', auto_pack_bytes=0)


def insert(index, x):
    for first in range(0, len(x), 256):
        index.insert_batch(x[first:first+256], group_commit=True)
    index.flush_fission()


def assert_one_read(query, vector, cells):
    ids, scores, stats = query.search(vector)
    assert stats['live']['code_reads'] == cells
    assert stats['live']['reads'] == cells + stats['live']['rerank_reads']
    return ids, scores, stats


@pytest.mark.parametrize('dim', [128, 768, 1024])
def test_one_read_per_cell_through_growth_fission_and_reopen(tmp_path, dim):
    root = tmp_path/'db'
    rng = np.random.default_rng(972)
    x = rng.normal(size=(6200, dim)).astype(np.float32)
    q = rng.normal(size=dim).astype(np.float32)
    with AnchorIndex.create(root, dim, cell_capacity=2048, auto_pack_bytes=0) as index:
        first = 0
        for end in (513, 1025, 1800, 3200, 6200):
            insert(index, x[first:end]); first = end
            cells = index.fission_stats['cells']
            with index.context(nprobe=1536, rerank=7000) as query:
                reference = assert_one_read(query, q, cells)
                data = x[:end].astype(np.float64)
                cosine = data @ q / np.linalg.norm(data, axis=1) / np.linalg.norm(q.astype(np.float64))
                wanted = np.argsort(-cosine)[:10]
                np.testing.assert_array_equal(reference[0], wanted)
                np.testing.assert_allclose(reference[1], cosine[wanted], atol=3e-6)
                for width, overlap, direct in [(1, False, False), (7, True, False), (64, True, True)]:
                    query.residual_io(batch_cells=width, overlap=overlap, direct=direct)
                    actual = assert_one_read(query, q, cells)
                    np.testing.assert_array_equal(actual[0], reference[0])
                    np.testing.assert_array_equal(actual[1], reference[1])
                    assert actual[2]['entries'] == 2*end
            index.pack_live()
            layout = assert_contiguous(root)
            assert max(c[2] for c in layout) > 512
        assert cells > 2 and index.fission_stats['splits'] > 0
    with reopen(root) as index, index.context(nprobe=1536, rerank=7000) as query:
        actual = assert_one_read(query, q, cells)
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])


def test_slot_growth_when_cell_limit_is_reached(tmp_path):
    root = tmp_path/'db'; x = np.random.default_rng(973).normal(size=(6200, 128)).astype(np.float32)
    with AnchorIndex.create(root, 128, cell_capacity=2048, max_cells=2, auto_pack_bytes=0) as index:
        for part in np.array_split(x, 4):
            insert(index, part)
            with index.context(nprobe=1536, rerank=400) as query:
                assert_one_read(query, part[-1], 2)
            index.pack_live(); assert_contiguous(root)
        assert index.fission_stats['largest_cell'] == len(x)
    with reopen(root) as index, index.context(nprobe=1536, rerank=400) as query:
        assert_one_read(query, x[-1], 2)


def test_oversized_cells_use_bounded_buffers(tmp_path):
    root = tmp_path/'db'; x = np.random.default_rng(979).normal(size=(110000, 8)).astype(np.float32)
    with AnchorIndex.create(root, 8, cell_capacity=2048, max_cells=2, auto_pack_bytes=0) as index:
        insert(index, x)
        with index.context(nprobe=2, rerank=400) as query:
            reference = None
            for width, direct in [(1, False), (64, False), (64, True)]:
                query.residual_io(batch_cells=width, overlap=True, direct=direct)
                result = query.search(x[-1])
                assert result[2]['live']['code_reads'] == 4  # two portions per >8 MiB cell
                assert result[2]['live']['buffer_bytes'] == 16*1024*1024
                assert result[2]['entries'] == 2*len(x)
                if reference is not None:
                    np.testing.assert_array_equal(result[0], reference[0])
                    np.testing.assert_array_equal(result[1], reference[1])
                reference = result


def legacy_fixture(root):
    x = np.random.default_rng(974).normal(size=(6200, 128)).astype(np.float32)
    with AnchorIndex.create(root, 128, cell_capacity=2048, auto_pack_bytes=0) as index:
        insert(index, x)
        with index.context(nprobe=1536, rerank=400) as query:
            expected = [query.search(q) for q in x[[0, 2030, 6199]]]
    return x, expected, fragment_as_v1(root)


def check_migrated(root, x, expected, before):
    with reopen(root) as index, index.context(nprobe=1536, rerank=400) as query:
        for q, want in zip(x[[0, 2030, 6199]], expected):
            actual = assert_one_read(query, q, len(before[1]))
            np.testing.assert_array_equal(actual[0], want[0])
            np.testing.assert_array_equal(actual[1], want[1])
            assert actual[2]['entries'] == want[2]['entries']
    after = read_layout(root)
    assert after[0][11] == before[0][11]  # same cells, not a journal rebuild
    assert after[0][13:] == before[0][13:]
    assert after[2] == before[2] and after[4] == before[4]  # exact center/code bytes
    assert_contiguous(root)
    assert not (root/'live/.fission.codes.before-contiguous').exists()
    assert not (root/'live/.fission.codes.slots.tmp').exists()


def test_legacy_conversion_preserves_every_code_and_representative(tmp_path):
    root = tmp_path/'db'; x, expected, before = legacy_fixture(root)
    check_migrated(root, x, expected, before)


@pytest.mark.parametrize('phase', [5, 6, 7])
def test_killed_migration_recovers_from_durable_checkpoint(tmp_path, instrumented_library, phase):
    root = tmp_path/'db'; x, expected, before = legacy_fixture(root)
    reached_r, reached_w = os.pipe(); resume_r, resume_w = os.pipe()
    code = '''
import ctypes as C,sys
from pathlib import Path
from fissiondb import AnchorIndex
from fissiondb.anchors import _lib
root=Path(sys.argv[1]);_lib.anchor_test_fission_gate.argtypes=[C.c_int,C.c_int,C.c_int]
_lib.anchor_test_fission_gate(*map(int,sys.argv[2:]))
AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live',auto_pack_bytes=0)
raise RuntimeError('migration did not stop at the barrier')
'''
    env = {**os.environ, 'FISSIONDB_ANCHOR_LIBRARY': str(instrumented_library)}
    child = subprocess.Popen([sys.executable, '-c', code, str(root), str(reached_w), str(resume_r), str(phase)],
                             env=env, pass_fds=(reached_w, resume_r))
    try:
        assert select.select([reached_r], [], [], 20)[0], 'migration barrier not reached'
        assert os.read(reached_r, 1) == b's'
    finally:
        child.kill(); child.wait(timeout=10)
        for fd in (reached_r, reached_w, resume_r, resume_w): os.close(fd)
    check_migrated(root, x, expected, before)


def test_conversion_keeps_uncheckpointed_journal_tail(tmp_path):
    root = tmp_path/'db'; x, _, _ = legacy_fixture(root)
    # Create durable tail using the new engine, then restore the earlier v1
    # checkpoint+arena. Its log inode is unchanged; tail replay must follow migration.
    saved_state = (root/'live/fission.state').read_bytes()
    old_code = root/'live/old-fixture.codes'
    os.link(root/'live/fission.codes', old_code)
    with reopen(root) as index:
        extra = index.insert(x[0])
        index.update(5, x[0], {'edited': True}); index.delete(6)
        with index.context(nprobe=1536, rerank=400) as query:
            reference = query.search(x[0], where={'edited': True})
    old_code.replace(root/'live/fission.codes')
    (root/'live/fission.state').write_bytes(saved_state)
    with reopen(root) as index, index.context(nprobe=1536, rerank=400) as query:
        assert index.count == extra+1
        actual = query.search(x[0], where={'edited': True})
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[1], reference[1])
        assert 6 not in query.search(x[6])[0]
        assert extra in query.search(x[0])[0]
    assert_contiguous(root)


def test_parallel_ingestion_preserves_codes_routing_and_metadata(tmp_path):
    code = '''
import sys,numpy as np
from pathlib import Path
from fissiondb import AnchorIndex
p=Path(sys.argv[1]);x=np.random.default_rng(978).normal(size=(4096,128)).astype(np.float32)
with AnchorIndex.create(p,128,cell_capacity=2048,auto_pack_bytes=0) as index:
 for first in range(0,len(x),256):
  index.insert_batch(x[first:first+256],[{'part': first//256}]*256,group_commit=True)
  index.flush_fission()  # identical publication boundaries for both thread counts
 with index.context(nprobe=1536,rerank=400) as q:
  results=[q.search(v,where={'part':1})[:2] for v in x[:3]]
  np.savez(p/'results.npz',ids=np.stack([r[0] for r in results]),scores=np.stack([r[1] for r in results]))
'''
    for threads in (1, 4):
        subprocess.run([sys.executable, '-c', code, str(tmp_path/str(threads))],
                       env={**os.environ, 'FISSIONDB_INGEST_THREADS': str(threads),
                            'OMP_WAIT_POLICY': 'PASSIVE'}, check=True, timeout=30)
    serial = read_layout(tmp_path/'1'); parallel = read_layout(tmp_path/'4')
    assert serial[2] == parallel[2] and serial[4] == parallel[4]
    for name in ('ids', 'scores'):
        np.testing.assert_array_equal(np.load(tmp_path/'1/results.npz')[name],
                                      np.load(tmp_path/'4/results.npz')[name])
    assert_contiguous(tmp_path/'4')
