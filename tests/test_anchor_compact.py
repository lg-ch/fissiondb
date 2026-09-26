"""Checkpoint preservation, bounded storage and interrupted publication."""
import concurrent.futures
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from test_anchor_live import frozen, open_live, ROOT


def populate(frozen, directory):
    with open_live(frozen, directory) as index:
        index.insert_batch(frozen[2][:30], [{'value': 'old'} for _ in range(30)])
        for generation in range(12):
            for doc in (0, 1, 5120, 5121):
                index.set_metadata(doc, {'value': str(generation), 'keep': True})
        index.set_metadata(5122, {})


def check_state(frozen, directory):
    with open_live(frozen, directory) as index:
        assert index.count == 5150
        with index.context(nprobe=32, rerank=300) as query:
            ids, _, _ = query.search(frozen[2][0], where={'value': '11'})
            assert set(ids) == {0, 1, 5120, 5121}
            ids, _, _ = query.search(frozen[2][0], where={'value': 'old'}, allowed_ids=[5122])
            assert not len(ids)


def test_compact_preserves_queries_and_future_writes(frozen, tmp_path):
    populate(frozen, tmp_path)
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=32, rerank=300) as query:
            expected = [query.search(q)[:2] for q in frozen[2][:10]]
            sizes = index.compact()
            assert sizes['after_bytes'] < sizes['before_bytes']
            assert (tmp_path / 'live.log').stat().st_size == sizes['after_bytes']
            for q, (ids, scores) in zip(frozen[2][:10], expected):
                actual, score, _ = query.search(q)
                np.testing.assert_array_equal(actual, ids)
                np.testing.assert_array_equal(score, scores)
            assert index.compact()['saved_bytes'] == 0
            with pytest.raises(OSError):
                open_live(frozen, tmp_path)
    check_state(frozen, tmp_path)
    with open_live(frozen, tmp_path) as index:
        assert index.insert(frozen[2][30], {'next': True}) == 5150
        index.set_metadata(0, {'next': True})
        with index.context(nprobe=1, rerank=10) as query:
            assert set(query.search(frozen[2][0], where={'next': True})[0]) == {0, 5150}
        index.compact()
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=1, rerank=10) as query:
            assert set(query.search(frozen[2][0], where={'next': True})[0]) == {0, 5150}


def test_compact_empty_and_leftover_temp(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        (tmp_path / '.live.compact').write_bytes(b'interrupted checkpoint')
        assert index.compact() == {'before_bytes': 64, 'after_bytes': 64, 'saved_bytes': 0}
        assert not (tmp_path / '.live.compact').exists()


def test_checkpoint_metadata_spans_multiple_chunks(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        for doc in range(5120):
            index.set_metadata(doc, {'group': 'shared'})
        index.insert_batch(np.repeat(frozen[2][0:1], 3200, axis=0),
                           [{'group': 'shared'} for _ in range(3200)])
        assert index.compact()['saved_bytes'] > 0
    with open_live(frozen, tmp_path) as index:
        assert index.count == 8320
        with index.context(nprobe=1, rerank=10) as query:
            for doc in (0, 8191, 8192, 8319):
                ids, _, _ = query.search(frozen[2][0], where={'group': 'shared'}, allowed_ids=[doc])
                assert list(ids) == [doc]
            index.set_metadata(8192, {})
            assert not len(query.search(frozen[2][0], where={'group': 'shared'}, allowed_ids=[8192])[0])


@pytest.mark.parametrize('offset', [0, 64 + 80 + 7])
def test_compact_rejects_changed_header_or_corrupt_record(frozen, tmp_path, offset):
    populate(frozen, tmp_path)
    with open_live(frozen, tmp_path) as index:
        path = tmp_path / 'live.log'
        valid = path.read_bytes()
        corrupt = bytearray(valid)
        corrupt[offset] ^= 1
        path.write_bytes(corrupt)
        with pytest.raises(OSError):
            index.compact()
        assert path.read_bytes() == corrupt
        path.write_bytes(valid)
        assert index.compact()['saved_bytes'] > 0


def test_compact_write_failure_keeps_old_journal(frozen, tmp_path):
    import resource
    populate(frozen, tmp_path)
    old = (tmp_path / 'live.log').read_bytes()
    with open_live(frozen, tmp_path) as index:
        previous = resource.getrlimit(resource.RLIMIT_FSIZE)
        handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (128, previous[1]))
            with pytest.raises(OSError):
                index.compact()
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, previous)
            signal.signal(signal.SIGXFSZ, handler)
        assert index.count == 5150
        assert (tmp_path / 'live.log').read_bytes() == old
        assert not (tmp_path / '.live.compact').exists()
    check_state(frozen, tmp_path)


@pytest.fixture(scope='module')
def publication_faults(tmp_path_factory):
    directory = tmp_path_factory.mktemp('compact_faults')
    source = directory / 'faults.c'
    source.write_text(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
static int replaced;
int renameat(int a,const char* b,int c,const char* d) {
    int (*real)(int,const char*,int,const char*)=dlsym(RTLD_NEXT,"renameat");
    const char* mode=getenv("COMPACT_FAULT");
    int target=mode&&!strcmp(b,".live.compact");
    if(target&&!strcmp(mode,"before"))_exit(86);
    int rc=real(a,b,c,d);
    if(target&&!rc) {
        replaced=1;
        if(!strcmp(mode,"after"))_exit(87);
    }
    return rc;
}
int fsync(int fd) {
    int (*real)(int)=dlsym(RTLD_NEXT,"fsync");
    const char* mode=getenv("COMPACT_FAULT");
    if(replaced&&mode&&!strcmp(mode,"sync")){errno=EIO;return -1;}
    return real(fd);
}
''')
    library = directory / 'faults.so'
    subprocess.run(['gcc', '-shared', '-fPIC', str(source), '-o', str(library), '-ldl'], check=True)
    return library


@pytest.mark.parametrize('mode,code', [('before', 86), ('after', 87), ('sync', 0)])
def test_interrupted_publication(frozen, tmp_path, publication_faults, mode, code):
    populate(frozen, tmp_path)
    script = '''
import sys
from fissiondb.anchors import AnchorIndex
with AnchorIndex(sys.argv[1],sys.argv[2],live_dir=sys.argv[3]) as index:
    try: index.compact()
    except OSError:
        try: index.count
        except OSError: pass
        else: raise AssertionError('ambiguous publication must poison the handle')
    else: raise AssertionError('fault was not triggered')
'''
    # Keep libasan first when the suite is run with a sanitizer preload.
    preload = ':'.join(x for x in (os.environ.get('LD_PRELOAD'), str(publication_faults)) if x)
    result = subprocess.run([sys.executable, '-c', script, str(frozen[0]), str(frozen[1]), str(tmp_path)],
        env={**os.environ, 'LD_PRELOAD': preload, 'COMPACT_FAULT': mode, 'PYTHONPATH': str(ROOT / 'scripts')},
        capture_output=True, text=True)
    assert result.returncode == code, result.stderr
    check_state(frozen, tmp_path)


def test_compaction_serializes_with_queries_and_writes(frozen, tmp_path):
    populate(frozen, tmp_path)
    with open_live(frozen, tmp_path) as index:
        with index.context(nprobe=32, rerank=300) as query:
            def reader():
                for _ in range(30):
                    ids, _, _ = query.search(frozen[2][0], where={'keep': True})
                    assert set(ids) == {0, 1, 5120, 5121}
            def writer():
                for _ in range(20):
                    index.insert(frozen[2][5], {'other': True})
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                tasks = [pool.submit(reader), pool.submit(writer), pool.submit(index.compact)]
                for task in tasks:
                    task.result(timeout=60)
            assert index.count == 5170
