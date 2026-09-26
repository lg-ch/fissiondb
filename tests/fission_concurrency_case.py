"""Child process for deterministic native fission concurrency tests."""
import concurrent.futures
import ctypes as C
import os
from pathlib import Path
import select
import sys
import time

import numpy as np
from fissiondb import AnchorIndex
from fissiondb.anchors import _lib

root=Path(sys.argv[1]);mode=sys.argv[2]
x=np.random.default_rng(830).normal(size=(400,128)).astype(np.float32)
replacement=np.random.default_rng(831).normal(size=128).astype(np.float32)


def open_index():
    return AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live',auto_pack_bytes=0)


def check(query,n,mutated=False):
    data=x[:n].astype(np.float64).copy()
    if mutated:data[5]=replacement
    q=replacement.astype(np.float64) if mutated else x[0].astype(np.float64)
    expected=(data@q)/(np.linalg.norm(data,axis=1)*np.linalg.norm(q))
    if mutated:expected[6]=-np.inf
    wanted=np.argsort(-expected)[:10];ids,scores,_=query.search(q)
    np.testing.assert_array_equal(ids,wanted)
    np.testing.assert_allclose(scores,expected[wanted],atol=3e-6)
    if mutated:
        ids,_,_=query.search(q,where={'edited':True})
        np.testing.assert_array_equal(ids,[5])


if mode=='recover':
    with open_index() as index:
        assert index.count==104
        with index.context(rerank=1000) as query:check(query,104,True)
        index.flush_fission()
        assert index.fission_stats['largest_cell']<=64
    sys.exit(0)

reached_read,reached_write=os.pipe();resume_read,resume_write=os.pipe()
_lib.anchor_test_fission_gate.argtypes=[C.c_int,C.c_int,C.c_int]
_lib.anchor_test_fission_gate.restype=None
_lib.anchor_test_fission_gate(reached_write,resume_read,2)
with AnchorIndex.create(root,128,cell_capacity=64,auto_pack_bytes=0) as index:
    index.insert_batch(x[:96],group_commit=True)
    assert select.select([reached_read],[],[],10)[0], 'split did not reach unpublished stage'
    assert os.read(reached_read,1)==b's'
    with index.context(rerank=1000) as query:
        check(query,96)  # Completes while the split worker is deliberately paused.
        if mode=='failure':
            index.insert_batch(x[96:352],group_commit=True)
            check(query,352)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                waiting=pool.submit(index.insert,x[352])
                time.sleep(.1);assert not waiting.done(), 'ingestion should apply backpressure'
                check(query,352)  # A blocked writer must not hold the reader lock.
                os.write(resume_write,b'!')
                try:waiting.result(timeout=10)
                except OSError:pass
                else:raise AssertionError('writer accepted a poisoned handle')
        else:
            index.update(5,replacement,{'edited':True});index.delete(6)
            index.insert_batch(x[96:104],group_commit=True)
            check(query,104,True)  # Mutations acknowledged before publication.
            progress=index.fission_stats
            assert progress['preparing']==1 and progress['queries_during_prepare']>=2
            if mode=='crash':os._exit(0)  # Staged arena exists; journal owns recovery.
            if mode in ('checkpoint','compact'):
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    maintenance=pool.submit(index.pack_live if mode=='checkpoint' else index.compact)
                    time.sleep(.1);assert not maintenance.done()
                    check(query,104,True)  # Maintenance waits without the live lock.
                    os.write(resume_write,b'g');maintenance.result(timeout=10)
            else:os.write(resume_write,b'g')
            index.flush_fission()
            check(query,104,True)
            progress=index.fission_stats
            assert progress['delta_records']>0 and progress['pending_cells']==0
            assert progress['largest_cell']<=64
            index.pack_live()
with open_index() as index:
    count=352 if mode=='failure' else 104
    assert index.count==count
    with index.context(rerank=1000) as query:check(query,count,mode!='failure')
for fd in [reached_read,reached_write,resume_read,resume_write]:os.close(fd)
