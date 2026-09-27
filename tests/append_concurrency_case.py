"""Pause staged appends outside the live write lock; exercise crash boundaries."""
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
growth=mode.startswith('growth_')
if growth:mode=mode[len('growth_'):]
initial,total=(496,752) if growth else (96,160)
x=np.random.default_rng(959).normal(size=(total,128)).astype(np.float32)
def check(query,n):
    data=x[:n].astype(np.float64);q=x[initial+7].astype(np.float64)
    scores=data@q/(np.linalg.norm(data,axis=1)*np.linalg.norm(q))
    want=np.argsort(-scores)[:10];ids,actual,stats=query.search(q)
    if growth:assert stats['live']['code_reads']==2
    np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(actual,scores[want],atol=3e-6)
    ids,_,_=query.search(q,where={'batch':'next'})
    assert len(ids)==(0 if n==initial else 10)
    if len(ids):assert (ids>=initial).all()
if mode=='recover':
    with AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live') as index:
        assert index.count==total
        with index.context(rerank=1000) as query:check(query,total)
    sys.exit(0)

reached_read,reached_write=os.pipe();resume_read,resume_write=os.pipe()
_lib.anchor_test_fission_gate.argtypes=[C.c_int,C.c_int,C.c_int]
_lib.anchor_test_fission_gate.restype=None
with AnchorIndex.create(root,128,cell_capacity=2048 if growth else 64,
                        max_cells=2 if growth else 300000,auto_pack_bytes=0) as index:
    index.insert_batch(x[:initial],group_commit=True);index.flush_fission();index.pack_live()
    _lib.anchor_test_fission_gate(reached_write,resume_read,3 if mode in ('prepared','prepared_failure') else 4)
    with index.context(rerank=1000) as query, concurrent.futures.ThreadPoolExecutor(2) as pool:
        append=pool.submit(index.insert_batch,x[initial:],[{'batch':'next'}]*(total-initial),group_commit=True)
        assert select.select([reached_read],[],[],10)[0],'append never reached unpublished stage'
        assert os.read(reached_read,1)==b's'
        # A regression that holds live.write here blocks this search; subprocess timeout fails.
        check(query,initial)
        if mode=='crash':os._exit(0)
        checkpoint=None
        if mode=='checkpoint':
            checkpoint=pool.submit(index.pack_live);time.sleep(.05);assert not checkpoint.done()
            check(query,initial)
        failed=mode in ('failure','prepared_failure')
        os.write(resume_write,b'!' if failed else b'g')
        if failed:
            try:append.result(timeout=10)
            except OSError:pass
            else:raise AssertionError('failed staged append was acknowledged')
            if mode=='prepared_failure':check(query,initial)
        else:
            np.testing.assert_array_equal(append.result(timeout=10),np.arange(initial,total))
            if checkpoint:checkpoint.result(timeout=10)
            index.flush_fission();check(query,total)
if mode=='failure':
    with AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live') as index:
        assert index.count==total  # Durable but unacknowledged writes may recover.
        with index.context(rerank=1000) as query:check(query,total)
for fd in [reached_read,reached_write,resume_read,resume_write]:os.close(fd)
