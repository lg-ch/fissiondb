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
x=np.random.default_rng(959).normal(size=(160,128)).astype(np.float32)
def check(query,n):
    data=x[:n].astype(np.float64);q=x[103].astype(np.float64)
    scores=data@q/(np.linalg.norm(data,axis=1)*np.linalg.norm(q))
    want=np.argsort(-scores)[:10];ids,actual,_=query.search(q)
    np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(actual,scores[want],atol=3e-6)
    ids,_,_=query.search(q,where={'batch':'next'})
    assert len(ids)==(0 if n==96 else 10)
    if len(ids):assert (ids>=96).all()
if mode=='recover':
    with AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live') as index:
        assert index.count==160
        with index.context(rerank=400) as query:check(query,160)
    sys.exit(0)

reached_read,reached_write=os.pipe();resume_read,resume_write=os.pipe()
_lib.anchor_test_fission_gate.argtypes=[C.c_int,C.c_int,C.c_int]
_lib.anchor_test_fission_gate.restype=None
with AnchorIndex.create(root,128,cell_capacity=64,auto_pack_bytes=0) as index:
    index.insert_batch(x[:96],group_commit=True);index.flush_fission();index.pack_live()
    _lib.anchor_test_fission_gate(reached_write,resume_read,3 if mode in ('prepared','prepared_failure') else 4)
    with index.context(rerank=400) as query, concurrent.futures.ThreadPoolExecutor(2) as pool:
        append=pool.submit(index.insert_batch,x[96:],[{'batch':'next'}]*64,group_commit=True)
        assert select.select([reached_read],[],[],10)[0],'append never reached unpublished stage'
        assert os.read(reached_read,1)==b's'
        # A regression that holds live.write here blocks this search; subprocess timeout fails.
        check(query,96)
        if mode=='crash':os._exit(0)
        checkpoint=None
        if mode=='checkpoint':
            checkpoint=pool.submit(index.pack_live);time.sleep(.05);assert not checkpoint.done()
            check(query,96)
        failed=mode in ('failure','prepared_failure')
        os.write(resume_write,b'!' if failed else b'g')
        if failed:
            try:append.result(timeout=10)
            except OSError:pass
            else:raise AssertionError('failed staged append was acknowledged')
            if mode=='prepared_failure':check(query,96)
        else:
            np.testing.assert_array_equal(append.result(timeout=10),np.arange(96,160))
            if checkpoint:checkpoint.result(timeout=10)
            index.flush_fission();check(query,160)
if mode=='failure':
    with AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live') as index:
        assert index.count==160  # Durable but unacknowledged writes may recover.
        with index.context(rerank=400) as query:check(query,160)
for fd in [reached_read,reached_write,resume_read,resume_write]:os.close(fd)
