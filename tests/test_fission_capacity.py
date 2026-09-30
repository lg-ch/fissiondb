"""Changing a split threshold must retain durable data and direct retrieval."""
import concurrent.futures
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np
import pytest
from fissiondb import AnchorIndex
from test_fission import exact, ingest, reopen
from test_fission_concurrency import instrumented_library
from fission_layout_fixture import read_layout


@pytest.mark.parametrize('dim',[128,768,1024])
def test_retarget_preserves_search_and_survives_reopen(tmp_path,dim):
    root=tmp_path/'db';rng=np.random.default_rng(970)
    x=rng.normal(size=(4096,dim)).astype(np.float32)
    with AnchorIndex.create(root,dim,cell_capacity=512,auto_pack_bytes=0) as index:
        ingest(index,x);before=index.fission_stats
        with index.context(nprobe=10000,rerank=8192) as query:
            index.set_fission_capacity(384)
            assert index.fission_capacity==384
            for q in x[[1,2000]]:
                ids,scores,stats=query.search(q);want,expected=exact(q,x)
                np.testing.assert_array_equal(ids,want)
                np.testing.assert_allclose(scores,expected,atol=3e-6)
                assert stats['live']['direct_reads']==stats['live']['code_reads']>0
            index.flush_fission();after=index.fission_stats
            assert after['cells']>before['cells'] and after['largest_cell']<=384
            assert after['records']==before['records']==len(x)
            extra=index.insert(x[15]);assert extra==len(x)
            index.flush_fission();index.set_fission_capacity(512)
            assert index.fission_stats['cells']>=after['cells'] # raising never merges cells
    with reopen(root,auto_pack_bytes=0) as index:
        assert index.fission_capacity==512 and index.count==len(x)+1
        with index.context(nprobe=10000,rerank=8192) as query:
            ids,scores,_=query.search(x[15]);assert extra in ids
            _,expected=exact(x[15],np.vstack((x,x[15])))
            np.testing.assert_allclose(scores,expected,atol=3e-6)


def test_retarget_rejects_unbounded_scratch_and_invalid_values(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(971).normal(size=(1600,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=2048,auto_pack_bytes=0) as index:
        ingest(index,x);index.pack_live();saved=(root/'live/fission.config').read_bytes()
        for cap in (0,63,65537):
            with pytest.raises(ValueError):index.set_fission_capacity(cap)
        with pytest.raises(ValueError,match='stages'):index.set_fission_capacity(64)
        assert index.fission_capacity==2048 and (root/'live/fission.config').read_bytes()==saved
        index.set_fission_capacity(1536);index.flush_fission()
        index.set_fission_capacity(1024);index.flush_fission()
        assert index.fission_stats['largest_cell']<=1024


@pytest.mark.parametrize('phase,expected',[ (8,512),(9,384) ])
def test_threshold_crash_keeps_checkpoint_codes(tmp_path,instrumented_library,phase,expected):
    root=tmp_path/'db';x=np.random.default_rng(972).normal(size=(2048,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=512,auto_pack_bytes=0) as index:
        ingest(index,x);index.pack_live()
    checkpoint=(root/'live/fission.state').read_bytes()
    reached_r,reached_w=os.pipe();resume_r,resume_w=os.pipe()
    code='''
import ctypes as C,sys
from pathlib import Path
from fissiondb import AnchorIndex
from fissiondb.anchors import _lib
p=Path(sys.argv[1]);index=AnchorIndex(p,p/'base.f16bin',residual_dir=p/'residual',live_dir=p/'live',auto_pack_bytes=0)
_lib.anchor_test_fission_gate.argtypes=[C.c_int,C.c_int,C.c_int]
_lib.anchor_test_fission_gate(*map(int,sys.argv[2:]))
index.set_fission_capacity(384)
raise RuntimeError('capacity barrier was not reached')
'''
    env={**os.environ,'FISSIONDB_ANCHOR_LIBRARY':str(instrumented_library)}
    child=subprocess.Popen([sys.executable,'-c',code,str(root),str(reached_w),str(resume_r),str(phase)],env=env,pass_fds=(reached_w,resume_r))
    try:
        assert select.select([reached_r],[],[],20)[0]
        assert os.read(reached_r,1)==b's'
    finally:
        child.kill();child.wait(timeout=10)
        for fd in (reached_r,reached_w,resume_r,resume_w):os.close(fd)
    assert (root/'live/fission.state').read_bytes()==checkpoint
    with reopen(root,auto_pack_bytes=0) as index:
        assert index.count==len(x) and index.fission_capacity==expected
        assert index.fission_stats['records']==len(x)
        assert index.fission_stats['largest_cell']<=expected
        with index.context(nprobe=10000,rerank=8192) as query:
            ids,scores,_=query.search(x[57]);want,values=exact(x[57],x)
            np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(scores,values,atol=3e-6)


def test_capacity_changes_serialize_with_appends(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(973).normal(size=(4096,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=512,auto_pack_bytes=0) as index:
        ingest(index,x[:2048])
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            writer=pool.submit(ingest,index,x[2048:])
            change=pool.submit(index.set_fission_capacity,384)
            writer.result(timeout=20);change.result(timeout=20)
        index.flush_fission()
        assert index.count==4096 and index.fission_stats['largest_cell']<=384
        with index.context(nprobe=10000,rerank=8192) as query:
            ids,scores,_=query.search(x[3500]);want,values=exact(x[3500],x)
            np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(scores,values,atol=3e-6)


def test_config_ahead_of_checkpoint_reuses_exact_cells(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(974).normal(size=(2048,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=512,auto_pack_bytes=0) as index:
        ingest(index,x)
    before=read_layout(root)
    code='''
import os,sys
from pathlib import Path
from fissiondb import AnchorIndex
p=Path(sys.argv[1])
index=AnchorIndex(p,p/'base.f16bin',residual_dir=p/'residual',live_dir=p/'live',auto_pack_bytes=0)
index.set_fission_capacity(640)
os._exit(0) # acknowledged config, no close/checkpoint
'''
    subprocess.run([sys.executable,'-c',code,str(root)],check=True,timeout=20)
    with reopen(root,auto_pack_bytes=0) as index:assert index.fission_capacity==640
    after=read_layout(root)
    assert after[0][9]==640 and before[0][9]==512
    assert after[0][11:]==before[0][11:] # exact cells/chunks/split history, no rebuild
    # Reopen rebuilds free-run links; only descriptors and reachable records
    # are part of the retained cell topology.
    for field in (1,2,4):assert after[field]==before[field]
