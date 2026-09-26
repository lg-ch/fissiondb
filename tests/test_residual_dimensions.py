"""Original dimensions, bounded overlapped IO and durable tombstones."""
import concurrent.futures
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from mangrove.anchors import AnchorIndex


def build(root,dim,n=96,m=2,tqbits=1):
    root.mkdir(exist_ok=True)
    rng=np.random.default_rng(527+dim)
    x=rng.normal(size=(n,dim)).astype(np.float32)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    x=x.astype('<f2');base=root/'base.f16bin'
    base.write_bytes(struct.pack('<II',n,dim)+x.tobytes())
    directory=root/'index';directory.mkdir()
    subprocess.run([str(ROOT/'mangrove-engine'),'abuild',str(base),str(directory),'8',
                    '--m',str(m),'--eps','999','--tqbits',str(tqbits),'--seed','52'],
                   check=True,capture_output=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    out=root/'residual'
    with AnchorIndex(directory,base) as idx:
        assert idx.dim==dim
        idx.build_residual(out)
    return directory,base,out,x.astype(np.float32)


@pytest.mark.parametrize('dim',[1,3,7,8,15,31,63,64,96,127,128,129,192,255,256,257,384,511,512,513,767,768,1000,1023,1024])
def test_dimension_search_live_delete_reopen(tmp_path,dim):
    directory,base,out,x=build(tmp_path,dim)
    # No full-original padding on disk. Cosine scores agree with independent GT.
    assert base.stat().st_size==8+len(x)*dim*2
    unit=x.astype(np.float64);unit/=np.linalg.norm(unit,axis=1,keepdims=True)
    gt=np.sort(unit@unit[13])[-10:][::-1]
    with AnchorIndex(directory,base,residual_dir=out,live_dir=tmp_path/'live') as idx:
        with idx.context() as defaults:assert defaults.rerank==400
        with idx.context(nprobe=8,rerank=96) as ctx:
            ids,scores,_=ctx.search(x[13])
            np.testing.assert_allclose(scores,gt,atol=2e-6)
            added=idx.insert_batch(x[:3],[{'group':'live'}]*3)
            idx.update(15,x[12],{'group':'updated'})
            idx.pack_live()
            idx.delete(13);idx.delete(int(added[0]));idx.delete(int(added[0]))
            assert idx.deleted_count==2
            assert 13 not in ctx.search(x[13])[0]
            assert ctx.search(x[0],where={'group':'live'})[0].tolist()==ctx.search(x[0],allowed_ids=added[1:])[0].tolist()
            assert ctx.search(x[12],where={'group':'updated'})[0].tolist()==[15]
            idx.compact();idx.pack_live()
    with AnchorIndex(directory,base,residual_dir=out,live_dir=tmp_path/'live') as idx:
        assert idx.count==99 and idx.deleted_count==2
        with idx.context(nprobe=8,rerank=96) as ctx:
            assert set(ctx.search(x[0],where={'group':'live'})[0])==set(added[1:])
            assert 13 not in ctx.search(x[13])[0]
            with pytest.raises(OSError):idx.update(13,x[13])


@pytest.mark.parametrize('copies,tqbits',[(1,4),(3,2),(4,1)])
def test_residual_source_layouts(tmp_path,copies,tqbits):
    directory,base,out,x=build(tmp_path,192,m=copies,tqbits=tqbits)
    with AnchorIndex(directory,base,residual_dir=out) as idx:
        with idx.context(nprobe=8,rerank=96) as ctx:
            assert ctx.search(x[13],top_k=1)[0].tolist()==[13]


@pytest.mark.parametrize('dim',[63,128,192,256,384,512,768,1024])
def test_compressed_ranking_small_candidate_budget(tmp_path,dim):
    directory,base,out,x=build(tmp_path,dim,n=512)
    q=np.random.default_rng(dim+914).normal(size=dim).astype(np.float32)
    q/=np.linalg.norm(q)
    with AnchorIndex(directory,base,residual_dir=out,live_dir=tmp_path/'live') as idx:
        with idx.context(nprobe=8,rerank=4) as ctx:
            # Only four of 512 survive approximate ranking. Self neighbors
            # must survive; exhaustive reranking cannot conceal a bad codec.
            for row in [5,77,219]:assert ctx.search(x[row],top_k=1)[0].tolist()==[row]
            added=idx.insert(q)
            idx.pack_live()
            assert ctx.search(q,top_k=1)[0].tolist()==[added]
            idx.delete(added)
            assert added not in ctx.search(q,top_k=4)[0]


def test_pipeline_modes_and_concurrent_contexts(tmp_path):
    directory,base,out,x=build(tmp_path,768,n=1024)
    with AnchorIndex(directory,base,residual_dir=out,live_dir=tmp_path/'live') as idx:
        idx.delete(13)
        with idx.context(nprobe=8,rerank=400) as ctx:
            expected=ctx.residual_io(batch_cells=1,overlap=False,direct=False).search(x[13])
            for batch in [1,3,64,256]:
                for overlap in [False,True]:
                    for direct in [False,True]:
                        got=ctx.residual_io(batch_cells=batch,overlap=overlap,direct=direct).search(x[13])
                        np.testing.assert_array_equal(got[0],expected[0])
                        np.testing.assert_array_equal(got[1],expected[1])
        def query(_):
            with idx.context(nprobe=8,rerank=400,memory_bytes=2_000_000) as ctx:
                return ctx.search(x[13])[:2]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for ids,scores in pool.map(query,range(12)):
                np.testing.assert_array_equal(ids,expected[0])
                np.testing.assert_array_equal(scores,expected[1])


@pytest.mark.parametrize('fault',['truncate','bad_id'])
def test_failed_pipeline_drains_and_poison_context(tmp_path,fault):
    directory,base,out,x=build(tmp_path,384)
    with AnchorIndex(directory,base,residual_dir=out) as idx:
        with idx.context(nprobe=8,rerank=96) as ctx:
            ctx.residual_io(batch_cells=1,overlap=True,direct=True)
            with (out/'res512.bin').open('r+b') as f:
                if fault=='truncate':f.truncate(3)
                else:
                    data=bytearray(f.read())
                    for offset in range(0,len(data),72):struct.pack_into('<I',data,offset,2**32-1)
                    f.seek(0);f.write(data)
            with pytest.raises(OSError):ctx.search(x[13])
            with pytest.raises(RuntimeError,match='closed'):ctx.search(x[13])
