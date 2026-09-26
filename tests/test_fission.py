"""End-to-end contracts: adaptive ingestion, journal durability and filtering."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from fissiondb import AnchorIndex
from fissiondb.backup import create as backup, restore

def reopen(directory,**kwargs):
    return AnchorIndex(directory,directory/'base.f16bin',residual_dir=directory/'residual',live_dir=directory/'live',**kwargs)

def ingest(index,x,metadata=None):
    output=[]
    for i in range(0,len(x),128):
        output.extend(index.insert_batch(x[i:i+128],metadata[i:i+128] if metadata else None,group_commit=True))
    return np.asarray(output)

def exact(q,x,k=10,eligible=None):
    q=np.asarray(q,np.float64);v=np.asarray(x,np.float64)
    scores=(v@q)/(np.linalg.norm(v,axis=1)*np.linalg.norm(q))
    if eligible is not None:scores[~eligible]=-np.inf
    ids=np.argsort(-scores,kind='stable')[:k]
    return ids,scores[ids]

@pytest.mark.parametrize('dim',[1,64,128,256,768,1024])
def test_dynamic_cells_and_reopen(tmp_path,dim):
    root=tmp_path/'db';rng=np.random.default_rng(42)
    x=rng.normal(size=(384,dim)).astype(np.float32)
    with AnchorIndex.create(root,dim,cell_capacity=64,auto_pack_bytes=0) as index:
        assert index.count==0 and index.fission
        with index.context(nprobe=1536,rerank=1000) as query:
            ids,_,_=query.search(x[0]);assert len(ids)==0
            np.testing.assert_array_equal(ingest(index,x),np.arange(len(x)))
            info=index.fission_stats
            assert info['cells']>2 and info['splits']>0 and info['largest_cell']<=64
            for q in rng.normal(size=(4,dim)).astype(np.float32):
                ids,scores,_=query.search(q)
                want,expected=exact(q,x)
                if dim>1:np.testing.assert_array_equal(ids,want)
                np.testing.assert_allclose(scores,expected,atol=2e-6)
            index.pack_live()
            assert index.unpacked_bytes==0
        cells=info['cells']
    with reopen(root) as index:
        assert index.fission and index.count==len(x) and index.fission_stats['cells']==cells
        with index.context(nprobe=1536,rerank=1000) as query:
            q=x[17];ids,scores,_=query.search(q)
            _,want=exact(q,x)
            np.testing.assert_allclose(scores,want,atol=2e-6)

def test_updates_deletes_compaction_and_retry(tmp_path):
    root=tmp_path/'db';rng=np.random.default_rng(43);x=rng.normal(size=(700,64)).astype(np.float32)
    with AnchorIndex.create(root,64,cell_capacity=64,auto_pack_bytes=0) as index:
        ingest(index,x,[{'group':'a' if i%2 else 'b'} for i in range(len(x))])
        query=index.context(nprobe=1536,rerank=1500)
        for i in range(50):
            x[i]=rng.normal(size=64);index.update(i,x[i],{'group':'a'})
        for i in range(50,100):index.delete(i)
        extra=index.insert(x[0],{'group':'a'},idempotency_key='repeat')
        assert index.insert(x[0],{'group':'a'},idempotency_key='repeat')==extra
        x=np.vstack([x,x[0]])
        allowed=np.array([(i<50 or i%2==1 or i==extra) and not 50<=i<100 for i in range(len(x))])
        for action in [lambda:None,index.pack_live,index.compact]:
            action()
            for q in rng.normal(size=(3,64)).astype(np.float32):
                ids,scores,_=query.search(q,where={'group':'a'})
                _,expected=exact(q,x,eligible=allowed)
                assert allowed[ids].all();np.testing.assert_allclose(scores,expected,atol=2e-6)
        query.close()
    with reopen(root) as index:
        assert index.count==701 and index.fission
        with index.context(nprobe=1536,rerank=1500) as query:
            ids,_,_=query.search(x[51]);assert not np.isin(ids,np.arange(50,100)).any()

def test_checkpoint_then_unclosed_tail(tmp_path):
    root=tmp_path/'db'
    code='''
from pathlib import Path
import os,numpy as np
from fissiondb import AnchorIndex
p=Path(__import__('sys').argv[1]);x=np.random.default_rng(44).normal(size=(900,128)).astype(np.float32)
index=AnchorIndex.create(p,128,cell_capacity=64,auto_pack_bytes=0)
for i in range(0,900,100):
    index.insert_batch(x[i:i+100],group_commit=True)
    if i==200:index.pack_live()
os._exit(0)
'''
    subprocess.run([sys.executable,'-c',code,str(root)],check=True)
    x=np.random.default_rng(44).normal(size=(900,128)).astype(np.float32)
    with reopen(root) as index:
        assert index.count==900 and index.fission_stats['largest_cell']<=64
        with index.context(nprobe=1536,rerank=2000) as q:
            for v in x[[40,350,850]]:
                ids,scores,_=q.search(v);want,expected=exact(v,x)
                np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(scores,expected,atol=2e-6)

def test_parallel_search_during_fission(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(45).normal(size=(1200,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=64,auto_pack_bytes=0) as index:
        ingest(index,x[:200]);queries=[index.context(nprobe=1536,rerank=3000) for _ in range(3)]
        barrier=threading.Barrier(4)
        def reader(which):
            barrier.wait()
            for i in range(20):
                v=x[which*20+i];ids,scores,_=queries[which].search(v)
                assert len(ids)==10 and len(set(ids))==10 and (ids<len(x)).all()
                actual=np.sum(x[ids].astype(np.float64)*v,axis=1)/(np.linalg.norm(x[ids].astype(np.float64),axis=1)*np.linalg.norm(v.astype(np.float64)))
                np.testing.assert_allclose(scores,actual,atol=2e-6)
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            pending=[pool.submit(reader,i) for i in range(3)];barrier.wait();ingest(index,x[200:])
            for task in pending:task.result(timeout=60)
        assert index.count==1200 and index.fission_stats['cells']>20
        for query in queries:query.close()

def test_backup_rebuilds_fission_and_keeps_metadata(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(46).normal(size=(400,768)).astype(np.float32)
    with AnchorIndex.create(root,768,cell_capacity=64,auto_pack_bytes=0) as index:
        ingest(index,x,[{'source':'a'}]*len(x));index.delete(7)
        backup(index,tmp_path/'backup')
    kwargs=restore(tmp_path/'backup',tmp_path/'restored')
    with AnchorIndex(**kwargs) as index:
        assert index.fission and index.count==400 and index.fission_stats['cells']>2
        with index.context(nprobe=1536,rerank=1000) as q:
            ids,scores,_=q.search(x[7],where={'source':'a'})
            eligible=np.ones(len(x),bool);eligible[7]=False;_,expected=exact(x[7],x,eligible=eligible)
            assert 7 not in ids;np.testing.assert_allclose(scores,expected,atol=2e-6)

@pytest.mark.parametrize('damage',['state','codes','code_checksum'])
def test_truncated_derived_cache_recovers_from_journal(tmp_path,damage):
    root=tmp_path/'db';x=np.random.default_rng(47).normal(size=(300,128)).astype(np.float32)
    with AnchorIndex.create(root,128,cell_capacity=64,auto_pack_bytes=0) as index:ingest(index,x)
    if damage=='code_checksum':
        path=root/'live'/'fission.codes'
        with path.open('r+b') as f:
            # Corrupt all slots, including reachable ones; retain file length.
            for offset in range(0,path.stat().st_size,512*84):
                f.seek(offset);value=f.read(1);f.seek(offset);f.write(bytes([value[0]^1]))
    else:(root/'live'/('fission.'+damage)).write_bytes(b'torn')
    with reopen(root) as index:
        assert index.count==300 and index.fission
        with index.context(nprobe=1536,rerank=1000) as q:
            ids,scores,_=q.search(x[10]);want,expected=exact(x[10],x)
            np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(scores,expected,atol=2e-6)

def test_selective_filter_before_candidate_cap(tmp_path):
    root=tmp_path/'db';rng=np.random.default_rng(48);x=rng.normal(size=(6200,64)).astype(np.float32)
    with AnchorIndex.create(root,64,cell_capacity=128,auto_pack_bytes=0) as index:
        ingest(index,x,[{'group':'yes' if i<5000 else 'no'} for i in range(len(x))])
        with index.context(nprobe=1536,rerank=6200) as q:
            for v in x[[6001,6050]]:
                ids,scores,stats=q.search(v,where={'group':'yes'})
                want,expected=exact(v,x,eligible=np.arange(len(x))<5000)
                assert stats['entries']>5000
                np.testing.assert_array_equal(ids,want);np.testing.assert_allclose(scores,expected,atol=2e-6)

def test_distribution_changes_without_future_anchor_sampling(tmp_path):
    root=tmp_path/'db';rng=np.random.default_rng(49);x=rng.normal(0,.07,size=(2000,128)).astype(np.float32)
    x[:1000,0]+=1;x[1000:,1]+=1
    with AnchorIndex.create(root,128,cell_capacity=128,auto_pack_bytes=0) as index:
        ingest(index,x[:1000]);before=index.fission_stats['cells']
        # Exercise the documented 1536-cell policy across a distribution change.
        with index.context(rerank=400) as query:
            ingest(index,x[1000:]);assert index.fission_stats['cells']>before
            hits=0
            for v in x[1000:1020]:
                ids,_,_=query.search(v);expected,_=exact(v,x)
                hits+=len(set(ids)&set(expected))
            assert hits/200>=.95

def test_fission_memory_limit_continues_ingestion(tmp_path):
    root=tmp_path/'db';x=np.random.default_rng(50).normal(size=(200,64)).astype(np.float32)
    with AnchorIndex.create(root,64,cell_capacity=64,max_cells=2,auto_pack_bytes=0) as index:
        ingest(index,x);assert index.fission_stats['cells']==2 and index.count==200
        with index.context(nprobe=2,rerank=500) as query:
            ids,scores,_=query.search(x[30]);expected,values=exact(x[30],x)
            np.testing.assert_array_equal(ids,expected);np.testing.assert_allclose(scores,values,atol=2e-6)


from test_anchor_live import frozen


def test_fission_over_existing_frozen_index(frozen, tmp_path):
    directory,base,original=frozen
    residual=tmp_path/'residual';live=tmp_path/'live'
    with AnchorIndex(directory,base) as index:index.build_residual(residual)
    rng=np.random.default_rng(51)
    extra=rng.normal(size=(300,128)).astype(np.float32)
    options=dict(residual_dir=residual,live_dir=live,auto_pack_bytes=0)
    with AnchorIndex(directory,base,**options) as index:
        ingest(index,extra[:100]);index.pack_live()
    x=np.vstack([original,extra]);x[0]=extra[150]
    with AnchorIndex(directory,base,fission_cell_capacity=64,**options) as index:
        ingest(index,extra[100:]);index.update(0,x[0]);index.delete(1)
        assert index.count==len(x) and index.fission_stats['cells']>2
        with index.context(nprobe=1536,rerank=10000) as query:
            for v in [x[0],x[1],x[-1],original[100]]:
                ids,scores,_=query.search(v)
                _,expected=exact(v,x,eligible=np.arange(len(x))!=1)
                assert 1 not in ids and len(set(ids))==10
                np.testing.assert_allclose(scores,expected,atol=3e-6)
        index.compact()
    with AnchorIndex(directory,base,**options) as index:
        assert index.fission and index.count==len(x)
        with index.context(nprobe=1536,rerank=10000) as query:
            ids,scores,_=query.search(x[0])
            assert 0 in ids and scores[0]==pytest.approx(1,abs=2e-6)
