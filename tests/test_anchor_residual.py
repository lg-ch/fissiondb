"""Residual loading, exact reranking and existing live/filter semantics."""
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mangrove.anchors import AnchorIndex


def fingerprint(anchors, offsets, seed):
    lib = C.CDLL(ctypes.util.find_library('xxhash'))
    lib.XXH64.argtypes = [C.c_void_p, C.c_size_t, C.c_uint64]
    lib.XXH64.restype = C.c_uint64
    return lib.XXH64(anchors.ctypes.data, anchors.nbytes, seed) ^ lib.XXH64(offsets.ctypes.data, offsets.nbytes, 17)


@pytest.fixture(scope='module')
def residual(tmp_path_factory):
    root = tmp_path_factory.mktemp('residual')
    rng = np.random.default_rng(83)
    x = rng.normal(size=(5120, 1024)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = x.astype(np.float16)
    base = root/'base.f16bin'
    base.write_bytes(struct.pack('<II', *x.shape)+x.tobytes())
    index=root/'index'; index.mkdir()
    subprocess.run([str(ROOT/'mangrove-engine'),'abuild',str(base),str(index),'32',
                    '--m','2','--eps','999','--tqbits','1','--seed','52'],
                   check=True,capture_output=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    a=np.fromfile(index/'anchors.bin',np.float32)
    a8=np.rint(a*np.float32(127/(np.max(np.abs(a))+np.float32(1e-9)))).astype(np.int8)
    offsets=np.fromfile(index/'offs.bin',np.uint64)
    old=np.fromfile(index/'blocks.bin',np.uint8).reshape(-1,132)
    # Deliberately uninformative approximation: exhaustive candidate budget must
    # still recover the exact top results independently of the encoded signs.
    blocks=np.zeros((len(old),72),np.uint8);blocks[:,:4]=old[:,:4]
    out=root/'residual';out.mkdir();(out/'res512.bin').write_bytes(blocks.tobytes())
    (out/'residual.meta').write_text(f'MGR512V1 5120 32 1024 52 {fingerprint(a8,offsets,52)}\n')
    return index,base,out,x.astype(np.float64)


def test_residual_exact_live_delete_reopen(residual,tmp_path):
    index,base,out,x=residual
    q=x[7];unit=x/np.linalg.norm(x,axis=1,keepdims=True)
    gt=np.argsort(-(unit@(q/np.linalg.norm(q))))[:10]
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,scores,_=ctx.search(q)
            assert list(ids)==list(gt)
            np.testing.assert_allclose(scores,unit[ids]@(q/np.linalg.norm(q)),atol=2e-6)
            added=idx.insert(q,{'lang':'fr'})
            idx.set_metadata(7,{'lang':'fr'})
            ids,_,_=ctx.search(q,where={'lang':'fr'})
            assert set(ids)=={7,added}
            idx.delete(7)
            ids,_,_=ctx.search(q)
            assert 7 not in ids and added in ids
            idx.compact()
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(q,where={'lang':'fr'})
            assert list(ids)==[added]


def test_residual_large_allowed_filter(residual):
    index,base,out,x=residual
    allowed=np.arange(4097,dtype=np.uint32)
    with AnchorIndex(index,base,residual_dir=out) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(x[4500],allowed_ids=allowed)
            unit=x[:4097]/np.linalg.norm(x[:4097],axis=1,keepdims=True)
            assert list(ids)==list(np.argsort(-(unit@x[4500]))[:10])


def test_residual_rejects_invalid_format(residual,tmp_path):
    index,base,out,_=residual
    import shutil
    shutil.copy2(out/'res512.bin',tmp_path/'res512.bin')
    meta=(out/'residual.meta').read_text()
    for bad in [meta.replace('MGR512V1','MGR512V2'),meta.replace('5120','5121'),meta+'garbage',meta.rsplit(' ',1)[0]+' 0\n']:
        (tmp_path/'residual.meta').write_text(bad)
        with pytest.raises(OSError):AnchorIndex(index,base,residual_dir=tmp_path)
    (tmp_path/'residual.meta').write_text(meta)
    with open(tmp_path/'res512.bin','r+b') as f:f.truncate(72)
    with pytest.raises(OSError):AnchorIndex(index,base,residual_dir=tmp_path)
    with pytest.raises(OSError):AnchorIndex(index,base,residual_dir=out,int8=False)


def test_residual_rejects_unsupported_transport_threads(residual):
    index,base,out,_=residual
    with AnchorIndex(index,base,residual_dir=out) as idx:
        for kwargs in [dict(threads=2),dict(s3_url='https://example.invalid')]:
            with pytest.raises((OSError,ValueError)):
                idx.context(nprobe=32,rerank=20,**kwargs)


def test_converter_resume_and_immutable_publication(residual,tmp_path):
    index,base,_,x=residual
    out=tmp_path/'converted'
    with AnchorIndex(index,base) as idx:
        idx.build_residual(out)
        saved=(out/'res512.bin').read_bytes()
        with pytest.raises(OSError):idx.build_residual(out)
        assert (out/'res512.bin').read_bytes()==saved
        # Simulate a crash before manifest publication and a partial tail after
        # a checkpoint: resume must reconstruct exactly the original bytes.
        (out/'residual.meta').unlink()
        checkpoint=(out/'residual.checkpoint').read_text().split()
        checkpoint[-1]='16'
        (out/'residual.checkpoint').write_text(' '.join(checkpoint)+'\n')
        offsets=np.fromfile(index/'offs.bin',np.uint64)
        with open(out/'res512.bin','r+b') as f:f.truncate(int(offsets[16])//132*72+11)
        idx.build_residual(out)
        assert (out/'res512.bin').read_bytes()==saved
    with AnchorIndex(index,base,residual_dir=out) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,scores,_=ctx.search(x[13])
            assert ids[0]==13 and scores[0]==pytest.approx(1,abs=2e-6)


def test_converter_preserves_unknown_existing_file(residual,tmp_path):
    index,base,_,_=residual
    (tmp_path/'res512.bin').write_bytes(b'preexisting payload')
    with AnchorIndex(index,base) as idx:
        with pytest.raises(OSError):idx.build_residual(tmp_path)
    assert (tmp_path/'res512.bin').read_bytes()==b'preexisting payload'


@pytest.mark.parametrize('compressed',[False,True])
def test_update_frozen_and_live_preserves_ids(residual,tmp_path,compressed):
    index,base,out,x=residual
    args=dict(residual_dir=out) if compressed else {}
    q=x[15];new=-q
    with AnchorIndex(index,base,live_dir=tmp_path,**args) as idx:
        inserted=idx.insert(q,{'version':0})
        for doc_id in [15,inserted]:
            idx.update(doc_id,new,{'version':1})
        assert idx.count==5121
        with idx.context(nprobe=32,rerank=5120) as ctx:
            # Sparse exact route must read replacements, including frozen IDs.
            ids,scores,_=ctx.search(new,where={'version':1})
            assert set(ids)=={15,inserted}
            np.testing.assert_allclose(scores,1,atol=2e-6)
            # Broad route must suppress old vectors before candidate selection.
            ids,scores,_=ctx.search(new)
            assert {15,inserted}<=set(ids)
            ids,_,_=ctx.search(q)
            assert 15 not in ids and inserted not in ids
            for version in range(2,7):idx.update(15,new,{'version':version})
            before=idx.count
            idx.compact()
            assert idx.count==before
            ids,_,_=ctx.search(new,where={'version':6})
            assert list(ids)==[15]
    with AnchorIndex(index,base,live_dir=tmp_path,**args) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,scores,_=ctx.search(new,allowed_ids=[15,inserted])
            assert set(ids)=={15,inserted}
            np.testing.assert_allclose(scores,1,atol=2e-6)
            idx.delete(15)
            with pytest.raises(OSError):idx.update(15,q)
            with pytest.raises(OSError):idx.update(999999,q)
            idx.compact()
    with AnchorIndex(index,base,live_dir=tmp_path,**args) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(new)
            assert 15 not in ids and inserted in ids


def test_live_pack_tail_updates_reopen_and_compaction(residual,tmp_path):
    index,base,out,x=residual
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        added=idx.insert_batch(x[:40], [{'group':'old'}]*40)
        idx.update(15,-x[15],{'group':'changed'})
        with idx.context(nprobe=32,rerank=5120) as ctx:
            before=ctx.search(-x[15])[:2]
            idx.pack_live()
            assert idx.unpacked_bytes==0 and (tmp_path/'live.pack').exists()
            after=ctx.search(-x[15])[:2]
            assert set(before[0])==set(after[0])
            np.testing.assert_allclose(np.sort(before[1]),np.sort(after[1]),atol=2e-6)
            # These operations happen after the snapshot boundary.
            idx.update(int(added[0]),-x[0],{'group':'new'})
            late=idx.insert(-x[1],{'group':'new'})
            idx.delete(int(added[1]))
            ids,_,_=ctx.search(-x[0])
            assert added[0] in ids
            ids,_,_=ctx.search(-x[1])
            assert late in ids
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        assert 0<idx.unpacked_bytes<(tmp_path/'live.log').stat().st_size
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,scores,_=ctx.search(-x[0],where={'group':'new'})
            assert set(ids)=={int(added[0]),late}
            for n in range(10):idx.set_metadata(late,{'revision':n})
            idx.compact()  # Invalidates the old packed journal offsets.
            ids,_,_=ctx.search(-x[0])
            assert added[0] in ids
            idx.pack_live()
            assert idx.unpacked_bytes==0
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(-x[1])
            assert late in ids and added[1] not in ids


def test_live_pack_corruption_is_not_served(residual,tmp_path):
    index,base,out,x=residual
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        idx.insert(x[0]);idx.pack_live()
        # Flip a byte in a complete code entry, preserving file shape.
        with open(tmp_path/'live.pack','r+b') as f:
            f.seek(-10,2);b=f.read(1);f.seek(-1,1);f.write(bytes([b[0]^1]))
        with idx.context(nprobe=32,rerank=5120) as ctx:
            with pytest.raises(OSError):ctx.search(x[0])
        idx.pack_live()  # Rebuild from authoritative checksummed journal.
        with idx.context(nprobe=32,rerank=5120) as ctx:
            assert 5120 in ctx.search(x[0])[0]


def test_compaction_reclaims_deleted_live_vectors(residual,tmp_path):
    index,base,out,x=residual
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        added=idx.insert_batch(x[:64])
        for doc_id in added[:60]:idx.delete(int(doc_id))
        before=(tmp_path/'live.rows').stat().st_blocks*512
        result=idx.compact()
        assert result['after_bytes']<result['before_bytes']//2
        assert (tmp_path/'live.rows').stat().st_blocks*512<before//2
        assert idx.count==5184 and idx.deleted_count==60
        assert idx.insert(x[100])==5184
        idx.pack_live()
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        assert idx.count==5185 and idx.deleted_count==60
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(x[63],allowed_ids=added)
            assert set(ids)==set(added[60:])


def test_pack_allows_concurrent_writes_and_queries(residual,tmp_path):
    import concurrent.futures
    index,base,out,x=residual
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        idx.insert_batch(x[:200])
        with idx.context(nprobe=32,rerank=5120) as ctx:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                packed=pool.submit(idx.pack_live)
                writes=pool.submit(idx.insert_batch,x[200:240])
                searches=pool.submit(lambda:[ctx.search(x[i],top_k=1)[0].tolist() for i in range(5)])
                packed.result(timeout=30)
                added=writes.result(timeout=30)
                assert len(searches.result(timeout=30))==5
            ids,_,_=ctx.search(x[239],allowed_ids=added)
            assert ids[0]==added[-1]


def test_portable_backup_restore_and_checksums(residual,tmp_path):
    from mangrove.backup import create,restore
    from mangrove.metatypes import FloatSpec
    index,base,out,x=residual
    live=tmp_path/'live';backup=tmp_path/'backup'
    with AnchorIndex(index,base,residual_dir=out,live_dir=live,float_specs={'price':FloatSpec(2)}) as idx:
        added=idx.insert(-x[0],{'price':1.25})
        idx.update(15,-x[15],{'price':1.25})
        idx.pack_live()
        create(idx,backup)
        idx.delete(added)  # Must not alter the completed snapshot.
    kwargs=restore(backup,tmp_path/'restored')
    assert not (tmp_path/'restored/index/blocks.bin').exists()
    with AnchorIndex(**kwargs) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(-x[0],where={'price':('range',1.2,1.3)})
            assert set(ids)=={15,added}
            assert added in ctx.search(-x[0])[0]
    with open(backup/'base.f16bin','r+b') as f:
        f.seek(16);b=f.read(1);f.seek(16);f.write(bytes([b[0]^1]))
    with pytest.raises(OSError,match='checksum'):
        restore(backup,tmp_path/'bad-restore')
    assert not (tmp_path/'bad-restore/backup.json').exists()


def test_group_commit_idempotency_and_partial_conflict(residual,tmp_path):
    from mangrove.anchors import AnchorBatchError,IdempotencyConflict
    index,base,out,x=residual
    keys=[f'row-{i}' for i in range(300)]
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        ids=idx.insert_batch(x[:300],idempotency_keys=keys,group_commit=True)
        again=idx.insert_batch(x[:300],idempotency_keys=keys,group_commit=True)
        np.testing.assert_array_equal(ids,again)
        assert idx.count==5420
        with pytest.raises(AnchorBatchError) as caught:
            idx.insert_batch([x[301],x[302]],idempotency_keys=['new-row','row-0'],group_commit=True)
        assert isinstance(caught.value.__cause__,IdempotencyConflict)
        assert caught.value.committed_ids==[5420]
    with AnchorIndex(index,base,residual_dir=out,live_dir=tmp_path) as idx:
        assert idx.count==5421
        assert idx.insert(x[301],idempotency_key='new-row')==5420
        with idx.context(nprobe=32,rerank=5120) as ctx:
            assert 5420 in ctx.search(x[301])[0]


def test_product_http_client_pack_and_metrics(residual, tmp_path):
    import threading
    from serve_anchors import AnchorServer
    from mangrove.client import Client, ServiceError
    index, base, out, x = residual
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        server = AnchorServer(('127.0.0.1', 0), idx, nprobe=32, rerank=5120, api_key='test-key')
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            client = Client('http://127.0.0.1:%d' % server.server_port, api_key='test-key')
            assert client.health()['status'] == 'ok'
            docs = client.insert_batch(x[:2], [{'group':'a'}, {'group':'b'}], idempotency_keys=['a','b'])
            assert client.pack_live()['packed']
            client.add_metadata([docs[0]], {'rank':3})
            assert client.search(x[0], where={'rank':{'range':[1,4]}})['ids']==[docs[0]]
            assert client.search(x[0], where={'group':'a'})['ids'] == [docs[0]]
            client.update(docs[0], x[1], {'group':'c'})
            assert client.search(x[1], where={'group':'c'})['ids'] == [docs[0]]
            stats = client.stats()
            assert stats['http']['requests'] >= 5
            assert stats['http']['request_seconds'] > 0
            assert stats['unpacked_bytes'] > 0
            with pytest.raises(ServiceError) as error:
                Client(client.url, api_key='wrong').stats()
            assert error.value.status == 401
        finally:
            server.shutdown(); thread.join(); server.server_close()


def test_bulk_metadata_native_before_candidates_and_replay(residual, tmp_path):
    index, base, out, x = residual
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        ids = list(range(4097))
        idx.add_metadata(ids, {'category':'a'})
        idx.add_metadata(ids, {'year':2025})
        idx.add_metadata(ids, {'category':'a'})
        idx.add_metadata([4096], {'category':'b'})
        idx.set_metadata(0, {'category':'replaced'})
        idx.delete(1)
        with pytest.raises(OSError):idx.add_metadata([2,1], {'category':'invalid'})
        with pytest.raises(ValueError):idx.add_metadata([-1], {'category':'bad'})
        with pytest.raises(ValueError):idx.add_metadata([2], {'a':1, 'b':2})
        with idx.context(nprobe=32, rerank=5120) as ctx:
            assert len(ctx.search(x[2], where={'category':'invalid'})[0]) == 0
        idx.compact()
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        with idx.context(nprobe=32, rerank=5120) as ctx:
            got = ctx.search(x[30], where={'category':'a'})[0]
            candidates=np.arange(2,4097); unit=x/np.linalg.norm(x,axis=1,keepdims=True)
            expected=candidates[np.argsort(-(unit[candidates]@unit[30]))[:10]]
            assert list(got)==list(expected)
            assert ctx.search(x[30], where={'year':('range',2024,2026)})[0].tolist()==list(expected)
            assert ctx.search(x[4096], where={'category':'b'})[0].tolist()==[4096]


def test_bulk_metadata_upgrades_v1_journal(residual, tmp_path):
    index, base, out, x = residual
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path):pass
    with (tmp_path/'live.log').open('r+b') as f:
        f.seek(8);f.write(struct.pack('<Q',1))
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        idx.add_metadata([5,6], {'legacy':'imported'})
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        with idx.context(nprobe=32, rerank=5120) as ctx:
            assert set(ctx.search(x[5], where={'legacy':'imported'})[0]) == {5,6}


def test_bulk_metadata_retry_after_partial_tail(residual, tmp_path):
    index, base, out, x = residual
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        idx.add_metadata([5,6], {'year':2025})
    journal=tmp_path/'live.log'
    with journal.open('r+b') as f:f.truncate(journal.stat().st_size-20)
    with AnchorIndex(index, base, residual_dir=out, live_dir=tmp_path) as idx:
        idx.add_metadata([5,6], {'year':2025})
        with idx.context(nprobe=32, rerank=5120) as ctx:
            assert set(ctx.search(x[5], where={'year':('range',2020,2030)})[0]) == {5,6}
            assert set(ctx.search(x[5], where={'year':('exists',)})[0]) == {5,6}


def test_converter_accepts_valid_base_with_heldout_tail(residual,tmp_path):
    index,base,_,x=residual
    payload=base.read_bytes()
    extended=tmp_path/'extended.f16bin'
    extended.write_bytes(struct.pack('<II',len(x)+1,1024)+payload[8:]+payload[8:8+2048])
    output=tmp_path/'converted'
    with AnchorIndex(index,extended) as idx:idx.build_residual(output)
    assert int((output/'residual.meta').read_text().split()[1])==len(x)
    with AnchorIndex(index,extended,residual_dir=output) as idx:
        with idx.context(nprobe=32,rerank=5120) as ctx:
            ids,_,_=ctx.search(x[13]);assert ids[0]==13 and max(ids)<len(x)
    with extended.open('r+b') as stream:stream.truncate(extended.stat().st_size-2)
    with pytest.raises(OSError):
        with AnchorIndex(index,extended) as idx:idx.build_residual(tmp_path/'invalid')


def test_residual_query_projection_removes_parallel_noise(tmp_path):
    from test_tq_scaling import inverse_rotation
    dim=1024;seed=52
    anchor_rot=np.ones(dim,np.float32)/32
    first_half=np.r_[np.ones(512),-np.ones(512)].astype(np.float32)/32
    alternating=np.tile(np.array([1,-1],np.float32),512)/32
    anchor=inverse_rotation(anchor_rot,seed)
    x=np.array([inverse_rotation(.9*anchor_rot+np.sqrt(1-.9**2)*first_half,seed),
                inverse_rotation(.95*anchor_rot+np.sqrt(1-.95**2)*alternating,seed)],dtype='<f2')
    base=tmp_path/'base.f16bin';base.write_bytes(struct.pack('<II',2,dim)+x.tobytes())
    directory=tmp_path/'index';directory.mkdir()
    (directory/'meta.txt').write_text('2 1024 2 1 999 2 52 1024\n')
    np.tile(anchor,(2,1)).astype('<f4').tofile(directory/'anchors.bin')
    np.ones(dim,dtype='<f4').tofile(directory/'scale.bin')
    blocks=np.zeros((4,132),np.uint8)
    for i,doc in enumerate([0,1,0,1]):blocks[i,:4]=np.frombuffer(struct.pack('<I',doc),np.uint8)
    blocks.tofile(directory/'blocks.bin');np.array([0,264,528],dtype='<u8').tofile(directory/'offs.bin')
    residual_dir=tmp_path/'residual'
    with AnchorIndex(directory,base) as idx:idx.build_residual(residual_dir)
    # Query equals the canonical anchor. Exact residual contributions are zero;
    # the truncated sign sketch spuriously aligns the first document otherwise.
    assert np.argmax(x.astype(np.float64)@anchor)==1
    with AnchorIndex(directory,base,residual_dir=residual_dir) as idx:
        with idx.context(nprobe=2,rerank=1) as ctx:
            assert ctx.search(anchor,top_k=1)[0].tolist()==[1]
