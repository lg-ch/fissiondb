"""Changing IO scheduling must preserve live ranking, filters and corruption checks."""
import numpy as np
import pytest
from fissiondb import AnchorIndex


@pytest.mark.parametrize('dim',[128,768,1024])
def test_live_batched_reads_match_serial_with_tail_chunks(tmp_path,dim):
    x=np.random.default_rng(957).normal(size=(6200,dim)).astype(np.float32)
    with AnchorIndex.create(tmp_path/'db',dim,cell_capacity=512,auto_pack_bytes=0) as index:
        for first in range(0,len(x),256):
            rows=x[first:first+256]
            index.insert_batch(rows,[{'group':'yes' if i<5000 else 'no'} for i in range(first,first+len(rows))],group_commit=True)
        index.flush_fission()
        with index.context(nprobe=1536,rerank=400) as query:
            default=query.search(x[6060])[2]['live']
            assert default['direct_reads']==0
            assert default['max_pending']>1 and default['submits']<default['reads']
            if default['code_reads']>64:assert default['overlap_batches']>0
            for where in (None,{'group':'yes'}):
                reference=query.residual_io(batch_cells=1,overlap=False,direct=False).search(x[6060],where=where)
                assert reference[2]['live']['max_pending']==1
                for width,overlap,direct in [(64,True,False),(64,True,True),(7,False,True),(256,True,True),(1,True,True)]:
                    actual=query.residual_io(batch_cells=width,overlap=overlap,direct=direct).search(x[6060],where=where)
                    np.testing.assert_array_equal(actual[0],reference[0])
                    np.testing.assert_array_equal(actual[1],reference[1])
                    assert actual[2]['entries']==reference[2]['entries']
                    io=actual[2]['live']
                    assert io['code_reads']>1 and io['rerank_reads']<=400
                    assert io['max_pending']<=width
                    assert io['direct_reads']==io['code_reads'] if direct else io['direct_reads']==0
                    if width>1:assert io['submits']<io['reads']
                    if width==1 and overlap:assert io['overlap_batches']>0


def test_live_io_error_drains_inflight_reads(tmp_path):
    x=np.random.default_rng(958).normal(size=(1200,128)).astype(np.float32)
    with AnchorIndex.create(tmp_path/'db',128,cell_capacity=64,auto_pack_bytes=0) as index:
        for start in range(0,len(x),128):index.insert_batch(x[start:start+128],group_commit=True)
        index.flush_fission()
        with index.context(nprobe=1536,rerank=400) as query:
            query.residual_io(batch_cells=1,overlap=True,direct=True)
            query.search(x[0])
            path=tmp_path/'db/live/fission.codes'
            with path.open('r+b') as f:
                for offset in range(0,path.stat().st_size,512*84):
                    f.seek(offset);value=f.read(1);f.seek(offset);f.write(bytes([value[0]^1]))
            with pytest.raises(OSError):query.search(x[0])
            with pytest.raises(RuntimeError):query.search(x[0])
