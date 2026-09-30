"""Automatic native-width thresholds must drive real, durable splits."""
import json
import subprocess
import sys

import numpy as np
import pytest
from fissiondb import AnchorIndex
from test_fission import exact, ingest, reopen
from test_residual_dimensions import build


@pytest.mark.parametrize('dim,expected',[
    (1,64),(31,64),(32,64),(33,66),(128,256),(384,768),
    (768,1536),(1000,2000),(1024,2048),
])
def test_default_splits_at_native_threshold_and_reopens(tmp_path,dim,expected):
    root=tmp_path/'db'
    x=np.random.default_rng(730+dim).normal(size=(expected+1,dim)).astype(np.float32)
    with AnchorIndex.create(root,dim,auto_pack_bytes=0) as index:
        assert index.fission_capacity==expected
        ingest(index,x)
        stats=index.fission_stats
        assert stats['splits']>0 and stats['largest_cell']<=expected
        assert stats['center_dim']==dim and index.count==len(x)
        with index.context(nprobe=10000,rerank=len(x)) as query:
            _,scores,stats=query.search(x[7])
            _,wanted=exact(x[7],x)
            np.testing.assert_allclose(scores,wanted,atol=3e-6)
            assert stats['live']['direct_reads']==stats['live']['code_reads']>0
    with reopen(root,auto_pack_bytes=0) as index:
        assert index.fission_capacity==expected and index.count==len(x)
        assert index.fission_stats['largest_cell']<=expected


@pytest.mark.parametrize('dim,expected',[(33,66),(128,256),(384,768),(768,1536),(1000,2000)])
def test_native_auto_enable_on_existing_frozen_index(tmp_path,dim,expected):
    directory,base,residual,_=build(tmp_path,dim)
    with AnchorIndex(directory,base,residual_dir=residual,live_dir=tmp_path/'live',
                     fission_cell_capacity=0,auto_pack_bytes=0) as index:
        assert index.fission_capacity==expected


def test_auto_reopen_preserves_explicit_capacity_and_later_change(tmp_path):
    root=tmp_path/'db'
    with AnchorIndex.create(root,768,cell_capacity=2048,auto_pack_bytes=0) as index:
        assert index.fission_capacity==2048
    with reopen(root,fission_cell_capacity=0,auto_pack_bytes=0) as index:
        assert index.fission_capacity==2048
        index.set_fission_capacity(1536)
    with reopen(root,fission_cell_capacity=0,auto_pack_bytes=0) as index:
        assert index.fission_capacity==1536


@pytest.mark.parametrize('dim,setting,expected',[(768,None,1536),(384,0,768),(128,2048,2048)])
def test_cli_reports_and_persists_resolved_capacity(tmp_path,dim,setting,expected):
    root=tmp_path/'db'
    args=[sys.executable,'-m','fissiondb.cli','create','--index',str(root),'--dim',str(dim)]
    if setting is not None:args+=['--cell-capacity',str(setting)]
    result=subprocess.run(args,capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(result.stdout)['cell_capacity']==expected
    with reopen(root,auto_pack_bytes=0) as index:assert index.fission_capacity==expected


@pytest.mark.parametrize('setting',[-1,1,63,65537])
def test_invalid_capacity_does_not_create_collection(tmp_path,setting):
    root=tmp_path/'db'
    with pytest.raises(ValueError):AnchorIndex.create(root,768,cell_capacity=setting)
    assert not root.exists()
