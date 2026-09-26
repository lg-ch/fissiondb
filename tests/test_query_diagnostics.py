import numpy as np
import pytest
from test_anchor_live import frozen
from test_anchor_residual import residual
from fissiondb.anchors import AnchorIndex


@pytest.mark.parametrize('kind',['tq','residual'])
def test_exact_unique_counts_and_reset(frozen,residual,kind):
    directory,base,x=frozen if kind=='tq' else (residual[0],residual[1],residual[3])
    options={} if kind=='tq' else {'residual_dir':residual[2]}
    with AnchorIndex(directory,base,**options) as index,index.context(nprobe=32,rerank=20) as ctx:
        baseline=ctx.search(x[0])[0]
        ids,_,stats=ctx.search(x[0],diagnostic_ids=[0],diagnostic_unique=True)
        assert ids.tolist()==baseline.tolist()
        assert stats['diagnostic']['unique_vectors']==len(x)
        assert stats['diagnostic']['eligible_entries']>len(x)
        for allowed,n in [([],0),([0,1,2],3)]:
            d=ctx.search(x[0],allowed_ids=allowed,diagnostic_ids=[0],diagnostic_unique=True)[2]['diagnostic']
            assert d['eligible_entries']==d['unique_vectors']==n
        assert 'unique_vectors' not in ctx.search(x[0],diagnostic_ids=[0])[2]['diagnostic']
        with pytest.raises(ValueError):ctx.search(x[0],diagnostic_unique=True)


@pytest.mark.parametrize('kind',['tq','residual'])
def test_trace_matches_exhaustive_candidates_and_resets(frozen,residual,kind):
    directory,base,x=frozen if kind=='tq' else (residual[0],residual[1],residual[3])
    options={} if kind=='tq' else {'residual_dir':residual[2]}
    v=x.astype(np.float64);v/=np.linalg.norm(v,axis=1,keepdims=True)
    gt=np.argsort(-(v[0]@v.T))[:10]
    with AnchorIndex(directory,base,**options) as index,index.context(nprobe=32,rerank=20000) as ctx:
        baseline=ctx.search(v[0])
        ids,_,stats=ctx.search(v[0],diagnostic_ids=gt)
        assert ids.tolist()==baseline[0].tolist()
        assert stats['diagnostic']['routed']==10
        assert stats['diagnostic']['candidates']==10
        assert 'diagnostic' not in ctx.search(v[0])[2]
        assert ctx.search(v[0],allowed_ids=[],diagnostic_ids=gt)[2]['diagnostic']['routed']==0
        filtered=ctx.search(v[0],allowed_ids=gt[:3],diagnostic_ids=gt)[2]['diagnostic']
        assert filtered['routed']==filtered['candidates']==3
        with pytest.raises(ValueError):ctx.search(v[0],diagnostic_ids=[0,0])
        with pytest.raises(ValueError):ctx.search(v[0],diagnostic_ids=[index.count])


def test_trace_exposes_candidate_loss(frozen):
    x=frozen[2].astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    gt=np.argsort(-(x[0]@x.T))[:10]
    with AnchorIndex(frozen[0],frozen[1]) as index,index.context(nprobe=32,rerank=1) as ctx:
        ids,_,stats=ctx.search(x[0],top_k=1,diagnostic_ids=gt)
        trace=stats['diagnostic']
        assert trace['routed']==10
        assert trace['candidates']==len(set(ids)&set(gt))<=1


def test_trace_rejects_live(frozen,tmp_path):
    with AnchorIndex(frozen[0],frozen[1],live_dir=tmp_path) as index,index.context(nprobe=4,rerank=20) as ctx:
        with pytest.raises(ValueError):ctx.search(frozen[2][0],diagnostic_ids=[0])
        assert len(ctx.search(frozen[2][0])[0])==10
