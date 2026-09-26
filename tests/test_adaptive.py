import numpy as np
import pytest
from test_anchor_live import frozen
from test_anchor_residual import residual
from mangrove.anchors import AnchorIndex
from mangrove.adaptive import calibrate


def test_autocalibration_expands_routing_without_validation_feedback(frozen):
    from mangrove.adaptive import autocalibrate
    x=frozen[2].astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    q=x[:8];gt=np.argsort(-(q@x.T),axis=1)[:,:10]
    split=['calibration']*4+['validation']*4
    with AnchorIndex(frozen[0],frozen[1]) as index:
        kwargs=dict(partitions=split,initial_probes=1,maximum_probes=32,initial_rerank=10240,maximum_rerank=10240,target_recall=1)
        profile=autocalibrate(index,q,gt,**kwargs)
        assert profile['validated'] and len(profile['measurements'])>1
        assert all(r['configuration']['rerank']==10240 for r in profile['measurements'])
        assert all('validation_rows' not in r for r in profile['measurements'][:-1])
        assert all(row['query']<4 for r in profile['measurements'] for row in r['rows'])
        bad=gt.copy();bad[4:]=np.arange(100,110)
        failed=autocalibrate(index,q,bad,**kwargs)
        assert not failed['validated']
        assert failed['configuration']==profile['configuration']
        assert len(failed['measurements'])==len(profile['measurements'])


def test_explicit_workload_partitions_and_latency_constraint(frozen):
    x=frozen[2].astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    q=x[:8];gt=np.argsort(-(q@x.T),axis=1)[:,:10]
    config=dict(minimum=32,maximum=32,filtered_minimum=32,rerank=10240,gap=0)
    split=['validation']*3+['calibration']*5
    with AnchorIndex(frozen[0],frozen[1]) as index:
        profile=calibrate(index,q,gt,[config],partitions=split,query_kind='query-to-document',target_recall=1)
        assert profile['validated'] and profile['partitions']==split
        assert profile['measurements'][0]['summary']['validation']['null']['queries']==3
        rejected=calibrate(index,q,gt,[config],partitions=split,latency_target_ms=1e-12)
        assert not rejected['validated'] and rejected['configuration'] is None
        with pytest.raises(ValueError):calibrate(index,q,gt,[config],partitions=['calibration']*8)


@pytest.mark.parametrize('kind',['tq','residual'])
def test_adaptive_matches_fixed_and_budget_reset(frozen,residual,kind):
    directory,base,x=frozen if kind=='tq' else (residual[0],residual[1],residual[3])
    options={} if kind=='tq' else {'residual_dir':residual[2]}
    with AnchorIndex(directory,base,**options) as index:
        with index.context(nprobe=4,rerank=100) as fixed,index.context(nprobe=32,rerank=100) as adaptive:
            adaptive.adapt(minimum=4)
            for q in x[:3]:
                expected=fixed.search(q)
                actual=adaptive.search(q)
                assert actual[0].tolist()==expected[0].tolist()
                assert actual[2]['bytes']==expected[2]['bytes']
                assert actual[2]['probes']==4
            adaptive.adapt(minimum=4,gap=1e6)
            assert adaptive.search(x[0])[2]['probes']==32
            adaptive.adapt(minimum=32,code_bytes=1)
            partial=adaptive.search(x[0])[2]
            assert partial['probes']==1 and partial['budget_limited']
            adaptive.adapt(minimum=4)
            assert not adaptive.search(x[0])[2]['budget_limited']
            with pytest.raises(ValueError):adaptive.adapt(minimum=33)
            with pytest.raises(ValueError):adaptive.adapt(minimum=2**32+1)


def test_filter_floor_and_exact_bypass(frozen,tmp_path):
    with AnchorIndex(frozen[0],frozen[1],live_dir=tmp_path) as index:
        with index.context(nprobe=32,rerank=100) as ctx:
            ctx.adapt(minimum=2,filtered_minimum=16)
            assert ctx.search(frozen[2][0])[2]['probes']==2
            assert ctx.search(frozen[2][0],allowed_ids=range(5000))[2]['probes']==16
            assert ctx.search(frozen[2][0],allowed_ids=[0])[2]['probes']==0
            assert ctx.search(frozen[2][0],allowed_ids=[])[0].tolist()==[]


def test_calibration_disjoint_validation_and_identity(frozen):
    x=frozen[2].astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    q=x[:8];gt=np.argsort(-(q@x.T),axis=1)[:,:10]
    config=dict(minimum=32,maximum=32,filtered_minimum=32,rerank=10240,gap=0)
    with AnchorIndex(frozen[0],frozen[1]) as index:
        profile=calibrate(index,q,gt,[config],target_recall=1)
        assert profile['validated']
        assert profile['measurements'][0]['summary']['validation']['null']['queries']==4
        with index.calibrated_context(profile) as ctx:
            assert ctx.search(q[0])[0].tolist()==gt[0].tolist()
            with pytest.raises(ValueError):ctx.search(q[0],top_k=11)
        import json,threading,urllib.request
        from serve_anchors import AnchorServer
        server=AnchorServer(('127.0.0.1',0),index,calibration=profile)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/search',
                data=json.dumps({'qvec':q[0].tolist()}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=10) as response:result=json.load(response)
            assert result['ids']==gt[0].tolist()
        finally:
            server.shutdown();thread.join();server.server_close()
        profile['fingerprint']='wrong'
        with pytest.raises(ValueError):index.calibrated_context(profile)
        bad=gt.copy();bad[1::2]=np.arange(100000,100010)
        failed=calibrate(index,q,bad,[config],target_recall=1)
        assert not failed['validated']
        with pytest.raises(ValueError):index.calibrated_context(failed)
