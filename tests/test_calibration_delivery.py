import numpy as np
import pytest
from fissiondb.calibration import recall_lower_bound,calibrate_snapshot


def test_bound_does_not_accept_tiny_perfect_sample():
    assert recall_lower_bound(np.ones(64))<.95
    assert recall_lower_bound(np.ones(600))>.98
    dispersed=np.r_[np.ones(540),np.zeros(60)]
    constant=np.full(600,.9)
    assert recall_lower_bound(dispersed)<recall_lower_bound(constant)<.9
    with pytest.raises(ValueError):recall_lower_bound([1,float('nan')])


class FakeIndex:
    dim=2
    count=10000
    def __init__(self,fail_audit=False):self.calls=[];self.fail_audit=fail_audit
    def context(self,**kwargs):return FakeContext(self,kwargs['nprobe'])


class FakeContext:
    def __init__(self,index,probes):self.index=index;self.probes=probes
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def adapt(self,**kwargs):pass
    def search(self,q,top_k=10,diagnostic_ids=None):
        i=int(q[0])-1;self.index.calls.append((i,self.probes,diagnostic_ids is not None))
        good=self.probes>=2 and not(self.index.fail_audit and i>=300)
        ids=np.arange(10) if good else np.r_[np.arange(9),99]
        return ids,np.ones(10),{'diagnostic':{'routed':10 if good else 9}}


def workload():
    q=np.c_[np.arange(1,601),np.ones(600)].astype(np.float32)
    return q,np.tile(np.arange(10),(600,1)),['calibration']*300+['validation']*300


def test_end_to_end_selection_audit_and_resume(monkeypatch,tmp_path):
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    index=FakeIndex();q,gt,labels=workload();checkpoint=tmp_path/'progress.json'
    options=dict(initial_probes=1,maximum_probes=2,initial_rerank=10,maximum_rerank=10,checkpoint=checkpoint)
    result=calibrate_snapshot(index,q,gt,labels,**options)
    assert result['validated'] and result['status']=='validated'
    assert result['configuration']['maximum']==2
    first_audit=next(j for j,x in enumerate(index.calls) if x[0]>=300)
    assert all(x[0]>=300 for x in index.calls[first_audit:])
    calls=len(index.calls)
    resumed=calibrate_snapshot(index,q,gt,labels,**options)
    assert resumed==result and len(index.calls)==calls
    with pytest.raises(ValueError):calibrate_snapshot(index,q,gt,labels,target_recall=.94,**options)
    with pytest.raises(ValueError):calibrate_snapshot(index,q,gt,labels,protocol_id="different-cache",**options)


def test_failure_and_latency_tradeoff_are_explicit(monkeypatch):
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    q,gt,labels=workload();options=dict(initial_probes=1,maximum_probes=2,initial_rerank=10,maximum_rerank=10)
    index=FakeIndex(fail_audit=True);failed=calibrate_snapshot(index,q,gt,labels,**options)
    assert failed['status']=='quality_not_validated' and not failed['validated']
    first_audit=next(j for j,x in enumerate(index.calls) if x[0]>=300)
    assert all(x[0]>=300 for x in index.calls[first_audit:])
    tradeoff=calibrate_snapshot(FakeIndex(),q,gt,labels,latency_target_ms=1e-12,**options)
    assert tradeoff['status']=='latency_tradeoff' and not tradeoff['validated']
    assert tradeoff['audit']['quality']['summary']['recall_lower']>=.95


def test_rejects_group_leakage_and_small_samples(monkeypatch):
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    q,gt,labels=workload()
    with pytest.raises(ValueError):calibrate_snapshot(FakeIndex(),q,gt,labels,groups=['same']*600)
    with pytest.raises(ValueError):calibrate_snapshot(FakeIndex(),q[:64],gt[:64],labels[:64])


from test_anchor_live import frozen


def test_native_cli_profile_can_serve_and_resume(frozen,tmp_path):
    import json,os,subprocess,sys
    from fissiondb.anchors import AnchorIndex
    rng=np.random.default_rng(81)
    q=rng.normal(size=(600,128)).astype(np.float32)
    x=frozen[2].astype(np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True)
    gt=np.argsort(-(q.astype(np.float64)@x.T),axis=1)[:,:10]
    workload_path=tmp_path/'workload.npz';profile_path=tmp_path/'profile.json'
    np.savez(workload_path,queries=q,ids=gt,partitions=['calibration']*300+['validation']*300)
    command=[sys.executable,'-m','fissiondb.calibration',str(frozen[0]),str(frozen[1]),str(workload_path),
             '--output',str(profile_path),'--initial-rerank','10240','--max-rerank','10240','--p95-ms','10000','--cold']
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    subprocess.run(command,check=True,capture_output=True,env=environment)
    result=json.loads(profile_path.read_text());assert result['validated']
    assert result['audit']['quality']['summary']['recall_lower']>=.95
    with AnchorIndex(frozen[0],frozen[1]) as index:
        with index.calibrated_context(result) as context:
            assert context.search(q[0])[0].tolist()==gt[0].tolist()
    subprocess.run(command,check=True,capture_output=True,env=environment)
    assert json.loads(profile_path.read_text())==result
    rejected=subprocess.run(command[:-1],capture_output=True,env=environment)
    assert rejected.returncode!=0 and b'Checkpoint belongs' in rejected.stderr



def test_resume_after_audit_interruption_keeps_frozen_choices(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    q,gt,labels=workload();index=FakeIndex();checkpoint=tmp_path/'checkpoint.json'
    original=FakeContext.search
    def interrupted(self,q,**kwargs):
        if int(q[0])==311:raise RuntimeError('interrupted audit')
        return original(self,q,**kwargs)
    monkeypatch.setattr(FakeContext,'search',interrupted)
    options=dict(initial_probes=1,maximum_probes=2,initial_rerank=10,maximum_rerank=10,checkpoint=checkpoint)
    with pytest.raises(RuntimeError,match='interrupted audit'):calibrate_snapshot(index,q,gt,labels,**options)
    saved=json.loads(checkpoint.read_text());assert saved['stage']=='validation'
    monkeypatch.setattr(FakeContext,'search',original);index.calls.clear()
    result=calibrate_snapshot(index,q,gt,labels,**options)
    assert result['validated'] and all(i>=300 for i,_,_ in index.calls)
    assert result['configuration']==saved['choices']['quality']


def test_rejects_unachievable_sample_bound_and_mutable_index(monkeypatch):
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    q,gt,labels=workload()
    with pytest.raises(ValueError,match='even at perfect recall'):
        calibrate_snapshot(FakeIndex(),q[:400],gt[:400],['calibration']*200+['validation']*200)
    index=FakeIndex();index.live=True
    with pytest.raises(ValueError,match='immutable'):calibrate_snapshot(index,q,gt,labels)
    with pytest.raises(ValueError,match='protocol_id'):
        calibrate_snapshot(FakeIndex(),q,gt,labels,before_query=lambda:None)



def test_controller_expands_the_stage_losing_true_neighbors(monkeypatch):
    monkeypatch.setattr('fissiondb.calibration.fingerprint',lambda index:'snapshot-a')
    q,gt,labels=workload()
    class RerankIndex(FakeIndex):
        def context(self,**kwargs):
            parent=self
            class Context(FakeContext):
                def search(self,q,top_k=10,diagnostic_ids=None):
                    parent.calls.append((kwargs['nprobe'],kwargs['rerank']))
                    ids=np.arange(10) if kwargs['rerank']>=20 else np.r_[np.arange(9),99]
                    return ids,np.ones(10),{'diagnostic':{'routed':10}}
            return Context(self,kwargs['nprobe'])
    index=RerankIndex()
    result=calibrate_snapshot(index,q,gt,labels,initial_probes=1,maximum_probes=2,initial_rerank=10,maximum_rerank=20)
    assert result['validated'] and result['configuration']['maximum']==1 and result['configuration']['rerank']==20
    assert {p for p,r in index.calls}=={1}



def test_dimension_default_routes_native_and_server_without_calibration(frozen,tmp_path):
    import os,subprocess
    from test_anchor_live import ROOT
    from fissiondb.anchors import AnchorIndex
    from serve_anchors import AnchorServer
    directory=tmp_path/'index';directory.mkdir()
    subprocess.run([str(ROOT/'fissiondb-engine'),'abuild',str(frozen[1]),str(directory),'128','--m','2','--eps','999','--tqbits','1','--seed','52'],
                   check=True,capture_output=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    with AnchorIndex(directory,frozen[1]) as index:
        assert index.default_nprobe==64
        with index.context(rerank=1000) as automatic,index.context(nprobe=64,rerank=1000) as explicit:
            actual=automatic.search(frozen[2][0]);expected=explicit.search(frozen[2][0])
            assert actual[0].tolist()==expected[0].tolist() and actual[2]['probes']==64
        server=AnchorServer(('127.0.0.1',0),index,rerank=1000)
        try:assert server._all_contexts[0].search(frozen[2][0])[2]['probes']==64
        finally:
            for context in server._all_contexts:context.close()
            server.server_close()
        with index.context(nprobe=8) as overridden:assert overridden.search(frozen[2][0])[2]['probes']==8
    with AnchorIndex(frozen[0],frozen[1]) as small:
        with small.context() as query:assert query.search(frozen[2][0])[2]['probes']==32
