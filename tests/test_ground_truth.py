import struct
import numpy as np
import pytest
from fissiondb.ground_truth import exact_top_k


def test_streamed_reference_filters_ties_and_prefix(tmp_path):
    rng=np.random.default_rng(71)
    x=rng.normal(size=(73,16)).astype(np.float16);x[9]=x[3]
    q=np.array([x[3],x[8]],dtype=np.float32)
    path=tmp_path/'base';path.write_bytes(struct.pack('<II',*x.shape)+x.tobytes())
    allowed=[None,list(range(0,60,2))]
    ids,scores=exact_top_k(path,q,top_k=5,count=60,block_rows=7,query_batch=1,allowed_ids=allowed)
    v=x[:60].astype(np.float64);v/=np.linalg.norm(v,axis=1,keepdims=True)
    qq=q.astype(np.float64);qq/=np.linalg.norm(qq,axis=1,keepdims=True)
    for i in range(2):
        eligible=np.arange(60) if allowed[i] is None else np.array(allowed[i])
        s=v[eligible]@qq[i];expected=np.lexsort((eligible,-s))[:5]
        assert ids[i].tolist()==eligible[expected].tolist()
        np.testing.assert_allclose(scores[i],s[expected],atol=1e-14)
    assert ids[0,:2].tolist()==[3,9]
    with pytest.raises(ValueError):exact_top_k(path,q,allowed_ids=[[1],[2]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):exact_top_k(path,q)
