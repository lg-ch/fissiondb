"""Anisotropic quantization must retain the geometry of decoded coordinates."""
import struct
import numpy as np
import pytest
from fissiondb.anchors import AnchorIndex


def inverse_rotation(v, seed):
    mask=(1<<64)-1;state=seed^0x51CA;signs=[]
    for _ in v:
        state=(state+0x9E3779B97F4A7C15)&mask;z=state
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&mask
        z=((z^(z>>27))*0x94D049BB133111EB)&mask
        signs.append(1 if (z^(z>>31))&1 else -1)
    a=np.asarray(v,np.float32).copy();h=1
    while h<len(a):
        for i in range(0,len(a),2*h):
            left=a[i:i+h].copy();right=a[i+h:i+2*h].copy()
            a[i:i+h]=left+right;a[i+h:i+2*h]=left-right
        h*=2
    return a/np.sqrt(np.float32(len(a)))*np.asarray(signs,np.float32)


@pytest.mark.parametrize('case',['inverse_scale','decoded_norm'])
@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('threads',[1,2])
def test_inverse_quantization_scale_before_candidate_admission(tmp_path,bits,threads,case):
    dim=128;seed=52
    rotated=np.zeros((2,dim),np.float32)
    rotated[0,0]=.01;rotated[0,2]=np.sqrt(1-.01**2)
    rotated[1,1]=1
    if case=='decoded_norm':
        rotated[:]=0
        rotated[0,:2]=[.8,.6]
        rotated[1,:2]=[.95,-np.sqrt(1-.95**2)]
    x=np.asarray([inverse_rotation(v,seed) for v in rotated],dtype='<f2')
    base=tmp_path/'base.f16bin';base.write_bytes(struct.pack('<II',2,dim)+x.tobytes())
    directory=tmp_path/'index';directory.mkdir()
    (directory/'meta.txt').write_text(f'1 {dim} 1 {bits} 0 2 {seed} {dim}\n')
    x[0].astype('<f4').tofile(directory/'anchors.bin')
    scale=np.ones(dim,np.float32)
    if case=="inverse_scale":scale[0]=100
    scale.astype('<f4').tofile(directory/'scale.bin')
    codes=np.rint(rotated*scale).astype(np.int32)
    payload=bytearray()
    for doc,code in enumerate(codes):
        payload+=struct.pack('<I',doc)
        if bits==2:
            packed=((code+2)&3).reshape(-1,4)
            payload+=np.sum(packed<<np.arange(0,8,2),axis=1).astype(np.uint8).tobytes()
        else:
            packed=(code&15).reshape(-1,2)
            payload+=(packed[:,0]|(packed[:,1]<<4)).astype(np.uint8).tobytes()
    (directory/'blocks.bin').write_bytes(payload)
    np.array([0,len(payload)],dtype='<u8').tofile(directory/'offs.bin')
    query=np.zeros(dim,np.float32);query[:2]=[2,1] if case=="inverse_scale" else [1,0];query=inverse_rotation(query,seed)
    # Document 1 is the true nearest neighbour. Incorrect scale weighting or
    # ignoring the changed reconstruction norm admits only document 0.
    assert np.argmax(x.astype(np.float64)@query)==1
    with AnchorIndex(directory,base) as index:
        with index.context(nprobe=1,rerank=1,threads=threads) as ctx:
            assert ctx.search(query,top_k=1)[0].tolist()==[1]
