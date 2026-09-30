"""Exercise integer SIMD exactness, safe tails and process-wide fallback."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def x86_checker(tmp_path_factory):
    if platform.machine() != 'x86_64':
        pytest.skip('x86 instruction tests')
    binary = tmp_path_factory.mktemp('integer_kernels') / 'check'
    subprocess.run(['gcc', '-O2', '-std=c11', '-march=x86-64',
                    str(ROOT / 'tests/check_x86_kernels.c'), '-o', str(binary)], check=True)
    return binary


@pytest.mark.parametrize('disabled', [False, True])
def test_integer_kernels_and_dispatch(x86_checker, disabled):
    env = {**os.environ, 'FISSIONDB_DISABLE_AVX512': '1' if disabled else '0'}
    result = subprocess.run([str(x86_checker)], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    supported = 'avx512=1 ' in result.stdout
    assert f'selected_avx512={int(supported and not disabled)}' in result.stdout
    assert 'guard pages passed' in result.stdout


def test_dispatch_preserves_retrieval_and_encoded_cells(tmp_path, x86_checker):
    probe = subprocess.run([str(x86_checker)], capture_output=True, text=True)
    if probe.returncode == 77:
        pytest.skip(probe.stdout)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    if 'avx512=1 ' not in probe.stdout:
        pytest.skip('AVX-512BW unavailable on this runner; AVX2 tested separately')
    script = '''
import json,sys,numpy as np
from pathlib import Path
from fissiondb import AnchorIndex,integer_backend
root=Path(sys.argv[1]);expected=sys.argv[2]
assert integer_backend()==expected
rng=np.random.default_rng(8512)
x=rng.normal(size=(640,768)).astype(np.float32)
q=rng.normal(size=(12,768)).astype(np.float32)
with AnchorIndex.create(root,768,cell_capacity=64,auto_pack_bytes=0) as index:
 for start in range(0,len(x),64):
  index.insert_batch(x[start:start+64],group_commit=True);index.flush_fission()
 index.delete(7);index.update(9,x[10]);index.flush_fission();index.pack_live()
 with index.context(nprobe=7,rerank=100) as context:
  rows=[context.search(query) for query in q]
 (root/'answers.json').write_text(json.dumps([dict(ids=r[0].tolist(),scores=r[1].tolist(),entries=r[2]['entries']) for r in rows]))
'''
    for name, disabled in [('avx512bw', '0'), ('avx2', '1')]:
        env = {**os.environ, 'FISSIONDB_DISABLE_AVX512': disabled,
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'FISSIONDB_INGEST_THREADS': '1'}
        subprocess.run([sys.executable, '-c', script, str(tmp_path/name), name], env=env, check=True, timeout=45)
    from fission_layout_fixture import read_layout
    a = read_layout(tmp_path/'avx512bw'); b = read_layout(tmp_path/'avx2')
    assert a[0][8:] == b[0][8:]
    assert a[1:] == b[1:]  # exact cells, representatives and residual bytes
    assert json.loads((tmp_path/'avx512bw/answers.json').read_text()) == json.loads((tmp_path/'avx2/answers.json').read_text())
