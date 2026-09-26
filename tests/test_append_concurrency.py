import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_fission_concurrency import instrumented_library

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('mode',['prepared','durable','checkpoint','crash','failure','prepared_failure'])
def test_staged_append_does_not_block_search(instrumented_library,tmp_path,mode):
    env={**os.environ,'FISSIONDB_ANCHOR_LIBRARY':str(instrumented_library),
         'PYTHONPATH':str(ROOT/'scripts'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    subprocess.run([sys.executable,str(ROOT/'tests/append_concurrency_case.py'),str(tmp_path/'db'),mode],
                   env=env,check=True,timeout=30)
    if mode=='crash':
        subprocess.run([sys.executable,str(ROOT/'tests/append_concurrency_case.py'),str(tmp_path/'db'),'recover'],
                       env=env,check=True,timeout=30)
