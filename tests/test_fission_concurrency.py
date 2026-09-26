"""Deterministic reader/writer/recovery checks with a paused native split.

The barrier only exists in the separately compiled FISSIONDB_TESTING library.
Every case runs in a subprocess so a broken lock cannot hang the test runner.
"""
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def instrumented_library(tmp_path_factory):
    target=tmp_path_factory.mktemp('concurrent_native')/'libfissiondb_test.so'
    arch='armv8.2-a+dotprod+fp16' if platform.machine()=='aarch64' else 'x86-64'
    cmd=['gcc','-O1','-g','-std=c11','-fPIC','-shared','-fopenmp',f'-march={arch}',
         '-DFISSIONDB_TESTING',str(ROOT/'src/anchor.c'),str(ROOT/'src/anchor_live.c'),
         '-lm','-luring','-lroaring','-lxxhash','-lpthread','-lcurl','-o',str(target)]
    if 'asan' in os.environ.get('LD_PRELOAD',''):cmd+=['-fsanitize=address,undefined']
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    return target


@pytest.mark.parametrize('mode',['publish','checkpoint','compact','crash','failure'])
def test_search_and_writes_while_daughters_are_unpublished(instrumented_library,tmp_path,mode):
    env={**os.environ,'FISSIONDB_ANCHOR_LIBRARY':str(instrumented_library),
         'PYTHONPATH':str(ROOT/'scripts'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    subprocess.run([sys.executable,str(ROOT/'tests/fission_concurrency_case.py'),
                    str(tmp_path/'db'),mode],env=env,check=True,timeout=30)
    if mode=='crash':
        subprocess.run([sys.executable,str(ROOT/'tests/fission_concurrency_case.py'),
                        str(tmp_path/'db'),'recover'],env=env,check=True,timeout=30)
