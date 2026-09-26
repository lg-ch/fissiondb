"""Regression tests for failed IO, cache identity, and retained writes."""
import asyncio
import ctypes as C
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('http1', ['0', '1'])
def test_s3_session_token_is_sent(native_helpers, http1):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            seen.append(self.headers.get('x-amz-security-token'))
            self.send_response(206)
            self.send_header('Content-Range', 'bytes 1024-1039/2048')
            self.send_header('Content-Length', '16')
            self.end_headers()
            self.wfile.write(b'x' * 16)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        result = subprocess.run([str(native_helpers/'s3_range_regression'),
            f'http://127.0.0.1:{server.server_port}/object'],
            env={**os.environ, 'AWS_ACCESS_KEY_ID': 'test-access',
                 'AWS_SECRET_ACCESS_KEY': 'test-secret', 'AWS_SESSION_TOKEN': 'test-session',
                 'AWS_REGION': 'us-east-1', 'FISSIONDB_S3_HTTP1': http1}, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert seen == ['test-session']
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_s3_invalid_protocol_setting_fails_closed(native_helpers):
    result = subprocess.run([str(native_helpers/'s3_range_regression'),
        'http://127.0.0.1:1/not-contacted'],
        env={**os.environ, 'FISSIONDB_S3_HTTP1': 'invalid'}, timeout=5)
    assert result.returncode == 3








@pytest.fixture
def tiny(tmp_path):
    n, d = 256, 128
    x = np.random.default_rng(51).normal(size=(n, d)).astype(np.float16)
    (tmp_path/'base.f16bin').write_bytes(struct.pack('<II', n, d) + x.tobytes())
    (tmp_path/'q.fbin').write_bytes(struct.pack('<II', 10, d) + x[:10].astype(np.float32).tobytes())
    return tmp_path


def command(*args):
    return subprocess.run([str(ROOT/'fissiondb-engine'), *map(str, args)], capture_output=True,
                          text=True, timeout=60, env={**os.environ, 'OMP_NUM_THREADS': '1'})


def build(t, name, seed=51, extra=()):
    out = t/name
    out.mkdir(exist_ok=True)
    r = command('abuild', t/'base.f16bin', out, 16, '--m', 2, '--eps', 999,
                '--tqbits', 1, '--seed', seed, *extra)
    assert r.returncode == 0, r.stderr
    return out, r


def test_assignment_cache_tracks_seed_and_reuses_identical_input(tiny):
    cached, _ = build(tiny, 'cached')
    _, same = build(tiny, 'cached')
    assert 'reprise du cache' in same.stderr
    _, changed = build(tiny, 'cached', seed=52)
    assert 'reprise du cache' not in changed.stderr
    fresh, _ = build(tiny, 'fresh', seed=52)
    assert (cached/'assign.bin').read_bytes() == (fresh/'assign.bin').read_bytes()


def test_short_block_read_fails(tiny):
    out, _ = build(tiny, 'index')
    (out/'blocks.bin').write_bytes(b'')
    r = command('abench', out, tiny/'base.f16bin', tiny/'q.fbin', 10, 8, 16)
    assert r.returncode != 0


def test_coarse_small_k_and_four_assignments(tiny):
    coarse, _ = build(tiny, 'coarse', extra=('--m', 4))
    out, _ = build(tiny, 'fine', seed=52,
                   extra=('--m', 4, '--coarse', coarse, '--neighbors', 256))
    ids = np.fromfile(out/'assign.bin', np.int32, offset=16, count=256*4)
    assert np.all((ids >= 0) & (ids < 16))










@pytest.fixture(scope='module')
def native_helpers(tmp_path_factory):
    out = tmp_path_factory.mktemp('native_helpers')
    for name in ('s3_range_regression',):
        args = ['gcc', '-O2', '-std=c11', '-fopenmp', '-I', str(ROOT/'src'),
                str(ROOT/'tests'/f'{name}.c'), '-o', str(out/name)]
        if name.startswith('hot'):
            args += ['-L', str(ROOT), f'-Wl,-rpath,{ROOT}', '-lfissiondb']
        else:
            import platform
            if platform.machine() == 'aarch64':
                args += ['-march=armv8.2-a+dotprod+fp16']
            args += [str(ROOT/'src/anchor_live.c'), '-lm', '-luring', '-lcurl', '-lroaring', '-lxxhash', '-lpthread']
        subprocess.run(args, check=True, capture_output=True)
    return out




def test_packed_simd_scores_include_tail_dimensions(native_helpers):
    subprocess.run([str(native_helpers/'s3_range_regression')], check=True, timeout=10)




@pytest.mark.parametrize('status,range_header,length,success', [
    (206, 'bytes 1024-1039/2048', 16, True),
    (403, 'bytes 1024-1039/2048', 16, False),
    (200, None, 16, False),
    (206, 'bytes 0-15/2048', 16, False),
    (206, 'bytes 1024-1039/2048', 8, False),
    (206, 'bytes 1024-1039/2048', 32, False),
])
def test_s3_validates_http_and_range(native_helpers, status, range_header, length, success):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            if range_header: self.send_header('Content-Range', range_header)
            self.send_header('Content-Length', str(length))
            self.end_headers()
            self.wfile.write(b'x'*length)
        def log_message(self, *args): pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith('AWS_')}
        r = subprocess.run([str(native_helpers/'s3_range_regression'),
            f'http://127.0.0.1:{server.server_port}/object'], env=env, timeout=10)
        assert (r.returncode == 0) == success
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_anchor_api_reuses_context_and_matches_cli(tiny):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from fissiondb.anchors import AnchorIndex
    out, _ = build(tiny, 'index')
    cli = tiny/'result.bin'
    r = command('abench', out, tiny/'base.f16bin', tiny/'q.fbin', 10, 8, 16, '--out', cli)
    assert r.returncode == 0, r.stderr
    expected = np.fromfile(cli, np.uint32).reshape(10, 11)
    vectors = np.fromfile(tiny/'q.fbin', np.float32, offset=8).reshape(10, 128)
    with AnchorIndex(out, tiny/'base.f16bin') as index:
        with index.context(nprobe=8, rerank=16, threads=1) as query:
            for i, vector in enumerate(vectors):
                ids, scores, stats = query.search(vector, top_k=11)
                assert np.array_equal(ids, expected[i, :len(ids)])
                assert np.isfinite(scores).all() and stats['bytes'] > 0
        with pytest.raises(ValueError):
            index.context(nprobe=8, rerank=16, memory_bytes=1)


def test_anchor_api_memory_limit_during_query(tiny):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from fissiondb.anchors import AnchorIndex
    out, _ = build(tiny, 'index')
    # A skewed index with one cell larger than the available IO buffer.
    entry = (out/'blocks.bin').read_bytes()[:20]
    (out/'blocks.bin').write_bytes(entry*100000)
    (out/'offs.bin').write_bytes(struct.pack('<17Q', 0, *([2000000]*16)))
    vector = np.fromfile(tiny/'q.fbin', np.float32, offset=8, count=128)
    with AnchorIndex(out, tiny/'base.f16bin') as index:
        with index.context(nprobe=16, rerank=16, memory_bytes=1000000) as query:
            with pytest.raises(MemoryError): query.search(vector)


def test_streamed_cells_preserve_candidates_across_buffer_reuse(tiny):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from fissiondb.anchors import AnchorIndex
    out, _ = build(tiny, 'index')
    data = (out/'blocks.bin').read_bytes()
    offsets = np.fromfile(out/'offs.bin', np.uint64)
    (out/'blocks.bin').write_bytes(b''.join(
        data[int(a):int(b)]*100 for a,b in zip(offsets[:-1], offsets[1:])))
    (offsets*100).tofile(out/'offs.bin')
    vectors = np.fromfile(tiny/'q.fbin', np.float32, offset=8).reshape(10,128)
    with AnchorIndex(out, tiny/'base.f16bin') as index:
        with index.context(nprobe=16, rerank=16) as whole, \
             index.context(nprobe=16, rerank=16, memory_bytes=1000000) as streamed:
            for vector in vectors:
                a, sa, _ = whole.search(vector)
                b, sb, _ = streamed.search(vector)
                assert np.array_equal(a, b)
                assert np.array_equal(sa, sb)
