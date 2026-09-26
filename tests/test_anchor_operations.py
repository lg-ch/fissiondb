import concurrent.futures
import http.client
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from test_anchor_live import frozen, open_live, ROOT
from fissiondb.anchors import _tags
from serve_anchors import AnchorServer


def test_idempotency_survives_metadata_delete_compact_reopen(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        v = frozen[2][0]
        doc = index.insert(v, {'x': 'one'}, idempotency_key='request-1')
        assert index.insert(v, {'x': 'one'}, idempotency_key='request-1') == doc
        with pytest.raises(ValueError):
            index.insert(v, {'x': 'two'}, idempotency_key='request-1')
        assert index.count == 5121
        for n in range(100):
            index.set_metadata(doc, {'x': str(n)})
        index.delete(doc)
        index.delete(doc)
        index.delete(0)
        assert index.deleted_count == 2
        assert index.compact()['saved_bytes'] > 0
        stats = index.stats()
        assert stats['active_count'] == 5119 and stats['idempotency_records'] == 1
    with open_live(frozen, tmp_path) as index:
        assert index.insert(v, {'x': 'one'}, idempotency_key='request-1') == doc
        assert index.count == 5121 and index.deleted_count == 2
        with index.context(nprobe=32, rerank=300) as query:
            assert not set(query.search(v)[0]) & {0, doc}
            assert not len(query.search(v, allowed_ids=[0, doc])[0])
            assert not len(query.search(v, where={'x': '99'})[0])
        with pytest.raises(OSError):
            index.set_metadata(doc, {'x': 'resurrection'})


def test_native_simultaneous_retry_is_one_insert(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        vector = frozen[2][0].copy()
        tags = _tags({'x': 'race'}, {})
        request = index._request(vector, tags, 'same-key')
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            # Bypass the Python lock to exercise the native race boundary.
            ids = list(pool.map(lambda _: index._insert_prepared(vector, tags, request), range(32)))
        assert set(ids) == {5120} and index.count == 5121


def test_batch_retry_and_empty_key_rejected(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        ids = index.insert_batch(frozen[2][:3], idempotency_keys=['a', 'b', 'c'])
        assert list(index.insert_batch(frozen[2][:3], idempotency_keys=['a', 'b', 'c'])) == list(ids)
        with pytest.raises(ValueError):
            index.insert(frozen[2][0], idempotency_key='')
        assert index.count == 5123


def test_retry_after_process_exit(frozen, tmp_path):
    script = '''
import os,sys,numpy as np
from fissiondb.anchors import AnchorIndex
i=AnchorIndex(sys.argv[1],sys.argv[2],live_dir=sys.argv[3])
v=np.fromfile(sys.argv[2],np.float16,offset=8,count=128).astype(np.float32)
assert i.insert(v, idempotency_key='lost-response')==5120
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(frozen[0]), str(frozen[1]), str(tmp_path)],
                   check=True, env={**os.environ, 'PYTHONPATH': str(ROOT / 'scripts')})
    with open_live(frozen, tmp_path) as index:
        assert index.insert(frozen[2][0], idempotency_key='lost-response') == 5120
        assert index.count == 5121


def test_automatic_compaction_stops_cleanly(frozen, tmp_path):
    with open_live(frozen, tmp_path, auto_compact_bytes=1000, auto_compact_interval=0.02) as index:
        for n in range(100):
            index.set_metadata(0, {'n': str(n)})
        deadline = time.monotonic() + 5
        while index._maintenance.runs == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert index._maintenance.runs > 0
        assert index._maintenance.last_error is None
        thread = index._maintenance._thread
    assert not thread.is_alive()


def test_pure_ingestion_does_not_trigger_repeated_compaction(frozen, tmp_path):
    with open_live(frozen, tmp_path, auto_compact_bytes=100, auto_compact_interval=0.01) as index:
        index.insert_batch(frozen[2][:20])
        time.sleep(0.1)
        assert index.maintenance_bytes == 0
        assert index._maintenance.runs == 0


def test_http_mutations_filters_auth_and_limits(frozen, tmp_path):
    with open_live(frozen, tmp_path) as index:
        server = AnchorServer(('127.0.0.1', 0), index, workers=2, nprobe=32,
                              rerank=300, api_key='test-only', max_body=100000)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        def request(path, data=None, auth=True):
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            headers = {'X-API-Key': 'test-only'} if auth else {}
            connection.request('POST' if data is not None else 'GET', path,
                               json.dumps(data) if data is not None else None, headers)
            response = connection.getresponse()
            status, payload = response.status, json.loads(response.read())
            connection.close()
            return status, payload
        try:
            assert request('/stats', auth=False)[0] == 401
            assert request('/health', auth=False)[0] == 200
            body = {'vec': frozen[2][0].tolist(), 'metadata': {'lang': 'fr', 'year': 2026},
                    'idempotency_key': 'http-1'}
            status, payload = request('/insert', body)
            assert status == 200 and payload['doc_id'] == 5120
            assert request('/insert', body)[1] == payload
            assert request('/insert', {**body, 'metadata': {'lang': 'en'}})[0] == 409
            status, payload = request('/search', {'qvec': body['vec'],
                'where': {'lang': 'fr', 'year': {'range': [2025, 2027]}}})
            assert status == 200 and payload['ids'] == [5120]
            assert request('/delete', {'doc_id': 5120})[0] == 200
            assert request('/search', {'qvec': body['vec'], 'where': {'lang': 'fr'}})[1]['ids'] == []
            assert request('/compact', {})[0] == 200
            assert request('/stats')[1]['deleted_count'] == 1
            assert request('/insert', {'vec': [0], 'padding': 'x' * 100001})[0] == 413
            assert request('/search', {'qvec': [float('nan')]})[0] == 400
            assert request('/insert', {**body, 'metadata': ['bad']})[0] == 400
            server._all_contexts[0].close()
            assert request('/health')[0] == 503
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


def test_s3_upload_keeps_body_and_refuses_existing_prefix(tmp_path):
    import hashlib
    from check_anchor_s3 import upload_fixture
    source = tmp_path / 'object'
    source.write_bytes(b'payload to upload')
    class Client:
        exists = False
        def list_objects_v2(self, **kwargs):
            return {'KeyCount': int(self.exists)}
        def put_object(self, **kwargs):
            assert kwargs['IfNoneMatch'] == '*'
            self.body = kwargs['Body'].read()
            assert self.body == source.read_bytes()
            assert kwargs['Metadata']['sha256'] == hashlib.sha256(self.body).hexdigest()
        def head_object(self, **kwargs):
            return {'ContentLength': len(self.body)}
    client = Client()
    upload_fixture(client, 'test-bucket', 'new', {'object': source})
    client.exists = True
    with pytest.raises(RuntimeError):
        upload_fixture(client, 'test-bucket', 'new', {'object': source})
