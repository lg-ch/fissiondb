"""HTTP API client. Mutations are never retried implicitly."""
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

class ServiceError(RuntimeError):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        super().__init__(f'HTTP {status}: {payload}')

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')

class Client:
    def __init__(self, url='http://127.0.0.1:8080', *, api_key=None, timeout=30):
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('Expected an HTTP(S) service URL without embedded credentials')
        self.url, self.api_key, self.timeout=url.rstrip('/'), api_key, timeout
        self._opener=build_opener(_NoRedirect())

    def _request(self, path, body=None):
        headers={'Accept':'application/json'}
        if self.api_key is not None:headers['X-API-Key']=self.api_key
        data=None
        if body is not None:
            data=json.dumps(body,allow_nan=False,default=_json_default).encode()
            headers['Content-Type']='application/json'
        request=Request(self.url+path,data=data,headers=headers,method='POST' if body is not None else 'GET')
        try:
            with self._opener.open(request,timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:payload=json.load(exc)
            except (ValueError,UnicodeError):payload={'error':'Non-JSON error response'}
            raise ServiceError(exc.code,payload) from exc

    def health(self):return self._request('/health')
    def stats(self):return self._request('/stats')
    def search(self, qvec, *, top_k=10, where=None, allowed_ids=None):
        return self._request('/search',dict(qvec=qvec,top_k=top_k,where=where,allowed_ids=allowed_ids))
    def insert(self, vector, metadata=None, *, idempotency_key=None):
        return self._request('/insert',dict(vec=vector,metadata=metadata,idempotency_key=idempotency_key))['doc_id']
    def insert_batch(self, vectors, metadata=None, *, idempotency_keys=None, group_commit=True):
        return self._request('/insert_batch',dict(vecs=vectors,metadata=metadata,idempotency_keys=idempotency_keys,group_commit=group_commit))['doc_ids']
    def update(self, doc_id, vector, metadata=None):
        return self._request('/update',dict(doc_id=doc_id,vec=vector,metadata=metadata))
    def delete(self, doc_id):return self._request('/delete',dict(doc_id=doc_id))
    def set_metadata(self, doc_id, metadata):return self._request('/metadata',dict(doc_id=doc_id,metadata=metadata))
    def add_metadata(self, doc_ids, metadata):return self._request('/metadata/add',dict(doc_ids=doc_ids,metadata=metadata))
    def compact(self):return self._request('/compact',{})
    def pack_live(self):return self._request('/pack',{})
