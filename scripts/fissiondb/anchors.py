"""In-process anchor search. One reusable context per concurrent caller.

    with AnchorIndex(index_dir, base_path) as index:
        with index.context(nprobe=1024, rerank=400, memory_bytes=1_800_000_000) as query:
            ids, scores, stats = query.search(vector, top_k=10)

Set live_dir to enable durable insertion and native metadata filtering.
The allocation limit excludes dynamic live metadata, filter bitmaps and the OS
page cache; use a cgroup for a process limit. See README.md for guarantees.
"""
from __future__ import annotations
import ctypes as C
import threading
import weakref
from pathlib import Path
import operator
import re
import hashlib
import struct
import os
from .metatypes import encode_meta, compile_where

import numpy as np
from ._anchor_native import _lib

_lib.anchor_integer_backend.argtypes = []
_lib.anchor_integer_backend.restype = C.c_char_p


def integer_backend():
    """Selected int8 routing/TQ1 backend (short vectors may use narrower SIMD)."""
    return _lib.anchor_integer_backend().decode('ascii')


class _Stats(C.Structure):
    _fields_ = [(name, C.c_double) for name in
                ('anchor_ms', 'io_ms', 'score_ms', 'rerank_ms', 'total_ms')] + [
                ('entries', C.c_uint64), ('bytes', C.c_uint64)]


_lib.anchor_index_open.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
_lib.anchor_index_open.restype = C.c_void_p
_lib.anchor_index_close.argtypes = [C.c_void_p]
_lib.anchor_index_close.restype = None
_lib.anchor_index_dim.argtypes = [C.c_void_p]
_lib.anchor_index_dim.restype = C.c_int
_lib.anchor_index_bytes.argtypes = [C.c_void_p]
_lib.anchor_index_bytes.restype = C.c_uint64
_lib.anchor_index_enable_fission.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]
_lib.anchor_index_enable_fission.restype = C.c_int
_lib.anchor_index_fission_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_double)]
_lib.anchor_index_fission_stats.restype = C.c_int
_lib.anchor_index_fission_progress.argtypes = _lib.anchor_index_fission_stats.argtypes
_lib.anchor_index_fission_progress.restype = C.c_int
_lib.anchor_index_fission_flush.argtypes = [C.c_void_p]
_lib.anchor_index_fission_flush.restype = C.c_int
_lib.anchor_index_fission_set_capacity.argtypes = [C.c_void_p,C.c_uint32]
_lib.anchor_index_fission_set_capacity.restype = C.c_int
_lib.anchor_index_fission_capacity.argtypes = [C.c_void_p]
_lib.anchor_index_fission_capacity.restype = C.c_uint32
_lib.anchor_index_fission_center_info.argtypes = [C.c_void_p,C.POINTER(C.c_uint64)]
_lib.anchor_index_fission_center_info.restype = C.c_int
_lib.anchor_query_create.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int,
                                   C.c_uint64, C.c_char_p, C.c_int]
_lib.anchor_query_create.restype = C.c_void_p
_lib.anchor_query_close.argtypes = [C.c_void_p]
_lib.anchor_query_close.restype = None
_lib.anchor_query_search.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int,
                                    C.POINTER(C.c_uint32), C.POINTER(C.c_float),
                                    C.POINTER(_Stats)]
_lib.anchor_query_search.restype = C.c_int


for name, args, result in (
    ('anchor_query_residual_io', [C.c_void_p,C.c_int,C.c_int,C.c_int], C.c_int),
    ('anchor_query_adapt', [C.c_void_p,C.c_int,C.c_int,C.c_float,C.c_uint64,C.c_double], C.c_int),
    ('anchor_query_adapt_stats', [C.c_void_p,C.POINTER(C.c_double)], C.c_int),
    ('anchor_query_trace', [C.c_void_p,C.POINTER(C.c_uint32),C.c_int], C.c_int),
    ('anchor_query_trace_stats', [C.c_void_p,C.POINTER(C.c_uint64)], C.c_int),
    ('anchor_query_unique', [C.c_void_p,C.c_int], C.c_int),
    ('anchor_query_unique_stats', [C.c_void_p,C.POINTER(C.c_uint64)], C.c_int),
    ('anchor_query_live_stats', [C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_double)], C.c_int),
    ('anchor_index_snapshot_live', [C.c_void_p, C.c_char_p], C.c_int),
    ('anchor_index_pack_live', [C.c_void_p], C.c_int),
    ('anchor_index_unpacked_bytes', [C.c_void_p], C.c_uint64),
    ('anchor_index_build_residual', [C.c_void_p, C.c_char_p], C.c_int),
    ('anchor_index_enable_residual', [C.c_void_p, C.c_char_p], C.c_int),
    ('anchor_index_enable_live', [C.c_void_p, C.c_char_p], C.c_int),
    ('anchor_index_compact', [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)], C.c_int),
    ('anchor_index_insert_batch', [C.c_void_p, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_char_p), C.POINTER(C.c_int), C.POINTER(C.c_char_p), C.POINTER(C.c_char_p), C.POINTER(C.c_uint32), C.POINTER(C.c_int)], C.c_int),
    ('anchor_index_insert_once', [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_char_p), C.c_int, C.c_char_p, C.c_char_p, C.POINTER(C.c_uint32)], C.c_int),
    ('anchor_index_update', [C.c_void_p, C.c_uint32, C.POINTER(C.c_float), C.POINTER(C.c_char_p), C.c_int], C.c_int),
    ('anchor_index_delete', [C.c_void_p, C.c_uint32], C.c_int),
    ('anchor_index_live_stats', [C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
    ('anchor_index_maintenance_bytes', [C.c_void_p], C.c_uint64),
    ('anchor_index_deleted_count', [C.c_void_p], C.c_uint64),
    ('anchor_index_count', [C.c_void_p], C.c_uint64),
    ('anchor_index_insert', [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_char_p), C.c_int, C.POINTER(C.c_uint32)], C.c_int),
    ('anchor_index_add_tag', [C.c_void_p, C.POINTER(C.c_uint32), C.c_int, C.POINTER(C.c_char_p), C.c_int], C.c_int),
    ('anchor_index_set_tags', [C.c_void_p, C.c_uint32, C.POINTER(C.c_char_p), C.c_int], C.c_int),
    ('anchor_index_tag_keys', [C.c_void_p, C.c_void_p, C.c_int], C.c_int),
    ('anchor_query_search_filtered', [C.c_void_p, C.POINTER(C.c_float), C.c_int,
        C.POINTER(C.c_uint32), C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_int), C.c_int,
        C.POINTER(C.c_uint32), C.POINTER(C.c_float), C.POINTER(_Stats)], C.c_int),
):
    fn = getattr(_lib, name)
    fn.argtypes, fn.restype = args, result


def _fields(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Metadata and filters must be objects')
    for field in metadata:
        if not isinstance(field, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', field):
            raise ValueError('Metadata fields must match [A-Za-z_][A-Za-z0-9_-]*')


def _tags(metadata, specs):
    _fields(metadata)
    keys = list(dict.fromkeys(k.encode() for k in encode_meta(metadata, specs)))
    if len(keys) > 256 or any(not 0 < len(k) < 256 or any(b < 32 or b == 127 for b in k) for k in keys):
        raise ValueError('Metadata exceeds 256 keys, 255 UTF-8 bytes per key, or contains controls')
    return (C.c_char_p * len(keys))(*keys)


def _vector(vector, dim, *, nonzero=False):
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape != (dim,) or not np.isfinite(vector).all() or (nonzero and not np.any(vector)):
        raise ValueError('Expected a finite nonzero vector with the index dimension')
    return vector


class AnchorBatchError(OSError):
    """A durable prefix succeeded; the failed record may also have committed.

    Reopen and reconcile before retrying. Batch insertion is not atomic.
    """
    def __init__(self, committed_ids):
        self.committed_ids = list(committed_ids)
        super().__init__('Live batch failed; reopen and reconcile before retrying')


class IdempotencyConflict(ValueError):
    """A key is already committed with a different vector or metadata payload."""


class AnchorIndex:
    def __init__(self, directory, base_path=None, *, int8=True, live=False,
                 live_dir=None, float_specs=None, auto_compact_bytes=0, auto_compact_interval=30,
                 residual_dir=None, auto_pack_bytes=None, auto_pack_interval=30,
                 fission_cell_capacity=None, fission_max_cells=300_000):
        self.directory = Path(directory).resolve()
        self.int8_mode = bool(int8)
        self.base_path = Path(base_path).resolve() if base_path is not None else None
        self.int8 = bool(int8)
        self._lock = threading.Lock()
        self._contexts = weakref.WeakSet()
        self._maintenance = None
        self._packer = None
        self._packing_lock = threading.Lock()
        self.live_dir = None
        if fission_cell_capacity is not None:
            fission_cell_capacity=operator.index(fission_cell_capacity)
            fission_max_cells=operator.index(fission_max_cells)
            if (fission_cell_capacity != 0 and not 64 <= fission_cell_capacity <= 65536) or not 2 <= fission_max_cells <= 1_000_000:
                raise ValueError('Invalid fission cell or memory capacity')
            if not (live or live_dir is not None) or residual_dir is None:
                raise ValueError('Automatic fission requires live and residual storage')
        self._handle = _lib.anchor_index_open(str(directory).encode(),
            str(base_path).encode() if base_path is not None else None, bool(int8))
        if not self._handle:
            raise OSError('Cannot load anchor index')
        self._finalizer = weakref.finalize(self, _lib.anchor_index_close, self._handle)
        self.residual_dir = Path(residual_dir) if residual_dir is not None else None
        if self.residual_dir is not None:
            if _lib.anchor_index_enable_residual(self._handle, str(self.residual_dir).encode()):
                self._finalizer()
                self._handle = None
                raise OSError('Cannot load residual index: unsupported format, mismatched anchors, truncated blocks or incompatible mode')
        self.live = bool(live or live_dir is not None)
        self.float_specs = dict(float_specs or {})
        if self.live:
            destination = Path(live_dir) if live_dir is not None else Path(directory) / 'live'
            self.live_dir = destination
            if _lib.anchor_index_enable_live(self._handle, str(destination).encode()):
                self._finalizer()
                self._handle = None
                raise OSError('Cannot open live overlay: locked, corrupt, mismatched index or IO failure')
        self.dim = _lib.anchor_index_dim(self._handle)
        if fission_cell_capacity is not None:
            if _lib.anchor_index_enable_fission(self._handle,fission_cell_capacity,fission_max_cells):
                self._finalizer();self._handle=None
                raise OSError('Cannot enable fission: mismatched saved configuration or IO failure')
        values=(C.c_uint64*8)();timings=(C.c_double*3)()
        self.fission = _lib.anchor_index_fission_stats(self._handle,values,timings)==0
        # Constant-cost starting budget; no GT, fitting, or scan on ingestion.
        metadata = (self.directory / "meta.txt").read_text().split()
        self.indexed_dim = int(metadata[1])
        self.default_nprobe = min(int(metadata[0]), max(1, self.indexed_dim // 2))
        if self.fission:self.default_nprobe=1536
        self.memory_bytes = _lib.anchor_index_bytes(self._handle)
        if auto_pack_bytes is None:auto_pack_bytes=256*1024*1024 if self.fission else 0
        if auto_compact_bytes:
            from .anchor_maintenance import AutoCompactor
            try:
                self._maintenance = AutoCompactor(self, auto_compact_bytes, auto_compact_interval)
            except Exception:
                self.close()
                raise

        if auto_pack_bytes:
            from .anchor_maintenance import AutoPacker
            try:
                self._packer = AutoPacker(self, auto_pack_bytes, auto_pack_interval)
            except Exception:
                self.close()
                raise

    @classmethod
    def create(cls, directory, dim, *, cell_capacity=None, max_cells=300_000,
               auto_pack_bytes=256*1024*1024, **kwargs):
        """Create an empty local collection with automatic live-cell fission.

        The small empty frozen header is a format bootstrap, not a training set.
        Live representatives are chosen from inserted vectors and grow by splits.
        Omitted/None or zero capacity selects max(64, 2*dim) in the native engine.
        An explicit 64..65536 overrides it. The resolved threshold is persisted.
        """
        dim=operator.index(dim)
        if not 1<=dim<=1024:raise ValueError('Dimensions must be in 1..1024')
        cell_capacity=0 if cell_capacity is None else operator.index(cell_capacity)
        if (cell_capacity != 0 and not 64<=cell_capacity<=65536) or not 2<=operator.index(max_cells)<=1_000_000:
            raise ValueError('Invalid fission capacity')
        directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
        padded=max(8,1<<(dim-1).bit_length())
        centers=np.zeros((2,padded),'<f4');centers[0,0]=1;centers[1,0]=-1
        files={'meta.txt':f'2 {padded} 2 1 999 0 52 {padded} {dim}\n'.encode(),
               'anchors.bin':centers.tobytes(),'offs.bin':np.zeros(3,'<u8').tobytes(),
               'scale.bin':np.ones(padded,'<f4').tobytes(),'blocks.bin':b'',
               'base.f16bin':struct.pack('<II',0,dim)}
        for name,data in files.items():
            with (directory/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        with cls(directory,directory/'base.f16bin') as frozen:
            frozen.build_residual(directory/'residual')
        return cls(directory,directory/'base.f16bin',residual_dir=directory/'residual',
                   live_dir=directory/'live',fission_cell_capacity=cell_capacity,
                   fission_max_cells=max_cells,auto_pack_bytes=auto_pack_bytes,**kwargs)

    @property
    def fission_stats(self):
        with self._lock:
            self._require_live()
            if not self.fission:return None
            values=(C.c_uint64*8)();timings=(C.c_double*3)()
            if _lib.anchor_index_fission_stats(self._handle,values,timings):
                raise OSError('Fission state unavailable')
            result=dict(zip(('cells','splits','rewritten','records','largest_cell',
                             'owned_bytes','arena_bytes','max_cells'),map(int,values)))
            result.update(zip(('split_total_ms','split_max_ms','last_split_ms'),map(float,timings)))
            if _lib.anchor_index_fission_progress(self._handle,values,timings):
                raise OSError('Fission progress unavailable')
            result.update(zip(('pending_cells','preparing','queries_during_prepare','delta_records',
                              'backpressure_waits','scratch_bytes','peak_scratch_bytes','publishes'),map(int,values)))
            result.update(zip(('publish_total_ms','publish_max_ms','last_publish_ms'),map(float,timings)))
            if _lib.anchor_index_fission_center_info(self._handle,values):
                raise OSError('Fission center memory unavailable')
            result.update(zip(('center_dim','center_bytes','center_allocated_bytes'),map(int,values[:3])))
            return result

    def flush_fission(self):
        """Wait for pending splits while queries continue on published cells."""
        with self._lock:
            self._require_live()
            if not self.fission:raise ValueError('Automatic fission is not enabled')
            if _lib.anchor_index_fission_flush(self._handle):
                raise OSError('Fission failed; reopen the index')

    @property
    def unpacked_bytes(self):
        with self._lock:
            self._require_live()
            result = _lib.anchor_index_unpacked_bytes(self._handle)
            if result == 2**64-1:
                raise OSError('Cannot read live pack state')
            return result

    def snapshot_live(self, output_directory):
        """Copy a committed journal prefix; frozen data must be backed up too."""
        with self._packing_lock:
            with self._lock:
                self._require_live()
                handle = self._handle
            destination = Path(output_directory)
            destination.mkdir(parents=True, exist_ok=True)
            if _lib.anchor_index_snapshot_live(handle, str(destination).encode()):
                raise OSError('Live snapshot failed or destination already contains a snapshot')

    @property
    def fission_capacity(self):
        with self._lock:
            self._require_live()
            capacity=_lib.anchor_index_fission_capacity(self._handle)
            if not capacity:raise ValueError('Automatic fission is unavailable')
            return capacity

    def set_fission_capacity(self,capacity):
        """Persist a threshold and queue existing oversized cells for splitting.

        Does not wait for those splits. Call flush_fission() before measuring a
        drained topology. Reductions requiring more than capacity+512 scratch
        entries are rejected; reduce in stages and flush between them.
        """
        capacity=operator.index(capacity)
        if not 64<=capacity<=65536:raise ValueError('Fission capacity must be in 64..65536')
        with self._lock:
            self._require_live()
            if not self.fission:raise ValueError('Automatic fission is not enabled')
            rc=_lib.anchor_index_fission_set_capacity(self._handle,capacity)
            if rc==-2:raise ValueError('Cells exceed capacity+512; reduce in stages and flush fission')
            if rc:raise OSError('Cannot persist fission capacity; reopen after an IO failure')

    def pack_live(self):
        """Persist a derived live snapshot or an adaptive-cell checkpoint.

        Automatic fission checkpoints hold the live write lock; readers and
        writers wait. Fixed-cell packing allows concurrent mutations, but
        compaction may invalidate its attempt. The journal remains authoritative.
        """
        with self._packing_lock:
            with self._lock:
                self._require_live()
                if self.residual_dir is None:
                    raise ValueError('Compressed live snapshots require residual mode')
                handle = self._handle
            if _lib.anchor_index_pack_live(handle):
                raise OSError('Live packing failed or snapshot invalidated by compaction')

    def build_residual(self, output_directory):
        """Convert immutable TQ to a resumable residual sidecar, one thread.

        Published output is never overwritten. Keep the source immutable.
        """
        with self._lock:
            if not self._handle:
                raise RuntimeError('Index is closed')
            if self.live or self._contexts or self.residual_dir is not None:
                raise ValueError('Build requires a frozen TQ handle without contexts')
            destination = Path(output_directory)
            destination.mkdir(parents=True, exist_ok=True)
            if _lib.anchor_index_build_residual(self._handle, str(destination).encode()):
                raise OSError('Residual build failed: incompatible source, locked/published output or IO failure; unfinished builds can resume with the same source')
            return destination

    def _require_live(self):
        if not self._handle:
            raise RuntimeError('Index is closed')
        if not self.live:
            raise RuntimeError('Open the index with live=True or live_dir=...')

    @property
    def count(self):
        with self._lock:
            if not self._handle:
                raise RuntimeError('Index is closed')
            count = _lib.anchor_index_count(self._handle)
            if count == 2**64-1:
                raise OSError('Live state unavailable; reopen the index')
            return count

    @staticmethod
    def _request(vector, tags, key):
        if key is None:
            return None
        if not isinstance(key, str) or not key or len(key.encode()) > 256:
            raise ValueError('Idempotency key must contain 1..256 UTF-8 bytes')
        digest = hashlib.sha256(vector.tobytes())
        for tag in sorted(tags):
            digest.update(struct.pack('<I', len(tag)))
            digest.update(tag)
        return hashlib.sha256(key.encode()).digest(), digest.digest()

    def _insert_prepared(self, vector, tags, request):
        output = C.c_uint32()
        pointer = vector.ctypes.data_as(C.POINTER(C.c_float))
        if request is None:
            rc = _lib.anchor_index_insert(self._handle, pointer, tags, len(tags), C.byref(output))
        else:
            rc = _lib.anchor_index_insert_once(self._handle, pointer, tags, len(tags),
                                               *request, C.byref(output))
        if rc == -3:
            raise IdempotencyConflict('Idempotency key already used with a different payload')
        if rc:
            raise OSError('Live insert failed; reopen before retrying with the same idempotency key')
        return output.value

    def insert(self, vector, metadata=None, *, idempotency_key=None):
        vector = _vector(vector, self.dim, nonzero=True)
        tags = _tags(metadata or {}, self.float_specs)
        request = self._request(vector, tags, idempotency_key)
        with self._lock:
            self._require_live()
            return self._insert_prepared(vector, tags, request)

    def insert_batch(self, vectors, metadata=None, *, idempotency_keys=None, group_commit=False):
        if type(group_commit) is not bool:
            raise ValueError('group_commit must be a boolean')
        vectors = [_vector(v, self.dim, nonzero=True) for v in vectors]
        metadata = [{} for _ in vectors] if metadata is None else list(metadata)
        keys = [None for _ in vectors] if idempotency_keys is None else list(idempotency_keys)
        if len(metadata) != len(vectors) or len(keys) != len(vectors):
            raise ValueError('Metadata and idempotency keys must match the vector count')
        tags = [_tags(m or {}, self.float_specs) for m in metadata]
        requests = [self._request(v, t, k) for v, t, k in zip(vectors, tags, keys)]
        committed = []
        with self._lock:
            self._require_live()
            if group_commit:
                for begin in range(0, len(vectors), 256):
                    vv = np.ascontiguousarray(vectors[begin:begin+256], dtype=np.float32)
                    tt = tags[begin:begin+256]; rr = requests[begin:begin+256]
                    flat = [key for tag in tt for key in tag]
                    ckeys = (C.c_char_p * len(flat))(*flat)
                    counts = (C.c_int * len(tt))(*(len(tag) for tag in tt))
                    tokens = (C.c_char_p * len(rr))(*(r[0] if r else None for r in rr))
                    digests = (C.c_char_p * len(rr))(*(r[1] if r else None for r in rr))
                    output = np.empty(len(vv), dtype=np.uint32); done = C.c_int()
                    rc = _lib.anchor_index_insert_batch(self._handle, len(vv), vv.ctypes.data_as(C.POINTER(C.c_float)),
                        ckeys, counts, tokens, digests, output.ctypes.data_as(C.POINTER(C.c_uint32)), C.byref(done))
                    committed.extend(map(int, output[:done.value]))
                    if rc:
                        cause = IdempotencyConflict('Idempotency key already used') if rc == -3 else OSError('Group commit failed; reopen and reconcile')
                        raise AnchorBatchError(committed) from cause
                return np.asarray(committed, dtype=np.uint32)
            for vector, tag, request in zip(vectors, tags, requests):
                try:
                    committed.append(self._insert_prepared(vector, tag, request))
                except (OSError, ValueError) as exc:
                    raise AnchorBatchError(committed) from exc
        return np.asarray(committed, dtype=np.uint32)

    def update(self, doc_id, vector, metadata=None):
        """Durably replace vector and all metadata, preserving an existing ID.

        Deleted IDs cannot be resurrected. Last committed update wins.
        """
        doc_id = operator.index(doc_id)
        if not 0 <= doc_id < 2**32 - 1:
            raise ValueError('Invalid document ID')
        vector = _vector(vector, self.dim, nonzero=True)
        tags = _tags(metadata or {}, self.float_specs)
        with self._lock:
            self._require_live()
            if _lib.anchor_index_update(self._handle, doc_id, vector.ctypes.data_as(C.POINTER(C.c_float)), tags, len(tags)):
                raise OSError('Update failed: unknown/deleted ID or IO failure; reopen after IO errors')

    def delete(self, doc_id):
        doc_id = operator.index(doc_id)
        if not 0 <= doc_id < 2**32 - 1:
            raise ValueError('Invalid document ID')
        with self._lock:
            self._require_live()
            count = _lib.anchor_index_count(self._handle)
            if count == 2**64-1:
                raise OSError('Live state unavailable; reopen the index')
            if doc_id >= count:
                raise ValueError('Unknown document ID')
            if _lib.anchor_index_delete(self._handle, doc_id):
                raise OSError('Delete failed; reopen before retrying')

    def stats(self):
        with self._lock:
            self._require_live()
            values = (C.c_uint64 * 5)()
            if _lib.anchor_index_live_stats(self._handle, values):
                raise OSError('Live state unavailable; reopen the index')
            result = dict(zip(('allocated_count', 'deleted_count', 'maintenance_bytes',
                               'journal_bytes', 'idempotency_records'), values))
            result['active_count'] = result['allocated_count'] - result['deleted_count']
            if self.fission:
                numbers=(C.c_uint64*8)();timings=(C.c_double*3)()
                if _lib.anchor_index_fission_stats(self._handle,numbers,timings):
                    raise OSError('Fission state unavailable')
                result['fission'] = dict(zip(('cells','splits','rewritten','records','largest_cell',
                    'owned_bytes','arena_bytes','max_cells'),map(int,numbers)))
                result['fission'].update(zip(('split_total_ms','split_max_ms','last_split_ms'),map(float,timings)))
                if _lib.anchor_index_fission_progress(self._handle,numbers,timings):
                    raise OSError('Fission progress unavailable')
                result['fission'].update(zip(('pending_cells','preparing','queries_during_prepare','delta_records',
                    'backpressure_waits','scratch_bytes','peak_scratch_bytes','publishes'),map(int,numbers)))
                result['fission'].update(zip(('publish_total_ms','publish_max_ms','last_publish_ms'),map(float,timings)))
                if _lib.anchor_index_fission_center_info(self._handle,numbers):
                    raise OSError('Fission center memory unavailable')
                result['fission'].update(zip(('center_dim','center_bytes','center_allocated_bytes'),map(int,numbers[:3])))
            return result

    @property
    def maintenance_bytes(self):
        with self._lock:
            self._require_live()
            size = _lib.anchor_index_maintenance_bytes(self._handle)
            if size == 2**64 - 1:
                raise OSError('Live state unavailable; reopen the index')
            return size

    @property
    def deleted_count(self):
        with self._lock:
            self._require_live()
            count = _lib.anchor_index_deleted_count(self._handle)
            if count == 2**64 - 1:
                raise OSError('Live state unavailable; reopen the index')
            return count

    def add_metadata(self, doc_ids, metadata):
        """Union one field/value posting with 1..8192 existing, nondeleted IDs.

        Other values remain attached. Repeating the operation is idempotent;
        use set_metadata to replace all metadata of an individual document.
        """
        values = list(doc_ids)
        if not 1 <= len(values) <= 8192:
            raise ValueError('Expected 1..8192 document IDs')
        values = [operator.index(value) for value in values]
        if any(not 0 <= value < 2**32-1 for value in values):
            raise ValueError('Document ID outside uint32 domain')
        ids = np.asarray(values, dtype=np.uint32)
        keys = _tags(metadata, self.float_specs)
        if len(metadata) != 1:
            raise ValueError('Expected exactly one metadata field/value')
        with self._lock:
            self._require_live()
            if _lib.anchor_index_add_tag(self._handle, ids.ctypes.data_as(C.POINTER(C.c_uint32)), len(ids), keys, len(keys)):
                raise OSError('Metadata posting failed; reopen after an ambiguous IO failure')

    def set_metadata(self, doc_id, metadata):
        doc_id = operator.index(doc_id)
        if not 0 <= doc_id < 2**32 - 1:
            raise ValueError('Document ID outside uint32 domain')
        keys = _tags(metadata, self.float_specs)
        with self._lock:
            self._require_live()
            count = _lib.anchor_index_count(self._handle)
            if count == 2**64-1:
                raise OSError('Live state unavailable; reopen the index')
            if doc_id >= count:
                raise ValueError('Unknown document ID')
            if _lib.anchor_index_set_tags(self._handle, doc_id, keys, len(keys)):
                raise OSError('Metadata update failed; reopen and reconcile before retrying')

    def compact(self):
        """Blocking maintenance; return journal sizes before and after checkpoint."""
        with self._lock:
            self._require_live()
            before, after = C.c_uint64(), C.c_uint64()
            if _lib.anchor_index_compact(self._handle, C.byref(before), C.byref(after)):
                raise OSError('Live compaction failed; reopen after an ambiguous IO failure')
            return {'before_bytes': before.value, 'after_bytes': after.value,
                    'saved_bytes': before.value - after.value}

    def keys(self):
        with self._lock:
            self._require_live()
            capacity = 0
            while True:
                output = C.create_string_buffer(capacity) if capacity else None
                needed = _lib.anchor_index_tag_keys(self._handle, output, capacity)
                if needed < 0:
                    raise OSError('Cannot read metadata dictionary')
                if needed < capacity:
                    return output.value.decode().splitlines()
                capacity = needed + 1

    def calibrated_context(self, profile, *, memory_bytes=0, latency_budget_ms=0, code_bytes=0):
        from .adaptive import calibrated_context
        return calibrated_context(self,profile,memory_bytes=memory_bytes,
                                  latency_budget_ms=latency_budget_ms,code_bytes=code_bytes)

    def context(self, *, nprobe=None, rerank=400, threads=1, memory_bytes=0,
                s3_url=None, hedge_ms=0):
        if memory_bytes < 0:
            raise ValueError('memory_bytes must be nonnegative')
        with self._lock:
            if not self._handle:
                raise RuntimeError('Index is closed')
            if nprobe is None:
                nprobe = self.default_nprobe
            handle = _lib.anchor_query_create(self._handle, nprobe, rerank, threads,
                memory_bytes, s3_url.encode() if s3_url else None, hedge_ms)
            if not handle:
                raise ValueError('Cannot create context: invalid parameters, memory limit or IO setup')
            query = AnchorQuery(self, handle, rerank)
            self._contexts.add(query)
            return query

    def close(self):
        if self._maintenance is not None:
            self._maintenance.close()
        if self._packer is not None:
            self._packer.close()
        with self._packing_lock, self._lock:
            for query in list(self._contexts):
                query.close()
            if self._handle:
                self._finalizer()
                self._handle = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class AnchorQuery:
    def __init__(self, index, handle, rerank):
        self.index, self._handle, self.rerank = index, handle, rerank
        self._finalizer = weakref.finalize(self, _lib.anchor_query_close, handle)
        self._lock = threading.Lock()
        self.calibration_top_k = None
        self.latency_budget_ms = 0.0

    def residual_io(self, *, batch_cells=64, overlap=True, direct=True):
        """Configure local residual reads; each context owns its IO buffers."""
        batch_cells = operator.index(batch_cells)
        if not 1 <= batch_cells <= 256:
            raise ValueError('batch_cells must be between 1 and 256')
        with self._lock:
            if not self._handle:
                raise RuntimeError('Query context is closed')
            if _lib.anchor_query_residual_io(self._handle, batch_cells, bool(overlap), bool(direct)):
                raise ValueError('Residual IO unavailable for this context or filesystem')
        return self

    def adapt(self, *, minimum, filtered_minimum=None, gap=0.0,
              code_bytes=0, latency_budget_ms=0.0):
        """Configure calibrated routing; budgets are soft, recall is not guaranteed."""
        minimum=operator.index(minimum)
        filtered_minimum=minimum if filtered_minimum is None else operator.index(filtered_minimum)
        code_bytes=operator.index(code_bytes)
        if not 1 <= minimum <= filtered_minimum < 2**31:raise ValueError('Invalid probe budget')
        if not 0 <= code_bytes < 2**64:raise ValueError('Invalid byte budget')
        with self._lock:
            if not self._handle:raise RuntimeError('Query context is closed')
            if _lib.anchor_query_adapt(self._handle,minimum,filtered_minimum,gap,code_bytes,latency_budget_ms):
                raise ValueError('Invalid adaptive policy or context is not single-threaded')
            self.latency_budget_ms = latency_budget_ms
        return self

    def search(self, vector, *, top_k=10, where=None, allowed_ids=None, diagnostic_ids=None, diagnostic_unique=False):
        if self.calibration_top_k is not None and top_k > self.calibration_top_k:
            raise ValueError('top_k exceeds the calibrated quality target')
        vector = _vector(vector, self.index.dim)
        keys, lengths = [], []
        if where:
            if not self.index.live:
                raise RuntimeError('Metadata filters require live=True or live_dir=...')
            _fields(where)
            regex = any(isinstance(c, tuple) and c and c[0] == 're' for c in where.values())
            keys, lengths = compile_where(where, self.index.keys() if regex else None,
                                          self.index.float_specs)
            if len(keys) > 65536:
                raise ValueError('Filter exceeds 65536 keys')
        encoded = [k.encode() for k in keys]
        if any(b'\x00' in k for k in encoded):
            raise ValueError('Filter contains a NUL byte')
        ckeys = (C.c_char_p * len(encoded))(*encoded)
        groups = (C.c_int * len(lengths))(*lengths)
        allowed = None
        if allowed_ids is not None:
            values = [operator.index(i) for i in allowed_ids]
            if len(values) > 2**31 - 1 or any(i < 0 or i >= 2**32 - 1 for i in values):
                raise ValueError('Invalid allowed document IDs')
            allowed = np.asarray(values, dtype=np.uint32)
        if not 1 <= top_k <= self.rerank:
            raise ValueError('top_k must be between 1 and rerank')
        ids = np.empty(top_k, dtype=np.uint32)
        scores = np.empty(top_k, dtype=np.float32)
        stats = _Stats()
        watched = [] if diagnostic_ids is None else [operator.index(i) for i in diagnostic_ids]
        if len(watched)>64 or len(set(watched))!=len(watched) or any(i<0 or i>=2**32-1 for i in watched):
            raise ValueError('Diagnostic IDs must be unique uint32 IDs, at most 64')
        watched = np.asarray(sorted(watched),dtype=np.uint32)
        if diagnostic_unique and not len(watched):raise ValueError('Unique diagnostic requires watched IDs')
        with self._lock:
            if not self._handle:
                raise RuntimeError('Query context is closed')
            if _lib.anchor_query_trace(self._handle,watched.ctypes.data_as(C.POINTER(C.c_uint32)),len(watched)):
                raise ValueError('Diagnostics require frozen local single-thread index and existing IDs')
            if _lib.anchor_query_unique(self._handle,bool(diagnostic_unique)):
                raise MemoryError('Cannot allocate unique-ID diagnostic')
            n = _lib.anchor_query_search_filtered(self._handle,
                vector.ctypes.data_as(C.POINTER(C.c_float)), top_k,
                allowed.ctypes.data_as(C.POINTER(C.c_uint32)) if allowed is not None else None,
                len(allowed) if allowed is not None else -1, ckeys, groups, len(lengths),
                ids.ctypes.data_as(C.POINTER(C.c_uint32)),
                scores.ctypes.data_as(C.POINTER(C.c_float)), C.byref(stats))
            if n < 0:
                self._finalizer()
                self._handle = None
                if n == -2:
                    raise MemoryError('Anchor query exceeded its allocation limit')
                raise OSError('Anchor query failed; context has been closed')
            policy=(C.c_double*3)()
            _lib.anchor_query_adapt_stats(self._handle,policy)
            result_stats={name: getattr(stats, name) for name, _ in stats._fields_}
            result_stats.update(probes=int(policy[0]),budget_limited=bool(policy[1]),anchor_gap=float(policy[2]),
                                deadline_exceeded=bool(self.latency_budget_ms and stats.total_ms>self.latency_budget_ms))
            if self.index.fission:
                counters=(C.c_uint64*8)();timings=(C.c_double*5)()
                if _lib.anchor_query_live_stats(self._handle,counters,timings):raise OSError('Live IO counters unavailable')
                result_stats['live']=dict(zip(('reads','code_reads','rerank_reads','submits','overlap_batches',
                    'max_pending','direct_reads','buffer_bytes'),map(int,counters)))
                result_stats['live'].update(zip(('lock_wait_ms','route_ms','io_ms','score_ms','rerank_ms'),map(float,timings)))
            if diagnostic_ids is not None:
                trace=(C.c_uint64*3)()
                if _lib.anchor_query_trace_stats(self._handle,trace):raise OSError('Diagnostic counters unavailable')
                result_stats['diagnostic']={'watched':int(trace[0]),'routed':int(trace[1]),
                    'candidates':int(trace[2]),'timing_includes_diagnostic_overhead':True}
                if diagnostic_unique:
                    coverage=(C.c_uint64*2)()
                    if _lib.anchor_query_unique_stats(self._handle,coverage):raise OSError('Unique counts unavailable')
                    result_stats['diagnostic'].update(eligible_entries=int(coverage[0]),unique_vectors=int(coverage[1]))
        return ids[:n], scores[:n], result_stats

    def close(self):
        with self._lock:
            if self._handle:
                self._finalizer()
                self._handle = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
