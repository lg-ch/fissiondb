<p align="center">
  <img src="fissiondb-logo.png" alt="FissionDB logo: progressive cell division" width="180">
</p>

# FissionDB

Vector retrieval for large collections with a small RAM footprint. FissionDB keeps
compact cell representatives in memory and streams compressed vector codes from
local SSD storage. It reranks a short candidate list against the original vectors.

The engine is written in C, with a Python API and an HTTP service. It supports
live insertions with automatic cell splitting, updates, durable deletions and
native metadata filters. An empty collection grows its representatives as data
arrives, without an offline training set.

## Current architecture

1. Score the in-memory int8 cell representatives and select cells to explore.
2. Read their compact residual codes from SSD.
3. Apply metadata filters before candidate admission and rank eligible codes.
4. Read the best candidates' original vectors and compute exact cosine scores.

Residuals use up to 512 bits per vector, with an encoding adapted to dimensions
1–1024. Frozen records occupy 72 bytes; live records occupy 84 bytes including
journal offsets and checksums. Multiple cell assignments duplicate records.
Exact reranking defaults to 400 candidates.

Each query has its own bounded scratch buffers and IO context. The index is
shared. The native engine uses ARM NEON or runtime-selected x86 AVX2/F16C kernels.

[Architecture and data layout](docs/ARCHITECTURE.md)

## Measured retrieval performance

| Corpus | Vectors | Dimensions | Recall@10 | Median | p95 | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| MS MARCO v2 | 113,520,750 | 1024 | **96.25%** | **52.69 ms** | **62.59 ms** | **261.74 MB** |

Measured on one GB10 CPU core, with 200,000 representatives, 1,536 explored cells
and 400 reranks. The full original vectors were stored on local NVMe. The run
used 600 real query-to-document queries, requested page-cache eviction before
each query and enforced a 1 GB total-memory cgroup with no swap. Hardware device
caches were not flushed. These are native retrieval timings, without concurrent
ingestion or HTTP/network overhead. The query panel was reused during development.

[Protocol, evidence and scope](docs/BENCHMARKS.md)

## Build and run

Linux, Python 3.10+, GCC, OpenMP, liburing, CRoaring, xxHash and libcurl are required.
Install from this repository:

```sh
make product
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Create an empty collection and start ingesting through the HTTP API:

```sh
fissiondb-build create --index /data/collection --dim 768
fissiondb-serve --collection /data/collection --workers 4
```

Or create and populate a collection from Python:

```python
from fissiondb import AnchorIndex

with AnchorIndex.create('/data/collection', dim=768) as index:
    ids = index.insert_batch(vectors, group_commit=True)
    with index.context(nprobe=1536, rerank=400) as query:
        ids, scores, stats = query.search(query_vector, top_k=10)
    print(index.fission_stats)
```

The default cell capacity is 2,048 assignments. A background worker prepares
overflowing cells' daughters while queries continue on the published parent.
Concurrent inserts remain searchable there until their codes have caught up;
a short write lock publishes the daughters. A bounded backlog applies ingestion
backpressure when needed. A configurable cell-count ceiling limits growth.

For an existing frozen index and its original vectors:

```sh
fissiondb-build convert --index /data/index --base /data/base.f16bin --output /data/residual
fissiondb-serve --index /data/index --base /data/base.f16bin \
  --residual-dir /data/residual --live-dir /data/live --rerank 400 \
  --fission-cell-capacity 2048
```

```python
from fissiondb import AnchorIndex

with AnchorIndex('/data/index', '/data/base.f16bin',
                 residual_dir='/data/residual', live_dir='/data/live',
                 fission_cell_capacity=2048) as index:
    with index.context(nprobe=1536, rerank=400, threads=1) as query:
        ids, scores, stats = query.search(query_vector, top_k=10,
                                         where={'language': 'fr'})
    doc_id = index.insert(document_vector, {'language': 'fr'})
    index.delete(doc_id)
```

[Index creation, service configuration, filters and maintenance](docs/OPERATIONS.md)

## Scope

- Dimensions 1–1024 are supported. Recall depends on the data, filters and search
  budgets; a dimension-based default is a starting point, not a quality guarantee.
- New collections split live cells automatically. Existing frozen cells remain
  immutable; their results merge with an independently routed live index.
  The 113M benchmark above measures frozen retrieval, not this new live path.
- Filters support equality, membership, conjunction, numeric ranges, existence
  and regex over metadata values. Regex matching is not full-text retrieval.
  Metadata postings consume RAM and need separate sizing.
- Deletion hides records durably. Frozen files are immutable; their space is not
  reclaimed by deleting a document. Compaction reclaims obsolete live records.
- The measured low-memory result concerns one query context. Concurrent query
  throughput and large live populations require additional validation. The live
  journal and its derived row file currently store padded float32 vectors;
  storage costs differ from the compact frozen format.

This is a development release. The default documented deployment uses local
storage. [Run the tests](docs/OPERATIONS.md#validation) before deployment.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
