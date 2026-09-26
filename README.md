# FissionDB

FissionDB is a disk-backed approximate vector search engine with native metadata filters and durable live mutations. This repository starts from the validated current engine with an independent Git history.

The Python package (`mangrove`), command names (`mangrove-*`), environment variables and native symbols retain their existing names in this initial import. See [source provenance](docs/SOURCE_PROVENANCE.md).

## Build and install (Linux)

Requires GCC, OpenMP, liburing, CRoaring, xxHash and libcurl development packages, Python 3.10+ and NumPy.

```sh
make product
python3 -m pip install .
./mangrove-engine abuild
mangrove-build convert --index /data/index --base /data/base.f16bin --output /data/residual
mangrove-serve --index /data/index --base /data/base.f16bin --live-dir /data/live --residual-dir /data/residual --rerank 400 --auto-pack-bytes 1048576
```

`abuild` prints its required arguments when invoked without them. Raw input is a little-endian uint32 row count and dimension followed by float16 vectors. Residual conversion supports every input dimension from 1 through 1024, int8 anchors, TQ1/2/4 source codes and 1–4 assignments. Rotation pads internally to a power of two (minimum eight); original vectors stay unpadded on disk and in the public API. The original frozen index must already exist. Conversion is resumable; published residual files are immutable. Existing 1024d v1 residual files remain readable.

## Python

```python
from mangrove import AnchorIndex
from mangrove.backup import create, restore

with AnchorIndex('/data/index', '/data/base.f16bin',
                 live_dir='/data/live', residual_dir='/data/residual') as index:
    doc_id = index.insert(vector, {'language': 'fr', 'year': 2025})
    with index.context(nprobe=1024, rerank=400, threads=1) as query:
        ids, scores, stats = query.search(vector, top_k=10,
                                         where={'language': 'fr', 'year': ('range', 2020, 2026)})
    index.update(doc_id, replacement_vector, {'language': 'fr'})
    index.pack_live()
    create(index, '/backups/snapshot-001')

# Returns arguments for reopening a verified independent copy:
options = restore('/backups/snapshot-001', '/data/restored')
```

HTTP endpoints: `/health`, `/stats`, `/search`, `/insert`, `/insert_batch`, `/update`, `/delete`, `/metadata`, `/metadata/add`, `/compact`, `/pack`. Mutations use POST JSON. Set `MANGROVE_API_KEY` and send `X-API-Key` when binding beyond loopback; terminate TLS at a reverse proxy. The Python `mangrove.Client` uses these endpoints. Batch group commit defaults to enabled over HTTP; use stable idempotency keys to resolve ambiguous insertion acknowledgements.

`index.add_metadata(ids, {'category': 'news'})` (HTTP `/metadata/add`) adds a field/value to up to 8192 existing IDs per call, with one group fsync. It preserves other values. Repeating the call is idempotent; after an ambiguous IO failure, reopen and retry the entire call. A crash can retain a prefix of the encoded keys. Use `set_metadata` to replace all metadata for one ID.

Filters execute natively before approximate candidate admission and exact reranking. Available predicates include equality, membership, conjunction, inclusive numeric ranges, existence and regex over metadata values. Use one consistent type per field; configure decimal precision for float fields. Regex metadata matching is not full-text retrieval. Filter postings consume RAM and require separate large-scale validation.

## Default search budget

Python `index.context()` and HTTP serving now default to `nprobe = indexed_dimension // 2`, bounded to `[1, cell_count]`: 64 cells in 128d, 512 in 1024d. This is a starting heuristic, not a recall guarantee. An explicit `nprobe` / `--nprobe` overrides it. Computing this default needs no GT, training scan or ingestion work. Padded vectors use the indexed dimension.

Reranking defaults to **400** in Python and HTTP. The older MS MARCO 512-cell measurements use rerank **1000**; Common Crawl's 64-cell measurements use **4000**. Set those values to reproduce those configurations. Filter selectivity and concurrent load still require their own evaluation.

Residual codes use at most 512 bits: four bits per rotated component through padded dimension 128, two through 256, then one through 512. For padded dimension 1024, the first 512 rotated components are encoded. Records retain a 72-byte stride including ID and scales. This width policy supports the dimensions; it does not guarantee a given recall on every geometry.

The residual NVMe reader overlaps scoring with the next batch of up to 64 cells. Each context owns its ring and normally two 8 MiB buffers, reduced to fit its allocation budget or enlarged for an unusually large cell within that same budget. Dense cells require one logical request each; direct reads round boundaries to 4 KiB. `query.residual_io(batch_cells=64, overlap=True, direct=True)` configures this path; `direct=False` uses buffered reads. Unsupported direct-open filesystems default to buffered mode. AVX2/F16C kernels are selected at runtime on x86, with portable fallbacks; ARM retains NEON.

The heavy `mangrove-calibrate` workflow is optional offline validation, not a prerequisite for creating or serving an index.

## Measurements and limits

On the GB10, 247,154,006 text vectors (1024d), 200 doc-to-doc queries with exact GT and self exclusion: the final integrated residual engine achieved recall@10 **0.9595**, p50 **91.7 ms**, p95 **130.8 ms**, p99 **174.0 ms** on cold NVMe. One query thread, total cgroup limit **999,997,440 bytes**, swap disabled, zero OOM. This is a doc-to-doc benchmark, not a universal latency guarantee.

With live ingestion and automatic packing under the same memory limit: 1,000 vectors in batches of 64 achieved **78.84 vectors/s** while 120 unfiltered queries measured p50 **99.63 ms**, p95 **145.29 ms**. This is a small concurrency sample, not a steady-state capacity claim. Residual conversion of the existing index took 4,764 seconds; this is not the full raw-data build time.

On Common Crawl web graph embeddings (52,903,544 frozen 128d vectors), TQ4 with corrected dequantized L2 scoring achieves **0.992 recall@10, p50 42.6 ms, p95 65.6 ms** at nprobe=64 / rerank=4000, under the same decimal 1 GB memory limit and cold-cache protocol, without concurrent ingestion. The 200 held-out document queries were also used for parameter tuning; these are not independent unseen evaluation queries. At nprobe=32 / rerank=2000, recall is 0.969 at p50 24.6 ms. Filter budgets must be tuned separately: a rare-TLD predicate reaches only 0.91 recall at 64 probes, versus 0.99 at 256 probes in the preceding grid. Each filter currently has only ten query samples. See [measurements](docs/validation/commoncrawl-tq4.json).

The same Common Crawl TQ4 configuration ingested 10,000 real held-out vectors in 64-vector durable groups at **1,292 vectors/s** while queries ran. Concurrent unfiltered latency was p50 **48.2 ms**, p95 **64.1 ms** (127 queries); the common-TLD filter measured p50 49.6 ms, p95 62.1 ms (31 queries). This short load test used natural cache, one query caller plus one writer, and a 1 GB total-memory cap; it does not establish full-corpus recall during ingestion or sustained throughput. Process peak RSS was 319 MB.

The compressed live snapshot accelerates reads but the float32 journal remains authoritative. Compaction reclaims obsolete/deleted records; it does not remove the current vectors from the journal. Packing rewrites the snapshot. Backups copy frozen files, vectors and the committed journal and verify SHA-256 on restore; provision space for a full independent copy. Residual search currently supports local storage and one thread per query context. TQ storage retains the separate S3 path.

`index.delete(id)` and HTTP `/delete` persist a tombstone before acknowledgement. Repeating a deletion is safe; IDs are never reused. Deleted frozen, updated and packed live records are excluded before candidate admission, including filtered searches, and stay deleted after restart and compaction. Deletion does **not** erase or reclaim immutable frozen vector/code files; journal compaction reclaims obsolete live records.

`--memory-bytes` bounds context allocations, not the entire process or page cache; enforce the deployment limit with a cgroup. More query workers require more RAM. `/stats` includes maintenance errors, packing progress and cumulative HTTP counters; latency distributions and total memory should be measured externally.

On MS MARCO v2 (113,520,750 passages, 1024d), the simple 512-cell / rerank-1000 configuration measures **0.95883 recall@10, p50 152.4 ms, p95 188.3 ms** on 600 reserved real query-to-document questions. Peak cgroup memory was **599,916,544 bytes** under a 999,997,440-byte cap, no swap/OOM, one native thread, with a `POSIX_FADV_DONTNEED` request before each query. The separate 600 calibration questions measured 0.96033 recall and p50 138.8 ms. This is held-out empirical recall, not a statistical guarantee of recall above 0.95: the conservative 95% lower bound is 0.9304. The 100 ms target is not met. See [audit](docs/validation/msmarco-pragmatic.json).

## Verification

```sh
make test
```

The current product tree passes 136 tests on ARM64; 67 residual, dimension and quantization tests pass with AddressSanitizer and UndefinedBehaviorSanitizer (leak detection disabled for the Python host). Every dimension from 1 through 1024 also passes a separate exhaustive width check. The earlier ARM64 wheel and Docker checks predate this integration; remote CI execution remains unverified.

The integrated NVMe reader on the existing MS MARCO 113.5M/1024d, 200k-anchor index achieves **96.25% recall@10, 52.69 ms median, 62.59 ms p95 and 261.74 MB peak RAM**, using 1,536 probes and 400 reranks on 600 reused real queries, one GB10 core, under a 1 GB cgroup. Full local vectors, cache eviction requested, no ingestion. The pre-change reader produces the same results at 88.65 ms median and 659.31 MB peak. See the [protocol, evidence and limitations](docs/validation/nvme-integration-20260926.md).

Tests cover journal recovery, updates/deletes, filters, residual conversion, live packing, concurrent access, backup/restore and group commit. CI builds and tests the minimal engine. Large-corpus validation and the subsequent dataset campaign are tracked in `docs/PRODUCT_WORKPLAN.md`.

## Docker io_uring profile

The default Docker seccomp profile blocks the io_uring calls used by this engine.
Generate an explicit profile from this pinned official Moby source before running
queries in the container (tested on Docker 29.2.1, ARM64):

```sh
curl -fsSL https://raw.githubusercontent.com/moby/profiles/61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31/seccomp/default.json -o docker-default.json
python3 scripts/allow_uring_seccomp.py docker-default.json mangrove-seccomp.json
docker build -t mangrove:dev .
docker run --rm --security-opt seccomp="$(pwd)/mangrove-seccomp.json" \
  --memory 1g --memory-swap 1g -p 127.0.0.1:8080:8080 \
  --mount type=bind,src=/absolute/data,dst=/data \
  -e MANGROVE_API_KEY mangrove:dev --host 0.0.0.0 \
  --index /data/index --base /data/base.f16bin --live-dir /data/live
```

The generator preserves the supplied profile except for allowing
`io_uring_setup`, `io_uring_enter` and `io_uring_register`. The host kernel must
support io_uring. Export the API key in the launching shell before using the
example. The container memory example is 1 GiB; benchmark cgroups used decimal
limits reported above.

See [calibrated adaptive search](docs/ADAPTIVE_SEARCH.md) for per-index budget selection, native geometry thresholds, filtered budgets and soft IO/time limits.

See [automatic snapshot calibration](docs/AUTOMATIC_CALIBRATION.md) for the end-to-end `mangrove-calibrate` command, independent audit, resumable measurements and explicit quality/latency tradeoffs.
