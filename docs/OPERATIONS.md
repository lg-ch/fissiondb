# Using FissionDB

## Installation

On Debian/Ubuntu, install the native build dependencies:

```sh
sudo apt-get update
sudo apt-get install build-essential liburing-dev libroaring-dev libxxhash-dev \
  libcurl4-openssl-dev python3-venv
make product
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]' build
```

This builds `fissiondb-engine` and `libfissiondb_anchor.so`, and installs the
`fissiondb` Python package with `fissiondb-build`, `fissiondb-serve` and
`fissiondb-calibrate`. These instructions install the local repository.

## Create an index

For live ingestion from an empty collection:

```sh
fissiondb-build create --index /data/collection --dim 768 \
  --cell-capacity 2048 --max-cells 300000
fissiondb-serve --collection /data/collection --workers 4
```

`AnchorIndex.create(path, dim=768)` provides the same operation in Python.
The destination must not exist. The native engine creates representatives from
inserted vectors and splits overflowing cells automatically. Supported input
dimensions are 1–1024. Cell capacity must be 64–65,536 and the cell-count ceiling
2–1,000,000. These values persist with the collection.

`max_cells` bounds representative growth, not total RAM. For example, 300,000
representatives at a padded dimension of 1024 require about 307 MB for int8
centers, plus metadata, allocation capacity, query buffers and the live journal's
in-memory bookkeeping. Once the ceiling is reached, ingestion continues into
larger cells. Monitor `/stats` or `index.fission_stats` for cell count, largest
cell, split timings, arena size and memory owned by the adaptive structure.
This memory counter excludes query buffers, journal bookkeeping and page cache.
Split counts persist; timing counters restart when the process reopens.

Fission runs on one native background worker. New writes are acknowledged after
durable commit and remain searchable on the parent while daughters are prepared.
`index.flush_fission()` waits for the queue to drain without holding the reader
lock. The cell capacity is an eventual target: a pending split can temporarily
exceed it. The cell-count ceiling can prevent further splitting altogether.
An overloaded split queue applies backpressure to vector writes, not readers.

Progress counters include `pending_cells`, `preparing`, `queries_during_prepare`,
`delta_records`, `backpressure_waits`, `scratch_bytes` and `peak_scratch_bytes`.
Scratch counters exclude the worker's 1 MiB stack and representative-array growth.
`publish_max_ms` measures the final descriptor swap with the write lock held;
it excludes lock acquisition, chunk reservations, journal commits and maintenance.
End-to-end query latency must still be measured under the intended workload.

To reopen from Python, supply the collection directory, its `base.f16bin`,
`residual` and `live` paths to `AnchorIndex`, as in the examples below. Fission
configuration loads automatically. To add adaptive live ingestion to an existing
residual index, use `fission_cell_capacity=2048`, or pass
`--fission-cell-capacity 2048` to the server. Frozen cells remain immutable.

For an offline frozen build:

The input format is a little-endian uint32 row count and dimension, followed by
float16 vectors. Choose the representative count for the corpus and RAM budget;
the following example uses 4,096 and requires at least that many input vectors.

```sh
./fissiondb-engine abuild /data/base.f16bin /data/index 4096 \
  --m 2 --eps 999 --tqbits 1 --seed 52
fissiondb-build convert --index /data/index --base /data/base.f16bin \
  --output /data/residual
```

Residual conversion supports dimensions 1–1024, TQ1/2/4 source codes and one to
four assignments. It is resumable; published residual files are immutable.
Build memory and speed must be sized separately from retrieval memory.

## Serve and query

```sh
fissiondb-serve --index /data/index --base /data/base.f16bin \
  --residual-dir /data/residual --live-dir /data/live \
  --nprobe 1536 --rerank 400 --auto-pack-bytes 1048576
```

The 1,536/400 setting reproduces the documented MS MARCO retrieval budget; it is
not a universal quality guarantee. Adaptive collections default to 1,536 cells
and 400 reranks. Fixed collections default to half the padded dimension, bounded
to the number of cells. Validate recall on
representative queries and exact ground truth, including each intended filter.
`fissiondb-calibrate` provides optional offline calibration from a supplied
workload; ingestion does not need to run that workflow.
The optional query-adaptation policies currently apply to the frozen reader;
adaptive live cells use the requested `nprobe` budget.

HTTP endpoints include `/health`, `/stats`, `/search`, `/insert`, `/insert_batch`,
`/update`, `/delete`, `/metadata`, `/metadata/add`, `/compact` and `/pack`.
Mutations use POST JSON. For a non-loopback bind, set `FISSIONDB_API_KEY` and send
`X-API-Key`; terminate TLS at a reverse proxy. Use stable idempotency keys for
batch ingestion to resolve ambiguous acknowledgements.

Metadata filters run natively before approximate candidate admission and exact
reranking. Supported predicates include equality, membership, conjunction,
inclusive numeric ranges, existence and regex over values. Use one consistent
type per field and configure decimal precision for float fields. Metadata
postings consume RAM and are not a general full-text index.

## Updates, deletion and backups

```python
from fissiondb import AnchorIndex
from fissiondb.backup import create, restore

with AnchorIndex('/data/index', '/data/base.f16bin',
                 residual_dir='/data/residual', live_dir='/data/live') as index:
    doc_id = index.insert(vector, {'category': 'news'})
    index.update(doc_id, replacement_vector, {'category': 'news'})
    index.delete(doc_id)
    index.pack_live()
    create(index, '/backups/snapshot-001')

options = restore('/backups/snapshot-001', '/data/restored')
```

IDs are stable and never reused. Repeating a deletion is safe, and tombstones
survive restart and compaction. Immutable frozen files retain their allocated
space. The compressed live snapshot accelerates reads; current live vectors
remain in the authoritative journal.

For adaptive collections, `/pack` and `index.pack_live()` persist a checkpoint.
Automatic checkpointing defaults to a 256 MiB journal-growth threshold checked
every 30 seconds; `--auto-pack-bytes 0` disables it. Checkpoints allow retired
code chunks to be reused but do not shrink the arena file. Split preparation
overlaps readers; publication briefly holds the live write lock. Checkpointing
and compaction also acquire the worker's maintenance mutex and hold the live
write lock, so requests can still wait for those operations.
Compaction rebuilds adaptive cells and can be expensive on a large live corpus.
Close normally to attempt a final checkpoint; journal recovery also handles
unclean shutdowns. Restore rebuilds derived files from the committed journal.

Size live storage separately: journal and row files both hold padded float32
vectors. A 768-dimensional vector is padded to 1024, using roughly 8 KiB across
those two files before metadata and codes. Large frozen retrieval measurements
do not establish the disk footprint or throughput of a large live collection.

Backups copy all required frozen files, originals and a committed live prefix;
restore verifies their SHA-256 checksums. Provision disk space for an independent
copy. Existing version-one backup manifests remain readable after the rename.

## Memory and Docker

`--memory-bytes` limits context allocations rather than the whole process.
Enforce total memory with a cgroup. Additional workers need their own contexts;
measure total memory and tail latency at the intended concurrency.

Docker's default seccomp profile blocks the io_uring calls required by the
engine. Generate an explicit profile from this pinned Moby source:

```sh
curl -fsSL https://raw.githubusercontent.com/moby/profiles/61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31/seccomp/default.json -o docker-default.json
python scripts/allow_uring_seccomp.py docker-default.json fissiondb-seccomp.json
docker build -t fissiondb:dev .
docker run --rm --security-opt seccomp="$(pwd)/fissiondb-seccomp.json" \
  --memory 1g --memory-swap 1g -p 127.0.0.1:8080:8080 \
  --mount type=bind,src=/absolute/data,dst=/data \
  -e FISSIONDB_API_KEY fissiondb:dev --host 0.0.0.0 \
  --index /data/index --base /data/base.f16bin \
  --residual-dir /data/residual --live-dir /data/live
```

The generator only adds `io_uring_setup`, `io_uring_enter` and
`io_uring_register` to the supplied profile. The host kernel must support them.
Export the API key in the launching shell first. Docker's `1g` is one GiB;
published measurements use the explicit decimal-byte cap shown in their report.

## Validation

For bulk loading, set `FISSIONDB_INGEST_THREADS` before opening the collection
(default 1, valid range 1–256). It parallelizes routing and residual encoding
within insertion batches; journal commits and the split worker remain serialized.
For example, `FISSIONDB_INGEST_THREADS=20 OMP_WAIT_POLICY=PASSIVE python ingest.py`
uses up to 20 ingestion threads. The process must have affinity to the intended
CPUs. This does not increase the number of search threads. Measure actual CPU use
and throughput: IO, fsync and split backpressure can limit scaling.

```sh
make test
python -m build
```

The test suite exercises query correctness, residuals, dimensions, filters, live
mutations, automatic fission, concurrent readers, deletion, journal recovery,
cache corruption, packing, compaction and backup/restore.
Concurrency tests pause a worker with unpublished daughters and require reads,
writes and deletions to remain visible; they also exercise crashes and worker
failure while a writer is waiting for backpressure to clear.
Small-corpus correctness tests do not establish large-corpus recall or throughput.

Adaptive IO tests compare IDs and scores across unit/grouped reads, overlap on
and off, and buffered/direct reads. They also corrupt a code while another
batch is in flight and check that all IO drains before the context closes.
Append concurrency tests pause before writing and after durability but before
publication: searches must complete against the old state, then include the
new records after acknowledgment. Crash/failure tests verify journal recovery.

### Rebuild-based performance regression gate

Run `tests/bench_live_regression.py` on both revisions using the same host,
CPU pair, memory cgroup and input files. Each run refuses an existing output
directory and builds a fresh adaptive index. The last quarter is ingested with
a concurrent reader; a quiescent pass then measures recall against exact GT for
the complete selected prefix. Concurrent prefix queries are not compared to
full-corpus GT. For example, with externally supplied real embeddings:

```sh
PYTHONPATH=scripts python tests/bench_live_regression.py \
  --source /data/base.f16bin --queries /data/queries.npy \
  --truth /data/gt-1000000.npz --rows 1000000 \
  --ingest-cpu 0 --search-cpu 1 --output /results/baseline
# Switch/build the candidate revision, then use a different output directory:
PYTHONPATH=scripts python tests/bench_live_regression.py \
  --source /data/base.f16bin --queries /data/queries.npy \
  --truth /data/gt-1000000.npz --rows 1000000 \
  --ingest-cpu 0 --search-cpu 1 --output /results/candidate \
  --baseline /results/baseline/report.json --require-contiguous --require-direct
```

The default gate fails if mean recall@10 drops by more than 0.005, median latency
increases by more than 10%, or p95 increases by more than 20%, in either the
quiescent or concurrent pass. Thresholds are explicit command-line options.
Use an external cgroup (for example `MemoryMax=2000000000`, `MemorySwapMax=0`)
and avoid other disk-intensive work when establishing performance baselines.
There is no forced cache eviction in this gate. Different rebuilds may produce
slightly different split topologies; the correctness suite separately requires
identical query results across IO schedules on the same index.

The GitHub Actions workflow **Rebuilt live performance comparison** runs on
engine pull requests and can also be dispatched manually with a baseline ref. It
rebuilds both revisions on a seeded 65,536-vector fixture with independent
queries and exhaustive FP64 GT. It enforces the same 2 GB cgroup for both runs.
Its 25% median / 50% p95 tolerance accounts for shared-runner noise; dedicated
real-corpus measurements should use the stricter defaults above. Regular CI
runs the deterministic IO scheduling and publication correctness checks.

The candidate CI run additionally requires exactly one code read per selected
nonempty cell in the quiescent pass, and verifies that all live code reads use
direct IO in both passes. These gates fail independently of recall or timing
noise, including if direct IO silently falls back to buffered reads. The default
IO policy is also checked at dimensions 128, 768 and 1024. Direct IO is the live
code default; `query.residual_io(direct=False)` remains available for workloads
that benefit from a resident page cache. Layout tests cross 512/1024/2048-entry boundaries, exercise fission and
cell-count ceilings, and verify the invariant again after checkpoint/reopen.
Legacy migration tests compare every encoded byte and representative and kill
the opener before arena replacement, after replacement and after checkpoint
publication. A separate recovery test retains an uncheckpointed journal tail.

For adaptive queries, `stats['live']` reports the direct lock-wait timer, routing,
IO submit/wait, code scoring and reranking, plus read/submission/overlap counts.
IO time includes rerank waits and overlaps scoring, so these timers are not an
additive wall-time breakdown. The legacy `rerank_ms` field still aggregates the
adaptive live path; use `stats['live']` to diagnose it.
