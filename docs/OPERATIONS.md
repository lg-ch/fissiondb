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
not a universal choice. Without an explicit probe count, the starting heuristic
is half the padded dimension, bounded to the number of cells. Validate recall on
representative queries and exact ground truth, including each intended filter.
`fissiondb-calibrate` provides optional offline calibration from a supplied
workload; ingestion does not need to run that workflow.

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

```sh
make test
python -m build
```

The test suite exercises query correctness, residuals, dimensions, filters, live
mutations, deletion, journal recovery, packing, compaction and backup/restore.
Small-corpus correctness tests do not establish large-corpus recall or throughput.
