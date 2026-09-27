# FissionDB architecture

FissionDB partitions vectors into cells. Each cell has a
representative; vectors can belong to multiple cells to improve retrieval
coverage. The representatives route queries, while vector payloads stay on disk.

## Retrieval

```mermaid
flowchart LR
    Q[Query vector] --> R[Score int8 representatives]
    R --> C[Select cells]
    C --> I[Stream residual codes]
    I --> F[Native metadata filter]
    F --> S[Approximate candidate ranking]
    S --> E[Exact rerank from original vectors]
    E --> K[Top-k results]
```

Routing scans the representatives. The query budget controls how many cells are
read. Highly selective filters with at most 4,096 allowed IDs can use an exact
shortcut.

Frozen cell payloads are contiguous in the residual file. The reader issues one logical
request per selected cell, with direct reads aligned to 4 KiB. It overlaps the
next batch of up to 64 cells with scoring the current batch. Device or filesystem
layers may split a logical request into multiple physical operations.

Adaptive live cells occupy contiguous slots, allocated in powers of two of
512-record chunks. Used records form a dense prefix. The reader issues one code
request per ordinary cell; cells exceeding the bounded buffer are read in pieces.
It reuses the query's aligned buffers and io_uring queue, with up to 64 reads per
batch and the next batch in flight during scoring. Live codes use direct IO by
default when the filesystem supports opening a direct descriptor. Explicit
`residual_io(direct=False)` selects buffered IO; an unavailable direct descriptor
also falls back to buffered reads. Actual mode is visible in `live.direct_reads`.
Direct reads include boundary pages; only published used records are scored and
checksum-validated. Live reranking also batches buffered journal reads.
`residual_io()` configures both frozen and adaptive live code reads. Frozen and
live cells are routed independently.

Candidate IDs are deduplicated before exact reranking. The default rerank budget
is 400 per source. Exact scoring uses stored float16 originals for frozen vectors
and normalized float32 journal vectors for live data. Mixed collections can
rerank up to 400 from each source. Approximate routing and candidate selection
can still miss neighbors.

## Compact residuals

Vectors are rotated with a seeded transform and encoded relative to their cell
representative. Input dimensions are padded internally to a power of two, with a
minimum of eight. Public vectors and frozen original storage retain the input
width. The live journal and derived row file use the padded width.

| Padded dimension | Residual encoding |
|---:|---|
| 8–128 | Four bits per rotated component |
| 256 | Two bits per rotated component |
| 512 | One bit per rotated component |
| 1024 | One bit for each of the first 512 rotated components |

The fixed record stride is 72 bytes: an ID, scales and a 64-byte code area.
The code budget is at most 512 bits. Multiple assignments multiply the number
of records, so index sizing must include the assignment count.
Live records add an eight-byte journal offset and four-byte checksum: 84 bytes
per assignment. Each live vector is assigned to two representatives.

## Memory

Int8 representatives cost approximately `cell_count × indexed_dimension` bytes,
plus per-cell metadata and scales. A query context normally owns two 8 MiB IO
buffers and ranking scratch space. It can reduce its buffers to fit its budget,
or enlarge them for a large cell within that budget.

The allocation budget controls owned query memory. It does not bound the entire
process, metadata postings or the operating-system page cache. Enforce total
deployment memory with a cgroup and measure concurrent contexts explicitly.

## Live mutations and persistence

New collections start empty. Their first insertion initializes live
representatives; subsequent insertions scan int8 representatives and attach to
the two best cells. Once a cell exceeds its configured capacity (2,048 by
default), the engine reads its current vectors, removes obsolete versions and
splits it if it still overflows.

Splitting samples 16 pairs on 64 seeded rotated coordinates, chooses the most
distant sampled pair, and partitions at the median of the resulting projection.
Each child's mean direction becomes an int8 representative. The engine recomputes
the residual codes relative to those representatives. It does not run global
training or compute ground truth. At the configured representative ceiling,
insertions continue and cells may grow beyond the capacity target.

One native worker prepares a split outside the live write lock. The parent stays
published while the worker reads its vectors, computes child representatives and
writes new codes into reserved, unpublished chunks. Concurrent insertions and
updates append to the parent and remain searchable immediately after commit.
The worker collects and encodes this delta before publication. Deletes and
overrides are checked against the current journal state during retrieval.

Once the worker has caught up, a short write lock swaps the cell descriptors and
representatives. Acquiring that lock waits out readers of the previous version;
old chunks remain intact until a later durable checkpoint. A query therefore
sees a consistent topology. Preparation overlaps queries and ingestion; this is
not a lock-free engine. Chunk reservation, publication, checkpointing and
compaction still need synchronized sections.

After initial bootstrap, insertions prepare bounded batches of at most 256
vectors. Routing and residual encoding run under a shared read lock; journal,
row and unpublished code writes and their durability barriers run without the
live write lock. A writer mutex serializes journal mutations. A separate
publication mutex pins representatives and protects reserved slots from
checkpoint reclamation, while the split worker can continue preparing daughters.
After the journal and rows are durable, a short write section publishes counts,
metadata and chunk descriptors together. Queries see the previous committed
state until publication, and acknowledged inserts are searchable immediately.
The existing journal format and recovery rules are unchanged; durable but
unacknowledged inserts may be recovered after a crash.

Readers still hold a shared live lock during retrieval. A publisher can wait
for those readers, and readers can wait for reservation/publication. Initial
bootstrap, updates, metadata mutations, deletes and maintenance still have
exclusive sections. This change removes expensive insertion routing and fsync
from the reader exclusion period; it does not eliminate all synchronization.

Only one cell is prepared at a time. A parent exceeding capacity plus 256
assignments pauses subsequent vector writes until the worker makes progress;
an already-running batch can add at most 256 more. Waiting writers release the
live lock so searches can continue. The bounded delta lives in the published
parent's disk records. Scratch buffers cover at most capacity plus 512 records,
and the worker has a 1 MiB stack. Growing the representative arrays can still
temporarily hold both old and new allocations; the large center copy happens
outside the live lock. No complete vector corpus is duplicated for a split.

`flush_fission()` waits for queued splits with readers still active. Stop or
quiesce ingestion when a completely drained topology is required. Closing the
collection drains the worker before its final checkpoint. Recovery from the
journal remains valid if the process dies with unpublished daughters on disk.
Checkpointing and compaction serialize with the worker before acquiring the live
write lock; they remain blocking maintenance operations.

Existing frozen files remain immutable. Enabling fission indexes live insertions
and vector overrides independently and merges their results with frozen results.
Without fission enabled, the original fixed-cell live packing path remains
available.

Updates keep a stable document ID. Deletes persist a tombstone; deleted entries
are excluded before candidate admission, including through filtered searches.
The journal is authoritative and fsynced before acknowledging writes. Adaptive
codes and topology are rebuildable caches. Packing an adaptive collection saves
a checkpoint bound to the journal prefix and file identities, then permits reuse
of chunks unreachable from that checkpoint. Retired chunks remain intact until
the replacement checkpoint is durable. Opening validates checksums and replays
the journal tail; a missing or invalid cache is rebuilt from the journal.

Compaction removes obsolete live records and rebuilds the adaptive cache because
journal offsets change. Backups include the fission configuration and committed
journal; derived caches rebuild on restore. Reopening still reads the journal
and validates cached codes, so restart cost grows with live data volume.

Live storage currently includes both normalized float32 journal vectors and a
derived float32 row file, at the padded width. At input dimension 768, padding
to 1024 means roughly 8 KiB per vector for these two copies alone, before
metadata, codes and retired chunks. This is larger than the frozen layout;
provision live storage separately. Reclaiming immutable frozen files is not
implemented.

## On-disk files

Adaptive cells own contiguous power-of-two slots, allocated in units of 512
records. Used records form a dense prefix: a normal cell takes one code read,
independent of allocation-chunk boundaries. Batched io_uring reads and overlap
remain enabled. A cell larger than an entire query IO buffer is read in bounded
portions instead of growing per-query memory.

An append that outgrows its slot copies existing codes into a larger unpublished
slot, then publishes it with the new records. Split daughters also own contiguous
slots. Old slots stay intact until a newer checkpoint is durable. Free runs are
reused; reserved capacity and retired slots contribute to the arena's disk size.

Live checkpoint version 2 records this invariant. Opening a version 1 checkpoint
streams its codes into contiguous slots without changing cell IDs, representatives,
membership, encoded bytes or the saved journal prefix. A hard link keeps the old
arena recoverable until the replacement checkpoint is durable. Migration needs
temporary disk space for the new arena and delays opening; it does not run inside
a search request. A killed migration recovers from the checkpoint's file identity.
Journal and residual formats remain unchanged. Older binaries do not understand
the v2 checkpoint and can rebuild it from the journal; avoid downgrading an active
collection when preserving its exact topology matters.

- `meta.txt`, `anchors.bin`, `scale.bin`, `offs.bin`: index configuration,
  representatives, scales and offsets.
- `blocks.bin`: source quantized cell payloads; the residual converter reads them.
- `residual.meta`, `res512.bin`: converted residual snapshot used for retrieval.
- `base.f16bin`: little-endian uint32 row count and dimension, followed by original
  float16 vectors.
- Live directory: `live.log`, derived `live.rows`, and optional fixed-cell
  `live.pack`. Adaptive collections also contain `fission.config`,
  `fission.state` and `fission.codes`.

Existing vector/index formats remain readable. Renaming the project does not
require rebuilding an existing index.
