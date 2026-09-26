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

Live cells use linked chunks of 512 records, read through buffered `pread` calls.
The live reader currently reads chunks serially; it does not yet use the frozen
reader's overlapping IO batches. Frozen and live cells are routed independently.

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

The shared live read/write lock covers publication and queries. A split holds
the write lock, so queries wait and cannot observe a half-rewritten cell.
Checkpointing and compaction also hold this lock. Separate query contexts allow
concurrent readers; this version does not provide lock-free splitting.

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
