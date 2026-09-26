# FissionDB architecture

FissionDB partitions an immutable vector snapshot into cells. Each cell has a
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

Cell payloads are contiguous in the residual file. The reader issues one logical
request per selected cell, with direct reads aligned to 4 KiB. It overlaps the
next batch of up to 64 cells with scoring the current batch. Device or filesystem
layers may split a logical request into multiple physical operations.

Candidate IDs are deduplicated before exact reranking. The default rerank budget
is 400. Exact scoring uses the stored float16 original vectors, not the residual
codes. Approximate routing and candidate selection can still miss neighbors.

## Compact residuals

Vectors are rotated with a seeded transform and encoded relative to their cell
representative. Input dimensions are padded internally to a power of two, with a
minimum of eight. Public vectors and original storage retain the input width.

| Padded dimension | Residual encoding |
|---:|---|
| 8–128 | Four bits per rotated component |
| 256 | Two bits per rotated component |
| 512 | One bit per rotated component |
| 1024 | One bit for each of the first 512 rotated components |

The fixed record stride is 72 bytes: an ID, scales and a 64-byte code area.
The code budget is at most 512 bits. Multiple assignments multiply the number
of records, so index sizing must include the assignment count.

## Memory

Int8 representatives cost approximately `cell_count × indexed_dimension` bytes,
plus per-cell metadata and scales. A query context normally owns two 8 MiB IO
buffers and ranking scratch space. It can reduce its buffers to fit its budget,
or enlarge them for a large cell within that budget.

The allocation budget controls owned query memory. It does not bound the entire
process, metadata postings or the operating-system page cache. Enforce total
deployment memory with a cgroup and measure concurrent contexts explicitly.

## Live mutations and persistence

The current serving snapshot has a fixed set of representatives. Insertions
attach to that structure and persist in a journal. Reads combine immutable
payloads, a compressed live snapshot and the more recent journal tail.

Updates keep a stable document ID. Deletes persist a tombstone; deleted entries
are excluded before candidate admission, including through filtered searches.
Packing builds the compressed live snapshot. Compaction removes obsolete live
journal entries, while current vectors remain authoritative in the journal.

Automatic cell splitting, representative growth and reclaiming immutable frozen
files are not implemented in this serving snapshot. These are separate from the
optional adaptive query budget, which changes how many cells a query explores.

## On-disk files

- `meta.txt`, `anchors.bin`, `scale.bin`, `offs.bin`: index configuration,
  representatives, scales and offsets.
- `blocks.bin`: source quantized cell payloads; the residual converter reads them.
- `residual.meta`, `res512.bin`: converted residual snapshot used for retrieval.
- `base.f16bin`: little-endian uint32 row count and dimension, followed by original
  float16 vectors.
- Live directory: mutation journal and derived live snapshots.

Existing vector/index formats remain readable. Renaming the project does not
require rebuilding an existing index.
