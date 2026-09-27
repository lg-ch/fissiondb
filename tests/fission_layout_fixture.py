"""Independent reader and v1 fragmented fixture for the persisted live format."""
import ctypes as C
import ctypes.util
import os
import struct

NONE = 2**32 - 1
CHUNK_BYTES = 512 * 84
HEADER = struct.Struct('<16Q')


def checksum(data):
    lib = C.CDLL(ctypes.util.find_library('xxhash'))
    lib.XXH64.argtypes = [C.c_void_p, C.c_size_t, C.c_uint64]
    lib.XXH64.restype = C.c_uint64
    return struct.pack('<Q', lib.XXH64(data, len(data), 0))


def read_layout(root):
    state = (root / 'live/fission.state').read_bytes()
    assert checksum(state[:-8]) == state[-8:]
    h = list(HEADER.unpack_from(state))
    dim, cells, chunks = h[8], h[11], h[12]
    descriptors = list(struct.iter_unpack('<III', state[128:128 + cells * 12]))
    start = 128 + cells * 12
    representatives = state[start:start + cells * (dim + 4)]
    start += len(representatives)
    links = list(struct.iter_unpack('<II', state[start:-8]))
    assert len(links) == chunks
    records = []
    with (root / 'live/fission.codes').open('rb') as f:
        for head, tail, count in descriptors:
            parts = []; seen = set(); block = head
            while block != NONE:
                assert block not in seen and block < chunks
                seen.add(block)
                nxt, used = links[block]
                assert used <= 512
                f.seek(block * CHUNK_BYTES)
                part = f.read(used * 84)
                assert len(part) == used * 84
                parts.append(part); last = block; block = nxt
            assert (last if parts else NONE) == tail
            data = b''.join(parts)
            assert len(data) == count * 84
            records.append(data)
    return h, descriptors, representatives, links, records


def assert_contiguous(root):
    h, cells, _, links, _ = read_layout(root)
    assert h[1] == 2
    owned = set()
    for head, tail, count in cells:
        if head == NONE:
            assert tail == NONE and count == 0
            continue
        blocks = tail - head + 1
        assert blocks > 0 and blocks & (blocks - 1) == 0
        left = count
        for b in range(head, tail + 1):
            assert b not in owned
            owned.add(b)
            used = min(left, 512)
            assert links[b] == (b + 1 if b < tail else NONE, used)
            left -= used
        assert left == 0
    return cells


def fragment_as_v1(root):
    """Round-robin the actual records into noncontiguous old-format chains.

    Cell IDs, record order, encoded bytes, representatives and checkpoint prefix
    remain unchanged. This is the old wire format, not the new allocator.
    """
    h, cells, representatives, _, records = read_layout(root)
    blocks = [[] for _ in cells]; payloads = []
    for part in range(max((len(r) + CHUNK_BYTES - 1) // CHUNK_BYTES for r in records)):
        for k, data in enumerate(records):
            piece = data[part * CHUNK_BYTES:(part + 1) * CHUNK_BYTES]
            if piece:
                blocks[k].append(len(payloads)); payloads.append(piece)
    links = [(NONE, 0)] * len(payloads); descriptors = []
    for k, chain in enumerate(blocks):
        descriptors.append((chain[0] if chain else NONE, chain[-1] if chain else NONE, len(records[k]) // 84))
        for i, b in enumerate(chain):
            links[b] = (chain[i + 1] if i + 1 < len(chain) else NONE, len(payloads[b]) // 84)
    path = root / 'live/fission.codes'; temporary = root / 'live/fixture.codes'
    with temporary.open('wb') as f:
        for data in payloads:
            f.write(data); f.write(bytes(CHUNK_BYTES - len(data)))
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)
    st = path.stat(); h[1] = 1; h[6] = st.st_dev; h[7] = st.st_ino; h[12] = len(links)
    state = HEADER.pack(*h) + b''.join(struct.pack('<III', *c) for c in descriptors)
    state += representatives + b''.join(struct.pack('<II', *c) for c in links)
    (root / 'live/fission.state').write_bytes(state + checksum(state))
    assert len(links) > len(cells) * 2, 'fixture must expose the old read amplification'
    return read_layout(root)
