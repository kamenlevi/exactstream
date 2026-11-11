"""Direct-pread row gather for MTPLX's _SidecarGather decode miss path (drop-in for the warm + fancy-index pair).
Each pool task preads one row from every map and returns the bytes, so the rows never go through the page cache
twice and the generation thread never touches the memmaps. The bytes come from the same file offsets that the
memmap would read, so the result is identical by construction (and checked in bench_gather.py)."""
import os
import numpy as np


def direct_rows(sg, rows, names):
    """rows: int64 array of row ids (unique or not). Returns {name: (len(rows), cols) array}, like
    {n: sg._maps[n][0][rows]}. Planar layout only (one record per map, in map order)."""
    fd = sg._fd
    metas = dict(zip(sg._maps, sg._row_meta))

    def read(r):
        return [os.pread(fd, metas[n][1], metas[n][0] + int(r) * metas[n][1]) for n in names]

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if sg._pool is not None and len(rows) > 1:
        parts = list(sg._pool.map(read, rows))
    else:
        parts = [read(r) for r in rows]
    out = {}
    for j, n in enumerate(names):
        mm = sg._maps[n][0]
        buf = b"".join(p[j] for p in parts)
        out[n] = np.frombuffer(buf, dtype=mm.dtype).reshape(len(rows), mm.shape[1])
    return out
