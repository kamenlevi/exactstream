"""P3 unit test (Wed 11 Nov, no model load): PF-D's staging with R57_PFD_FOLD's one-eval allocation (stage_many) holds the
same bytes as the per-array path (stage_reads) and as a plain read of the records, for B (LSB parts) and C (whole records)
rows of three layers, with spare rows; then gather() on a stage_many stage fills a miss into a spare row like stage_reads'.
A few experts per layer, a few MB. Exit 0 only if every byte matches.
usage: ~/BigMoE/.venv/bin/python scripts/pfd_fold_unit.py [n_experts_per_layer]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser(f"~/BigMoE/ext/{os.environ.get('BIGMOE_EXT_BUILD', 'build_qom')}"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))           # install_pfd imports bigmoe.stream
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import pflm  # noqa: E402

D = os.path.expanduser("~/BigMoE/models/bare/records3")
ix = json.load(open(os.path.join(D, "index.json")))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = np.random.default_rng(1111)


class _Tier:                                             # install_pfd only wraps prefill_call; the staging needs nothing else
    def prefill_call(self, *a, **kw):
        raise AssertionError("not called")


tier = _Tier()
pflm.install_pfd(tier)
dbg = tier._pfd_dbg
bad, rows = 0, 0


def done(st):
    for f_ in st[2].values():
        f_.result()
    return np.asarray(memoryview(st[0])).view(np.uint8).reshape(int(st[0].shape[0]), -1)


for L in (0, 23, 47):
    r = ix["layers"][str(L)]
    t = dict(path=os.path.join(D, r["file"]), recsz=int(r["recsz"]), msb_sz=int(r["msb_sz"]))
    m, rec = t["msb_sz"], t["recsz"]
    idsB = [int(e) for e in np.sort(rng.choice(int(r["E"]), n, replace=False))]
    idsC = [int(e) for e in np.sort(rng.choice(int(r["E"]), n + 1, replace=False))]
    want = [("B", idsB, m, rec - m, max(8, n // 2)), ("C", idsC, 0, rec, max(8, (n + 1) // 2))]
    many = dbg["stage_many"](t, want)
    for tag, ids_, off, size, sp in want:
        one = dbg["stage_reads"](t, ids_, off, size, sp)
        a, b = done(many[tag]), done(one)
        ref = pflm.read_parts(dict(t, layout=2), np.array(ids_), off, size)
        ref = np.asarray(ref).view(np.uint8).reshape(len(ids_), size) if not isinstance(ref, np.ndarray) else ref.view(np.uint8).reshape(len(ids_), size)
        ok = a.shape == b.shape and np.array_equal(a[:len(ids_)], b[:len(ids_)]) and np.array_equal(a[:len(ids_)], ref)
        ok = ok and not a[len(ids_):].any()                  # spare rows still zero
        bad += 0 if ok else 1
        rows += len(ids_)
        print(f"layer {L:2d} {tag}: {len(ids_)} rows of {size} bytes + {sp} spare: "
              f"{'== stage_reads == the records' if ok else 'MISMATCH'}")
    # a miss on a stage_many stage: gather reads it into a spare row (as on a stage_reads stage)
    extra = next(int(e) for e in range(int(r["E"])) if e not in set(idsC))
    arr, rowsix = dbg["gather"](many["C"], t, idsC[:2] + [extra], 0, rec, "C")
    got = np.asarray(memoryview(arr)).view(np.uint8).reshape(int(arr.shape[0]), rec)[rowsix]
    ref = pflm.read_parts(dict(t, layout=2), np.array(idsC[:2] + [extra]), 0, rec)
    ref = np.asarray(ref).view(np.uint8).reshape(3, rec) if not isinstance(ref, np.ndarray) else ref.view(np.uint8).reshape(3, rec)
    ok = np.array_equal(got, ref) and int(rowsix[-1]) == len(idsC)
    bad += 0 if ok else 1
    print(f"layer {L:2d} C gather with a miss (row {int(rowsix[-1])}): {'== the records' if ok else 'MISMATCH'}")
print(f"PFD FOLD UNIT: {'PASS' if bad == 0 else 'FAIL'} ({bad} mismatches; {rows} staged rows over 3 layers + 3 gathers)")
sys.exit(0 if bad == 0 else 1)
