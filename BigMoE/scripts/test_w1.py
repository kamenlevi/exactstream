"""Wall 1, lever 1: model-free test of ext/build_w1 (Leap's W1-PICKUP patch + S0 instrumentation + the direct path),
CPU only. Two layers like the 45 GB tier pools (a 9-piece MSB-3 part and a 6-piece LSB-1 part of one 16 KB-aligned record,
the pieces at today's sizes scaled down), E = 24 experts, C = 6 slots, driven through ext.ensure() like the loader, in two
processes: the default bounce path, BIGMOE_RECORDS_DIRECT=1 and BIGMOE_COPY_PAR=1 (switches are read once per process). After every call
each resident slot must hold exactly its record part, piece by piece. With BIGMOE_TRACE=1 every read event carries
t0 <= tp0 <= tp1 <= t1 (issue, pread start / end, bytes in place), on both paths.
usage: .venv/bin/python scripts/test_w1.py [BUILD=build_w1]"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = sys.argv[1] if len(sys.argv) > 1 else "build_w1"


def child(build, direct):
    sys.path.insert(0, os.path.join(HERE, "..", "ext", build))
    import numpy as np
    import mlx.core as mx
    import bigmoe_ext as ext
    E, C = 24, 6
    MSB = [61440, 5120, 5120] * 3                        # w3 / s3 / b3 per projection (today's 614,400 / 51,200 / 51,200 / 10)
    LSB = [20480, 5120] * 3                              # l1 / b per projection
    msb_sz, lsb_sz = 16384 * 14, 16384 * 5               # 16 KB-aligned parts (>= the pieces: 215,040 / 76,800 B)
    rec = msb_sz + lsb_sz
    rng = np.random.default_rng(3)
    recs = rng.integers(0, 256, size=(E, rec), dtype=np.uint8)
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    path = os.path.join(d, "L00.bin"); recs.tofile(path)
    layers = []
    LSBP = [20480] * 3 + [5120] * 3                      # native+p11+pad's LSB order: the aligned planes first, then the b's
    for pers, off, ln, split in ((MSB, 0, msb_sz, None), (LSB, msb_sz, lsb_sz, None), (LSBP, msb_sz, lsb_sz, 3)):
        pools = [mx.zeros((C, p), mx.uint8) for p in pers]
        mx.eval(pools)
        L = ext.add_layer(C, E, [-1] * len(pers), [0] * len(pers), pers, pools)
        ext.set_records_part(L, path, rec, off, ln)
        if split is not None:
            ext.set_direct_split(L, split)              # with BIGMOE_RECORDS_DIRECT: 3 pieces straight in, 3 via a bounce
        layers.append((L, pools, pers, off))
    n = 0
    for step in range(60):
        ids = [int(x) for x in rng.choice(E, rng.integers(1, C + 1), replace=False)]
        for L, pools, pers, off in layers:
            ext.ensure(L, mx.array(ids, dtype=mx.int32))
            host = [np.array(p) for p in pools]
            for e, s in enumerate(ext.lookup(L, list(range(E)))):
                if s < 0:
                    continue
                o = off
                for h, per in zip(host, pers):
                    assert np.array_equal(h[s], recs[e, o:o + per]), (direct, step, L, e, s)
                    o += per
                n += 1
    rd = ext.trace_reads()
    bad = [r for r in rd if not (len(r) == 7 and r[3] <= r[5] <= r[6] <= r[4])]
    assert rd and not bad, (len(rd), bad[:3])
    st = ext.stats()
    print(json.dumps(dict(direct=direct, checks=n, reads=len(rd), misses=st["misses"],
                          pread_us=round(1e6 * float(np.median([r[6] - r[5] for r in rd])), 1),
                          read_to_place_us=round(1e6 * float(np.median([r[4] - r[3] for r in rd])), 1))))


if __name__ == "__main__":
    if os.environ.get("W1_CHILD"):
        child(os.environ["W1_CHILD"], os.environ.get("BIGMOE_RECORDS_DIRECT") is not None or
              ("copy_par" if os.environ.get("BIGMOE_COPY_PAR") else False))
        sys.exit(0)
    out = []
    for direct in (False, True, "copy_par"):
        env = dict(os.environ, W1_CHILD=BUILD, BIGMOE_TRACE="1")
        env.pop("BIGMOE_RECORDS_DIRECT", None); env.pop("BIGMOE_COPY_PAR", None)
        if direct is True:
            env["BIGMOE_RECORDS_DIRECT"] = "1"
        elif direct == "copy_par":
            env["BIGMOE_COPY_PAR"] = "1"
        p = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[-600:])
        assert p.returncode == 0, f"direct={direct} failed"
        out.append(json.loads(p.stdout.strip().splitlines()[-1]))
    print(f"PASS: bounce, direct and bounce + COPY_PAR keep every slot == its record part ({out[0]['checks']} / "
          f"{out[1]['checks']} / {out[2]['checks']} slot checks); every read event has t0 <= tp0 <= tp1 <= t1")
