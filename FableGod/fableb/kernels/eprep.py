"""eprep.py (Fable B, DENSE_REBUILD v2, Fri 2026-11-06): DRAFT of the expert id / mask glue as ONE kernel. NOT RUN ON A GPU.
It replaces the tiny integer ops ref2._ooo_1c_call builds before its posts (astype, where, |, &, ~, >=, zeros_like,
the nolsb select: 4-12 launches per layer, by arm) with one dispatch of M x K threads. Integer and select logic only, so it
is bitwise == the mx ops it replaces: usable in the exact arm too.
Per use (row m, rank r), with id = ids[m, r]:
  opt   = HAS_Q and qopt[m, r]                                  (QOM: an optional use)
  sdrop = HAS_SPLIT and opt and qsure[m] and cf[m] >= thr[0]    (QOM-SPLIT-1C: a strict row drops it)
  ids_k = sdrop ? ids[m, 0] : id                                (the kernel's ids)
  ids_p = (opt and not sdrop) ? ids_k | (1 << 21) : ids_k       (the loader's ids)
  sc_o  = sdrop ? 0 : sc[m, r]
  ids_s = (HAS_NL and nl[m]) ? ids[first row with nl false, 0] : id      for r < R4   (U3: sure rows post a duplicate)
What stays outside (Leap's call): the two posts, the slices of their results, and pre_l's -2 select for nolsb rows, which
belongs in kernel A's prologue (one compare per pair) rather than in its own launch.
Dtypes to confirm against the engine: qopt / qsure / nl bool, cf float32, sc as TS (template)."""
HDR = ""
SRC = r"""
    const uint i = thread_position_in_grid.x;
    const uint m = i / K_C, r = i % K_C;
    const int id = ids[i];
    const bool opt = HAS_Q && qopt[i];
    const bool sdrop = HAS_SPLIT && opt && qsure[m] && (cf[m] >= thr[0]);
    const int idk = sdrop ? ids[m * K_C] : id;
    ids_k[i] = idk;
    ids_p[i] = (opt && !sdrop) ? (idk | (1 << 21)) : idk;
    sc_o[i] = sdrop ? static_cast<TS>(0) : sc[i];
    if (r < (uint)R4) {
        int dup = id;
        if (HAS_NL && nl[m]) {
            uint u = 0;
            for (uint j = 0; j < (uint)M_C; j++) if (!nl[j]) { u = j; break; }
            dup = ids[u * K_C];
        }
        ids_s[m * R4 + r] = dup;
    }
"""
IN = ["ids", "qopt", "qsure", "cf", "thr", "nl", "sc"]
OUT = ["ids_p", "ids_k", "ids_s", "sc_o"]
TYPES = {"ids": "int", "qopt": "bool", "qsure": "bool", "cf": "float", "thr": "float", "nl": "bool", "sc": "TS",
         "ids_p": "int", "ids_k": "int", "ids_s": "int", "sc_o": "TS"}


def emulate(ids, qopt, qsure, cf, thr, nl, sc, R4):
    """the kernel's logic in numpy, and beside it ref2._ooo_1c_call's own expressions (returns both for comparison)"""
    import numpy as np
    M, K = ids.shape
    has_q, has_split, has_nl = qopt is not None, qopt is not None and thr is not None, nl is not None
    ids_k, ids_p, sc_o, ids_s = ids.copy(), ids.copy(), sc.copy(), ids[:, :R4].copy()
    for m in range(M):                                          # kernel form
        for r in range(K):
            opt = has_q and bool(qopt[m, r])
            sdrop = has_split and opt and bool(qsure[m]) and cf[m] >= thr
            idk = ids[m, 0] if sdrop else ids[m, r]
            ids_k[m, r] = idk
            ids_p[m, r] = (idk | (1 << 21)) if (opt and not sdrop) else idk
            sc_o[m, r] = 0 if sdrop else sc[m, r]
            if r < R4:
                if has_nl and nl[m]:
                    un = [j for j in range(M) if not nl[j]]
                    ids_s[m, r] = ids[un[0] if un else 0, 0]
                else:
                    ids_s[m, r] = ids[m, r]
    # ref2's expressions (ref2.py _ooo_1c_call, numpy for mx)
    if has_split:
        sd = qopt & (qsure & (cf >= thr))[:, None]
        rk = np.where(sd, ids[:, :1], ids)
        rp = np.where(qopt & ~sd, rk | (1 << 21), rk)
        rsc = np.where(sd, np.zeros_like(sc), sc)
    else:
        rk = ids
        rp = ids if not has_q else np.where(qopt, ids | (1 << 21), ids)
        rsc = sc
    if has_nl:
        un = np.where(~nl)[0]
        dup = ids[int(un[0]) if len(un) else 0, 0]
        rs = np.where(nl[:, None], dup, ids[:, :R4])
    else:
        rs = ids[:, :R4]
    return (ids_p, ids_k, ids_s, sc_o), (rp, rk, rs, rsc)


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(11)
    n = bad = 0
    for trial in range(4000):
        M, K, R4 = int(rng.integers(1, 17)), 10, int(rng.integers(0, 3))
        ids = rng.integers(0, 512, (M, K)).astype(np.int32)
        sc = rng.random((M, K)).astype(np.float32)
        mode = trial % 4
        qopt = None if mode == 0 else rng.random((M, K)) < 0.3
        thr = None if mode < 2 else float(rng.random())
        qsure = rng.random(M) < 0.5
        cf = rng.random(M).astype(np.float32)
        nl = None if mode == 1 else (rng.random(M) < (1.0 if trial % 17 == 0 else 0.4))
        a, b = emulate(ids, qopt, qsure, cf, thr, nl, sc, R4)
        n += 1
        bad += not all(np.array_equal(x, y) for x, y in zip(a, b))
    print("eprep logic vs ref2's expressions: %d cases, %d differ -> %s" % (n, bad, "PASS" if bad == 0 else "FAIL"))
