"""S10c (b33): the miss-kind split of an S10b trace, a cross-check against the ext's own counters, and whether the
passes reproduce another trace (b32: then b32's files and b33's labels share the (seq, L) keys). CPU only (numpy).
miss.bin entries are e | miss << 16 | kind << 17, kind 0 demand (or resident), 1 on time, 2 late, 3 queued (cancelled to
demand); see ext/bigmoe_ext.cpp s10b_rec. Verify passes and first calls as in s10b_check.py.
usage: s10b_labels.py B33_TRACE_DIR [B32_TRACE_DIR]"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json")))
NL = meta["n_layers"]
raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
first, allk, i = {}, np.zeros(5, np.int64), 0               # allk: [ids, misses, on time, late, queued] over all records
while i < len(raw):
    seq, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
    v = raw[i + 3:i + 3 + n]
    m, k = (v >> 16) & 1, (v >> 17) & 3
    allk += [n, m.sum(), ((m == 1) & (k == 1)).sum(), ((m == 1) & (k == 2)).sum(), ((m == 1) & (k == 3)).sum()]
    first.setdefault((seq, L), (m, k))
    i += 3 + n


def first_calls(f):
    z = np.load(f)
    S, Ls, rows, off, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    by = {}
    for j in range(len(S)):
        by.setdefault(int(S[j]), {}).setdefault(int(Ls[j]), j)
    return {s: {L: ids[off[c[L]]:off[c[L]] + int(rows[c[L]])] for L in range(NL)}
            for s, c in by.items() if all(L in c for L in range(NL)) and len({int(rows[c[L]]) for L in range(NL)}) == 1}


tot, same, diff, gone = np.zeros(6, np.int64), 0, 0, 0     # tot: [pass-layers, ids, on time, late, queued, demand]
for f in sorted(glob.glob(os.path.join(D, "p*.npz"))):
    P = first_calls(f)
    for s, lay in P.items():
        for L in range(NL):
            if (s, L) in first:
                m, k = first[(s, L)]
                tot += [1, len(m), ((m == 1) & (k == 1)).sum(), ((m == 1) & (k == 2)).sum(),
                        ((m == 1) & (k == 3)).sum(), ((m == 1) & (k == 0)).sum()]
    if len(sys.argv) > 2:                                    # same passes, same routing as the other trace?
        g = os.path.join(sys.argv[2], os.path.basename(f))
        Q = first_calls(g) if os.path.exists(g) else {}
        for s, lay in P.items():
            if s not in Q:
                gone += 1
            elif all(np.array_equal(lay[L], Q[s][L]) for L in range(NL)):
                same += 1
            else:
                diff += 1
n, ms = max(tot[0], 1), max(tot[2:].sum(), 1)
print(f"verify pass-layers {tot[0]}: per pass-layer {tot[1] / n:.2f} unique routed, {tot[2:].sum() / n:.2f} misses = "
      f"on time {tot[2] / n:.2f} ({100 * tot[2] / ms:.1f} %) + late {tot[3] / n:.2f} ({100 * tot[3] / ms:.1f} %) + "
      f"queued {tot[4] / n:.2f} ({100 * tot[4] / ms:.1f} %) + demand {tot[5] / n:.2f} ({100 * tot[5] / ms:.1f} %)")
st = meta.get("ext_stats", {})
if st:
    print(f"all records vs the ext's counters: misses {allk[1]} vs {st.get('misses')}; on time {allk[2]} vs pf_ontime "
          f"{st.get('pf_ontime')}; late {allk[3]} vs pf_late {st.get('pf_late')}; queued {allk[4]} vs pf_qdemand "
          f"{st.get('pf_qdemand')}; demand incl. queued {allk[1] - allk[2] - allk[3]} vs miss_pred_unissued + miss_unpred "
          f"{st.get('miss_pred_unissued', 0) + st.get('miss_unpred', 0)}")
else:
    print("no ext_stats in meta.json:", meta.get("ext_stats_error"))
if len(sys.argv) > 2:
    print(f"passes vs {sys.argv[2]}: {same} identical routing at every layer, {diff} differ, {gone} missing there")
