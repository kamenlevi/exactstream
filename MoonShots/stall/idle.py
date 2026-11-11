"""SSD idle windows per layer from BigMoE's ext trace (anatomy.py with STALL_TRACE=1), and the best case for a look-ahead
prefetch that fills them. Per verify pass: per model layer, the gate events of the base pool (ext layer L) and the LSB sidecar
(49+L) are merged; a call with misses (kind 4) keeps the SSD busy from t_seen to t_rel; the window between layer L's release
and layer L+1's routing is SSD-idle (the GPU runs L's MoE and L+1's attention). Reads per busy call are estimated from the
SSD model (0.191 ms + n x 2.16 MB at 13.1 GB/s). Look-ahead: predicted reads for L+1 are issued at L's release and must land
by L+1's routing (no contention with demand reads); a top-k list has precision p(k) (NewIdeas' D1C capture: top-2 0.95,
top-3 0.93, top-4 0.90, top-8 0.76, top-10 0.69); each useful read removes one demand read from L+1's batch.
usage: idle.py [json glob, default data/stall/anatomy_*_trace.json]"""
import glob, json, os, sys
import numpy as np

LAT, BW, MB = 0.191, 13.1, 2.162688                          # ms, GB/s, MB per MSB-3 read
RD = MB / BW                                                  # ms per read at full bandwidth
PK = {1: 0.95, 2: 0.95, 3: 0.93, 4: 0.90, 8: 0.76, 10: 0.69, 16: 0.51}
pk = lambda k: float(np.interp(k, sorted(PK), [PK[j] for j in sorted(PK)])) if k > 0 else 0.0
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/MoonShots/data/stall/anatomy_*_trace.json")))
for f in files:
    r = json.load(open(f))
    ev = np.array(r["ext_trace"] or [])
    if not len(ev):
        print(f, "no trace"); continue
    L, seq, kind, t0, t1 = ev[:, 0].astype(int), ev[:, 1].astype(int), ev[:, 2].astype(int), ev[:, 3] * 1e3, ev[:, 4] * 1e3
    res = dict(passes=0, span=[], busy=[], idle=[], save={}, nreads=[], caps=[])
    for s in np.unique(seq):
        m = (seq == s) & (kind == 1)
        ml = L[m] % 49                                        # base L and sidecar 49+L -> model layer
        if len(np.unique(ml[ml < 48])) < 40:                  # MTP-only passes (layer 48) and partial passes: skip
            continue
        slow = set(zip(L[(seq == s) & (kind == 4)], np.round(t0[(seq == s) & (kind == 4)], 6)))
        lay = {}
        for li, a, b, raw in zip(ml, t0[m], t1[m], L[m]):
            if li >= 48:
                continue
            e = lay.setdefault(li, [a, b, 0.0])
            e[0], e[1] = min(e[0], a), max(e[1], b)
            if (raw, round(a, 6)) in slow:
                e[2] = max(e[2], b - a)                       # SSD busy time of this layer's reads
        ls = sorted(lay)
        busy = np.array([lay[i][2] for i in ls])
        nread = np.maximum(0, np.round((busy - LAT) / RD)) * (busy > 0)
        gaps = np.array([max(0.0, lay[ls[j + 1]][0] - lay[ls[j]][1]) for j in range(len(ls) - 1)])
        res["passes"] += 1
        res["span"].append(lay[ls[-1]][1] - lay[ls[0]][0]); res["busy"].append(busy.sum()); res["idle"].append(gaps.sum())
        res["nreads"].append(nread.sum())
        cap = np.floor(np.maximum(0, gaps - LAT) / RD).astype(int)   # reads that fit in the window before L+1's routing
        res["caps"] += cap.tolist()
        for kmax in (2, 3, 4, 8, 10):
            sv = 0.0
            for j in range(len(gaps)):
                n = nread[j + 1]
                if n <= 0:
                    continue
                k = min(cap[j], kmax)
                u = min(int(round(pk(k) * k)), int(n))
                b_old = busy[j + 1]
                b_new = LAT + (n - u) * RD if n - u > 0 else 0.0
                sv += max(0.0, b_old - b_new)
            res["save"].setdefault(kmax, []).append(sv)
    if not res["passes"]:
        print(f, "no full passes"); continue
    sp, bu, idl = np.mean(res["span"]), np.mean(res["busy"]), np.mean(res["idle"])
    caps = np.array(res["caps"])
    tag = os.path.basename(f).replace("anatomy_", "").replace(".json", "")
    tok = r["summary"]
    passes_ms = np.mean([p["t_verify"] for p in r["passes"]]) * 1e3
    print(f"== {tag}: {res['passes']} verify passes, verify {passes_ms:.1f} ms/pass; layer span {sp:.1f} ms = SSD busy {bu:.1f} "
          f"(~{np.mean(res['nreads']):.0f} demand reads) + idle windows {idl:.1f} ms between layers")
    print(f"   idle windows: median {np.median(caps):.0f} reads of capacity; {100 * np.mean(caps >= 1):.0f} % fit >= 1 read, "
          f"{100 * np.mean(caps >= 3):.0f} % >= 3, {100 * np.mean(caps >= 6):.0f} % >= 6")
    for kmax, sv in sorted(res["save"].items()):
        s_ = np.mean(sv)
        print(f"   look-ahead top-{kmax:<2d} per layer (precision {pk(kmax):.2f}): saves {s_:.1f} ms/pass -> {passes_ms - s_:.1f} ms "
              f"({passes_ms / max(1e-9, passes_ms - s_):.2f}x)")
