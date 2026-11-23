"""STEP 0 PART B read-out (Leap, Fri 2026-11-06 08:0x): the kernel-level records of a stepb run -> per piece and per
kernel type: calls per check, GPU ms, bytes, GB/s, and the size-matched read floor (the same synced method, so the
method's bias cancels in the ratio).
usage: [any python3 with numpy] stepb_read.py out/stepb_fri_224_flat2.json [more.json] > report.txt
Per record GPU ms = the synced eval minus one round trip (stepb_k.gpu_ms); glue = the flush before a call or at a piece's
end that found GPU work. A check = one kernel-level forward (all 48 layers); per-check numbers are medians over the
forwards of a row class (2 = the live check's rows, 16 = the stashed block at 16 rows)."""
import json
import sys
from collections import defaultdict

import numpy as np

PIECES = ["gdn", "att", "hc_a", "hc_m", "hc_f", "hc_w", "moe", "experts", "ple", "layer", "-"]


def gpu(ms, sync):
    return 0.0 if ms < 0.5 * sync else max(0.0, ms - sync)


def floor_us(nbytes, rd, sync, key="cold_ms"):
    """the read floor's GPU us at nbytes, interpolated in log-log over the calibration sizes"""
    x = np.array([r["kb"] * 1024.0 for r in rd])
    y = np.array([max(1e-3, (r[key] - sync) * 1e3) for r in rd])
    if nbytes <= x[0]:
        return float(y[0])
    if nbytes >= x[-1]:
        return float(y[-1] * nbytes / x[-1])
    return float(np.exp(np.interp(np.log(nbytes), np.log(x), np.log(y))))


def load(paths):
    fw = defaultdict(list)                               # row class -> [forward records]
    meta = None
    for p in paths:
        d = json.load(open(p))
        meta = meta or d["meta"]
        for r_ in d["runs"]:
            for pr in r_.get("probes") or []:
                for key, cls in (("k_real", "2"), ("k_16", "16")):
                    for kr in pr.get(key) or []:
                        if kr.get("krec"):
                            fw[cls].append(dict(check=pr["check"], rows=kr["rows"], wall=kr["wall"], equal=kr.get("equal"),
                                                skip=kr.get("kskip"), rec=kr["krec"]))
    return meta, fw


def main(paths):
    meta, fw = load(paths)
    cal = meta.get("stepb_cal") or {}
    sync = cal.get("sync_ms", meta.get("sync_ms", 0.13))
    rd = cal.get("read") or []
    print(f"STEP 0 PART B read-out: {paths}; tag {meta.get('tag')}, C {meta.get('C')}, arm {meta.get('arm')}, FAST {meta.get('fast')}")
    print(f"  calibration: round trip {sync * 1e3:.0f} us (p10-p90 {[round(v * 1e3) for v in cal.get('sync_p10_p90', [])]}), "
          f"dependent dispatch {cal.get('dispatch_us', float('nan')):.2f} us")
    if rd:
        print("  read floor (same synced method; KB: cold us / GB/s, warm us / GB/s): " + "; ".join(
            f"{r['kb']}: {(r['cold_ms'] - sync) * 1e3:.1f} / {r['cold_gbs']:.0f}, {(r['warm_ms'] - sync) * 1e3:.1f} / {r['warm_gbs']:.0f}"
            for r in rd))
    for cls in ("2", "16"):
        F = fw.get(cls) or []
        if not F:
            continue
        print(f"\n== {cls} rows: {len(F)} kernel-level forwards (checks {sorted(set(f['check'] for f in F))}; rows "
              f"{sorted(set(f['rows'] for f in F))}); logits == live {[f['equal'] for f in F]}; skipped calls "
              f"{[f['skip'] for f in F]}")
        per = defaultdict(lambda: defaultdict(list))     # piece -> metric -> per-forward values
        ops = defaultdict(lambda: dict(us=[], b=[], n=0, fl=set()))
        exl = []
        for f in F:
            acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])   # calls, call ms, glue ms, bytes, glue flushes
            for ctx, L, name, sig, b, fl, pre, bld, ev in f["rec"]:
                a = acc[ctx]
                g = gpu(pre, sync)
                if name == "glue_end":
                    a[2] += g; a[4] += g > 0
                    continue
                if name == "experts_inner_calls":
                    continue
                a[2] += g; a[4] += g > 0
                if name == "experts_unit":
                    exl.append((L, gpu(ev, sync), b, sig))
                a[0] += 1
                a[1] += gpu(ev, sync)
                if "P" not in fl:
                    a[3] += b
                o = ops[(ctx, name, sig)]
                o["us"].append(1e3 * gpu(ev, sync)); o["b"].append(b); o["n"] += 1; o["fl"].add(fl)
            for ctx, a in acc.items():
                for k_, v_ in zip(("calls", "ms", "glue", "bytes", "flushes"), a):
                    per[ctx][k_].append(v_)
        nF = len(F)
        tot_ms = sum(np.median(np.array(per[c]["ms"]) + np.array(per[c]["glue"])) for c in per)
        tot_b = sum(np.median(per[c]["bytes"]) for c in per if c != "experts")
        print(f"  per check (medians over forwards): GPU {tot_ms:.2f} ms in wrapped calls + glue; dense bytes (all but "
              f"experts, pools not counted) {tot_b / 1e9:.3f} GB")
        print("  piece     calls  glue-fl   call ms   glue ms   total ms      MB    GB/s(calls)")
        for c in sorted(per, key=lambda c: -np.median(np.array(per[c]["ms"]) + np.array(per[c]["glue"]))):
            m = per[c]
            ms, gl, b = np.median(m["ms"]), np.median(m["glue"]), np.median(m["bytes"])
            print(f"  {c:8s} {np.median(m['calls']):6.0f} {np.median(m['flushes']):7.0f} {ms:9.2f} {gl:9.2f} {ms + gl:10.2f} "
                  f"{b / 1e6:8.1f} {b / 1e6 / max(1e-6, ms):8.0f}")
        print(f"  kernel types by GPU ms per check (top 40): piece | name | signature | calls per check | median us per "
              f"call | KB per call | GB/s | read floor us (cold) | x floor")
        rows_ = []
        for (ctx, name, sig), o in ops.items():
            if name == "experts_unit":
                continue
            us = float(np.median(o["us"]))
            b = float(np.median(o["b"]))
            per_check = o["n"] / nF
            fl_ = floor_us(b, rd, sync) if rd and b > 0 and "P" not in o["fl"] else float("nan")
            rows_.append((per_check * us / 1e3, ctx, name, sig, per_check, us, b, fl_, o["fl"]))
        rows_.sort(key=lambda r: -r[0])
        for ms_c, ctx, name, sig, pc, us, b, fl_, fls in rows_[:40]:
            gbs = b / max(1e-9, us * 1e-6) / 1e9
            print(f"  {ms_c:6.2f} ms | {ctx:7s} | {name[:34]:34s} | {sig[:70]:70s} | {pc:5.1f} | {us:7.1f} | {b / 1024:8.1f} | "
                  f"{gbs:6.0f} | {fl_:6.1f} | {us / fl_ if fl_ == fl_ and fl_ > 0 else float('nan'):5.1f}{' ' + ''.join(sorted(fls)) if any(fls) else ''}")
        n_types = len(rows_)
        small = sum(r[0] for r in rows_ if r[5] < 20)
        print(f"  {n_types} kernel types; calls under 20 us take {small:.2f} ms per check")
        if exl:
            per_l = defaultdict(list)
            for L, ms, b, sig in exl:
                per_l[L].append((ms, b, sig))
            ms_l = [np.median([x[0] for x in v]) for v in per_l.values()]
            b_l = [np.median([x[1] for x in v]) for v in per_l.values()]
            tot_e, tot_eb = float(np.sum(ms_l)), float(np.sum(b_l))
            print(f"  experts (unit per layer, {len(per_l)} layers): {tot_e:.2f} ms per check, bytes (upper bound: distinct "
                  f"experts x MSB-3 record + ranks 0-1 x LSB) {tot_eb / 1e9:.3f} GB -> {tot_eb / 1e6 / max(1e-6, tot_e):.0f} GB/s; "
                  f"per layer median {np.median(ms_l) * 1e3:.0f} us, {np.median(b_l) / 1e6:.1f} MB; example {exl[0][3]}")
            if rd:
                fl_e = sum(floor_us(b, rd, sync) for b in b_l) / 1e3
                print(f"    read floor for those bytes, layer by layer (cold): {fl_e:.2f} ms per check -> experts at "
                      f"{tot_e / max(1e-6, fl_e):.2f}x the floor")


VIEWS = {"Broadcast", "Reshape", "Transpose", "ExpandDims", "Squeeze", "AsStrided", "StopGradient", "Flatten",
         "Unflatten", "Split", "Slice", "Depends", "NumberOfElements", "Shape"}


def fitline(pts):
    """least squares t_us = t_sync + bytes / B over (bytes, us) points; returns (t_sync us, B GB/s) or None"""
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    A = np.vstack([np.ones_like(x), x]).T
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (float(c0), (1.0 / c1) / 1e3 if c1 > 0 else float("inf"))


def extra(paths):
    meta = None
    rep = {"2": [], "16": []}
    dots = {"2": [], "16": []}
    for p in paths:
        d = json.load(open(p))
        meta = meta or d["meta"]
        for r_ in d["runs"]:
            for pr in r_.get("probes") or []:
                for key, cls in (("k_real", "2"), ("k_16", "16")):
                    for kr in pr.get(key) or []:
                        if kr.get("replay"):
                            rep[cls] = kr["replay"]
                if pr.get("dot"):
                    dots["2"] = pr["dot"]
                if pr.get("dot16"):
                    dots["16"] = pr["dot16"]
    for p in paths:                                      # the drafter (STEPB_DRAFT=1)
        d = json.load(open(p))
        for r_ in d["runs"]:
            for pr in r_.get("probes") or []:
                dr = pr.get("draft")
                if not dr:
                    continue
                if dr.get("error"):
                    print(f"\n== drafter at check {pr['check']}: ERROR {dr['error']}")
                    continue
                R = [r for r in dr.get("replay") or [] if "us_chain" in r]
                tot = sum((r["us_chain"] - r["us_null"]) * r["n"] for r in R) / 1e3
                print(f"\n== drafter at check {pr['check']}: propose walls {[round(w, 2) for w in dr.get('wall', [])]} ms; "
                      f"{len(dr.get('krec') or [])} wrapped calls; replayed kinds (>= 3 calls) sum {tot:.2f} ms")
                for r in sorted(R, key=lambda r: -(r["us_chain"] - r["us_null"]) * r["n"])[:25]:
                    us = r["us_chain"] - r["us_null"]
                    print(f"  {r['name'][:34]:34s} | {r['sig'][:70]:70s} | {r['n']:3d} | {r['bytes'] / 1024:8.1f} KB | "
                          f"{us:7.1f} us | {r['bytes'] / max(1e-9, us * 1e-6) / 1e9:6.0f} GB/s")
                from collections import Counter
                cnt = Counter(x[2] for x in dr.get("krec") or [])
                print("  wrapped calls by name: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common(14)))
    cal = meta.get("stepb_cal") or {}
    if cal.get("sync_chains"):
        print("\n== t_sync: dependent chains of a kernel with almost no bytes (threads x threadgroup: us per kernel)")
        print("  " + "; ".join(f"{c['threads']} x {c['tg']}: {c['us']:.2f}" for c in cal["sync_chains"]))
    for cls in ("2", "16"):
        R = [r for r in rep[cls] if "us_chain" in r]
        if not R:
            continue
        print(f"\n== {cls} rows: dependent-chain replay of the engine's own calls across layers (cold weights; per call = "
              f"chain - null links): piece | name | signature | n | KB | us per call | GB/s")
        for r in sorted(R, key=lambda r: -(r["us_chain"] - r["us_null"]) * r["n"]):
            us = r["us_chain"] - r["us_null"]
            print(f"  {r['ctx']:7s} | {r['name'][:34]:34s} | {r['sig'][:64]:64s} | {r['n']:3d} | {r['bytes'] / 1024:8.1f} | "
                  f"{us:7.1f} | {r['bytes'] / max(1e-9, us * 1e-6) / 1e9:6.0f}{' ' + r['flags'] if r.get('flags') else ''}")
        for fam, test in (("4-bit / weight GEMV-like (quantized_matmul, qmm*)", lambda r: "quantized_matmul" in r["name"] or "qmm" in r["name"].lower()),
                          ("HC kernels (names with hc)", lambda r: "hc" in r["name"].lower()),
                          ("all calls >= 256 KB", lambda r: r["bytes"] >= 256 * 1024)):
            pts = [(r["bytes"], r["us_chain"] - r["us_null"]) for r in R if test(r) and "P" not in (r.get("flags") or "")]
            f = fitline(pts)
            if f:
                print(f"  fit t = t_sync + bytes / B, {fam}: {len(pts)} kernel types -> t_sync {f[0]:.1f} us, B {f[1]:.0f} GB/s")
        bad = [r for r in rep[cls] if "error" in r]
        if bad:
            print(f"  replay errors: {[(r['name'], r['error'][:80]) for r in bad][:6]}")
    for cls in ("2", "16"):
        D = dots[cls]
        if not D:
            continue
        print(f"\n== {cls} rows: primitives per piece from the graph (mx.export_to_dot, one GDN and one attention layer; "
              f"views = {len(VIEWS)} no-kernel labels)")
        for kind, L, rows, cnt in D:
            if "error" in cnt:
                print(f"  L{L} {kind}: error {cnt['error']}")
                continue
            tot = sum(cnt.values()); nv = sum(v for k, v in cnt.items() if k not in VIEWS)
            top = ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if k not in VIEWS)
            print(f"  L{L:2d} {kind:8s}: {nv:3d} kernel primitives ({tot} with views): {top[:200]}")


if __name__ == "__main__":
    main(sys.argv[1:])
    extra(sys.argv[1:])
