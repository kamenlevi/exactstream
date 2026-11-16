"""GAP read-out (Leap, Fri 2026-11-06 21:xx; the orchestrator 21:0x: where do the engine's ~13-16 ms of dense time beyond the
kernel chains go?). Joins a stepb run's passes (t_start / t_end) with gpu_util_sampler.sh's ioreg samples.
Per pass: mean GPU 'Device Utilization %' over the pass, the loop's per-check draft / check + commit / build / wait, rows.
Per probe: the 16-row synced split (sum of per-piece GPU ms) vs the 16-row unsynced warm wall (the sweep's 16).
For a 32-row arm: check + commit by row band (<= 16 vs 17-32).
usage: gap_read.py <flat2.json> <sampler.txt> [more json]"""
import json
import sys

import numpy as np


def med(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def main(argv):
    samp = np.loadtxt(argv[1]) if len(argv) > 1 else np.zeros((0, 2))
    for p in [argv[0]] + argv[2:]:
        d = json.load(open(p))
        print(f"== {p.split('/')[-1]} (C {d['meta']['C']}; sync {d['meta'].get('sync_ms', 0):.3f} ms)")
        for r in d["runs"]:
            pc = r["pc"]
            n = len(pc["ck"])
            if not n:
                continue
            t0, t1 = r.get("t_start"), r.get("t_end")
            u = float("nan")
            if t0 is not None and len(samp):
                m = (samp[:, 0] >= t0) & (samp[:, 0] <= t1) & (samp[:, 1] >= 0)
                u = float(samp[m, 1].mean()) if m.any() else float("nan")
            rows = pc["rows"]
            line = (f"  {r['label'][:44]:44s} {r['mode']:5s}: {n:3d} checks, rows {min(rows)}-{max(rows)}, kept {r['kept']:.2f}; "
                    f"draft {med(pc['drf']):.2f}, check+commit {med(pc['ck']):.2f} (build {med(pc['bld']):.2f}, wait "
                    f"{med(pc['wt']):.2f}), misses {sum(pc['miss']):.0f}; GPU busy {u:.0f} % over "
                    f"{(t1 - t0) if t0 else float('nan'):.1f} s")
            print(line)
            if "ck_lw" in pc and n > 1:                  # XPT (xpt.py): the Python thread per check, checks 2.. (medians)
                m_ = {k: med(pc[k][1:]) for k in pc if k[:3] in ("ck_", "cm_", "cy_")}
                xn = max(1.0, m_["ck_xn"])
                print(f"      XPT check forward ({m_['ck_rows']:.0f} rows): 48 layer calls wall {m_['ck_lw']:.2f} / CPU {m_['ck_lc']:.2f} ms "
                      f"({m_['ck_lw'] / 48:.3f} / {m_['ck_lc'] / 48:.3f} per layer); inside them the expert calls wall {m_['ck_xw']:.2f} / CPU "
                      f"{m_['ck_xc']:.2f} ms ({xn:.0f} calls: {m_['ck_xw'] / xn:.3f} / {m_['ck_xc'] / xn:.3f} per call), the rest (dense + glue) "
                      f"wall {m_['ck_lw'] - m_['ck_xw']:.2f} / CPU {m_['ck_lc'] - m_['ck_xc']:.2f}; async_eval in its loop wall "
                      f"{m_['ck_aw']:.2f} / CPU {m_['ck_ac']:.2f} ms; engine build {med(pc['bld'][1:]):.2f}")
                print(f"      XPT cycle: the Python thread's CPU {m_.get('cy_cpu', float('nan')):.2f} ms of {m_.get('cy_wall', float('nan')):.2f} ms "
                      f"wall (draft + check + commit + loop): the floor if the GPU and every wait were free")
                print(f"      XPT commit ({m_['cm_f']:.0f} forwards): layers wall {m_['cm_lw']:.2f} / CPU {m_['cm_lc']:.2f}, experts wall "
                      f"{m_['cm_xw']:.2f} / CPU {m_['cm_xc']:.2f}, async_eval wall {m_['cm_aw']:.2f} / CPU {m_['cm_ac']:.2f}; every "
                      f"async_eval of the cycle wall {m_['cy_aw']:.2f} / CPU {m_['cy_ac']:.2f} ms")
            if max(rows) > 16:
                lo = [pc["ck"][i] for i in range(n) if rows[i] <= 16]
                hi = [pc["ck"][i] for i in range(n) if rows[i] > 16]
                print(f"      check+commit by rows: <= 16: {med(lo):.2f} ms (n {len(lo)}), 17-32: {med(hi):.2f} ms (n {len(hi)}); "
                      f"rows of the 17-32 checks {sorted(set(r_ for r_ in rows if r_ > 16))}")
            for pr in r.get("probes") or []:
                ab = pr.get("abl")
                if ab:                                   # S2b step 1: the in-situ price per dense piece (flat2_probe._abl_probe)
                    meta = ab.get("_meta") or {}

                    def busy(wins):
                        if not len(samp):
                            return float("nan")
                        m = np.zeros(len(samp), bool)
                        for a_, b_ in wins:
                            m |= (samp[:, 0] >= a_) & (samp[:, 0] <= b_)
                        m &= samp[:, 1] >= 0
                        return float(samp[m, 1].mean()) if m.any() else float("nan")

                    print(f"    ABL (S2b step 1) at check {pr['check']}: {meta.get('rows')} rows dense-only, {meta.get('rounds')} rounds x "
                          f"{meta.get('n')} forwards per variant; GDN {meta.get('gdn')} / attention {meta.get('att')} layers, in_proj objects "
                          f"{meta.get('inproj')}, PLE {meta.get('ple')}{'; ERROR ' + meta['error'] if meta.get('error') else ''}")
                    bw = med(ab["base"]["wall"])
                    bu = busy(ab["base"]["win"])
                    W_, G_ = {}, {}
                    for v in meta.get("names", []):
                        d_ = ab[v]
                        w_, b_ = med(d_["wall"]), med(d_["build"])
                        u_ = busy(d_["win"])
                        W_[v], G_[v] = w_, u_ / 100 * w_
                        print(f"      {v:10s} wall {w_:6.2f} ms (build {b_:5.2f}{', CPU-bound' if b_ > 0.9 * w_ else ''}): vs base {bw - w_:+6.2f} ms "
                              f"of wall; GPU busy {u_:3.0f} % -> GPU {u_ / 100 * w_:5.2f} ms, vs base {bu / 100 * bw - u_ / 100 * w_:+6.2f} ms of GPU")
                    for v in meta.get("names", []):      # the barrier-preserving price: cal:P (real + zero term) - P (stub + term)
                        if ":" in v or v in ("base", "d5") or f"cal:{v}" not in W_:
                            continue
                        print(f"      PRICE {v:7s} (barrier kept) = wall(cal) - wall(stub) = {W_['cal:' + v] - W_[v]:+6.2f} ms of wall, "
                              f"{G_['cal:' + v] - G_[v]:+6.2f} ms of GPU; the zero term itself costs {W_['cal:' + v] - bw:+5.2f} ms"
                              + (f"; the old depends stub said {bw - W_['dep:' + v]:+5.2f}" if f"dep:{v}" in W_ else ""))
                    if "d5" in W_:
                        print(f"      D5 (REF2_HC_PEND fold on) vs base: {bw - W_['d5']:+6.2f} ms of wall saved")
                s2 = pr.get("s2c")
                if s2:                                   # S2c step 1: the check's own block alone vs alternating blocks
                    print(f"    S2C at check {pr['check']} ({s2['rows']} rows; previous blocks {s2['prev']} rows):")
                    for lab in ("same", "alt2", "alt3", "same2"):
                        d_ = s2.get(lab)
                        if not d_:
                            continue
                        pos = sorted(set(d_["who"]))
                        parts = []
                        for p_ in pos:
                            ii = [i for i, x in enumerate(d_["who"]) if x == p_]
                            parts.append(f"block {p_}: wall {med([d_['wall'][i] for i in ii[1:]]):.2f} (build "
                                         f"{med([d_['build'][i] for i in ii[1:]]):.2f}), misses {sum(d_['miss'][i] for i in ii):.0f}")
                        print(f"      {lab:5s} ({len(d_['wall'])} forwards; the first per block dropped): " + "; ".join(parts))
                    mm = s2.get("mem") or {}             # S2c 1d: MLX's memory around the live check and the replay (GB)
                    for nm_ in ("live", "rep"):
                        q_ = mm.get(nm_) or {}
                        if q_:
                            print(f"      MLX memory, {nm_:4s}: active {q_.get('active0', float('nan')):.2f} -> {q_.get('active1', float('nan')):.2f}, "
                                  f"cache {q_.get('cache0', float('nan')):.3f} -> {q_.get('cache1', float('nan')):.3f}, peak {q_.get('peak', float('nan')):.2f} GB")
                    if s2.get("live_seg") and s2.get("replay_seg"):   # step 1b: the live check vs its block replayed, per piece
                        sync = d["meta"].get("sync_ms", 0.13)

                        def by_kind(seg):
                            o_ = {}
                            for kind, L_, b_, e_ in seg:
                                o_[kind] = o_.get(kind, 0.0) + max(0.0, e_ - sync)
                            return o_

                        def by_layer(seg):
                            o_ = {}
                            for kind, L_, b_, e_ in seg:
                                if L_ is not None:
                                    o_[L_] = o_.get(L_, 0.0) + max(0.0, e_ - sync)
                            return o_
                        lv, r1, r2 = by_kind(s2["live_seg"]), by_kind(s2["replay_seg"]), by_kind(s2.get("replay_seg2") or [])
                        print(f"      STEP 1b per piece, synced (ms; live check / its block replayed right after / again after the "
                              f"alternating forwards; live - replay): total {sum(lv.values()):.2f} / {sum(r1.values()):.2f} / "
                              f"{sum(r2.values()):.2f}")
                        for k_ in sorted(set(lv) | set(r1), key=lambda k: -(lv.get(k, 0) - r1.get(k, 0))):
                            print(f"        {k_:8s} {lv.get(k_, 0):7.2f} / {r1.get(k_, 0):7.2f} / {r2.get(k_, 0):7.2f}   {lv.get(k_, 0) - r1.get(k_, 0):+6.2f}")
                        ll, l1 = by_layer(s2["live_seg"]), by_layer(s2["replay_seg"])
                        top = sorted(ll, key=lambda L_: -(ll[L_] - l1.get(L_, 0)))[:6]
                        print("        layers with the largest live - replay: " + ", ".join(f"L{L_} {ll[L_] - l1.get(L_, 0):+.2f}" for L_ in top))
                        if s2.get("live_cpu") and s2.get("replay_cpu"):   # Sat 01:2x: the thread's CPU per piece (build + eval)
                            def cpu_kind(cl):
                                o_ = {}
                                for kind, L_, cb, ce in cl:
                                    o_[kind] = o_.get(kind, 0.0) + cb + ce
                                return o_

                            def per_layer(seg, kind, cpu=False):
                                o_ = {}
                                for row in seg:
                                    if row[0] == kind:
                                        o_[row[1]] = o_.get(row[1], 0.0) + ((row[2] + row[3]) if cpu else max(0.0, row[3] - sync))
                                return o_
                            cl, cr = cpu_kind(s2["live_cpu"]), cpu_kind(s2["replay_cpu"])
                            print("        thread CPU per piece (ms; live / replay; live - replay): " + ", ".join(
                                f"{k_} {cl.get(k_, 0):.2f} / {cr.get(k_, 0):.2f} ({cl.get(k_, 0) - cr.get(k_, 0):+.2f})"
                                for k_ in sorted(set(cl) | set(cr), key=lambda k: -abs(cl.get(k, 0) - cr.get(k, 0)))))
                            wl, wr = per_layer(s2["live_seg"], "moe_pre"), per_layer(s2["replay_seg"], "moe_pre")
                            cl_, cr_ = per_layer(s2["live_cpu"], "moe_pre", True), per_layer(s2["replay_cpu"], "moe_pre", True)
                            dw = [wl[L_] - wr.get(L_, 0) for L_ in wl]
                            dc = [cl_.get(L_, 0) - cr_.get(L_, 0) for L_ in wl]
                            ok_w, ok_c = med(dw) >= 0.08, abs(med(dc)) <= 0.03
                            print(f"        moe_pre per layer: live - replay wall median {med(dw):+.3f} ms, CPU median {med(dc):+.3f} ms -> "
                                  + ("EXTRA WAIT (wall >= 0.08, CPU flat)" if ok_w and ok_c else
                                     ("ENCODE CONTENTION (CPU rises)" if ok_w and med(dc) > 0.03 else "neither line")))
                for key_ in ("full16", "dense16"):
                    dn = pr.get(key_)
                    if not dn:
                        continue
                    if dn.get("error"):
                        print(f"    dense16 ERROR {dn['error']}")
                    else:
                        sync = d["meta"].get("sync_ms", 0.13)
                        ud = float("nan")
                        if len(samp):
                            m = (samp[:, 0] >= dn["t_start"]) & (samp[:, 0] <= dn["t_end"]) & (samp[:, 1] >= 0)
                            ud = float(samp[m, 1].mean()) if m.any() else float("nan")
                            nsm = int(m.sum())
                        pz = {}
                        for kind, L, b_, e_ in (dn["synced"].get("seg") or []):
                            pz[kind] = pz.get(kind, 0.0) + max(0.0, e_ - sync)
                        print(f"    {'FULL 16 rows (real experts, warm)' if key_ == 'full16' else 'DENSE-ONLY 16 rows (experts stubbed)'}, {len(dn['wall'])} unsynced forwards: wall {med(dn['wall']):.2f} ms "
                              f"(build {med(dn['build']):.2f}, so {med(dn['wall']) - med(dn['build']):.2f} after the build); GPU busy "
                              f"{ud:.0f} % over {dn['t_end'] - dn['t_start']:.1f} s ({nsm if len(samp) else 0} samples) -> GPU ~{ud / 100 * med(dn['wall']):.1f} ms per forward;"
                              f" synced split sum {sum(pz.values()):.2f} ms (" + ", ".join(f"{k} {v:.2f}" for k, v in sorted(pz.items(), key=lambda kv: -kv[1])) + ")")
                s16 = pr.get("synced_16")
                sw = (pr.get("sweep") or {}).get("16") or (pr.get("sweep") or {}).get(16)
                if not s16:
                    continue
                sync = d["meta"].get("sync_ms", 0.13)
                seg = s16.get("seg") or []
                pieces = {}
                for kind, L, b, e in seg:                   # every synced eval is GPU work since the last one (layer = its glue)
                    pieces[kind] = pieces.get(kind, 0.0) + max(0.0, e - sync)
                tot = sum(pieces.values())
                ww = med([x["wall"] for x in sw[1:]]) if sw else float("nan")
                bb = med([x["build"] for x in sw[1:]]) if sw else float("nan")
                print(f"    probe at check {pr['check']}: 16-row synced split sum {tot:.2f} ms (" + ", ".join(
                    f"{k} {v:.2f}" for k, v in sorted(pieces.items(), key=lambda kv: -kv[1])) + f"); 16-row unsynced warm wall "
                    f"{ww:.2f} ms (build {bb:.2f})")


if __name__ == "__main__":
    main(sys.argv[1:])
