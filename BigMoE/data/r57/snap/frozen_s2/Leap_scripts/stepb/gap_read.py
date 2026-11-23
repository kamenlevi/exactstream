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
