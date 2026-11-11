"""NEVERWAIT, the engine's own cold-vs-warm check (Fable FAR, Thu 2026-11-05 16:4x; CPU, numpy, seconds; reads logs only).
BigMoE's R57_WARMSPLIT samples (r57_screen.py: every Nth check is run on its real input cold at 16 rows, rolled back, then
warm (all its experts resident) at 1 / 4 / 8 / 16 rows, rolled back; the real check then commits). Per sample: cold ms,
misses, SSD busy ms, gate service ms; warm ms per row count. exposed = cold 16 - warm 16 = what the wait costs that check;
exposed / service = the share of the service time that is on the check's critical path, measured by the engine itself.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_warmsplit.py > far_warmsplit.txt"""
import glob, json, os, time
import numpy as np
R = os.path.expanduser("~/BigMoE/data/r57")
print(f"FAR WARMSPLIT ({time.strftime('%a %Y-%m-%d %H:%M')}): cold 16 rows vs warm 16 rows of the same check (BigMoE's R57_WARMSPLIT logs)")
for p in sorted(glob.glob(os.path.join(R, "*_warmsplit.json")), key=os.path.getmtime):
    tag = os.path.basename(p)[:-len("_warmsplit.json")]
    d = json.load(open(p))
    S = []
    for rec in d:
        for s in rec.get("samples", []):
            w = {int(a[0]): float(a[1]) for a in s["warm"]}
            c = s["cold"]
            if 16 in w and 1 in w and len(c) >= 5:
                S.append((c[1], c[2], c[3], c[4], w[1], w.get(4, np.nan), w.get(8, np.nan), w[16]))
    when = time.strftime('%a %H:%M', time.localtime(os.path.getmtime(p)))
    if len(S) < 3:
        print(f"  {tag} ({when}): {len(S)} samples"); continue
    A = np.array(S); cold, miss, io, svc, w1, w4, w8, w16 = A.T
    ex = cold - w16
    m = svc > 5
    co = np.linalg.lstsq(np.stack([np.ones(m.sum()), svc[m]], 1), ex[m], rcond=None)[0]
    print(f"  {tag} ({when}): {len(S)} samples over {len(d)} prompts | cold 16-row check median {np.median(cold):.1f} ms "
          f"(misses {np.median(miss):.0f}, SSD busy {np.median(io):.1f}, service {np.median(svc):.1f}) | warm 16 rows {np.median(w16):.1f} ms, 8 rows {np.nanmedian(w8):.1f}, "
          f"4 rows {np.nanmedian(w4):.1f}, 1 row {np.median(w1):.1f} (warm slope {np.median((w16 - w1) / 15):.2f} ms per row) | exposed = cold - warm 16: median "
          f"{np.median(ex):.1f} ms | exposed / service: median {np.median(ex[m] / svc[m]):.2f}, quartiles {np.quantile(ex[m] / svc[m], 0.25):.2f}-{np.quantile(ex[m] / svc[m], 0.75):.2f}, "
          f"pooled {ex[m].sum() / svc[m].sum():.2f} | exposed ~ a + b * service: b {co[1]:.2f}, a {co[0]:.1f} ms")
