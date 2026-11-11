"""MTPPICK exact official-loop simulation (MTP Head, Tue 11-03; REPORT 'MTPPICK'), CPU. Tables: DFlash2 L1-20 at EVERY position
(data/d02b_walk_d23h_dfl_<set>.npz, TL_WALK_EVERY + TL_WALK_CF: own picks / cf and teacher-forced picks / cf, 15 depths) and
the MTP head b5's teacher-forced chain (data/d24/<set>_b5.npz: top-1 p and the true token's rank, 15 depths; MBPP from
data/d23h/mbpp_b5.npz when DEPTH32's step 1 has run). Alignment: a check at anchor A drafts ids[A + 1 ..]; the live MTP chain
there starts from the trunk's hidden at A - 1, i.e. D-24's row r = A - 1 - a0 (d22_ttt.chain: row r depth c + 1 predicts
ids[a0 + r + c + 2]). Teacher-forced picks equal the live chain up to the first wrong pick, so kept is exact.
Arms, each its own loop (next anchor = A + k + 1): dfl (DFlash2's own chain), mtp (the MTP chain), pick (each depth takes the
token of the more confident drafter: DFlash2's selector cf vs the MTP's top-1 p, both fed the picked token), either (oracle).
16-row checks (15 drafts) and, as information, the shipped running-confidence cut (t 0.3: rows = 1 + leading depths whose
running product >= t; dfl on its own cf, mtp on p1, pick on the picked confidence; past the first miss the teacher-forced
confidences stand in for the live ones). Cost (D-24's model on the 45 GB default): F(type) + 7.82 ms x rows, + 12 ms for the
DFlash2 block pass, + 2.2 ms x drafted MTP steps (the cut arms stop drafting at the cut). Output: data/mtppick_sim.txt."""
import os
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
SETS = [("code", "d24/code_b5.npz", "code"), ("math", "d24/gsm_b5.npz", "math"), ("multi", "d24/multi_b5.npz", "chat"),
        ("ifeval", "d24/ifeval_b5.npz", "chat"), ("mbpp", "d23h/mbpp_b5.npz", "code")]
F = {"chat": 50.1, "math": 66.1, "code": 84.1}          # D-24's fit (data/d24_read.txt)
ROW, STEP, DFL, T = 7.82, 2.2, 12.0, 0.3
ARMS = ("dfl", "mtp", "pick", "either")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def lead(ok):
    """right drafts before the first wrong one"""
    k = 0
    while k < len(ok) and ok[k]:
        k += 1
    return k

def rows_cut(conf):
    return 1 + int((np.cumprod(np.asarray(conf, np.float64)) >= T).sum())

res = {}
say("MTPPICK exact official-loop simulation (every-position DFlash2 tables + the MTP head b5, aligned r = A - 1 - a0)")
for st, mf, typ in SETS:
    fw, fm = os.path.join(TD, f"d02b_walk_d23h_dfl_{st}.npz"), os.path.join(TD, mf)
    if not (os.path.exists(fw) and os.path.exists(fm)):
        say(f"  {st}: tables missing"); continue
    Wz, Mz = np.load(fw), np.load(fm)
    if "p1_0" not in Mz.files:
        say(f"  {st}: the MTP table has no top-1 p"); continue
    mk = {str(k): i for i, k in enumerate(Mz["keys"])}
    acc = {a: [0.0, 0.0, 0] for a in ARMS}              # kept, ms, checks (16 rows)
    acc_c = {a: [0.0, 0.0, 0, 0.0] for a in ("dfl", "mtp", "pick")}   # kept, ms, checks, rows (cut)
    for w, nm in enumerate(str(x) for x in Wz["names"]):
        mi = mk.get(nm[:-4])
        if mi is None:
            continue
        ids, off = Wz[f"{w}_ids"], int(Wz[f"{w}_off"])
        ck = Wz[f"{w}_checks"]
        at = {int(A): c for c, (k, m, A) in enumerate(ck)}
        a0 = int(Mz["a0"][mi]) if "a0" in Mz.files else max(0, int(Mz["gen"][mi]) - 2)
        p1 = Mz[f"p1_{mi}"][:, :15]
        rk = Mz[f"rk_{mi}"][:, :15] if f"rk_{mi}" in Mz.files else np.where(Mz[f"rt_{mi}"][:, :15] == 1, 0, np.where(Mz[f"rt_{mi}"][:, :15] < 0, -1, 1))
        dt, dc, tt, tc = (Wz[f"{w}_{x}"] for x in ("dtok", "dcf", "ttok", "tcf"))
        start = int(ck[0][2])

        def check(A):
            """per-depth flags at anchor A (or None past the tables): true, dfl own ok, mtp ok, pick ok, confidences"""
            c = at.get(A); r = A - 1 - a0
            if c is None or r < 0 or r >= rk.shape[0]:
                return None
            m = min(int(ck[c][1]), 15)
            m = min(m, int((rk[r] >= 0).sum()))
            if m <= 0:
                return None
            true = ids[A - off + 1:A - off + 1 + m].astype(np.int64)
            d_own = dt[c, :m].astype(np.int64) == true
            d_tf = tt[c, :m].astype(np.int64) == true
            m_ok = rk[r, :m] == 0
            dpick = tc[c, :m] >= p1[r, :m]
            p_ok = np.where(dpick, d_tf, m_ok)
            return dict(m=m, d_own=d_own, d_tf=d_tf, m_ok=m_ok, p_ok=p_ok, e_ok=d_tf | m_ok, dcf=dc[c, :m], p1=p1[r, :m],
                        pcf=np.where(dpick, tc[c, :m], p1[r, :m]))

        for arm in ARMS:
            A = start
            while True:
                x = check(A)
                if x is None:
                    break
                ok = {"dfl": x["d_own"], "mtp": x["m_ok"], "pick": x["p_ok"], "either": x["e_ok"]}[arm]
                k = lead(ok)
                ms = F[typ] + ROW * 16 + (DFL if arm in ("dfl", "pick", "either") else 0) + (STEP * 15 if arm in ("mtp", "pick", "either") else 0)
                acc[arm][0] += k + 1; acc[arm][1] += ms; acc[arm][2] += 1
                A += k + 1
        for arm in ("dfl", "mtp", "pick"):
            A = start
            while True:
                x = check(A)
                if x is None:
                    break
                ok = {"dfl": x["d_own"], "mtp": x["m_ok"], "pick": x["p_ok"]}[arm]
                cf = {"dfl": x["dcf"], "mtp": x["p1"], "pick": x["pcf"]}[arm]
                rows = rows_cut(cf)
                k = min(lead(ok), rows - 1)
                steps = min(15, rows)
                ms = F[typ] + ROW * rows + (DFL if arm != "mtp" else 0) + (STEP * steps if arm != "dfl" else 0)
                acc_c[arm][0] += k + 1; acc_c[arm][1] += ms; acc_c[arm][2] += 1; acc_c[arm][3] += rows
                A += k + 1
    if not acc["dfl"][2]:
        say(f"  {st}: no checks"); continue
    kp = {a: acc[a][0] / acc[a][2] for a in ARMS}; ts = {a: 1e3 * acc[a][0] / acc[a][1] for a in ARMS}
    kc = {a: acc_c[a][0] / acc_c[a][2] for a in acc_c}; tc_ = {a: 1e3 * acc_c[a][0] / acc_c[a][1] for a in acc_c}
    rc = {a: acc_c[a][3] / acc_c[a][2] for a in acc_c}
    res[st] = (kp, ts, kc, tc_, rc)
    say(f"  {st:6s} (F {F[typ]:.1f} ms, {acc['dfl'][2]} dfl checks): 16 rows: kept DFlash2 {kp['dfl']:.2f} / MTP {kp['mtp']:.2f} / pick "
        f"{kp['pick']:.2f} (x{kp['pick'] / kp['dfl']:.3f}) / oracle {kp['either']:.2f}; est. tok/s {ts['dfl']:.1f} / {ts['mtp']:.1f} / "
        f"{ts['pick']:.1f} (x{ts['pick'] / ts['dfl']:.3f})")
    say(f"  {'':6s} cut t {T}: kept {kc['dfl']:.2f} / {kc['mtp']:.2f} / {kc['pick']:.2f} at rows {rc['dfl']:.1f} / {rc['mtp']:.1f} / "
        f"{rc['pick']:.1f}; est. tok/s {tc_['dfl']:.1f} / {tc_['mtp']:.1f} / {tc_['pick']:.1f} (pick x{tc_['pick'] / tc_['dfl']:.3f} vs dfl, "
        f"x{tc_['pick'] / tc_['mtp']:.3f} vs mtp)")
open(os.path.join(TD, "mtppick_sim.txt"), "a").write("\n".join(lines) + "\n\n")
