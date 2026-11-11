"""CPU test for sticky/s10b_fit.py on a synthetic trace in b32's layout (8 layers, predictor layers 1 and 3, 64 experts,
top-4, hidden 32; 4 domains x 5 prompts; 1-3 rows per verify pass; one MTP-only seq per prompt; sidecar records in miss.bin;
res.bin: pools T = L+1..L+3 at each base layer's decision, T = NL the MTP pool, nT = 0 at L = NL).
The hidden state turns between layers (h_{L+1} = h_L M + noise, M a partial random rotation) and each layer routes top-4
of its own router (logit std ~1.5, as a real router) on its own state, so the plain router on x_L (M0) is weak at lead
>= 2 and a linear correction can learn the drift. Residency: ~35 % of the pool at random, every served expert resident
except the misses (5 % of those resident at the decision, evicted before T).
1. the loader: passes, rows per predictor layer as built; layer-1 re-runs (MTPLX's commit, same seq, fewer rows) and
   MTP-only seqs skipped, no pass lost to unequal rows (BigMoE's first-call join rule); res.bin: the first record per
   (pass, L) wins (a later all-resident record follows each re-run pass), passes without records counted;
2. M0 on all prompts = an independent recall@16-of-misses computation (s10b_check.py's definition), leads 1-3; the
   DEPLOYABLE recall at k = 2..16 (all prompts | held-out) and the read budget per pass-layer (recall, reads, unused,
   precision) = s10b_check.py's b32 loops;
   the per-pass catch at 4.8 reads (linear in reads between the budgets around it, both policy families) = an
   independent interpolation of the independent sweep;
3. the picked model beats M0 on the held-out prompts at lead 2 by >= 10 points: top-16 over all 512, deployable, and
   at 4.8 reads (the judge's M0 = the independent held-out number); selection by the catch at 4.8 reads;
4. the .txt and .json land, with a verdict; without res.bin the script falls back to b31's score (2 epochs);
5. S10c-1 (frozen recipe M1a lr 3e-4 wd 10, 5 epochs; S10B_FROZEN / S10B_TRAIN / S10B_LABEL=ld): a b33-like copy of the
   trace (the same rows with the passes in reverse order in every npz; miss.bin with kinds 0-3 on the misses, own random
   stream): the frozen run on the trace itself skips the selection; the cross run (trained on the trace, scored on the
   copy, late + demand) prints the kind counts, its M0 lines = the independent count without kind 1, its frozen models'
   lines on the training trace = the same-trace frozen run, and the S10c-1 verdict with the right differences; the cross
   run on all misses = the same-trace frozen run exactly (scoring path and standardization independent of row order);
   the export (S10B_EXPORT, BigMoE's format): m1a_8pairs.npz keys, shapes and dtypes; the exported models' held-out
   catch recomputed from the npz alone (s = x Wf + bf + sum of A over L's routed experts) = the script's folded number,
   close to the f32 models'; the check file's top-12 per row = an independent computation from the npz and the trace.
usage: [BigMoE venv] python tests/test_s10b_fit.py"""
import json, os, re, shutil, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NL, LP, E, K, H, SIDE0 = 8, [1, 3], 64, 4, 32, 9
DOMS = ["code", "prose", "edit", "math"]
KS, POLS = (2, 4, 8, 12, 16), ("k2", "k4", "k8", "k12", "k16", "all12")           # per row; the budget lines shown
KF, KA = (1, 2, 3, 4, 5, 6, 8, 12, 16), (4, 6, 8, 12, 16, 24, 32, 48, 64)           # the budget sweep behind 4.8 reads
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


def num(pat, o, i=0):
    f = re.findall(pat, o)
    return float(f[i]) if len(f) > i else float("nan")


g = np.random.default_rng(5)
gr = np.random.default_rng(7)                                                        # residency (own stream)
Q, _ = np.linalg.qr(g.normal(0, 1, (H, H)))
M = (0.4 * np.eye(H) + 0.9 * Q).astype(np.float32)                                  # the drift between layers
M /= np.abs(np.linalg.eigvals(M)).max()
WR = {L: g.normal(0, 1.5 / np.sqrt(H), (E, H)).astype(np.float32) for L in range(NL)}  # each layer's router
to_bits = lambda a: (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
from_bits = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
pack = lambda rb: np.packbits(rb, bitorder="little").view("<i4").tolist()           # bit e % 32 of word e // 32
d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
meta = {"layers_pred": LP, "n_layers": NL, "side0": SIDE0, "K": K, "E": E, "hidden": H, "prompts": [],
        "res": {"file": "res.bin", "W": E // 32}}
miss, seq, n_pass, n_rows, n_rerun, ref, held = [], 0, 0, 0, 0, {}, set()
resr, RESD, n_nores = [], {}, 0
for pi in range(20):
    dom, idx = DOMS[pi // 5], pi % 5
    name = f"p{pi:02d}_{dom}{idx}"
    meta["prompts"].append({"name": name, "domain": dom, "idx": idx})
    ent = {k: [] for k in ("seq", "L", "rows", "row_off", "x_off")}
    ids_all, x_all = [], []
    bias = g.normal(0, 0.8, H).astype(np.float32)                                # a per-prompt topic
    for _ in range(60):
        R = int(g.integers(1, 4))
        h = g.normal(0, 1, (R, H)).astype(np.float32) + bias
        hs = []
        for L in range(NL):
            hs.append(h)
            h = h @ M + 0.2 * g.normal(0, 1, (R, H)).astype(np.float32)
        for L in range(NL):
            xb = to_bits(hs[L])
            ids = np.argsort(-(from_bits(xb) @ WR[L].T), axis=1)[:, :K]
            ent["seq"].append(seq); ent["L"].append(L); ent["rows"].append(R)
            ent["row_off"].append(sum(len(a) for a in ids_all)); ids_all.append(ids.astype(np.int16))
            if L in LP:
                ent["x_off"].append(sum(len(a) for a in x_all)); x_all.append(xb)
            else:
                ent["x_off"].append(-1)
            u = np.unique(ids)
            fl = g.random(len(u)) < 0.5
            miss.append(np.concatenate([[seq, L, len(u)], u | (fl.astype(np.int64) << 16)]).astype(np.int32))
            ref[seq, L] = (ids.astype(np.int64), set(u[fl].tolist()), from_bits(xb))
            if L < 2:                                                              # sidecar records: ignored
                miss.append(np.array([seq, SIDE0 + L, 1, int(u[0]) | 1 << 16], np.int32))
        if idx == 4:
            held.add(seq)
        if seq % 37 != 5:                                       # residency at each base layer's prefetch decision
            for L in range(NL):
                Ts = [T for T in (L + 1, L + 2, L + 3) if T <= NL]               # T = NL: the MTP pool (ignored)
                r_ = [seq, L, len(Ts)]
                for T in Ts:
                    rb = gr.random(E) < 0.35
                    if T < NL:
                        ms_ = np.array(sorted(ref[seq, T][1]), np.int64)
                        rb[np.unique(ref[seq, T][0])] = True                     # served from the pool ...
                        rb[ms_] = gr.random(len(ms_)) < 0.05                     # ... but the misses (a few evicted later)
                        RESD[seq, L, T] = rb
                    r_ += [T] + pack(rb)
                resr.append(np.array(r_, np.int32))
        else:
            n_nores += 1
        if R > 1 and g.random() < 0.3:                                           # MTPLX's commit re-run of layer 1
            ent["seq"].append(seq); ent["L"].append(1); ent["rows"].append(R - 1)    # over the kept rows: skipped
            ent["row_off"].append(sum(len(a) for a in ids_all)); ids_all.append(np.full((R - 1, K), E - 1, np.int16))
            ent["x_off"].append(-1); n_rerun += 1
            if seq % 37 != 5:                                   # a later record for the same (pass, L): ignored
                resr.append(np.array([seq, 1, 3] + sum(([T] + pack(np.ones(E, bool)) for T in (2, 3, 4)), []), np.int32))
        seq += 1; n_pass += 1; n_rows += R
    ent["seq"].append(seq); ent["L"].append(NL); ent["rows"].append(1)               # an MTP-only draft call
    ent["row_off"].append(sum(len(a) for a in ids_all)); ids_all.append(np.zeros((1, K), np.int16)); ent["x_off"].append(-1)
    resr.append(np.array([seq, NL, 0], np.int32))
    seq += 1
    np.savez(f"{d}/{name}.npz", ids=np.concatenate(ids_all), x=np.concatenate(x_all),
             **{k: np.array(v, np.int32) for k, v in ent.items()})
np.concatenate(miss).tofile(f"{d}/miss.bin")
np.concatenate(resr).astype("<i4").tofile(f"{d}/res.bin")
np.savez(f"{d}/routers.npz", **{f"w{L + dd}": WR[L + dd].astype(np.float16) for L in LP for dd in (1, 2, 3)})
json.dump(meta, open(f"{d}/meta.json", "w"))

# the b33-like copy: kinds on the misses (0 demand, 1 on time, 2 late, 3 queued), passes in reverse order per prompt
gk = np.random.default_rng(9)
miss2, kinds, ms_ld = [], np.zeros(4, np.int64), {}
for rec_ in miss:
    v = rec_[3:].astype(np.int64)
    if rec_[1] < NL:
        fl = (v >> 16) & 1
        kd = gk.integers(0, 4, len(v)) * fl
        v = v | kd << 17
        kinds += np.bincount(kd[fl == 1], minlength=4)
        ms_ld[int(rec_[0]), int(rec_[1])] = set((v[(fl == 1) & (kd != 1)] & 0xFFFF).tolist())
    miss2.append(np.concatenate([rec_[:3], v]).astype(np.int32))
d2 = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
for f_ in ("routers.npz", "res.bin", "meta.json"):
    shutil.copy(f"{d}/{f_}", f"{d2}/{f_}")
np.concatenate(miss2).tofile(f"{d2}/miss.bin")
for q in meta["prompts"]:
    z_ = dict(np.load(f"{d}/{q['name']}.npz"))
    grp = {}
    for j, s_ in enumerate(z_["seq"].tolist()):
        grp.setdefault(s_, []).append(j)
    perm = np.array(sum(reversed(list(grp.values())), []))
    np.savez(f"{d2}/{q['name']}.npz", ids=z_["ids"], x=z_["x"], **{k: z_[k][perm] for k in ("seq", "L", "rows", "row_off", "x_off")})

# the independent M0 (s10b_check.py's loops, on the f16 routers as the script reads them): all 512, deployable, budget
def indep(msets, lgf=None, leads=(1, 2, 3)):
    m0, dref, bref = {}, {}, {}
    for dd in leads:
        hit = tot = 0
        for (s, L), (ids, ms, x) in ref.items():
            if L not in LP:
                continue
            lg = lgf(L, x, ids) if lgf else x @ WR[L + dd].astype(np.float16).astype(np.float32).T
            top = np.argpartition(-lg, 15, axis=1)[:, :16]
            idsT, msT = ref[s, L + dd][0], msets[s, L + dd]
            for r in range(len(ids)):
                rm = set(idsT[r].tolist()) & msT
                hit += len(rm & set(top[r].tolist())); tot += len(rm)
            rb = RESD.get((s, L, L + dd))
            if rb is None:
                continue
            order = np.argsort(-np.where(rb[None, :], -np.inf, lg), axis=-1, kind="stable")
            bags = ["all"] + (["held"] if s in held else [])
            for r in range(len(ids)):
                rm = set(idsT[r].tolist()) & msT
                for k in KS:
                    h_ = len(rm & set(order[r, :k].tolist()))
                    for b in bags:
                        v = dref.setdefault((dd, k, b), [0, 0]); v[0] += h_; v[1] += len(rm)
            RT = set(idsT.ravel().tolist())
            Mp = RT & msT
            oall = np.argsort(-lg, axis=-1, kind="stable")
            for pol, pk in [(f"k{k}", order[:, :k]) for k in KF] + [(f"all{k}", oall[:, :k]) for k in KA]:
                U = {e for e in pk.ravel().tolist() if not rb[e]}
                for b in bags:
                    v = bref.setdefault((dd, pol, b), [0, 0, 0, 0, 0])
                    v[0] += len(U & Mp); v[1] += len(Mp); v[2] += len(U); v[3] += 1; v[4] += len(U - RT)
        m0[dd] = 100 * hit / tot
    return m0, dref, bref


m0, dref, bref = indep({k: v[1] for k, v in ref.items()})
m0l, drefl, brefl = indep(ms_ld)                                                     # late + demand (the copy)


def interp(dd, fam, b, target=4.8, br=None):
    pts = sorted((v[2] / v[3], v[0] / v[1]) for (l_, pol, b_), v in (br or bref).items() if l_ == dd and b_ == b and pol.startswith(fam))
    for (r0, y0), (r1, y1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return 100 * (y0 + (target - r0) / (r1 - r0) * (y1 - y0))
    return float("nan")
env = dict(os.environ, TL_DATA=d, PYTHONDONTWRITEBYTECODE="1")
r = subprocess.run([sys.executable, os.path.join(HERE, "..", "sticky", "s10b_fit.py"), d, "test"], capture_output=True,
                   text=True, env=env)
o = r.stdout + r.stderr
print(o)
check(f"loader: {n_pass} passes, {n_rows} rows per predictor layer, {n_rerun} layer-1 re-runs and 20 MTP-only seqs skipped",
      f"{n_pass} verify passes, {n_rows} rows per predictor layer" in o and f"{n_rerun} re-run calls" in o
      and "20 MTP-only seqs" in o and "0 partial" in o and "0 unequal-row" in o and n_rerun > 0)
check(f"res.bin: {n_nores} passes without records -> {n_nores * len(LP) * 3} (pass, L -> T) unscored",
      f"scored (pass, L -> T) without one: {n_nores * len(LP) * 3}" in o and n_nores > 0)


def m0_checks(o, m0, dref, bref, lbl=""):
    got = [num(rf"lead {dd} recall@16 of misses ([\d.]+) %", o) for dd in (1, 2, 3)]
    check(f"M0 all prompts = the independent count{lbl} (script {got}, test {[round(m0[dd], 1) for dd in (1, 2, 3)]})",
          all(abs(got[i] - m0[dd]) < 0.06 for i, dd in enumerate((1, 2, 3))))
    bad = []
    for dd in (1, 2, 3):
        for k in KS:
            for j, b in enumerate(("all", "held")):
                v = dref[dd, k, b]
                s_ = float(re.search(rf"M0 DEPLOYABLE lead {dd} .*?k{k} ([\d.]+) \| ([\d.]+)", o).group(j + 1))
                if abs(s_ - 100 * v[0] / v[1]) >= 0.06:
                    bad.append(f"lead {dd} k{k} {b}: script {s_} test {100 * v[0] / v[1]:.2f}")
    check(f"M0 DEPLOYABLE recall at k = 2..16, all prompts and held-out, = s10b_check's loop{lbl} ({len(bad)} off: "
          f"{bad[:3]})", not bad)
    m0_budget(o, bref, lbl)


def m0_budget(o, bref, lbl):
    bad = []
    for dd in (1, 2, 3):
        for pol in POLS:
            m = re.search(rf"\n  lead {dd} +{pol}: recall +([\d.]+) % \| +([\d.]+) %, reads +([\d.]+) \( *([\d.]+) unused\) "
                          rf"vs misses +([\d.]+) per pass-layer, precision +([\d.]+) %", o)
            v, vh = bref[dd, pol, "all"], bref[dd, pol, "held"]
            want = (100 * v[0] / v[1], 100 * vh[0] / vh[1], v[2] / v[3], v[4] / v[3], v[1] / v[3], 100 * v[0] / v[2])
            if m is None or any(abs(float(m.group(i + 1)) - w) >= 0.06 for i, w in enumerate(want)):
                bad.append(f"lead {dd} {pol}: script {m.groups() if m else None} test {[round(w, 2) for w in want]}")
    check(f"M0 read budget per pass-layer (recall all | held-out, reads, unused, misses, precision) = s10b_check's loop"
          f"{lbl} ({len(bad)} off: {bad[:2]})", not bad)
    bad = []
    for dd in (1, 2, 3):
        m = re.search(rf"M0 AT 4.8 READS lead {dd} .*?: top-k among non-resident ([\d.]+) \| ([\d.]+) %, top-K over all "
                      rf"512 \(resident skipped\) ([\d.]+) \| ([\d.]+) %", o)
        want = tuple(interp(dd, f_, b_, br=bref) for f_ in ("k", "all") for b_ in ("all", "held"))
        if m is None or any(not abs(float(m.group(i + 1)) - w) < 0.06 for i, w in enumerate(want)):
            bad.append(f"lead {dd}: script {m.groups() if m else None} test {[round(w, 2) for w in want]}")
    check(f"M0 catch at 4.8 reads per pass-layer (top-k and top-K families, all | held-out) = an independent "
          f"interpolation{lbl} ({len(bad)} off: {bad[:2]})", not bad)


m0_checks(o, m0, dref, bref)
fit2, m02 = num(r"lead 2: \S+ recall@16 of misses ([\d.]+) % vs M0", o), num(r"lead 2: \S+ recall@16 of misses [\d.]+ % vs M0 ([\d.]+)", o)
check(f"held-out lead 2, top-16 over all 512: the fit {fit2} % beats M0 {m02} % by >= 10 points", fit2 >= m02 + 10)
df2 = num(r"DEPLOYABLE lead 2: \S+ recall@16 of misses ([\d.]+) % vs M0", o)
dm2 = num(r"DEPLOYABLE lead 2: \S+ recall@16 of misses [\d.]+ % vs M0 ([\d.]+)", o)
v = dref[2, 16, "held"]
check(f"held-out lead 2, DEPLOYABLE: the fit {df2} % beats M0 {dm2} % by >= 10 points; the judge's M0 = the held-out "
      f"count {100 * v[0] / v[1]:.2f}", df2 >= dm2 + 10 and abs(dm2 - 100 * v[0] / v[1]) < 0.06)
check("selection by the catch at 4.8 reads, M2 tried, budget lines for the fit",
      "validation lead-2 per-pass catch at 4.8 reads" in o and "\n  M2  lr" in o and re.search(r"\n    lead 2 +k4: recall", o))
fr = re.search(r"\n    lead 2 AT 4.8 READS: top-k among non-resident \S+ ([\d.]+) % \(M0 ([\d.]+)\)", o)
check(f"held-out lead 2 at 4.8 reads: the fit beats M0 by >= 10 points ({fr.groups() if fr else None}); the judge's M0 = "
      f"the held-out interpolation {interp(2, 'k', 'held'):.2f}", fr is not None and float(fr.group(1)) >= float(fr.group(2)) + 10
      and abs(float(fr.group(2)) - interp(2, "k", "held")) < 0.06 and "VERDICT (pre-registered; NewIdeas 13:46" in o
      or "S10B_TARGET" in os.environ)
js = json.load(open(f"{d}/s10b_fit_test.json")) if os.path.exists(f"{d}/s10b_fit_test.json") else {}
check("the .txt and .json are written, with a verdict", r.returncode == 0 and os.path.exists(f"{d}/s10b_fit_test.txt")
      and "VERDICT" in open(f"{d}/s10b_fit_test.txt").read() and "verdict" in js and "budget" in js["judge"]["2"]["fit"])
# 5. S10c-1: the frozen recipe; trained on the trace, scored on the b33-like copy (late + demand, then all misses)
FZ, SCRIPT = "M1a,3e-4,10,5", os.path.join(HERE, "..", "sticky", "s10b_fit.py")
run = lambda D_, tag, **kw: (lambda r_: (r_.returncode, r_.stdout + r_.stderr))(subprocess.run(
    [sys.executable, SCRIPT, D_, tag], capture_output=True, text=True, env=dict(env, **kw)))
rc1, o1 = run(d, "fz", S10B_FROZEN=FZ)
ex_d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
rc2, o2 = run(d2, "s10c", S10B_FROZEN=FZ, S10B_TRAIN=d, S10B_LABEL="ld", S10B_EXPORT=ex_d)
rc3, o3 = run(d2, "x", S10B_FROZEN=FZ, S10B_TRAIN=d)
print(o2)
AT = (r"\n    lead 2 AT 4.8 READS: top-k among non-resident \S+ ([\d.]+) % \(M0 ([\d.]+)\) \[[^\]]*\] \| top-K over all 512, "
      r"resident skipped: \S+ ([\d.]+) % \(M0 ([\d.]+)\)")
DEP = r"\n  DEPLOYABLE lead \d: \S+ recall@16 of misses ([\d.]+) % vs M0 ([\d.]+) %"
a1, a3 = re.findall(AT, o1), re.findall(AT, o3)
check(f"frozen run: no selection, the recipe printed, judge + leave-one-out lines ({a1})", rc1 == 0
      and "frozen recipe (S10B_FROZEN, no selection): M1a lr 3e-04 wd 10.0 5 epochs" in o1 and "selection: layer" not in o1
      and len(a1) == 2)
check(f"the copy's labels: kinds counted (demand / on time / late / queued {kinds.tolist()}), late + demand scored",
      rc2 == 0 and f"misses by kind: demand {kinds[0]}, on time {kinds[1]}, late {kinds[2]}, queued {kinds[3]}, scored: "
      "late + demand (S10B_LABEL=ld)" in o2 and kinds.min() > 0 and m0l[2] != m0[2])
m0_checks(o2, m0l, drefl, brefl, " (late + demand, b33-like copy)")
rp = re.findall(r"(held-out|leave-one-out): top-K over all 512 \(resident skipped\) ([\d.]+) % \(M0 ([\d.]+)\), top-k "
                r"among non-resident ([\d.]+) % \(M0 ([\d.]+)\)", o2)
check(f"cross run: the frozen models on the training trace = the same-trace frozen run ({rp} vs {a1})", len(rp) == 2
      and all(tuple(r_[1:]) == (a_[2], a_[3], a_[0], a_[1]) for r_, a_ in zip(rp, a1)))
vd = re.search(r"VERDICT S10c-1 \(pre-registered; NewIdeas 14:08\): .*?held-out M1a ([\d.]+) % vs router ([\d.]+) % "
               r"\(([+-][\d.]+) points\); leave-one-out M1a ([\d.]+) % vs router ([\d.]+) % \(([+-][\d.]+) points\) -> (\w+)", o2)
a2 = re.findall(AT, o2)
js2 = json.load(open(f"{d}/s10b_fit_s10c.json")) if os.path.exists(f"{d}/s10b_fit_s10c.json") else {}
check(f"S10c-1 verdict: the shipped-rule catches = the judge lines, differences right, verdict by the lines "
      f"({vd.groups() if vd else None})", vd is not None and len(a2) == 2
      and all((vd.group(1 + 3 * i), vd.group(2 + 3 * i)) == (a2[i][2], a2[i][3]) for i in (0, 1))
      and all(abs(float(vd.group(3 + 3 * i)) - (float(vd.group(1 + 3 * i)) - float(vd.group(2 + 3 * i)))) < 0.11 for i in (0, 1))
      and vd.group(7) == ("PASS" if min(float(vd.group(3)), float(vd.group(6))) >= 10 else "KILL"
                          if max(float(vd.group(3)), float(vd.group(6))) < 5 else "between")
      and "s10c" in js2 and js2["pick"].get("frozen") and js2["label"] == "late + demand")
check(f"cross run on all misses = the same-trace frozen run (passes reordered; {a3} vs {a1})", rc3 == 0 and a3 == a1
      and re.findall(DEP, o3) == re.findall(DEP, o1) and len(re.findall(DEP, o1)) == 6 and "VERDICT S10c-1" not in o3)
ex = dict(np.load(f"{ex_d}/m1a_8pairs.npz")) if os.path.exists(f"{ex_d}/m1a_8pairs.npz") else {}
check(f"export: m1a_8pairs.npz holds L{{L}}_Wf [H, E] f16, _bf [E] f32, _A [E, E] f16 for L in {LP} ({sorted(ex)})",
      sorted(ex) == sorted(f"L{L}_{k}" for L in LP for k in ("Wf", "bf", "A")) and all(
          (ex[f"L{L}_Wf"].shape, ex[f"L{L}_Wf"].dtype, ex[f"L{L}_bf"].shape, ex[f"L{L}_bf"].dtype, ex[f"L{L}_A"].shape,
           ex[f"L{L}_A"].dtype) == ((H, E), np.float16, (E,), np.float32, (E, E), np.float16) for L in LP))


def fold_lg(L, x, ids):                                       # the engine's form, from the exported file alone
    oh = np.zeros((len(x), E), np.float32)
    np.put_along_axis(oh, ids, 1.0, axis=1)
    return x @ ex[f"L{L}_Wf"].astype(np.float32) + ex[f"L{L}_bf"] + oh @ ex[f"L{L}_A"].astype(np.float32)


ef = re.search(r"EXPORT \(S10c \(2\), .*?shipped rule at 4.8 reads: folded ([\d.]+) % \(f32 ([\d.]+), router ([\d.]+)\).*?"
               r"identical to the f32 model's ([\d.]+) %", o2)
want = interp(2, "all", "held", br=indep(ms_ld, fold_lg, (2,))[2]) if ex else float("nan")
check(f"export: the held-out catch of the exported models (shipped rule, 4.8 reads, late + demand), recomputed from the "
      f"npz = the script's folded number, folded ~ f32 ({ef.groups() if ef else None}, test {want:.2f})", ef is not None
      and abs(float(ef.group(1)) - want) < 0.06 and abs(float(ef.group(1)) - float(ef.group(2))) < 1.0
      and float(ef.group(4)) > 95)
ck = json.load(open(f"{ex_d}/m1a_8pairs_check.json")) if os.path.exists(f"{ex_d}/m1a_8pairs_check.json") else {}
q4n = next(q["name"] for q in meta["prompts"] if q["idx"] == 4)
z4 = np.load(f"{d2}/{q4n}.npz")
okc = ex and len(ck.get("calls", [])) == 3 and ck.get("prompt") == q4n
for c_, j in zip(ck.get("calls", []), np.flatnonzero(z4["L"] == LP[0])[:3]):
    R_, xo, ro = int(z4["rows"][j]), int(z4["x_off"][j]), int(z4["row_off"][j])
    s_ = fold_lg(LP[0], from_bits(z4["x"][xo:xo + R_]), z4["ids"][ro:ro + R_].astype(np.int64))
    okc = okc and c_["call"] == int(j) and c_["rows"] == R_ and c_["top12"] == np.argsort(-s_, axis=1, kind="stable")[:, :12].tolist()
check(f"export: the check file's top-12 per row (first 3 calls of layer {LP[0]} in {q4n}) = an independent computation "
      f"from the npz and the trace", okc and os.path.exists(f"{ex_d}/m1a_8pairs.json"))
ex_d3 = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))                           # the all-layer mode: lead 2 only
rc4, o4 = run(d, "l2", S10B_FROZEN=FZ, S10B_LEADS="2", S10B_EXPORT=ex_d3, S10B_EXPORT_NAME="m1a_2pairs")
ex3 = dict(np.load(f"{ex_d3}/m1a_2pairs.npz")) if os.path.exists(f"{ex_d3}/m1a_2pairs.npz") else {}
check(f"S10B_LEADS=2: lead 2 only, the same lead-2 judge / leave-one-out lines as the 3-lead frozen run "
      f"({re.findall(AT, o4)} vs {a1}), export named by S10B_EXPORT_NAME with both layers",
      rc4 == 0 and re.findall(AT, o4) == a1 and "lead 1 recall@16" not in o4 and "lead 3 recall@16" not in o4
      and sorted(ex3) == sorted(f"L{L}_{k}" for L in LP for k in ("Wf", "bf", "A"))
      and os.path.exists(f"{ex_d3}/m1a_2pairs_check.json"))
os.rename(f"{d}/res.bin", f"{d}/res.off")                                          # b31: no residency
r = subprocess.run([sys.executable, os.path.join(HERE, "..", "sticky", "s10b_fit.py"), d, "test0"], capture_output=True,
                   text=True, env=dict(env, S10B_EPOCHS="2"))
o = r.stdout + r.stderr
check("without res.bin: b31's score and verdict, no deployable lines", r.returncode == 0 and "no res.bin" in o
      and "DEPLOYABLE" not in o and re.search(r"VERDICT \([^)]*\): held-out lead-2 recall@16", o))
print("ALL OK" if ok_all else "SOME FAILED")
