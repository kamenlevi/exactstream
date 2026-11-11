"""R-51 VD24-T (MoonShots Thu 19:3x; NON-exact pricing, CPU): price POM / LP1 / VD by predicted SSD busy and gate wait
per token, not bytes. b45 showed why bytes mislead: POM cut rusage bytes 19 % but gate wait only ~6 % (reads got smaller,
not fewer; 30 % of 2-plane reads came back as plane-3 upgrades on the critical path).
(1) Engine data (read-only): BigMoE's b44 (A = records3, S = records3m = M-01) and b45 (A = M-01 default, P = + POM R2 4,
    L = + LP1) per prompt: ssd_busy_s and gate_wait_s (BIGMOE_TRACE unions), reads per gate and tag (0 demand / 1 prefetch /
    2 joint; gates 0-47 MSB, 49+ the LSB sidecar), POM counters (2-plane / full / upgrade demand reads), rusage bytes.
    Sizes: records3 MSB 2,162,688 / LSB 770,048; records3m MSB 2,048,000 / LSB 720,896; records3p seg1 (2 planes)
    1,425,408, seg2 (plane 3) 704,512. Prefetch read sizes under POM are not counted: inferred from rusage (the non-expert
    bytes per token of the same prompt in the A arms).
(2) Cost models fit on the A and S arms only (full reads; b44's 5 % size step identifies per byte vs per read), then
    used to predict P and L out of sample: they must land within 2 points of the measured change vs A (busy and gate
    wait) before pricing anything.
(3) Replays (pomvs24 / lpom24 machinery, narrow path 4 / 3, C_m 159 / C_l 44) emit demand reads by size per token for
    POM R2 4, + LP1, + VD, + VD + LP1; their ratios to the replay's POM scale the engine's measured P arm; prefetch reads
    scale with demand misses. Quality at the engine's real shares: b45 served 31.0 % of rank >= 4 uses at 2 planes (replay
    20.8 %) and 37.5 % of rank-1 uses without the LSB (replay 26.1 %): engine share = replay share x those ratios; KLD per
    part from the three measured harness runs (M-03 F 0.20; M-07 F 0.20 + F1 0.26; VD + LP1 F 0.28 + F1 0.26) fit as
    v3 + alpha F + beta F1.
PASS (R-51 row): VD + LP1 (or a variant) at <= -15 % predicted SSD busy ms / token vs the M-01 default with predicted KLD
inside the gate at the engine's real shares. KILL: under -8 % for every variant inside the gate.
usage: [BigMoE venv] taskpolicy -b python r51_busy.py -> stdout + data/r51_busy.txt"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

lines = []


def say(s):
    print(s, flush=True); lines.append(s)


B = os.path.expanduser("~/BigMoE/data")
SZ = {"records3": (2162688, 770048), "records3m": (2048000, 720896)}
SEG1, SEG2 = 1425408, 704512
ARMS = {("b44", "A1"): "records3", ("b44", "A2"): "records3", ("b44", "S1"): "records3m", ("b44", "S2"): "records3m",
        ("b45", "A1"): "records3m", ("b45", "A2"): "records3m", ("b45", "P1"): "P", ("b45", "P2"): "P",
        ("b45", "L1"): "L", ("b45", "L2"): "L"}

# ---------------------------------------------------------------- (1) engine data per prompt
rows = []
for (b, r), kind in ARMS.items():
    m = json.load(open(os.path.join(B, b, r, "meta.json")))
    for p in m["prompts"]:
        t = p["trace"]
        rd = np.zeros((3, 3))                                # [msb / mtp / side, tag]
        for L, tag, c in t["reads"]:
            rd[0 if L < 48 else 1 if L == 48 else 2, tag] += c
        pom = p.get("pom") or [0] * 9
        rows.append(dict(b=b, r=r, kind=kind, name=p["name"], tok=p["n_tok"], busy=t["ssd_busy_s"], gw=t["gate_wait_s"],
                         disk=p["disk_read"], dem=rd[0, 0] + rd[0, 2], pf=rd[0, 1], side=rd[2].sum(),
                         mtp=rd[1].sum(), p2=pom[0], pfull=pom[1], pup=pom[2]))
# expert bytes with known sizes; prefetch size under POM inferred from rusage
other = {}
for x in rows:
    if x["kind"] in SZ:
        msb, lsb = SZ[x["kind"]]
        x["exp_bytes"] = (x["dem"] + x["pf"] + x["mtp"]) * msb + x["side"] * lsb
        other.setdefault((x["b"], x["name"]), []).append((x["disk"] - x["exp_bytes"]) / x["tok"])
oth = {k: float(np.mean(v)) for k, v in other.items()}
for x in rows:
    if x["kind"] in ("P", "L"):
        msb, lsb = SZ["records3m"]
        dem_b = x["p2"] * SEG1 + x["pfull"] * msb + x["pup"] * SEG2
        rest = x["disk"] - oth[("b45", x["name"])] * x["tok"] - dem_b - x["side"] * lsb - x["mtp"] * msb
        x["pf_size"] = rest / max(x["pf"], 1)
        x["exp_bytes"] = dem_b + x["side"] * lsb + x["mtp"] * msb + max(rest, 0)
say("(1) engine runs (b44 / b45, 4 prompts x 400 tokens each): per arm, per token")
for kind in ("records3", "records3m", "P", "L"):
    xs = [x for x in rows if x["kind"] == kind and (kind != "records3m" or x["b"] == "b45")]
    tok = sum(x["tok"] for x in xs)
    f = lambda k: sum(x[k] for x in xs) / tok
    pfs = f"; prefetch read size inferred {np.mean([x['pf_size'] for x in xs]) / 1e6:.2f} MB" if kind in ("P", "L") else ""
    say(f"  {kind:9s} busy {1e3 * f('busy'):5.2f} ms, gate wait {1e3 * f('gw'):5.2f} ms, rusage {f('disk') / 1e6:5.1f} MB, "
        f"expert bytes {f('exp_bytes') / 1e6:5.1f} MB; reads: demand {f('dem'):5.1f} (2-plane {f('p2'):4.1f}, full "
        f"{f('pfull'):4.1f}, upgrade {f('pup'):4.1f}), prefetch {f('pf'):5.1f}, sidecar {f('side'):4.1f}{pfs}")
xs44 = [x for x in rows if x["b"] == "b44"]
say(f"  b45 non-expert rusage bytes per token (A arms) by prompt: " + ", ".join(f"{k[1]} {v / 1e6:.1f} MB" for k, v in oth.items() if k[0] == "b45"))


# ---------------------------------------------------------------- (2) cost models: fit on A / S, predict P / L
def feats(x, model):
    n_all = x["dem"] + x["pf"] + x["side"] + x["mtp"]
    if model == "per read + per byte":
        return [n_all, x["exp_bytes"]]
    if model == "demand reads + bytes":
        return [x["dem"] + x["side"], x["pf"], x["exp_bytes"]]
    if model == "bytes only":
        return [x["exp_bytes"]]
    if model == "reads only":
        return [n_all]


fitset = [x for x in rows if x["kind"] in SZ]
say("(2) models fit per prompt on the A / S arms (b44 A, S; b45 A), then P and L predicted out of sample")
best = None
for target in ("busy", "gw"):
    for model in ("per read + per byte", "demand reads + bytes", "bytes only", "reads only"):
        X = np.array([feats(x, model) for x in fitset]); y = np.array([x[target] for x in fitset])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = lambda x: float(np.dot(feats(x, model), coef))
        out = []
        for kind in ("P", "L"):
            for prs in (("A1", "A2"),):
                A = [x for x in rows if x["b"] == "b45" and x["r"] in prs]
                K = [x for x in rows if x["kind"] == kind]
                ta, tk = sum(x["tok"] for x in A), sum(x["tok"] for x in K)
                meas = (sum(x[target] for x in K) / tk) / (sum(x[target] for x in A) / ta) - 1
                prd = (sum(pred(x) for x in K) / tk) / (sum(pred(x) for x in A) / ta) - 1
                out.append((kind, 100 * meas, 100 * prd))
        ok = all(abs(m - p) <= 2 for _, m, p in out)
        r2 = 1 - np.sum((X @ coef - y) ** 2) / np.sum((y - y.mean()) ** 2)
        say(f"  {target:4s} ~ {model:22s} coef {np.array2string(coef, precision=6)}  fit R2 {r2:.3f}; " + "; ".join(
            f"{k} vs A measured {m:+.1f} %, predicted {p:+.1f} %" for k, m, p in out) + ("  REPRODUCES" if ok else ""))
        if ok and target == "busy" and best is None:
            best = (model, coef)
        if ok and target == "gw":
            best_gw = (model, coef)
stop = best is None
if stop:
    say("  no busy model reproduces P and L within 2 points: STOP (no pricing), per the R-51 row; (3) and (4) below give "
        "read counts, bytes and KLD only (descriptive, no busy price)")
else:
    say(f"  pricing with: busy ~ {best[0]}")

# ---------------------------------------------------------------- (3) replay read counts per variant -> engine pricing
import heapq
import pomvs24 as P
import wall24 as W


def run_counts(checks, R2, slot2, dg, A):
    """pomvs24.run's policy, counting per measured check: [full reads, 2-plane reads, upgrades, rank >= R2 uses (accepted
    rows), of them served at 2 planes, measured checks]"""
    res, score, shi, w, used, Bud = {}, {}, {}, 1.0, 0, P.CM * P.SLOT3
    n = np.zeros(6)
    sz = {3: P.SLOT3, 2: slot2}
    for (meas, best, ids, g) in checks:
        w /= P.DEC
        for e, k in best.items():
            score[e] = score.get(e, 0.0) + w
            if k < R2:
                shi[e] = shi.get(e, 0.0) + w
        for e, k in best.items():
            need3 = k < R2
            p = res.get(e)
            if p is None:
                if meas:
                    n[0 if need3 else 1] += 1
                res[e] = 3 if need3 else 2
                used += sz[res[e]]
            elif need3 and p == 2:
                if meas:
                    n[2] += 1
                res[e] = 3
                used += P.SLOT3 - slot2
        if meas:
            for r in range(A):
                for kk in range(R2, 10):
                    n[3] += 1
                    n[4] += res[int(ids[r, kk])] == 2
            n[5] += 1
        if used > Bud and dg == 2:
            hp = [(shi.get(v, 0.0) * P.P3 / (P.SLOT3 - slot2), v) if q == 3 else (score[v] * P.P12 / slot2, v)
                  for v, q in res.items() if v not in best]
            heapq.heapify(hp)
            while used > Bud and hp:
                c, v = heapq.heappop(hp)
                if res[v] == 3:
                    res[v] = 2
                    used -= P.SLOT3 - slot2
                    heapq.heappush(hp, (score[v] * P.P12 / slot2, v))
                else:
                    used -= sz[res.pop(v)]
        elif used > Bud:
            vic = sorted((v for v in res if v not in best), key=lambda v: score[v])
            for v in vic:
                if used <= Bud:
                    break
                if dg and res[v] == 3:
                    res[v] = 2
                    used -= P.SLOT3 - slot2
                else:
                    used -= sz[res.pop(v)]
            for v in vic:
                if used <= Bud:
                    break
                if v in res:
                    used -= sz[res.pop(v)]
    return n


def lsb_counts(checks, mode, A):
    """lpom24.lsb_run's policy, counting: [LSB reads, rank-1 uses (accepted rows), of them without the LSB]"""
    Rs, score, w = set(), {}, 1.0
    n = np.zeros(3)
    for (meas, best, ids, g) in checks:
        w /= P.DEC
        need = {e: k for e, k in best.items() if k < 2}
        for e in need:
            score[e] = score.get(e, 0.0) + w
        for e, k in need.items():
            if e in Rs or (mode == "LP1" and k == 1):
                continue
            if len(Rs) >= P.CL:
                vic = [v for v in Rs if v not in need]
                if vic:
                    Rs.discard(min(vic, key=lambda v: score[v]))
            Rs.add(e)
            if meas:
                n[0] += 1
        if meas:
            for r in range(A):
                n[1] += 1
                n[2] += int(ids[r, 1]) not in Rs
        if len(Rs) > P.CL:
            Rs = set(sorted(Rs, key=lambda v: -score[v])[:P.CL])
    return n


per = P.build(4, 3)
VAR = {"v3": (10, P.SLOT3, 0, "v3"), "POM": (4, P.SLOT3, 0, "v3"), "POM + LP1": (4, P.SLOT3, 0, "LP1"),
       "POM + VD": (4, P.SLOT2V, 2, "v3"), "POM + VD + LP1": (4, P.SLOT2V, 2, "LP1")}
rep = {}
for name, (R2, slot2, dg, lmode) in VAR.items():
    m, l, ntok = np.zeros(6), np.zeros(3), 0
    for k, (lay, nt, lb) in per.items():
        ntok += nt
        for L in range(W.NL):
            m += run_counts(lay[L], R2, slot2, dg, 3)
            l += lsb_counts(lay[L], lmode, 3)
    rep[name] = dict(full=m[0] / ntok, two=m[1] / ntok, up=m[2] / ntok, share4=m[4] / max(m[3], 1),
                     lsb=l[0] / ntok, share1=l[2] / max(l[1], 1))
say("(3) replay (narrow path 4 / 3, C_m 159 / C_l 44), demand reads per token by size:")
for name, q in rep.items():
    say(f"  {name:15s} full {q['full']:5.2f}, 2-plane {q['two']:5.2f}, upgrade {q['up']:5.2f} ({100 * q['up'] / max(q['two'], 1e-9):4.1f} % "
        f"of 2-plane reads), LSB {q['lsb']:5.2f}; rank >= 4 uses at 2 planes {100 * q['share4']:4.1f} %, rank-1 uses without "
        f"the LSB {100 * q['share1']:4.1f} %")

# engine per-token reads of the measured P arm and the A arm (b45)
def arm_tok(kind):
    xs = [x for x in rows if x["kind"] == kind and x["b"] == "b45"]
    tok = sum(x["tok"] for x in xs)
    o = {k: sum(x[k] for x in xs) / tok for k in ("dem", "p2", "pfull", "pup", "pf", "side", "mtp", "busy", "gw")}
    o["pf_size"] = float(np.mean([x["pf_size"] for x in xs])) if kind in ("P", "L") else SZ["records3m"][0]
    return o


EA, EP, EL = arm_tok("records3m"), arm_tok("P"), arm_tok("L")
MSB3M, LSB3M = SZ["records3m"]


def engine(name):
    """the variant's engine reads per token: the P arm scaled by the replay's ratios to its POM"""
    q, b = rep[name], rep["POM"]
    d2 = EP["p2"] * q["two"] / b["two"]
    df = EP["pfull"] * q["full"] / b["full"]
    du = EP["pup"] * q["up"] / b["up"]
    pf = EP["pf"] * (q["two"] + q["full"]) / (b["two"] + b["full"])
    side = EP["side"] * q["lsb"] / b["lsb"]
    x = dict(dem=d2 + df + du, pf=pf, side=side, mtp=EP["mtp"],
             exp_bytes=d2 * SEG1 + df * MSB3M + du * SEG2 + pf * EP["pf_size"] + side * LSB3M + EP["mtp"] * MSB3M)
    return x


model, coef = best or (None, None)
ea = dict(dem=EA["dem"], pf=EA["pf"], side=EA["side"], mtp=EA["mtp"],
          exp_bytes=(EA["dem"] + EA["pf"] + EA["mtp"]) * MSB3M + EA["side"] * LSB3M)
busyA = float(np.dot(feats(ea, model), coef)) if not stop else EA["busy"]
say(f"  check: replay POM + LP1 scaled to the engine predicts sidecar reads {engine('POM + LP1')['side']:.1f} / token vs "
    f"the L arm's measured {EL['side']:.1f}")
# quality: KLD = v3 + alpha F + beta F1 per part, from the three measured harness runs
v3 = np.array([0.00172, 0.00275, 0.01367, 0.01097]); tgt = np.array([0.00253, 0.00449, 0.02041, 0.02055])
runs = [((0.20, 0.00), (0.00202, 0.00353, 0.01639, 0.01432)), ((0.20, 0.26), (0.00221, 0.00388, 0.01770, 0.01515)),
        ((0.28, 0.26), (0.00247, 0.00427, 0.01916, 0.01656))]
Fm = np.array([f for f, _ in runs]); Km = np.array([k for _, k in runs]) - v3
ab, *_ = np.linalg.lstsq(Fm, Km, rcond=None)           # [2, 4]: alpha, beta per part
say(f"  KLD slopes from the 3 harness runs (code / gsm / ifeval / multi): alpha {np.round(ab[0], 5).tolist()}, beta "
    f"{np.round(ab[1], 5).tolist()}")
say(f"(4) {'pricing' if not stop else 'reads and bytes (no busy price)'} vs the M-01 default (b45 A: busy "
    f"{1e3 * EA['busy']:.2f} ms / token measured" + (f", model {1e3 * busyA:.2f}" if not stop else "") + "); "
    f"engine shares = b45's measured 31.0 % / 37.5 % x the replay's ratio to POM / POM + LP1")
nA = ea["dem"] + ea["pf"] + ea["side"]
say(f"  A (measured): reads {nA:6.1f} / token (demand {ea['dem']:5.1f}, prefetch {ea['pf']:5.1f}, sidecar {ea['side']:5.1f}), "
    f"expert bytes {ea['exp_bytes'] / 1e6:6.1f} MB")
for name in ("POM", "POM + LP1", "POM + VD", "POM + VD + LP1"):
    x = engine(name)
    F = 0.310 * rep[name]["share4"] / rep["POM"]["share4"]
    F1 = 0.375 * rep[name]["share1"] / rep["POM + LP1"]["share1"] if "LP1" in name else 0.0
    kl = v3 + ab[0] * F + ab[1] * F1
    ok = (kl <= tgt).all()
    n = x["dem"] + x["pf"] + x["side"]
    head = (f"predicted SSD busy {100 * (float(np.dot(feats(x, model), coef)) / busyA - 1):+6.1f} % vs A " if not stop else
            f"reads {100 * (n / nA - 1):+6.1f} %, bytes {100 * (x['exp_bytes'] / ea['exp_bytes'] - 1):+6.1f} % vs A ")
    say(f"  {name:15s} {head}(reads {n:6.1f} / token: demand {x['dem']:5.1f}, prefetch {x['pf']:5.1f}, sidecar {x['side']:5.1f}; "
        f"expert bytes {x['exp_bytes'] / 1e6:6.1f} MB); engine shares F {100 * F:4.1f} %, F1 {100 * F1:4.1f} % -> KLD "
        + " / ".join(f"{k:.5f}" for k in kl) + (f" INSIDE (code {100 * kl[0] / tgt[0]:.0f} %)" if ok else
        " OVER on " + ", ".join(n for n, a, t in zip(("code", "gsm", "ifeval", "multi"), kl, tgt) if a > t)))
say(f"  measured for reference: P {100 * (EP['busy'] / EA['busy'] - 1):+.1f} %, L {100 * (EL['busy'] / EA['busy'] - 1):+.1f} % "
    f"SSD busy vs A (b45)")
open(os.path.expanduser("~/MoonShots/data/r51_busy.txt"), "w").write("\n".join(lines) + "\n")
