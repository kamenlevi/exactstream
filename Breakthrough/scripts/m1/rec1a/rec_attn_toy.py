"""rec_attn_toy.py (Breakthrough, Tue 17 Nov 2026 01:1x; MODULE 1a milestone 2 groundwork, docs/MODULE1A.md): what a
growing KV cache changes per check, pre-registered in docs/MODULE1A.md (Tue 01:1x).
Per check at kv: write 32 new K and V rows and the 32 x 32 tree-mask block at row / column kv of preallocated buffers,
then mx.fast.scaled_dot_product_attention over the first kv + 32 rows with the mask view.
(1) within a kL alignment class, records differ only in setBytes words and buffer offsets; (2) rules fitted on 6
unaligned + 3 aligned records (values only) predict the base record's (kv 1203) changes exactly; (3) patched mode-2 and ICB replays bitwise == MLX
on 16 new unaligned kv, stale control misses 16/16; (4) info: the unaligned record at 16 aligned kv. Tiny memory."""
import os
import resource
import struct
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()
import mlx.core as mx
import numpy as np

mx.set_cache_limit(0)
QL, H, KVH, D, MAXKV = 32, 8, 2, 128, 4096
SCALE = D ** -0.5
ALIGNED = [1056, 1120, 1184]               # kL = kv + 32 = 1088, 1152, 1216: multiples of 64
UNFIT = [993, 1119, 1137, 1215, 1289, 1363]  # kL residues mod 64: 1, 63, 17, 31, 41, 51 (all odd)
UBASE = 1203                                # kL 1235 (odd)
UTEST = [1301 + 34 * i for i in range(16)]  # kL odd
ATEST = [1504 + 64 * i for i in range(16)]  # kL multiples of 64
assert all((kv + QL) % 2 for kv in UNFIT + [UBASE] + UTEST) and all((kv + QL) % 64 == 0 for kv in ALIGNED + ATEST)
rng = np.random.default_rng(20261117)
OPC = {v: k for k, v in R.OPN.items()}


def bf(a):
    return mx.array(np.asarray(a, dtype=np.float32)).astype(mx.bfloat16)


BK = bf(rng.standard_normal((1, KVH, MAXKV, D)))
BV = bf(rng.standard_normal((1, KVH, MAXKV, D)))
BM = mx.array(np.ones((1, 1, QL, MAXKV), dtype=bool))
mx.eval(BK, BV, BM)


def u8(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8)


def lbits(a):
    """The bf16 bits in logical order (the SDPA output is a strided view, so no raw-buffer cast)."""
    return np.array(a.view(mx.uint16))


def new_state():
    ck, cv, mb = mx.array(BK), mx.array(BV), mx.array(BM)
    ck, cv, mb = ck + 0, cv + 0, mb & True       # own buffers
    mx.eval(ck, cv, mb)
    return ck, cv, mb


def tree_mask():
    par = [-1] + [int(rng.integers(0, i)) for i in range(1, QL)]
    m = np.zeros((QL, QL), dtype=bool)
    for i in range(QL):
        j = i
        while j >= 0:
            m[i, j] = True
            j = par[j]
    return m


def inputs():
    q = bf(rng.standard_normal((1, H, QL, D)))
    kn = bf(rng.standard_normal((1, KVH, QL, D)))
    vn = bf(rng.standard_normal((1, KVH, QL, D)))
    anc = mx.array(tree_mask()[None, None])
    mx.eval(q, kn, vn, anc)
    return q, kn, vn, anc


def f(kv, q, kn, vn, anc, ck, cv, mb):
    ck[..., kv:kv + QL, :] = kn
    cv[..., kv:kv + QL, :] = vn
    mb[..., kv:kv + QL] = anc
    kL = kv + QL
    return mx.fast.scaled_dot_product_attention(q, ck[..., :kL, :], cv[..., :kL, :], scale=SCALE, mask=mb[..., :kL])


def record(kv):
    ins = inputs()
    stt = new_state()
    mx.synchronize()
    R.start()
    o = f(kv, *ins, *stt)
    mx.eval(o)
    mx.synchronize()
    R.stop()
    return dict(kv=kv, ins=ins, stt=stt, out=o, ops=R.ops_dump(), bytes=R.bytes_dump(), names=R.disp_names())


def disp_of(ops):
    """For each op, the ordinal of the first dispatch at or after it (-1 if none)."""
    d = np.full(len(ops), -1)
    nxt, cnt = -1, int(np.isin(ops[:, 0], [OPC["DT"], OPC["DTG"]]).sum())
    for i in range(len(ops) - 1, -1, -1):
        if ops[i, 0] in (OPC["DT"], OPC["DTG"]):
            cnt -= 1
            nxt = cnt
        d[i] = nxt
    return d


def diff(A, B):
    oa, ob = A["ops"], B["ops"]
    out = dict(structure="same", pso=[], buf_off=[], words=[], grid=[], tg=[], tgmem=[], other=[], canon=None)
    if oa.shape != ob.shape or not np.array_equal(oa[:, 0], ob[:, 0]):
        out["structure"] = f"op sequences differ ({len(oa)} vs {len(ob)} ops)"
        return out
    dsp = disp_of(oa)
    nm = A["names"]
    kern = lambda i: nm[dsp[i]] if dsp[i] >= 0 else "?"
    ca, cb, sa, sb = {}, {}, [], []
    for i in np.where(oa[:, 0] == OPC["BUF"])[0]:
        sa.append(ca.setdefault(int(oa[i, 10]), len(ca)))
        sb.append(cb.setdefault(int(ob[i, 10]), len(cb)))
    pers = sum(1 for pa, ia in ca.items() for pb, ib in cb.items() if ia == ib and pa == pb)
    out["canon"] = dict(same=sa == sb, buffers=len(ca), persistent=pers)
    for i in range(len(oa)):
        op = R.OPN[int(oa[i, 0])]
        if op == "PSO" and oa[i, 10] != ob[i, 10]:
            out["pso"].append((i, kern(i)))
        elif op == "BUF" and oa[i, 2] != ob[i, 2]:
            out["buf_off"].append(dict(i=i, bind=int(oa[i, 1]), kernel=kern(i), a=int(oa[i, 2]), b=int(ob[i, 2])))
        elif op in ("DT", "DTG"):
            if not np.array_equal(oa[i, 4:7], ob[i, 4:7]):
                out["grid"].append((i, kern(i), oa[i, 4:7].tolist(), ob[i, 4:7].tolist()))
            if not np.array_equal(oa[i, 7:10], ob[i, 7:10]):
                out["tg"].append((i, kern(i), oa[i, 7:10].tolist(), ob[i, 7:10].tolist()))
        elif op == "TGMEM" and oa[i, 2] != ob[i, 2]:
            out["tgmem"].append((i, kern(i), int(oa[i, 2]), int(ob[i, 2])))
        elif op in ("ENC", "BAR", "USE") and (oa[i, 1] != ob[i, 1]):
            out["other"].append((i, op))
    for a, b in zip(A["bytes"], B["bytes"]):
        if a["length"] != b["length"] or a["bind"] != b["bind"]:
            out["other"].append((a["k"], "BYTES layout"))
            continue
        for off in range(0, a["length"] - a["length"] % 4, 4):
            if a["data"][off:off + 4] != b["data"][off:off + 4]:
                out["words"].append(dict(k=a["k"], off=off, kernel=nm[a["disp"]] if a["disp"] >= 0 else "?",
                                         a=struct.unpack("<i", a["data"][off:off + 4])[0],
                                         b=struct.unpack("<i", b["data"][off:off + 4])[0]))
    return out


def show(tag, d):
    print(f"[attn] diff {tag}: structure {d['structure']}; buffers {d['canon']}; pso changes {len(d['pso'])} "
          f"{sorted(set(k for _, k in d['pso']))}; grid {len(d['grid'])}; threadgroup {len(d['tg'])}; tgmem "
          f"{len(d['tgmem'])}; other {d['other'][:4]}; buffer offsets {len(d['buf_off'])}; setBytes words "
          f"{len(d['words'])}")
    for b in d["buf_off"]:
        print(f"[attn]   offset: BUF op {b['i']} bind {b['bind']} ({b['kernel']}): {b['a']} -> {b['b']}")
    for w in d["words"]:
        print(f"[attn]   word: setBytes k={w['k']} off={w['off']} ({w['kernel']}): {w['a']} -> {w['b']}")
    for g in d["grid"]:
        print(f"[attn]   grid: op {g[0]} ({g[1]}): {g[2]} -> {g[3]}")


def fit(xs, vs):
    """(a, b, d, m) with value = (((x * a + b) div d) mod m) (d / m = 0: none), or None."""
    x1, x2, v1, v2 = xs[0], xs[1], vs[0], vs[1]
    if (v2 - v1) % (x2 - x1) == 0:
        a = (v2 - v1) // (x2 - x1)
        b = v1 - a * x1
        if all(a * x + b == v for x, v in zip(xs, vs)):
            return (a, b, 0, 0)
    for d in (8, 16, 32, 64, 128, 256):
        for b in range(0, 4 * d + 64):
            if all((x + b) // d == v for x, v in zip(xs, vs)):
                return (1, b, d, 0)
        for b in range(0, d):
            if all((x + b) % d == v for x, v in zip(xs, vs)):
                return (1, b, 0, d)
    return None


def evalf(r, x):
    a, b, d, m = r
    v = x * a + b
    if d:
        v //= d
    if m:
        v %= m
    return v


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


# ---- (1) records and diffs
recA = [record(kv) for kv in ALIGNED]
recU = [record(kv) for kv in UNFIT]
print(f"[attn] kernels (aligned kv {ALIGNED[0]}): {recA[0]['names']}")
print(f"[attn] kernels (unaligned kv {UNFIT[0]}): {recU[0]['names']}")
dA = diff(recA[0], recA[1])
show(f"aligned {ALIGNED[0]} vs {ALIGNED[1]}", dA)
dA2 = diff(recA[0], recA[2])
dU = diff(recU[0], recU[1])
show(f"unaligned {UNFIT[0]} vs {UNFIT[1]}", dU)
dX = diff(recA[0], recU[0])
show(f"across classes {ALIGNED[0]} vs {UNFIT[0]}", dX)


def clean(d):
    return (d["structure"] == "same" and d["canon"]["same"] and not d["pso"] and not d["grid"] and not d["tg"]
            and not d["tgmem"] and not d["other"])


within = [clean(d) for d in (dA, dA2, dU)] + [clean(diff(recU[0], r)) for r in recU[2:]]
line1 = all(within)
print(f"[attn] (1) within-class records differ only in words / offsets: {within} -> {'PASS' if line1 else 'FAIL'}; "
      f"across classes: pso changes on {sorted(set(k for _, k in dX['pso']))}, structure {dX['structure']}")

# ---- (2) fit rules on the unaligned records + the aligned ones (values only; targets = every word / offset that
# changed within either class). The aligned points pin the floor-division offsets at kL residue 0.
fitset = recU + ([r for r in recA if diff(recU[0], r)["structure"] == "same"])
xs = [r["kv"] for r in fitset]
print(f"[attn] fit set: kv {xs}")
targets = {}
for r in recU[1:] + recA[1:]:
    d = diff(recU[0] if r in recU else recA[0], r)
    for w in d["words"]:
        targets[("w", w["k"], w["off"])] = w["kernel"]
    for b in d["buf_off"]:
        targets[("b", b["i"], b["bind"])] = b["kernel"]


def value(rec, t):
    if t[0] == "w":
        return struct.unpack("<i", rec["bytes"][t[1]]["data"][t[2]:t[2] + 4])[0]
    return int(rec["ops"][t[1], 2])


rules, unfit = {}, []
for t in sorted(targets):
    r = fit(xs, [value(rec, t) for rec in fitset])
    if r is None:
        unfit.append(t)
    else:
        rules[t] = r
        print(f"[attn] rule {t} ({targets[t]}): value = ((kv * {r[0]} + {r[1]}) div {r[2] or '-'}) mod {r[3] or '-'}; "
              f"values {[value(rec, t) for rec in fitset]}")
base = record(UBASE)
dB = diff(recU[0], base)
changed_b = {("w", w["k"], w["off"]) for w in dB["words"]} | {("b", b["i"], b["bind"]) for b in dB["buf_off"]}
pred_ok = [evalf(rules[t], UBASE) == value(base, t) for t in rules]
line2 = (not unfit and changed_b <= set(rules) and all(pred_ok) and clean(dB))
print(f"[attn] (2) {len(rules)} rules, {len(unfit)} unfitted {unfit}; base kv {UBASE}: predicted {sum(pred_ok)}/"
      f"{len(pred_ok)} exactly, changed targets covered {changed_b <= set(rules)} -> {'PASS' if line2 else 'FAIL'}")

# ---- (3) patched replays on new unaligned kv; (4) the same record at aligned kv
q0, kn0, vn0, anc0 = base["ins"]
ck0, cv0, mb0 = base["stt"]
o0 = base["out"]
n_icb, err = R.build_icb()
print(f"[attn] ICB build from the base record: {n_icb} commands {('(error: ' + err + ')') if n_icb < 0 else ''}")
for t, r in rules.items():
    if t[0] == "w":
        n = R.rule_add2("int32", t[1], off=t[2], var=0, a=r[0], b=r[1], d=r[2], m=r[3])
    else:
        n = R.rule_add2("bufoff", t[1], var=0, a=r[0], b=r[1], d=r[2], m=r[3])
    if n < 0:
        print(f"[attn] rule_add2 failed for {t}")


def restore():
    u8(ck0)[:] = u8(BK)
    u8(cv0)[:] = u8(BV)
    u8(mb0)[:] = u8(BM)


def run(kvs, control):
    ok2 = oki = miss = 0
    for kv in kvs:
        ins = inputs()
        ref = f(kv, *ins, *new_state())
        mx.eval(ref)
        ref_b = lbits(ref)
        for dst, src in zip((q0, kn0, vn0, anc0), ins):
            u8(dst)[:] = u8(src)
        if control:
            restore()
            R.replay(2)
            miss += not np.array_equal(lbits(o0), ref_b)
        R.rules_apply_v([kv])
        restore()
        rc2, _ = R.replay(2)
        m2 = rc2 == 0 and np.array_equal(lbits(o0), ref_b)
        mi = False
        if n_icb > 0:
            restore()
            rci, _ = R.replay_icb()
            mi = rci == 0 and np.array_equal(lbits(o0), ref_b)
        ok2 += m2
        oki += mi
        print(f"[attn]   kv {kv} (kL {kv + QL}, mod 32 = {(kv + QL) % 32}): mode 2 {m2}, ICB {mi}")
    return ok2, oki, miss


print("[attn] (3) new unaligned kv:")
ok2, oki, miss = run(UTEST, control=True)
line3 = ok2 == len(UTEST) and oki == len(UTEST) and miss == len(UTEST)
print(f"[attn] (3) patched bitwise == MLX: mode 2 {ok2}/{len(UTEST)}, ICB {oki}/{len(UTEST)}; stale control misses "
      f"{miss}/{len(UTEST)} -> {'PASS' if line3 else 'FAIL'}")
print("[attn] (4) info: the unaligned record at aligned kv:")
ok2a, okia, _ = run(ATEST, control=False)
print(f"[attn] (4) info: mode 2 {ok2a}/{len(ATEST)}, ICB {okia}/{len(ATEST)} bitwise == MLX's own (aligned) forward")
verdict = "PASS" if (line1 and line2 and line3) else ("KILL" if not line3 else "FAIL")
print(f"[attn] RESULT: {verdict}: (1) {line1} (2) {line2} (3) {line3}; (4) info {ok2a}/{okia} of {len(ATEST)}; "
      f"lifetime peak {peak_gb():.2f} GB")
