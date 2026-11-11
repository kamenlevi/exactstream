"""FAST-NAX GPU check (Leap, Thu 2026-11-05 19:xx, pre-registered in TABLE_350 'FAST-NAX'; run after Fri 00:15 under a
share lock, MEM ~4 GB): QOM drops + U3 sure rows on NAX6-N's one-call (scripts/ooo1cq.py), on the CONVERTED records3
(native+p11+pad) of layers 20 / 44, pools of all 512 experts through BigMoE's loader (tier.load_index / make_pools /
make_sidecar / ext.ensure; read-only use). 16-row calls, top-10, PLAN 4433333333. A fake release region as ooo1c_check.
Each scenario: ~70 % resident bases / ~80 % sidecars, 4 U3 sure rows (rank 0-1 sidecars -2 at post), 2 LP1 experts (sidecar
resolves to -2), 3 QOM drops (late experts used only at ranks >= 2: their base resolves to -2; one is -2 at post).
  reference: NN.tier_glu_nax6n on the final slots (a dropped pair reads its row's rank-0 slot, as 3c's QOM) + combine_phase
             with the dropped pairs' scores 0
  (a) released before the call, (b) late release (20 ms): one-call-Q + combine_phase == reference, bitwise; rs_m == the
      final slots (-2 = dropped); rs_l == each pair's own final sidecar; a dropped pair's rows are exactly 0; no timeout
  (d) OOQ_DIAG=1: ooo1c's existing NAX6-N one-call on the same scenarios without drops (diagnostic for its rs_l writes)
  (e) OOQ_TIME=1: ms per MoE layer call in 48-call chains (two layers' scenarios alternating, pre-released, ~96 % resident):
      NAX6-N 1C-Q + combine_phase vs 3c-N 1C QOM (fused combine), interleaved rounds. Line: <= 0.85x PASS, > 0.95x KILL.
-> ~/Leap/data/ooo1cq_check.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import tierglu_nax6 as N6
import tierglu_nax6n as NN
import ooo1c as OC
import ooo1cq as OQ
ext = BG.ext
TG = T._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/ooo1cq_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nFAST-NAX check {time.strftime('%a %H:%M:%S')} ({os.environ.get('OOQ_COND', 'share lock')}; MLX {mx.__version__})")
PLAN, M, K, R4 = "4433333333", 16, 10, 2
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
C = 512
d_, ix = T.load_index(BARE, layout_ok=True)
nxt = [0]
P = {}
for Lm in (20, 44):
    r = ix["layers"][str(Lm)]
    assert r.get("layout") == "native+p11+pad", r.get("layout")
    L = nxt[0]
    base, _ = T.make_pools(ext, L, 512, C, SH4, d_, r)
    Ls, side, _ = T.make_sidecar(ext, 512, C, SH4, d_, r)
    nxt[0] = Ls + 1
    allids = np.arange(512, dtype=np.int32)
    for j in range(0, 512, 64):
        ext.ensure(L, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
    sm_of = np.array(ext.lookup(L, allids.tolist()), np.int32)
    sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
    assert (sm_of >= 0).all() and (sl_of >= 0).all(), "an expert did not load"
    assert base["gate_proj"]["s3"].dtype == mx.uint32, "p11 pools expected on the converted store"
    P[Lm] = (base, side, sm_of, sl_of)
say(f"pools built: layers 20 / 44, 512 experts each, native+p11+pad (p11 pieces), C {C}")

NREG, PAIRS, RS, NL = 128, 1, 192, 96
region = mx.zeros((NREG * PAIRS * RS,), mx.int32)
status = mx.zeros((64,), mx.int32)
mx.eval(region, status)
REG = np.frombuffer(memoryview(region), dtype=np.int32)
REG.setflags(write=True)
STA = np.frombuffer(memoryview(status), dtype=np.int32)

def release(off, slots, seq):
    w = OC.release_words(slots, seq)
    for i, v in enumerate(w):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq

def scenario(rng, sm_of, sl_of, fm=0.7, fl=0.8, n_sure=4, n_lp1=2, n_drop=3, drops=True):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
    uniq = np.unique(ids)
    res_m = set(rng.choice(uniq, int(fm * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(fl * len(uniq)), replace=False).tolist())
    hi_only = [e for e in uniq.tolist() if e not in res_m and not np.isin(ids[:, :R4], [e]).any()]   # late, ranks >= 2 only
    drop = set(rng.choice(hi_only, min(n_drop, len(hi_only)), replace=False).tolist()) if drops and hi_only else set()
    post_drop = set(sorted(drop)[:1])                                        # one drop known at post (-2 in pre_m)
    nl = set(uniq.tolist()) - res_l
    lp1 = set(rng.choice(sorted(nl), min(n_lp1, len(nl)), replace=False).tolist()) if nl else set()
    sure = rng.choice(M, n_sure, replace=False)
    dropped = np.isin(ids, list(drop))
    sm = np.where(dropped, -2, sm_of[ids]).astype(np.int32)
    sl = np.where(np.isin(ids[:, :R4], list(lp1)), -2, sl_of[ids[:, :R4]]).astype(np.int32)
    sl[sure] = -2
    pre_m = np.where(np.isin(ids, list(res_m)), sm_of[ids], -1).astype(np.int32)
    pre_m[np.isin(ids, list(post_drop))] = -2
    pre_l = np.where(np.isin(ids[:, :R4], list(res_l)), sl_of[ids[:, :R4]], -1).astype(np.int32)
    pre_l[sure] = -2
    return ids, sm, sl, pre_m, pre_l, dropped

def ref(x, msb, lsb, sm, sl, dropped, sc):
    smr = mx.array(np.where(dropped, sm[:, :1], sm).astype(np.int32))      # a dropped pair: its row's rank-0 slot (3c's rule)
    slr = mx.array(sl)
    y = NN.tier_glu_nax6n(x, msb, lsb, smr, slr, plan=PLAN, grp=N6.gpu_grp(smr, slr, PLAN))
    scr = mx.where(mx.array(dropped), mx.zeros_like(sc), sc)
    return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), scr)

def one_q(x, msb, lsb, pre_m, pre_l, ids, offs, sc):
    y, rs_m, rs_l = OQ.tier_glu_nax6n_1cq(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), mx.array(ids), PLAN, region, offs, status)
    yc = TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)
    return yc, y, rs_m, rs_l

eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
rng = np.random.default_rng(1105)
ok_a, ok_b, n_a, ndrop, nsure = True, True, 0, 0, 0
seq = 7
for Lm in (20, 44):
    msb, lsb, sm_of, sl_of = P[Lm]
    for rep in range(6):
        seq += 1
        ids, sm, sl, pre_m, pre_l, dropped = scenario(rng, sm_of, sl_of)
        ndrop += int(dropped.sum()); nsure += 4
        off_m, off_l = OC.region_offset(seq, NL, Lm, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + Lm, NREG, PAIRS, RS)
        offs = mx.array([off_m, off_l, seq], mx.int32)
        x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
        mx.eval(x, sc)
        y_ref = ref(x, msb, lsb, sm, sl, dropped, sc)
        mx.eval(y_ref)
        late = rep >= 4
        yc, y, rs_m, rs_l = one_q(x, msb, lsb, pre_m, pre_l, ids, offs, sc)
        if not late:                                                       # (a) released before the call
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            mx.eval(yc, y, rs_m, rs_l)
            dt = 0.0
        else:                                                              # (b) late release, 20 ms after the launch
            t0 = time.perf_counter()
            mx.async_eval(yc, y, rs_m, rs_l)
            while time.perf_counter() - t0 < 0.020:
                pass
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            mx.eval(yc, y, rs_m, rs_l)
            dt = time.perf_counter() - t0
        zr = bool(mx.all(y.reshape(M * K, -1)[mx.array(np.where(dropped.reshape(-1))[0])] == 0).item()) if dropped.any() else True
        same = (eqb(yc, y_ref) and np.array_equal(np.array(rs_m), sm.reshape(-1)) and np.array_equal(np.array(rs_l), sl.reshape(-1))
                and zr and int(STA[0]) == 0 and (not late or dt >= 0.020))
        if late:
            ok_b &= same
        else:
            ok_a &= same; n_a += 1
        if not same or late:
            say(f"  L{Lm} rep {rep}{' late' if late else ''}: == ref {eqb(yc, y_ref)}, rs_m {np.array_equal(np.array(rs_m), sm.reshape(-1))}, "
                f"rs_l {np.array_equal(np.array(rs_l), sl.reshape(-1))}, dropped rows 0 {zr} ({int(dropped.sum())} pairs), "
                f"timeouts {int(STA[0])}" + (f", call {1e3 * dt:.1f} ms (release at 20.0)" if late else ""))
say(f"(a) released before the call ({n_a} calls; {ndrop} dropped pairs and {nsure} U3 sure rows over all calls): "
    f"one-call-Q + combine == NAX6-N on the final slots + combine (drops at weight 0), slots resolved, no timeout: "
    f"{'PASS' if ok_a else 'FAIL'}")
say(f"(b) late release (20 ms): {'PASS' if ok_b else 'FAIL'}")

if os.environ.get("OOQ_DIAG") == "1":                                      # (d) the existing NAX one-call on U3 scenarios
    bad = 0
    for Lm in (20, 44):
        msb, lsb, sm_of, sl_of = P[Lm]
        for rep in range(4):
            seq += 1
            ids, sm, sl, pre_m, pre_l, dropped = scenario(rng, sm_of, sl_of, drops=False)
            off_m, off_l = OC.region_offset(seq, NL, Lm, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + Lm, NREG, PAIRS, RS)
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
            y_ref = ref(x, msb, lsb, sm, sl, dropped, sc)
            y, rs_m, rs_l = OC.tier_glu_nax6n_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), mx.array(ids), PLAN, region,
                                                 mx.array([off_m, off_l, seq], mx.int32), status)
            yc = TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)
            mx.eval(y_ref, yc, rs_l)
            e1, e2 = eqb(yc, y_ref), np.array_equal(np.array(rs_l), sl.reshape(-1))
            bad += (not e1) + (not e2)
            say(f"  (d) ooo1c NAX6-N one-call, U3 rows, L{Lm} rep {rep}: == ref {e1}, rs_l == own sidecars {e2}")
    say(f"(d) diagnostic, ooo1c's existing NAX6-N one-call with U3 sure rows: {bad} mismatches over 8 calls "
        f"(its head writes rs_l with the group's first tier)")

if os.environ.get("OOQ_TIME") == "1":                                      # (e) NAX6-N 1C-Q vs 3c-N 1C QOM, same calls
    SC = []
    for Lm in (20, 44):
        msb, lsb, sm_of, sl_of = P[Lm]
        seq += 1
        ids, sm, sl, pre_m, pre_l, dropped = scenario(rng, sm_of, sl_of, fm=0.96, fl=0.97)
        off_m, off_l = OC.region_offset(seq, NL, Lm, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + Lm, NREG, PAIRS, RS)
        release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
        SC.append((Lm, ids, pre_m, pre_l, mx.array([off_m, off_l, seq], mx.int32), int(dropped.sum())))
    sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
    x0 = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    mx.eval(sc, x0)
    def chain(arm):
        x = x0
        for c in range(48):
            Lm, ids, pre_m, pre_l, offs, _ = SC[c % 2]
            msb, lsb = P[Lm][0], P[Lm][1]
            if arm == "naxq":
                y = one_q(x, msb, lsb, pre_m, pre_l, ids, offs, sc)[0]
            else:
                y = OC.tier_glu_3cn_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc, qom=True)[0]
            x = (y.reshape(M, -1).astype(mx.float32) * 0.1).astype(mx.bfloat16)
        return x
    TT = {"naxq": [], "c3nq": []}
    rounds = int(os.environ.get("OOQ_ROUNDS", "18"))
    for rr in range(rounds):
        for a in (("naxq", "c3nq") if rr % 2 == 0 else ("c3nq", "naxq")):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain(a)); mx.synchronize()
            TT[a].append(1e3 * (time.perf_counter() - t0) / 48)
    med = {a: float(np.median(v[3:])) for a, v in TT.items()}
    rt = med["naxq"] / med["c3nq"]
    v = "PASS" if rt <= 0.85 else ("KILL" if rt > 0.95 else "BETWEEN")
    say(f"(e) ms per MoE layer call (48-call chains, median of {rounds - 3}; ~96 % resident, drops {SC[0][5]} / {SC[1][5]} pairs): "
        f"NAX6-N 1C-Q + combine {med['naxq']:.3f}, 3c-N 1C QOM {med['c3nq']:.3f} -> {rt:.3f}x; timeouts {int(STA[0])}; "
        f"speed line (pre-registered: <= 0.85x PASS, > 0.95x KILL): {v} ({os.environ.get('OOQ_COND', 'share lock')})")
