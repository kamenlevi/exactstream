"""OOO-1C check (Leap, Wed 2026-11-04 15:xx; TABLE_350 'OOO-1C'): one-call OOO on the NAX path (scripts/ooo1c.py) on
real records3 pools (layers 20 / 44, all 512 experts, slot = expert id), 16-row calls, top-10, PLAN 4433333333, ~70 % of
each call's experts resident in the base pool and ~80 % in the sidecar, some late sidecars released as -2 (LP1):
 (1) bits, region released BEFORE the call: one-call + combine == NAX6 on the final slots + combine (one call) == today's
     two phases (gpu_grp_phase 1 / 2 + combine_phase, ref2._ooo_nax6's arithmetic), bitwise; rs_m / rs_l == the final
     slots; status[0] == 0 (no timeout);
 (2) a LATE release: the call is launched on an unreleased region, the CPU writes the release 20 ms later (words, a
     fence, then the tags, line 0 last: the loader's order): the same bits, no timeout, and the call took >= 20 ms;
 (3) OOC_TIME=1: ms per MoE layer call in a 48-call chain (two layers' pools alternating, regions pre-released): today's
     two phases (+ 2 one-thread copy kernels standing in for the ext's spins) vs one call; median of 15 after 3.
Usage: [BigMoE venv] [OOC_TIME=1] ooo1c_check.py -> ~/Leap/data/ooo1c_check.txt (GPU, MEM ~3.5 GB)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
import mlx.core as mx
import ooo1c as OC
import tierglu_nax6 as N6
import bigmoe.tier as BT
from tierglu_nax import rowsum64
TG = BT._tier_glu_mod()

OUT = os.path.expanduser("~/Leap/data/ooo1c_check.txt")
TIME = os.environ.get("OOC_TIME") == "1"
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nOOO-1C check {time.strftime('%a %H:%M:%S')} ({os.environ.get('OOC_COND', 'share lock')}; resident base / sidecar "
    f"{os.environ.get('OOC_RES_M', '0.7')} / {os.environ.get('OOC_RES_L', '0.8')}, rounds {os.environ.get('OOC_ROUNDS', '18')})")
PLAN, M, K, R4 = "4433333333", 16, 10, 2
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
idx = json.load(open(f"{RD}/index.json"))["layers"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}

def pools(r):
    E, REC = r["E"], r["recsz"]
    raw = np.memmap(f"{RD}/{r['file']}", dtype=np.uint8, mode="r", shape=(E, REC))
    msb, lsb = {}, {}
    for p, pr in enumerate(("gate_proj", "up_proj", "down_proj")):
        o, i = SH[pr]
        ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
        lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    del raw
    return msb, lsb

# the region: coherent mode (one line pair per entry), RS 192 (XLINES 4), 128 entries (as gated.Gate); entry (seq, L) at OC.region_offset
NREG, PAIRS, RS, NL = 128, 1, 192, 96
region = mx.zeros((NREG * PAIRS * RS,), mx.int32)
status = mx.zeros((64,), mx.int32)
mx.eval(region, status)
REG = np.frombuffer(memoryview(region), dtype=np.int32)
REG.setflags(write=True)
STA = np.frombuffer(memoryview(status), dtype=np.int32)

def release(off, slots, seq, early_tags=True):
    """the loader's release of one (seq, L) entry: stamped words, a fence (numpy writes are program-ordered on one CPU
    thread; the GPU sees them through coherent(system) loads), then the tags, line 0 last"""
    w = OC.release_words(slots, seq)
    for i, v in enumerate(w):
        REG[off + 32 * (i // 31) + 1 + i % 31] = np.int32(np.uint32(v).view(np.int32))
    for j in range(RS // 32 - 1, -1, -1):
        REG[off + 32 * j] = seq

FRAC_M, FRAC_L = float(os.environ.get("OOC_RES_M", "0.7")), float(os.environ.get("OOC_RES_L", "0.8"))
ROUNDS = int(os.environ.get("OOC_ROUNDS", "18"))
def scenario(rng, seq):
    ids = np.stack([rng.choice(512, K, replace=False) for _ in range(M)]).astype(np.int32)
    uniq = np.unique(ids)
    res_m = set(rng.choice(uniq, int(FRAC_M * len(uniq)), replace=False).tolist())
    res_l = set(rng.choice(uniq, int(FRAC_L * len(uniq)), replace=False).tolist())
    lp1 = set(rng.choice(sorted(set(uniq.tolist()) - res_l), 2, replace=False).tolist()) if len(set(uniq.tolist()) - res_l) >= 2 else set()
    sm = ids.copy()                                                          # final slots: slot = expert id
    sl = np.where(np.isin(ids[:, :R4], list(lp1)), -2, ids[:, :R4]).astype(np.int32)
    pre_m = np.where(np.isin(ids, list(res_m)), ids, -1).astype(np.int32)
    pre_l = np.where(np.isin(ids[:, :R4], list(res_l)), ids[:, :R4], -1).astype(np.int32)
    return ids, sm, sl, pre_m, pre_l

def ref_one(x, msb, lsb, sm, sl, sc):
    y = N6.tier_glu_nax6(x, msb, lsb, mx.array(sm), mx.array(sl), plan=PLAN, grp=N6.gpu_grp(mx.array(sm), mx.array(sl), PLAN))
    return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc)

def two_phase(x, msb, lsb, pre_m, pre_l, sm, sl, sc, spins=None):
    ok = (pre_m >= 0) & np.concatenate([pre_l >= 0, np.ones((M, K - R4), bool)], axis=1)
    phase = mx.array(np.where(ok, 1, 2).astype(np.int32))
    pm, pl = mx.array(pre_m), mx.array(np.maximum(pre_l, 0))
    xs = rowsum64(x)
    y1 = N6.tier_glu_nax6(x, msb, lsb, pm, pl, plan=PLAN, xs=xs, grp=N6.gpu_grp_phase(pm, pl, PLAN, phase, 1)).reshape(M * K, -1)
    smx, slx = mx.array(sm), mx.array(sl)
    if spins is not None:                                                    # the ext's two spin kernels (stand-ins)
        smx, slx = spins(mx.depends(smx, [y1])), spins(mx.depends(slx, [y1]))
    y2 = N6.tier_glu_nax6(x, msb, lsb, smx, slx, plan=PLAN, xs=xs, grp=N6.gpu_grp_phase(smx, slx, PLAN, phase, 2)).reshape(M * K, -1)
    return TG.combine_phase(y1, y2, phase, sc)

def one_call(x, msb, lsb, pre_m, pre_l, ids, offs, sc):
    y, rs_m, rs_l = OC.tier_glu_nax6_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), mx.array(ids), PLAN, region, offs, status)
    return TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc), rs_m, rs_l

eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
P = {L: pools(idx[str(L)]) for L in (20, 44)}
say("pools built (layers 20, 44; 512 experts each)")
rng = np.random.default_rng(1104)
ok1, ok2, n1 = True, True, 0
seq = 7
for L in (20, 44):
    msb, lsb = P[L]
    for rep in range(4):
        seq += 1
        ids, sm, sl, pre_m, pre_l = scenario(rng, seq)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        assert off_m != off_l
        offs = mx.array([off_m, off_l, seq], mx.int32)
        x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
        mx.eval(x, sc)
        y_ref = ref_one(x, msb, lsb, sm, sl, sc)
        y_2p = two_phase(x, msb, lsb, pre_m, pre_l, sm, sl, sc)
        if rep < 2:                                                          # (1) released before the call
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            y_1c, rs_m, rs_l = one_call(x, msb, lsb, pre_m, pre_l, ids, offs, sc)
            mx.eval(y_ref, y_2p, y_1c, rs_m, rs_l)
            same = eqb(y_1c, y_ref) and eqb(y_1c, y_2p) and np.array_equal(np.array(rs_m), sm.reshape(-1)) \
                and np.array_equal(np.array(rs_l), sl.reshape(-1)) and int(STA[0]) == 0
            ok1 &= same
            n1 += 1
            if not same:
                say(f"  L{L} rep {rep}: 1c == ref {eqb(y_1c, y_ref)}, == two-phase {eqb(y_1c, y_2p)}, rs_m "
                    f"{np.array_equal(np.array(rs_m), sm.reshape(-1))}, rs_l {np.array_equal(np.array(rs_l), sl.reshape(-1))}, timeouts {int(STA[0])}")
        else:                                                                # (2) late release, 20 ms after the launch
            mx.eval(y_ref, y_2p)
            y_1c, rs_m, rs_l = one_call(x, msb, lsb, pre_m, pre_l, ids, offs, sc)
            t0 = time.perf_counter()
            mx.async_eval(y_1c, rs_m, rs_l)
            while time.perf_counter() - t0 < 0.020:
                pass
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            mx.eval(y_1c, rs_m, rs_l)
            dt = time.perf_counter() - t0
            same = eqb(y_1c, y_ref) and eqb(y_1c, y_2p) and int(STA[0]) == 0 and dt >= 0.020
            ok2 &= same
            say(f"  L{L} late release: == ref {eqb(y_1c, y_ref)}, == two-phase {eqb(y_1c, y_2p)}, timeouts {int(STA[0])}, "
                f"call {1e3 * dt:.1f} ms (release at 20.0)")
say(f"(1) released before the call: one-call == NAX6 on the final slots == two phases, slots resolved, no timeout: "
    f"{'PASS' if ok1 else 'FAIL'} ({n1} calls)")
say(f"(2) late release (20 ms): {'PASS' if ok2 else 'FAIL'}")

if TIME:
    spin_k = mx.fast.metal_kernel(name="ooc_spin_standin", input_names=["a"], output_names=["o"],
                                  source="const uint i = thread_position_in_grid.x; if (i == 0) { for (uint j = 0; j < a_shape[0]; j++) o[j] = a[j]; }")
    spins = lambda a: spin_k(inputs=[a], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[a.shape], output_dtypes=[mx.int32])[0]
    SC = []
    for j, L in enumerate((20, 44)):                                         # two pre-released scenarios
        seq += 1
        ids, sm, sl, pre_m, pre_l = scenario(rng, seq)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
        SC.append((L, ids, sm, sl, pre_m, pre_l, mx.array([off_m, off_l, seq], mx.int32)))
    sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
    x0 = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
    mx.eval(sc, x0)
    def chain(arm):
        x = x0
        for c in range(48):
            L, ids, sm, sl, pre_m, pre_l, offs = SC[c % 2]
            msb, lsb = P[L]
            if arm == "2ph":
                y = two_phase(x, msb, lsb, pre_m, pre_l, sm, sl, sc, spins=spins)
            else:
                y = one_call(x, msb, lsb, pre_m, pre_l, ids, offs, sc)[0]
            x = (y.astype(mx.float32) * 0.1).astype(mx.bfloat16)
        return x
    T = {"2ph": [], "1c": []}
    for r in range(ROUNDS):
        for a in (("2ph", "1c") if r % 2 == 0 else ("1c", "2ph")):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain(a)); mx.synchronize()
            T[a].append(1e3 * (time.perf_counter() - t0) / 48)
    med = {a: float(np.median(v[3:])) for a, v in T.items()}
    say(f"(3) ms per MoE layer call (48-call chains, median of 15): two phases {med['2ph']:.3f}, one call {med['1c']:.3f} "
        f"-> {med['2ph'] - med['1c']:+.3f} ms per layer = {48 * (med['2ph'] - med['1c']):+.2f} ms per check (48 MoE layers); "
        f"timeouts {int(STA[0])}")

# ---------------------------------------------------------------- the 3c form (today's exact default)
def two_phase_3c(x, msb, lsb, pre_m, pre_l, sm, sl, sc, spins=None):
    ok = (pre_m >= 0) & np.concatenate([pre_l >= 0, np.ones((M, K - R4), bool)], axis=1)
    phase = mx.array(np.where(ok, 1, 2).astype(np.int32))
    y1 = TG.tier_glu_phase(x, msb, lsb, mx.array(pre_m), mx.array(np.maximum(pre_l, 0)), phase, 1, plan=PLAN)
    smx, slx = mx.array(sm), mx.array(sl)
    if spins is not None:
        smx, slx = spins(mx.depends(smx, [y1])), spins(mx.depends(slx, [y1]))
    y2 = TG.tier_glu_phase(x, msb, lsb, smx, slx, phase, 2, plan=PLAN)
    return TG.combine_phase(y1, y2, phase, sc)

def one_call_3c(x, msb, lsb, pre_m, pre_l, offs, sc):
    return OC.tier_glu_3c_1c(x, msb, lsb, mx.array(pre_m), mx.array(pre_l), PLAN, region, offs, status, sc)

ok3, ok3l = True, True
for L in (20, 44):
    msb, lsb = P[L]
    for rep in range(3):
        seq += 1
        ids, sm, sl, pre_m, pre_l = scenario(rng, seq)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        offs = mx.array([off_m, off_l, seq], mx.int32)
        x = (mx.random.normal((M, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
        mx.eval(x, sc)
        y_ref = TG.tier_glu(x, msb, lsb, mx.array(sm), mx.array(sl), plan=PLAN, scores=sc)
        y_2p = two_phase_3c(x, msb, lsb, pre_m, pre_l, sm, sl, sc)
        mx.eval(y_ref, y_2p)
        if rep < 2:
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            y_1c, rs_m, rs_l = one_call_3c(x, msb, lsb, pre_m, pre_l, offs, sc)
            mx.eval(y_1c, rs_m, rs_l)
            same = eqb(y_1c, y_ref) and eqb(y_1c, y_2p) and np.array_equal(np.array(rs_m), sm.reshape(-1)) \
                and np.array_equal(np.array(rs_l), sl.reshape(-1)) and int(STA[0]) == 0
            ok3 &= same
            if not same:
                say(f"  3c L{L} rep {rep}: == tier_glu {eqb(y_1c, y_ref)}, == two-phase {eqb(y_1c, y_2p)}, rs_m "
                    f"{np.array_equal(np.array(rs_m), sm.reshape(-1))}, rs_l {np.array_equal(np.array(rs_l), sl.reshape(-1))}, timeouts {int(STA[0])}")
        else:
            y_1c, rs_m, rs_l = one_call_3c(x, msb, lsb, pre_m, pre_l, offs, sc)
            t0 = time.perf_counter()
            mx.async_eval(y_1c, rs_m, rs_l)
            while time.perf_counter() - t0 < 0.020:
                pass
            release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
            mx.eval(y_1c, rs_m, rs_l)
            dt = time.perf_counter() - t0
            same = eqb(y_1c, y_ref) and eqb(y_1c, y_2p) and int(STA[0]) == 0 and dt >= 0.020
            ok3l &= same
            say(f"  3c L{L} late release: == tier_glu {eqb(y_1c, y_ref)}, == two-phase {eqb(y_1c, y_2p)}, timeouts {int(STA[0])}, call {1e3 * dt:.1f} ms")
say(f"(1-3c) released before: one-call 3c == tier_glu(scores) == BigMoE's two phases + combine_phase, slots resolved: "
    f"{'PASS' if ok3 else 'FAIL'}; (2-3c) late release: {'PASS' if ok3l else 'FAIL'}")

if TIME:
    SC = []                                                                  # fresh released regions (later seqs may reuse
    for j, L in enumerate((20, 44)):                                         # an entry of the NAX timing's scenarios)
        seq += 1
        ids, sm, sl, pre_m, pre_l = scenario(rng, seq)
        off_m, off_l = OC.region_offset(seq, NL, L, NREG, PAIRS, RS), OC.region_offset(seq, NL, NL // 2 + L, NREG, PAIRS, RS)
        release(off_m, sm.reshape(-1).tolist(), seq); release(off_l, sl.reshape(-1).tolist(), seq)
        SC.append((L, ids, sm, sl, pre_m, pre_l, mx.array([off_m, off_l, seq], mx.int32)))
    def chain3(arm):
        x = x0
        for c in range(48):
            L, ids, sm, sl, pre_m, pre_l, offs = SC[c % 2]
            msb, lsb = P[L]
            y = two_phase_3c(x, msb, lsb, pre_m, pre_l, sm, sl, sc, spins=spins) if arm == "2ph" else \
                one_call_3c(x, msb, lsb, pre_m, pre_l, offs, sc)[0]
            x = (y.astype(mx.float32) * 0.1).astype(mx.bfloat16)
        return x
    T3 = {"2ph": [], "1c": []}
    for r in range(ROUNDS):
        for a in (("2ph", "1c") if r % 2 == 0 else ("1c", "2ph")):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain3(a)); mx.synchronize()
            T3[a].append(1e3 * (time.perf_counter() - t0) / 48)
    m3 = {a: float(np.median(v[3:])) for a, v in T3.items()}
    say(f"(3-3c) ms per MoE layer call (48-call chains, median of 15): two phases {m3['2ph']:.3f}, one call {m3['1c']:.3f} "
        f"-> {m3['2ph'] - m3['1c']:+.3f} ms per layer = {48 * (m3['2ph'] - m3['1c']):+.2f} ms per check; timeouts {int(STA[0])}")
