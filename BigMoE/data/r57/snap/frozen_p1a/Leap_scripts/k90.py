"""K90 harness (Leap, Wed 2026-11-04 18:0x; MISSION 33, ~/FableGod/MISSION_33_KERNEL90.md): the shared setup for the expert-
kernel microbenchmarks. Layer 20's real records3 (all 512 experts) + the native repack (NAX6-N's layout), real routing
(Breakthrough's expA traces125, code / chat; a pass = S consecutive tokens' top-10, plan 4433333333), and per call:
  bytes   the call's record bytes = 2,150,400 B per distinct expert (MSB-3 codes + scales + biases, gate / up / down) +
          768,000 B per distinct expert used at a 4-bit rank (LSB-1 plane + LSB bias), i.e. what streaming must move;
  groups  the call's (expert, tier) groups (NAX6's work units);
  ms      interleaved medians (8 calls per eval), GB/s = bytes / ms, us per group;
  bits    a candidate's output vs NAX6 (== NAX6-N, bitwise), all bf16 bits.
Use: import k90 as H; H.load(native=True, records3=False); for trace, S, ids, x in H.passes(...): ...; H.timeit({...})."""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
from tierglu import PROJ
import tierglu_nax6 as N6
import tierglu_nax6n as NN

LAYER, PLAN = 20, "4433333333"
MSB_B, LSB_B = 2_150_400, 768_000
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
L = {}                                                  # msb / lsb (records3), msb_n / lsb_n (native)


def load(layer=LAYER, native=True, records3=False):
    RD = os.path.expanduser("~/BigMoE/models/bare/records3")
    idx = json.load(open(f"{RD}/index.json"))["layers"][str(layer)]
    assert idx.get("layout", "row-major") == "row-major", (                 # a raw row-major reader: refuse a converted
        f"records3 L{layer} is {idx.get('layout')}: k90 reads row-major bytes only (use the engine's read path)")
    E, REC = idx["E"], idx["recsz"]
    raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
    msb, lsb = {}, {}
    for p, pr in enumerate(PROJ):
        o, i = SH[pr]
        ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
        msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
                   "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
                   "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
        lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
        lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
                   "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    del raw
    if native:
        L["msb_n"], L["lsb_n"] = NN.repack(msb, lsb)
        mx.eval([v for d in (L["msb_n"], L["lsb_n"]) for pr in d for v in d[pr].values()])
    if records3:
        L["msb"], L["lsb"] = msb, lsb
        mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
    return L


def trace(name, layer=LAYER):
    return np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{name}_800_g.npz"))[f"L{layer}"].astype(np.int32)


def call_bytes(ids):
    """record bytes one call must stream: MSB pieces per distinct expert, LSB pieces per distinct 4-bit expert"""
    ids = np.asarray(ids)
    r4 = [i for i, b in enumerate(PLAN) if b == "4"]
    return MSB_B * len(np.unique(ids)) + LSB_B * len(np.unique(ids[:, r4]))


def call_groups(ids):
    ids = np.asarray(ids)
    t4 = np.array([b == "4" for b in PLAN])
    return len({(int(e), bool(t4[r])) for row in ids for r, e in enumerate(row[:len(PLAN)])})


def passes(S, n=6, traces=("code", "chat"), seed=0):
    """(trace, ids [S, 10], ms_, ls_, x [S, 2560] bf16 N(0, 0.5)) for n evenly spaced passes per trace"""
    mx.random.seed(seed)
    for name in traces:
        tr = trace(name)
        for t in np.linspace(0, tr.shape[0] - S, n).astype(int):
            ids = tr[t:t + S, :10].copy()
            yield name, ids, mx.array(ids), mx.array(ids[:, :2].copy()), (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)


def nax6n(x, ms_, ls_):
    return NN.tier_glu_nax6n(x, L["msb_n"], L["lsb_n"], ms_, ls_, plan=PLAN)


def nax6(x, ms_, ls_):
    return N6.tier_glu_nax6(x, L["msb"], L["lsb"], ms_, ls_, plan=PLAN)


def same_bits(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def bits_report(a, ref):
    """(share of bf16 outputs that differ, max |diff| in ulps of the reference's value)"""
    a16, r16 = np.array(a.view(mx.uint16)).reshape(-1), np.array(ref.view(mx.uint16)).reshape(-1)
    d = a16 != r16
    if not d.any():
        return 0.0, 0
    af, rf = np.array(a.astype(mx.float32)).reshape(-1)[d], np.array(ref.astype(mx.float32)).reshape(-1)[d]
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(rf), 1e-30))) - 7)
    return float(d.mean()), float((np.abs(af - rf) / ulp).max())


def timeit(fns, rounds=7, per_eval=8):
    """fns: name -> callable() returning one call's output; interleaved medians of `rounds`, s per call"""
    run = {k: (lambda f=f: [f() for _ in range(per_eval)]) for k, f in fns.items()}
    for f in run.values():
        mx.eval(*f())
    ts = {k: [] for k in run}
    keys = list(run)
    for r in range(rounds):
        for k in (keys if r % 2 == 0 else keys[::-1]):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(*run[k]()); mx.synchronize()
            ts[k].append((time.perf_counter() - t0) / per_eval)
    return {k: float(np.median(v)) for k, v in ts.items()}
