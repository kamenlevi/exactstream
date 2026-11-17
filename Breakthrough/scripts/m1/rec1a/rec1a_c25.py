"""rec1a_c25.py (Breakthrough, Tue 17 Nov 2026 07:1x; MODULE 1a milestone 2, the engine run). Pre-registered in
docs/MODULE1A.md, block 'Tue 06:5x' (b4e7e38) + its two 07:0x amendments; Leap (c25's owner) read it 06:59.

Launcher, for the real engine (inside the slot's lock; env = c25_env.py's exports + the run's knobs, see
run_rec1a_c25.sh):   python -I -B scripts/m1/rec1a/rec1a_c25.py <C> <CAP>   (the args c25_loop.py takes)
  1. loads libmtlrec before mlx;
  2. imports Leap's c25_engine.py (read-only) as sys.modules["c25_engine"] and wraps its window();
  3. runs Leap's c25_loop.py from its source with ONE line changed in memory: OUT = this run's folder, so nothing is
     written under ~/Leap (no copy of the loop is written anywhere).
  The wrapped window() runs Leap's window() (warm-up + timed round), then on the first window run_test() below;
  later windows are skipped.
rec1a_c25_stub.py runs run_test() on a toy engine with the same interfaces (the test of this file).

run_test(), per arm of REC1A_ARMS (default F32o,F32c6i), the pre-registered steps:
  - record A, record B: two consecutive checks of the arm inside the recorder (knobs; snapshot; verify forward; logits
    astype + eval), each followed by the engine's own commit + back() outside it. A's objects are kept alive through
    B (mtlrec keep()), so a buffer bound in both with the same pointer is persistent, and B's per-call buffers are new;
  - diff A vs B; fit each changed setBytes word / buffer offset / small-buffer word to a seq formula: the plain seq,
    or ((seq * nl + L) mod M) * S + c for the gate's table / region / slot-ring layouts (MODULE1A_ENGINE_MAP.md);
  - rules; the ICB from B (the mode-2 replay if the build refuses);
  - REC1A_ROUNDS (16) rounds: the engine's own check (logits -> numpy; ext 'gated' and status[0] / [16] deltas) ->
    back() -> s = new_pass() -> rewrite B's per-call small buffers from B's bind-time snapshots -> rules(s) -> knobs ->
    the one-buffer replay -> unknob; the states(P) hashes around round 1's replay.
Lines per arm: (1) same structure and 0 unexplained changes; (2) the replayed logits bitwise == the engine's on every
round; (3) each replay's gated delta == the engine check's, > 0, and status[0] timeouts 0, every round; (4) the
states(P) hashes equal around round 1's replay. Verdict: PASS = 1-4 on every arm; BETWEEN = the first arm passes and a
later one does not; FAIL = the first arm fails (a cause, not a KILL). Info: commits, Python-level evals inside the
record, retained bytes, walls, replay GPU time, fast hits."""
import hashlib
import importlib.util
import json
import os
import resource
import struct
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mtlrec as R  # noqa: E402

mx = None                                              # mlx.core, bound by bind_mx() after R.load()
SMALL = 4096                                           # mtlrec's SNAPMAX: bind-time snapshots of shared buffers <= 4 KB
OPC = {v: k for k, v in R.OPN.items()}
LEAP = os.path.expanduser("~/Leap/scripts/week98/c25")
OUT_LINE = 'OUT = os.path.expanduser("~/Leap/data/week98/c25"); os.makedirs(OUT, exist_ok=True)'
LINES = []
RESULT = dict(arms={}, notes=[])


def cfg():
    return dict(arms=[a for a in os.environ.get("REC1A_ARMS", "F32o,F32c6i").split(",") if a],
                rounds=int(os.environ.get("REC1A_ROUNDS", "16")),
                retain_gb=float(os.environ.get("REC1A_RETAIN_GB", "3")),
                out=os.path.expanduser(os.environ.get("REC1A_OUT", os.path.join(HERE, "..", "..", "..", "data", "m1",
                                                                                   "rec1a_engine"))),
                tag=os.environ.get("REC1A_TAG", "rec1a"))


def bind_mx():
    global mx
    import mlx.core as _mx
    mx = _mx
    return mx


def say(s):
    LINES.append(s)
    print(f"[rec1a] {s}", flush=True)


def write_out(C):
    os.makedirs(C["out"], exist_ok=True)
    with open(os.path.join(C["out"], f"{C['tag']}_rec1a.out"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    with open(os.path.join(C["out"], f"{C['tag']}_rec1a.json"), "w") as f:
        json.dump(RESULT, f, indent=1, default=str)


class EvalCount:
    """Counts mx.eval / mx.async_eval / mx.synchronize calls made through the mx module attribute while active (info:
    where the forward syncs or flushes inside the record; calls bound by `from mlx.core import eval` are not seen)."""
    NAMES = ("eval", "async_eval", "synchronize")

    def __init__(self):
        self.n = {k: 0 for k in self.NAMES}
        self.orig = {}

    def __enter__(self):
        for k in self.NAMES:
            f = getattr(mx, k)
            self.orig[k] = f

            def w(*a, _k=k, _f=f, **kw):
                self.n[_k] += 1
                return _f(*a, **kw)
            setattr(mx, k, w)
        return self

    def __exit__(self, *e):
        for k, f in self.orig.items():
            setattr(mx, k, f)
        return False


# ---------------------------------------------------------------- record, canonical buffers, diff, fit
def canon(ops):
    """Canonical buffers in order of first binding: dict(c, ptr, op = first BUF op index, len)."""
    first = {}
    for i in np.where(ops[:, 0] == OPC["BUF"])[0]:
        p = int(ops[i, 10])
        if p and p not in first:
            first[p] = int(i)
    return [dict(c=c, ptr=p, op=i, len=int(R.buf_len(i))) for c, (p, i) in enumerate(first.items())]


def record(E, arm, n, toks, label):
    """One check of `arm` inside the recorder (the forward + the logits' astype + eval), then the engine's own commit
    and back() outside it. Returns the record's dumps and the evaluated float32 logits array (kept: B's is the replay's
    output buffer)."""
    CE, G, P = E["CE"], E["G"], E["P"]
    model, ref2, CS, states, BG = G["model"], G["ref2"], G["CS"], G["states"], G["BG"]
    CE.knobs(G, arm, n)
    try:
        sn = CS.snapshot_untrimmable_cache_lazy(P)
        mx.synchronize()
        s0 = int(BG.seq)
        ec = EvalCount()
        t0 = time.perf_counter()
        R.start()
        with ec:
            with ref2.verify_scope(model):
                lg = model(mx.array([toks]), cache=P)
        seq = int(BG.seq)
        lg = lg[0].astype(mx.float32)
        mx.eval(lg)
        mx.synchronize()
        R.stop()
        wall = time.perf_counter() - t0
        rec = dict(label=label, arm=arm, s0=s0, seq=seq, ops=R.ops_dump(), bytes=R.bytes_dump(), names=R.disp_names(),
                   counts=R.counts(), opcounts=R.opcounts(), wall=wall, evals=dict(ec.n))
        rec["bufs"] = canon(rec["ops"])
        for b in rec["bufs"]:
            if 0 <= b["len"] <= SMALL:
                b["snap"] = R.buf_snap(b["op"])
                b["post"] = R.buf_read(b["op"])
        assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
        mx.eval(states(P))
        mx.synchronize()
    finally:
        R.stop()
        CE.unknob(G)
    CE.back(G, P, E["S0"], n)
    rec["lg"] = lg
    return rec


def families(BG):
    """(nl, [(name, M, S)]): value = ((seq * nl + L) mod M) * S + c for the gate's per-(pass, row) layouts."""
    nl = int(BG.nl)
    mod = sys.modules.get(type(BG).__module__)
    maxk = int(getattr(BG, "maxk", 0) or getattr(mod, "MAXK", 0) or 0)
    fams = []

    def add(name, M, S):
        if M > 0 and S > 0:
            fams.append((name + ".i32", int(M), int(S)))
            fams.append((name + ".bytes", int(M), int(S) * 4))
    add("tab", int(BG.ntab), int(BG.tables.size) // int(BG.ntab))
    add("reg", int(BG.nreg), int(BG.region.size) // int(BG.nreg))
    if maxk:
        add("ring", int(BG.ring.size) // maxk, maxk)
    return nl, fams


def fit(vA, vB, sA, sB, nl, fams):
    """Every seq formula that gives vA at seq sA and vB at seq sB: [(name, rule kwargs)]."""
    out = []
    if vA == sA and vB == sB:
        out.append(("seq", dict(a=1, b=0, d=0, m=0, s=1, c=0)))
    elif vA == (sA & 0x3FF) and vB == (sB & 0x3FF):
        out.append(("seq&0x3FF", dict(a=1, b=0, d=0, m=1024, s=1, c=0)))
    for name, M, S in fams:
        cA, cB = vA % S, vB % S
        if cA != cB:
            continue
        qA, qB = vA // S, vB // S
        if not (0 <= qA < M and 0 <= qB < M):
            continue
        L = (qA - sA * nl) % M
        if L < nl and (sA * nl + L) % M == qA and (sB * nl + L) % M == qB:
            out.append((f"{name} L={L}", dict(a=nl, b=int(L), d=0, m=M, s=S, c=int(cA))))
    return out


def words(a, b, step=4):
    """Offsets of the `step`-byte words that differ between two equal-length byte strings."""
    n = min(len(a), len(b)) // step * step
    if n == 0:
        return []
    x = np.frombuffer(a[:n], dtype=np.uint8).reshape(-1, step)
    y = np.frombuffer(b[:n], dtype=np.uint8).reshape(-1, step)
    return [int(i) * step for i in np.where((x != y).any(axis=1))[0]]


def i32(bs, off):
    return struct.unpack_from("<i", bs, off)[0]


def diff_and_plan(A, B, BG, C):
    """Structure check, every changed target classified, rules registered, the per-replay rewrite list.
    Returns (line1, plan)."""
    sA, sB = A["seq"], B["seq"]
    nl, fams = families(BG)
    oa, ob = A["ops"], B["ops"]
    struct_err = []
    plan = dict(rules=[], rewrite=[], unexplained=[], garbage=0, state=[], ambiguous=[], targets=[])
    if oa.shape != ob.shape or not np.array_equal(oa[:, 0], ob[:, 0]):
        struct_err.append(f"op sequences differ ({oa.shape[0]} vs {ob.shape[0]} ops)")
    else:
        if A["names"] != B["names"]:
            struct_err.append("dispatch kernel names differ")
        disp = np.isin(oa[:, 0], [OPC["DTG"], OPC["DT"]])
        g = [R.OPS_COLS.index(c) for c in ("gw", "gh", "gd", "tw", "th", "td")]
        if not np.array_equal(oa[disp][:, g], ob[disp][:, g]):
            struct_err.append("dispatch grid / threadgroup sizes differ")
        for opn, cols in (("TGMEM", ("i0", "n0")), ("BUF", ("i0",)), ("BUFOFF", ("i0",)), ("BARR", ("nb",)),
                          ("USE", ("nb", "i0"))):
            m = oa[:, 0] == OPC[opn]
            ci = [R.OPS_COLS.index(c) for c in cols]
            if not np.array_equal(oa[m][:, ci], ob[m][:, ci]):
                struct_err.append(f"{opn} fields {cols} differ")
        cidA = {b["ptr"]: b["c"] for b in A["bufs"]}
        cidB = {b["ptr"]: b["c"] for b in B["bufs"]}
        bi = np.where(oa[:, 0] == OPC["BUF"])[0]
        ca = np.array([cidA.get(int(oa[i, 10]), -1) for i in bi])
        cb = np.array([cidB.get(int(ob[i, 10]), -1) for i in bi])
        if not np.array_equal(ca, cb):
            struct_err.append(f"canonical bindings differ at {int((ca != cb).sum())} of {len(bi)} BUF ops")
        if len(A["bytes"]) != len(B["bytes"]) or any(
                (x["length"], x["bind"], x["disp"]) != (y["length"], y["bind"], y["disp"])
                for x, y in zip(A["bytes"], B["bytes"])):
            struct_err.append("setBytes layout differs")
    plan["struct"] = struct_err
    if struct_err:
        for s in struct_err:
            say(f"  STRUCTURE: {s}")
        return False, plan

    def take(kind, desc, vA, vB, rule_type, k, off):
        c = fit(vA, vB, sA, sB, nl, fams)
        t = dict(kind=kind, where=desc, vA=vA, vB=vB, fits=[x[0] for x in c])
        plan["targets"].append(t)
        if len(c) == 1:
            name, kw = c[0]
            plan["rules"].append((rule_type, k, off, kw, f"{kind} {desc}: {name}"))
            return True
        if len(c) > 1:
            plan["ambiguous"].append(t)
        return False

    # (a) setBytes words (inline constants: every change must be explained)
    for x, y in zip(A["bytes"], B["bytes"]):
        for off in words(x["data"], y["data"]):
            if take("setBytes", f"k {x['k']} off {off} (dispatch {x['disp']})", i32(x["data"], off),
                    i32(y["data"], off), "int32", x["k"], off):
                continue
            plan["unexplained"].append(f"setBytes k {x['k']} off {off}: {i32(x['data'], off)} -> {i32(y['data'], off)}")
    # (b) buffer offsets (BUF ops; a changed BUFOFF offset is not a supported rule target)
    for i in np.where(oa[:, 0] == OPC["BUF"])[0]:
        va, vb = int(oa[i, 2]), int(ob[i, 2])
        if va != vb and not take("bufoff", f"BUF op {i} index {int(oa[i, 1])}", va, vb, "bufoff", int(i), 0):
            plan["unexplained"].append(f"BUF op {i} offset: {va} -> {vb}")
    for i in np.where(oa[:, 0] == OPC["BUFOFF"])[0]:
        if int(oa[i, 2]) != int(ob[i, 2]):
            plan["unexplained"].append(f"BUFOFF op {i} offset {int(oa[i, 2])} -> {int(ob[i, 2])} (unsupported target)")
    # (c) small buffers: bind-time contents
    ptrA = {b["c"]: b["ptr"] for b in A["bufs"]}
    newA = newB = 0
    for b in B["bufs"]:
        a = A["bufs"][b["c"]]
        per_call = ptrA.get(b["c"]) != b["ptr"]
        if per_call:
            newB += max(b["len"], 0)
            newA += max(a["len"], 0)
        if b.get("snap") is None or a.get("snap") is None:
            continue
        if per_call:
            plan["rewrite"].append((b["op"], b["snap"]))
        for off in words(a["snap"], b["snap"]):
            vA, vB = i32(a["snap"], off), i32(b["snap"], off)
            if take("bufword", f"buffer c{b['c']} (BUF op {b['op']}, {'per-call' if per_call else 'persistent'}) off {off}",
                    vA, vB, "bufword32", b["op"], off):
                continue
            post = b.get("post")
            if post is not None and len(post) >= off + 4 and post[off:off + 4] != b["snap"][off:off + 4]:
                if per_call:                             # overwritten in the check: pre-run garbage
                    plan["garbage"] += 1
                else:                                    # state the check rewrites (e.g. a post row / release word
                    plan["state"].append(f"buffer c{b['c']} off {off}: {vA} -> {vB}")   # holding the last pass's seq)
                continue
            plan["unexplained"].append(f"buffer c{b['c']} off {off}: {vA} -> {vB} ({'per-call' if per_call else 'persistent'})")
    plan["retained_bytes"] = newA + newB
    plan["per_call_buffers"] = sum(1 for b in B["bufs"] if ptrA.get(b["c"]) != b["ptr"])
    say(f"  diff: {len(plan['targets'])} changed targets -> {len(plan['rules'])} rules, {len(plan['ambiguous'])} ambiguous, "
        f"{len(plan['unexplained'])} unexplained; {plan['garbage']} per-call words overwritten in the check (pre-run "
        f"garbage) and {len(plan['state'])} persistent state words the check rewrites, ignored; per-call buffers "
        f"{plan['per_call_buffers']} of {len(B['bufs'])} ({len(plan['rewrite'])} small, rewritten per replay); new bytes "
        f"in A + B {plan['retained_bytes'] / 1e9:.3f} GB")
    for r_ in plan["rules"][:60]:
        say(f"    rule: {r_[4]} -> {r_[3]}")
    for u in plan["state"][:20]:
        say(f"    state (rewritten in the check): {u}")
    for u in plan["unexplained"][:40]:
        say(f"    UNEXPLAINED: {u}")
    for t in plan["ambiguous"][:20]:
        say(f"    AMBIGUOUS: {t}")
    line1 = not plan["unexplained"] and not plan["ambiguous"]
    return line1, plan


# ---------------------------------------------------------------- hashes, counters, the rounds
def hash_states(G, P):
    arrs = []

    def flat(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                flat(y)
        elif x is not None:
            arrs.append(x)
    flat(G["states"](P))
    mx.eval(arrs)
    h = hashlib.blake2b(digest_size=16)
    nb = 0
    for a in arrs:
        u = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[a.itemsize]
        v = np.array(a.view(u))
        h.update(v.tobytes())
        nb += v.nbytes
    return h.hexdigest(), len(arrs), nb


def counters(BG):
    st = np.frombuffer(memoryview(BG.status), dtype=np.int32)
    return int(BG.ext.stats()["gated"]), int(st[0]), int(st[16])


def settle(BG, quiet_ms=5.0, cap_ms=200.0):
    """The counters once the loader's 'gated' count has stopped moving for quiet_ms (a fast-path pass does not wait for
    the loader, so it can still be serving rows when the GPU finishes)."""
    t_end = time.perf_counter() + cap_ms / 1e3
    last, t_last = counters(BG), time.perf_counter()
    while time.perf_counter() < t_end:
        time.sleep(0.001)
        c = counters(BG)
        if c[0] != last[0]:
            last, t_last = c, time.perf_counter()
        elif time.perf_counter() - t_last >= quiet_ms / 1e3:
            break
    return counters(BG)


def lbits(a):
    return np.array(a.view(mx.uint32))


def test_arm(E, arm, C):
    CE, G, P, BG = E["CE"], E["G"], E["P"], E["G"]["BG"]
    n = 16 if arm == "F16" else 32
    toks = E["gids"][E["gen"]:E["gen"] + n]
    res = dict(arm=arm, n=n)
    say(f"ARM {arm}: {n} rows; one warm check, then records A and B")
    CE.check(G, P, toks, n, arm, E["key"], "rec1a_warm", -1)
    CE.back(G, P, E["S0"], n)
    A = record(E, arm, n, toks, "A")
    kept = R.keep()
    B = record(E, arm, n, toks, "B")
    for X in (A, B):
        say(f"  record {X['label']}: seq {X['s0']} -> {X['seq']}; {X['counts']}; ops {X['opcounts']}; evals inside the "
            f"forward {X['evals']}; wall {1e3 * X['wall']:.1f} ms (encode-then-run: not a timing)")
    res.update(seqA=A["seq"], seqB=B["seq"], countsA=A["counts"], countsB=B["counts"], opcountsA=A["opcounts"],
               evalsA=A["evals"], evalsB=B["evals"], kept_refs=kept)
    line1, plan = diff_and_plan(A, B, BG, C)
    dropped = R.keep_drop()
    res.update(line1=line1, rules=[r_[4] for r_ in plan["rules"]], unexplained=plan["unexplained"],
               ambiguous=plan["ambiguous"], garbage=plan["garbage"], state=plan["state"],
               retained_gb=plan.get("retained_bytes", 0) / 1e9,
               structure=plan.get("struct"), rewrite=len(plan["rewrite"]), dropped_refs=dropped)
    if plan.get("retained_bytes", 0) > C["retain_gb"] * 1e9:
        say(f"  STOP: new bytes in A + B {plan['retained_bytes'] / 1e9:.2f} GB > {C['retain_gb']} GB (pre-registered "
            f"guard): no replay")
        res["stopped"] = "retained bytes over the guard"
        line1 = False
    say(f"  (1) record + diff: {'PASS' if line1 else 'FAIL'}")
    if not line1:
        res.update(line2=False, line3=False, line4=False)
        R.clear()
        return res
    n_icb, err = R.build_icb()
    use_icb = n_icb > 0
    say(f"  ICB: {n_icb} commands, {R.icb_nseg()} segments {('(refused: ' + err + ') -> the mode-2 replay') if not use_icb else ''}")
    res.update(icb=n_icb, icb_err=err, replay="icb" if use_icb else "mode2")
    R.rules_clear()
    for rt, k, off, kw, desc in plan["rules"]:
        if R.rule_add2(rt, k, off=off, var=0, **kw) < 0:
            say(f"  rule_add2 refused: {desc}")
    lgB = B["lg"]
    del A
    rows, h_eq = [], None
    for r in range(C["rounds"]):
        g0 = settle(BG)
        lg = CE.check(G, P, toks, n, arm, E["key"], "rec1a", r)
        ref = lbits(lg).copy()                           # Leap 1: the reference in numpy before any replay
        g1 = settle(BG)
        d_ref = tuple(y - x for x, y in zip(g0, g1))
        del lg
        CE.back(G, P, E["S0"], n)
        h0 = hash_states(G, P) if r == 0 else None
        s = int(BG.new_pass())
        for op, data in plan["rewrite"]:
            R.buf_write(op, data)
        R.rules_apply_v([s])
        g2 = counters(BG)
        CE.knobs(G, arm, n)
        try:
            rc, st = R.replay_icb() if use_icb else R.replay(2)
        finally:
            CE.unknob(G)
        got = lbits(lgB)
        g3 = settle(BG)
        d_rep = tuple(y - x for x, y in zip(g2, g3))
        if r == 0:
            h1 = hash_states(G, P)
            h_eq = h0[0] == h1[0]
            say(f"  (4) states(P): {h0[1]} arrays, {h0[2] / 1e6:.1f} MB; hash before {h0[0]}, after {h1[0]} -> "
                f"{'equal' if h_eq else 'DIFFERENT'}")
        eq = rc == 0 and got.shape == ref.shape and np.array_equal(got, ref)
        nrow = int((got == ref).all(axis=-1).sum()) if got.shape == ref.shape else -1
        ok3 = d_rep[0] == d_ref[0] and d_rep[0] > 0 and d_rep[1] == 0 and d_ref[1] == 0
        rows.append(dict(round=r, seq=s, rc=rc, bitwise=bool(eq), rows_equal=nrow, d_ref=d_ref, d_rep=d_rep, ok3=bool(ok3),
                         gpu_us=round(st.get("gpu_us", float("nan")), 1)))
        say(f"  round {r:2d}: seq {s}; replay rc {rc}, logits bitwise {eq} ({nrow}/{n} rows); gated / timeouts / fast "
            f"hits: engine {d_ref}, replay {d_rep} -> {'ok' if ok3 else 'NOT OK'}; replay GPU {st.get('gpu_us', 0):.0f} us")
    line2 = len(rows) == C["rounds"] and all(x["bitwise"] for x in rows)
    line3 = len(rows) == C["rounds"] and all(x["ok3"] for x in rows)
    line4 = bool(h_eq)
    say(f"  (2) replay == engine, bitwise: {sum(x['bitwise'] for x in rows)}/{C['rounds']} -> {'PASS' if line2 else 'FAIL'}")
    say(f"  (3) loader served each replay like the engine's check: {sum(x['ok3'] for x in rows)}/{C['rounds']} -> "
        f"{'PASS' if line3 else 'FAIL'}")
    say(f"  (4) states(P) unchanged by a replay: {'PASS' if line4 else 'FAIL'}")
    res.update(line2=line2, line3=line3, line4=line4, rounds=rows)
    R.rules_clear()
    R.clear()
    return res


def run_test(CE, G, key, P, S0, gids, gen, C=None):
    C = C or cfg()
    E = dict(CE=CE, G=G, key=key, P=P, S0=S0, gids=gids, gen=gen)
    for k in ("C25_PROF", "C25_CPUSPLIT", "C25_PASSK"):
        assert not os.environ.get(k), f"{k} must be unset (pre-registered)"
    old = mx.set_cache_limit(0)                          # Leap 1: freed buffers go back to Metal, never to live arrays
    mx.clear_cache()
    say(f"TEST on window {key}: arms {C['arms']}, {C['rounds']} rounds; MLX cache limit 0 (was {old}); "
        f"BIGMOE_POST_SYS={os.environ.get('BIGMOE_POST_SYS')} BIGMOE_SPIN_COHERENT={os.environ.get('BIGMOE_SPIN_COHERENT')}")
    t0 = time.time()
    try:
        for arm in C["arms"]:
            t_a = time.time()
            RESULT["arms"][arm] = res = test_arm(E, arm, C)
            res["wall_s"] = round(time.time() - t_a, 1)
            write_out(C)
    except Exception as e:
        say(f"TEST ERROR: {e!r}")
        RESULT["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        mx.set_cache_limit(old)
    ok = {a: all(r.get(f"line{i}") for i in (1, 2, 3, 4)) for a, r in RESULT["arms"].items()}
    arms = C["arms"]
    if "error" in RESULT or not arms or not ok.get(arms[0]):
        verdict = "FAIL"
    elif all(ok.get(a) for a in arms):
        verdict = "PASS"
    else:
        verdict = "BETWEEN"
    RESULT["verdict"] = verdict
    pk = G.get("footprint_peak_gb")
    RESULT["peak_gb"] = round(pk(), 2) if callable(pk) else round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    say(f"RESULT: {verdict}; per arm (lines 1-4): " + "; ".join(
        f"{a} {[bool(RESULT['arms'].get(a, {}).get(f'line{i}')) for i in (1, 2, 3, 4)]}" for a in arms)
        + f"; {time.time() - t0:.0f} s; lifetime peak {RESULT['peak_gb']} GB")
    write_out(C)
    return verdict


# ---------------------------------------------------------------- the launcher (the real engine)
def wrap(CE, C):
    orig = CE.window
    state = dict(done=False)

    def window(G, key, P, S0, gids, gen, Greq, first):
        if state["done"]:
            say(f"window {key}: skipped (the test ran on the first window)")
            return
        orig(G, key, P, S0, gids, gen, Greq, first)
        state["done"] = True
        run_test(CE, G, key, P, S0, gids, gen, C)
    CE.window = window


FLAG_MARK = "REC1A FLAGCHECK: reached EM.load"


def flagcheck_hook(g):
    """REC1A_FLAGCHECK=1 (Leap's c25_headtest pattern): the real loop runs up to its model load, where
    evalsuite.model.load prints the marker with the effective env and this run's OUT, then exits 97. No model load."""
    import importlib.abc

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "evalsuite.model":
                return None
            sys.meta_path.remove(self)
            s = importlib.util.find_spec(name)
            orig = s.loader.exec_module

            def exec_module(mod):
                orig(mod)

                def load(*a, **k):
                    keys = ("BIGMOE_POST_SYS", "BIGMOE_SPIN_COHERENT", "R57_VTIER", "R57_G32TIME", "R57_G32TIME_WARM",
                            "R57_G32TEST", "C25_ARMS", "C25_PROF", "C25_CPUSPLIT", "C25_PASSK", "BIGMOE_MAXK")
                    print(f"{FLAG_MARK}; OUT={g.get('OUT')!r}; c25_engine wrapped: "
                          f"{getattr(sys.modules.get('c25_engine'), 'window', None).__name__ == 'window' and sys.modules['c25_engine'].window.__module__ == __name__}; "
                          + " ".join(f"{k}={os.environ.get(k)!r}" for k in keys), flush=True)
                    os._exit(97)
                mod.load = load
            s.loader.exec_module = exec_module
            return s
    sys.meta_path.insert(0, Hook())


def main():
    C = cfg()
    for k in ("C25_PROF", "C25_CPUSPLIT", "C25_PASSK"):
        assert not os.environ.get(k), f"{k} must be unset (pre-registered)"
    R.load()
    bind_mx()
    spec = importlib.util.spec_from_file_location("c25_engine", os.path.join(LEAP, "c25_engine.py"))
    CE = importlib.util.module_from_spec(spec)
    sys.modules["c25_engine"] = CE
    spec.loader.exec_module(CE)
    wrap(CE, C)
    src_path = os.path.join(LEAP, "c25_loop.py")
    src = open(src_path).read()
    assert src.count(OUT_LINE) == 1, "c25_loop.py's OUT line changed: refusing to run (it would write under ~/Leap)"
    os.makedirs(C["out"], exist_ok=True)
    src = src.replace(OUT_LINE, f"OUT = {C['out']!r}; os.makedirs(OUT, exist_ok=True)")
    RESULT["notes"].append(dict(c25_loop_md5=hashlib.md5(open(src_path, "rb").read()).hexdigest(),
                                c25_engine_md5=hashlib.md5(open(os.path.join(LEAP, "c25_engine.py"), "rb").read()).hexdigest(),
                                mtlrec_md5=hashlib.md5(open(os.path.join(HERE, "libmtlrec.dylib"), "rb").read()).hexdigest(),
                                out=C["out"], argv=sys.argv[1:]))
    sys.argv = [src_path] + sys.argv[1:]
    g = {"__name__": "__main__", "__file__": src_path, "__builtins__": __builtins__}
    if os.environ.get("REC1A_FLAGCHECK") == "1":
        flagcheck_hook(g)
    exec(compile(src, src_path, "exec"), g)


if __name__ == "__main__":
    main()
