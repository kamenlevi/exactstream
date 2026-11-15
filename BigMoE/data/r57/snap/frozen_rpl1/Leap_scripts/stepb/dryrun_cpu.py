"""FLAT-2's CPU dry-run of the plumbing (zero session, Thu 5 Nov 2026). No GPU, no MLX, no model: numpy and the
standard library, one process, a few seconds, < 200 MB. What it proves before a GPU slot is spent:
  A  the patcher fits BigMoE's HEAD (anchors, the copied rules, the names the hooks read), refuses a changed loop, and
     nothing flat2.py binds collides with a global the harness reads;
  B  the REAL patched run() of that HEAD (cut out of the generated source by name, with the real stats() / states())
     runs under flat2_driver.main with the real hooks, probe and analysis, on a stub engine whose clock is virtual: every
     piece of a check has a set cost, so the cycle, the stages, the row slope, the split and FLAT have exact expected
     values. Three cost sets must land on PASS / BETWEEN / KILL, and the control without the pool prewarm on INVALID;
  C  run_flat2.sh's dry mode: the memory ladder, each arm's env and command, no lock taken, nothing started.
The stub is not the engine: it checks call order, names, bookkeeping and arithmetic, not kernels or timings.
Usage: ~/BigMoE/.venv/bin/python -I dryrun_cpu.py [-v] -> prints each check, writes out/dryrun_*; exit 1 on a failure."""
import ast
import builtins
import contextlib
import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flat2_core
import flat2_driver
import flat2_patch
import flat2_probe

VERBOSE = "-v" in sys.argv
SCREEN = os.path.expanduser(os.environ.get("FLAT2_SCREEN", "~/BigMoE/scripts/r57_screen.py"))
OUT = os.path.join(HERE, "out")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and (VERBOSE or not ok) else ""))
    if not ok:
        FAILS.append(name)


def near(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ the stub engine
V, NE, NL_, N_LIN = 4096, 512, 48, 36                    # token ids, experts per layer, layers, GDN layers (the rest attention)
COST = dict(hc=0.04, gdn=0.20, att=0.30, pre=0.05, post=0.02, hcw=0.03, glue=0.01, ple=0.10, hcf=0.06, head=1.50,
            e=0.025, draft=5.0, commit=1.8, miss=0.5, prefill=300.0)   # ms; e = experts per row per layer


def expect(c, rows=2):
    dense = NL_ * (2 * c["hc"] + c["pre"] + c["post"] + c["hcw"] + c["glue"]) + N_LIN * c["gdn"] + (NL_ - N_LIN) * c["att"] \
        + c["ple"] + c["hcf"] + c["head"]
    e_row = NL_ * c["e"]
    return dict(dense=dense, e_row=e_row, check=dense + rows * e_row, cycle=c["draft"] + c["commit"] + dense + rows * e_row,
                flat=c["draft"] + c["commit"] + dense)


def nxt(seq):                                            # the stub model's next token: fixed by the last 6 tokens
    return (sum((i + 3) * t for i, t in enumerate(seq[-6:])) * 2654435761 + 12345) % V


def experts(tok, L):                                     # a row's 10 routed experts at layer L, rank order, distinct
    h = (tok * 40503 + L * 9973 + 17) % NE
    return [(h + 37 * r) % NE for r in range(10)]


def make_ns(cost, C_m=226, C_l=112):
    """the names run() and FLAT-2 read from the harness, on a stub: lazy costs land on a virtual clock at each eval"""
    E = SimpleNamespace(t=1000.0, pending=0.0, tap=None, cache=None, rows=None,
                        S=dict(misses=0, io_s=0.0, slow_s=0.0, slow_calls=0, uses=0, stalled=0), pool={})
    clock = lambda: E.t

    def lazy(ms):
        E.pending += ms

    def flush(*a):
        E.t += E.pending / 1e3; E.pending = 0.0

    def now(ms):
        E.t += ms / 1e3

    class A(np.ndarray):                                 # mx.array: a numpy view; reading a value waits for the GPU
        def __new__(cls, data, dtype=None):
            return np.asarray(data, dtype=dtype).view(cls)

        def astype(self, dt, *a, **k):
            return self if isinstance(dt, str) and dt.startswith("mx.") else np.ndarray.astype(self, dt, *a, **k).view(A)

        def item(self):
            flush(); return np.ndarray.item(self)

        def tolist(self):
            flush(); return np.ndarray.tolist(self)

    mx = SimpleNamespace(array=A, eval=flush, async_eval=lambda *a: None, bfloat16="mx.bfloat16", float32="mx.float32",
                         stack=lambda xs: (flush(), A(np.stack([np.asarray(x_) for x_ in xs])))[1],   # read by the loop: a sync
                         argmax=lambda x, axis=None: A(np.argmax(np.asarray(x), axis=axis)),
                         array_equal=lambda a, b, equal_nan=False: A(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=equal_nan)),
                         reset_peak_memory=lambda: None, get_peak_memory=lambda: 0.0, get_active_memory=lambda: 0.0,
                         get_cache_memory=lambda: 0.0, clear_cache=lambda: None)
    FG = dict(on=False, keep=None, sure=None, sure_np=None, t3=False, cold=False, warm=None, cap=None, dump=None, chk=0,
              qom=False, qcf=None)
    TAP = {"on": False, "x": []}
    H1 = A(np.zeros((1, 1, 4), np.float32))

    def pool_use(L, ids, load=True):                     # LRU pool of capacity C: -> the ids that had to be read
        p, cap = E.pool[L]
        miss = []
        for e in ids:
            e = int(e)
            if e in p:
                p.move_to_end(e)
            elif load:
                miss.append(e); p[e] = 1
                if len(p) > cap:
                    p.popitem(last=False)
        return miss

    class GR:                                            # a hyper-connection read
        def __call__(self, h, pending=None):
            lazy(cost["hcf"] if self is model.final else cost["hc"]); return h

    class GDN:
        def __call__(self, h):
            lazy(cost["gdn"]); return h

    class Att:
        def __call__(self, h):
            lazy(cost["att"]); return h

    class PLE:
        def __call__(self, h):
            lazy(cost["ple"]); return h

    class Switch:                                        # the routed experts: QOM-like loads (sure rows read ranks 0-1)
        def __call__(self, x, indices):
            L, ids = self._bm["L"], np.asarray(indices)
            sure = np.asarray(FG["sure"]) if FG["on"] and FG["sure"] is not None else np.zeros(len(ids), bool)
            assert len(sure) == len(ids), ("the gate's rows != the check's rows", len(sure), len(ids))
            if FG["on"] and FG["qcf"] is not None:
                assert len(np.asarray(FG["qcf"])) == len(ids), "the split's row cf != the check's rows"
            loads = {int(e) for r in range(len(ids)) for e in ids[r, :2 if sure[r] else 10]}
            miss = pool_use(L, sorted(loads))
            opt = {int(e) for r in range(len(ids)) if sure[r] for e in ids[r, 2:] if int(e) in E.pool[L][0]}
            E.S["uses"] += len(loads | opt)
            if FG.get("qcnt") is not None:               # BigMoE's per-check QOM counters (Thu 19:0x), the live check only
                FG["qcnt"].append(A(np.array([len(loads | opt), 0, len(loads | opt)])))
            if miss:
                E.S["misses"] += len(miss); E.S["stalled"] += 1; E.S["slow_calls"] += 1
                E.S["io_s"] += len(miss) * cost["miss"] / 1e3; E.S["slow_s"] += len(miss) * cost["miss"] / 1e3
                lazy(len(miss) * cost["miss"])
            lazy(cost["e"] * len(ids))
            return x

    class MoE:
        def __call__(self, x):
            lazy(cost["pre"])
            inds = A(np.array([experts(t, self.switch_mlp._bm["L"]) for t in E.rows], np.int32))
            y = self.switch_mlp(x, inds)
            lazy(cost["post"])
            return y

    class Layer:
        def __contains__(self, k):
            return k == "ple" and self.ple is not None

        def __call__(self, h):
            if "ple" in self:
                h = h + self.ple(h)
            m = self.attn_hyper_connection(h)
            b = self.linear_attn(m) if self.is_linear else self.self_attn(m)
            m = self.mlp_hyper_connection(h, pending=(b, None))
            b = self.mlp(m)
            h = Q._hyper_residual_write(h, b, None)
            lazy(cost["glue"])                           # the layer's own ops, outside every piece
            return h

    def hrw(hyper, block_out, inject):
        lazy(cost["hcw"]); return hyper

    Q = SimpleNamespace(DecoderLayer=Layer, _hyper_residual_write=hrw)
    lay = []
    for i in range(NL_):
        l = Layer()
        l.attn_hyper_connection, l.mlp_hyper_connection, l.is_linear = GR(), GR(), i % 4 != 3
        l.linear_attn, l.self_attn, l.ple, l.mlp = GDN(), Att(), (PLE() if i == 1 else None), MoE()
        l.mlp.switch_mlp = Switch()
        l.mlp.switch_mlp._bm = dict(L=i, full=False, C=C_m, tier=dict(side=100 + i, C_l=C_l))
        E.pool[i], E.pool[100 + i] = (OrderedDict(), C_m), (OrderedDict(), C_l)
        lay.append(l)
    assert sum(l.is_linear for l in lay) == N_LIN

    class Model:
        final = GR()

        def __call__(self, x, cache=None):
            rows, ent = [int(t) for t in np.asarray(x)[0]], cache[0]
            E.rows = rows
            h = H1
            for l in lay:
                h = l(h)
            h = self.final(h)
            lazy(cost["head"])
            seq, out = list(ent.state), []
            for t in rows:
                seq.append(t); out.append(nxt(seq))
            ent.state = tuple(seq)
            lg = np.zeros((1, len(rows), V), np.float32)
            lg[0, np.arange(len(rows)), out] = 1.0
            if TAP["on"]:
                E.tap = A(np.zeros((len(rows), 8), np.float16))
            return A(lg)

        def commit_verified_window(self, cache, snapshot_states, *, keep_tokens, verified_tokens):
            ent, base = cache[0], snapshot_states[0]
            assert len(ent.state) == len(base) + verified_tokens, "commit: the cache does not hold the live check's rows"
            ent.state = tuple(ent.state[:len(base) + keep_tokens])
            lazy(cost["commit"])

    model = Model()

    def snapshot(cache):
        return SimpleNamespace(states=[c.state for c in cache])

    def rollback(cache, snap, n):
        for c, s in zip(cache, snap.states):
            assert len(c.state) == len(s) + n, ("rollback: the cache holds another row count", len(c.state) - len(s), n)
            c.state = s

    def prefill(ids, want_w=False, chunk=1024):          # as stream.py's staged prefill: the prompt's experts over the pools
        flush(); now(cost["prefill"])
        for i in range(NL_):
            p, cap = E.pool[i]
            p.clear()
            for j in range(cap):
                p[(7 * j + i) % NE] = 1
        cache = [SimpleNamespace(state=tuple(ids))]
        E.cache = cache
        last = np.zeros(V, np.float32); last[nxt(list(ids))] = 1.0
        return cache, A(last), A(np.zeros((len(ids), 8), np.float16)), []

    def propose(blk, ctx, dcache):                       # 70 % of the drafts right; a wrong one carries a low cf
        flush(); now(cost["draft"])
        n, seq = blk.shape[1] - 1, list(E.cache[0].state) + [int(np.asarray(blk)[0, 0])]
        dr, cf = [], []
        for j in range(n):
            good = (len(seq) * 7 + j * 3) % 10 < 7
            d = nxt(seq) if good else (nxt(seq) + 1) % V
            dr.append(d); cf.append(0.95 if good else 0.40); seq.append(d)
        dcache[0].offset += ctx.shape[1] + n + 1
        return A(np.array([dr])), A(np.array([cf], np.float32))

    def trim(dcache, n):
        for c in dcache:
            c.offset -= n

    ext = SimpleNamespace(lookup=lambda L, ids: [0 if int(e) in E.pool[L][0] else -1 for e in ids], ensure=None)

    def ensure(L, arr):
        ids = [int(e) for e in np.asarray(arr)]
        assert len(ids) <= E.pool[L][1], "ensure past the pool's size (the ext aborts)"
        m = pool_use(L, ids)
        E.S["misses"] += len(m); E.S["io_s"] += len(m) * 0.1 / 1e3; now(len(m) * 0.1)
    ext.ensure = ensure
    lines = []
    t_start = E.t
    fz = dict(thr=0.0, copy=False, rules=(), loop=None, q1u=True, t3=True, cold=False, qom=True)
    d4, d16 = SimpleNamespace(make_cache=lambda: [SimpleNamespace(offset=0)]), SimpleNamespace(make_cache=lambda: [SimpleNamespace(offset=0)])
    ns = dict(
        mx=mx, np=np, os=os, sys=sys, json=json, threading=threading,
        time=SimpleNamespace(perf_counter=clock, time=clock, sleep=lambda s: None), t0=t_start,
        model=model, Q=Q, lay=lay, NL=NL_, SW_=[l.mlp.switch_mlp for l in lay], TXT=None, FG=FG, TAP=TAP,
        tap_reset=lambda: None, taps=lambda: E.tap, CS=SimpleNamespace(snapshot_untrimmable_cache_lazy=snapshot,
                                                                      rollback_after_verify=rollback),
        ref2=SimpleNamespace(apply=lambda: None, remove=lambda: None, verify_scope=lambda m: contextlib.nullcontext(),
                             counts=lambda: {"stub": 1}),
        BG=SimpleNamespace(ext=ext, PAUSED={"pause_s": 0.0, "n_gap": 0}), BS=SimpleNamespace(ext=ext, _S10B=None),
        st=SimpleNamespace(stats=lambda: dict(E.S), gate=SimpleNamespace(check=lambda: None)),
        EM=SimpleNamespace(STATE=SimpleNamespace(capture=None)), ED=SimpleNamespace(load=lambda name: [
            dict(key="humaneval/HumanEval/68", prompt_ids=[(i * 97 + 11) % V for i in range(150)])]),
        rule=lambda item: "code", SETOF={"humaneval": "humaneval"}, CHATSW=True, DRAFT4=d4, DRAFT16=d16, draft=d16,
        DF=SimpleNamespace(_trim_recent_cache=trim), prefill=prefill, propose=propose, set_arm=lambda arm: None,
        ref_on=lambda on: None, disk_read_bytes=lambda: 0, footprint_now_gb=lambda: 0.0, footprint_peak_gb=lambda: 0.0,
        med=lambda v: float(np.median(v)) if len(v) else float("nan"), el=lambda: E.t - t_start, HARD=600,
        say=lambda s: (lines.append(s), print("      | " + s) if VERBOSE else None)[0], lines=lines,
        ARMS_=("gate+t",), CAP=256, C=208, R=16, K=16, MASK=1, SWIN=8, EOS={248044, 248046}, TRUNC=0.3, MTP_K=15,
        PAUSE="/nonexistent/PAUSE", TIER=True, VTIER="", A6=set(), A6CAP={}, A6_POOLS=24, WARMSPLIT=0, STALL=False,
        PREROUTE=False, PRDUMP=False, PR=dict(on=False, cur=None, y={}, blk={}, cap={}, dump=None), PRS={},
        PRD=dict(n=0, L=[], chk=[], row=[], X=[], Y=[]), LOGHASH=False, S10B_DIR=None, COPY="", COPYMODE={}, CGM=None,
        U3C_LEN=False, U3C_DUMP="", POP45=None, U3FORCE=False, FAST="QOM", QSPLIT=0.9, FASTARMS={"QOM": {"humaneval": fz}},
        RXC=SimpleNamespace(COPY_N=8, gate_rows=lambda cf, tail, dr, thr, grams: [1 if c >= thr else 0 for c in cf[:len(dr)]] + [0]),
        D17=True, D17M=SimpleNamespace(split17=lambda dr, cfl, R_: (dr[:R_ - 1], cfl[:R_ - 1], dr[R_ - 1], cfl[R_ - 1])),
        MTPHead=None, VT_MAP=[], _TKZ=None, pr_eval=None,   # paths run() does not take here (chat, vtier, relax, preroute)
        F2ON=[False], F2H=None, F2C=[], __builtins__=builtins)
    return ns, E


def load_run(ns, gen, path):
    """the real run(), stats() and states() of the generated harness source, executed in the stub's namespace"""
    tree = ast.parse(gen)
    want = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("run", "stats", "states")]
    assert {n.name for n in want} == {"run", "stats", "states"}
    fn = next(n for n in want if n.name == "run")
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = sorted(x for x in loads - flat2_patch._stores(fn) if not hasattr(builtins, x) and x not in ns
                  and x not in ("stats", "states"))
    exec(compile(ast.Module(body=want, type_ignores=[]), path, "exec"), ns)
    return free


def scenario(tag, cost, gen, info, prewarm=True, cap=100, passes=5, path=None):
    ns, E = make_ns(cost)
    free = load_run(ns, gen, path or os.path.join(OUT, "dryrun_generated.py"))
    cfg = dict(smoke=False, cap=cap, passes=passes, at=(8, 16, 24, 32), key="humaneval/HumanEval/68", tag=f"dryrun_{tag}",
               out=OUT, screen=SCREEN, no_git=True, real=2, real_cap=60)
    H = flat2_probe.Hooks(ns, clock=ns["time"].perf_counter)
    rc = flat2_driver.main(ns, cfg, info, hooks=H, prewarm=prewarm)
    J = json.load(open(os.path.join(OUT, f"dryrun_{tag}_flat2.json")))
    return rc, J, free, ns["lines"]


def part_a():
    print("A. the patcher on BigMoE's HEAD")
    src = open(SCREEN).read()
    try:
        gen, info = flat2_patch.build(src)
    except flat2_patch.PatchError as e:
        check("anchors, copied rules and names fit the HEAD", False, str(e))
        return None, None
    check("anchors, copied rules and names fit the HEAD", True, f"md5 {info['md5']}, hooks at {info['anchors']}")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "dryrun_generated.py"), "w").write(gen)
    hooks = [l for l in gen.split("\n") if l.endswith(flat2_patch.MARK)]
    check("six insertions: two cycle stamps, the prewarm call, three per-check hooks behind F2ON",
          len(hooks) == 6 and sum("if F2ON[0]:" in h for h in hooks) == 3 and sum("F2C.append" in h for h in hooks) == 2)
    for w in info["warnings"]:
        print(f"      warning (the run goes on): {w[:150]}")
    g2, i2 = flat2_patch.build(src.replace("sure_np=sq_, t3=bool(fz and fz.get(\"t3\")),", "sure_np=sq_, t3=bool(fz and fz.get(\"t3\")), zz9=1,"))
    check("accepts an ADDED gate argument, with a warning", any("zz9" in w for w in i2["warnings"]))
    check("the harness's main section is cut (no prompt loop, no file writes)", "REFROWS" not in gen and "json.dump(TOK" not in gen)
    for what, old, new in (("a changed gate rule", "                    sq_[len(dr)] = False", "                    sq_[len(dr)] = True"),
                           ("a renamed commit call", "model.commit_verified_window(cache, snap.states, keep_tokens=k + 1",
                            "model.commit_window(cache, snap.states, keep_tokens=k + 1"),
                           ("a changed gate setup", "sure_np=sq_, t3=bool(fz and fz.get(\"t3\")),   # (in the graph)",
                            "sure_np=sq_, t3=False,   # (in the graph)"),
                           ("a renamed local the hooks read", "        snap = CS.snapshot_untrimmable_cache_lazy(cache)\n        if not mtp:",
                            "        snp = CS.snapshot_untrimmable_cache_lazy(cache)\n        if not mtp:")):
        assert old in src, what
        try:
            flat2_patch.build(src.replace(old, new))
            check(f"refuses {what}", False, "accepted")
        except flat2_patch.PatchError as e:
            check(f"refuses {what}", True, str(e).split("\n")[0])
    mine = flat2_patch._module_names(ast.parse(open(os.path.join(HERE, "flat2.py")).read()).body)
    theirs = flat2_patch._module_names(ast.parse(gen).body)
    check("flat2.py binds no global the harness also binds (but os, sys)", mine & theirs <= {"os", "sys"}, str(sorted(mine & theirs)))
    for f in ("flat2.py", "flat2_driver.py", "flat2_probe.py", "flat2_core.py", "flat2_patch.py"):
        compile(open(os.path.join(HERE, f)).read(), f, "exec")
    check("every FLAT-2 file compiles", True)
    return gen, info


def part_b(gen, info):
    print("B. the real patched run() on the stub engine (virtual clock)")
    x = expect(COST)
    rc, J, free, lines = scenario("pass", COST, gen, info)
    S = J["summary"]
    check("run()'s globals are all in the stub (else the HEAD reads a new one)", not free, str(free))
    check("driver exit code 0, no hook errors", rc == 0 and not any(r["errors"] for r in J["runs"]), str([r["errors"] for r in J["runs"]]))
    check("every check of every 2-row pass has 2 rows", all(set(r["pc"]["rows"]) == {2} for r in J["runs"] if not r["real"]))
    meas = [r for r in J["runs"][1:] if r["mode"] == "meas" and not r["real"]]
    check("cold pass misses, prewarmed passes do not", sum(J["runs"][0]["pc"]["miss"]) > 100 and all(sum(r["pc"]["miss"]) == 0 for r in meas),
          f"cold {sum(J['runs'][0]['pc']['miss']):.0f}, measured {[sum(r['pc']['miss']) for r in meas]}")
    check("measured passes repeat the same tokens", all(S["passes_same_tokens"]) and len(S["passes_same_tokens"]) == 4)
    check(">= 200 steady zero-miss cycles", S["n"] >= 200, f"{S['n']}")
    check("cycle (start to start) = draft + commit + dense + 2 rows of experts", near(S["cycle"], x["cycle"]), f"{S['cycle']:.4f} vs {x['cycle']:.4f}")
    check("every pass has a start stamp per check and the end stamp", all(r["period"] and len(r["period"]) == len(r["pc"]["ck"]) for r in J["runs"]))
    check("the row slope and intercept", near(S["e_row"], x["e_row"]) and near(S["sweep_icpt"], x["dense"]) and S["sweep_resid"] < 1e-6,
          f"slope {S['e_row']:.4f} vs {x['e_row']:.4f}, intercept {S['sweep_icpt']:.4f} vs {x['dense']:.4f}")
    check("FLAT = cycle - 2 x slope", near(S["flat"], x["flat"]), f"{S['flat']:.4f} vs {x['flat']:.4f}")
    st = S.get("stage") or {}
    check("stages one after the other: draft, check, commit, no tail", bool(st) and near(st["draft"], COST["draft"]) and near(st["commit"], COST["commit"])
          and near(st["check"], x["check"]) and near(st["tail"], 0.0), str(st))
    sp = S["split2"] or {}
    want = {"GDN": N_LIN * COST["gdn"], "HC reads": 2 * NL_ * COST["hc"] + COST["hcf"], "attention": (NL_ - N_LIN) * COST["att"],
            "router + shared": NL_ * COST["pre"], "experts": 2 * x["e_row"], "MoE tail": NL_ * COST["post"], "HC write": NL_ * COST["hcw"],
            "PLE": COST["ple"], "layer glue": NL_ * COST["glue"], "embed + norm + head": COST["head"]}
    bad = {k: (sp.get(k), v) for k, v in want.items() if not near(sp.get(k, -1), v, 1e-5)}
    check("the synced split gives every piece its cost", not bad, str(bad))
    check("the 16-row split's experts = 16 rows", bool(S["split16"]) and near(S["split16"]["experts"], 16 * x["e_row"], 1e-5))
    probes = [p for r in J["runs"] if r["mode"] == "probe" for p in r["probes"]]
    check("4 probes, each ends on the live check's logits, warm repeats bitwise == live",
          len(probes) == 4 and all(p["final_equal"] and all(w["equal"] for w in p["warm"]) and not p.get("error") for p in probes))
    check("sweeps reach 16 rows, warm passes read nothing, walls rise with rows",
          all(sorted(int(k) for k in p["sweep"]) == [2, 4, 8, 12, 16] and all(v[1]["miss"] == 0 and v[2]["miss"] == 0
              for v in p["sweep"].values()) and [p["sweep"][k][1]["wall"] for k in ("2", "4", "8", "12", "16")]
              == sorted(p["sweep"][k][1]["wall"] for k in ("2", "4", "8", "12", "16")) for p in probes))
    rb = [r for r in J["runs"] if r["real"]]
    R = S.get("real") or {}
    check("real-block passes: 1 discarded + 2 measured + the commit pass, blocks of 2-16 rows",
          len(rb) == 4 and [r["mode"] for r in rb] == ["meas", "meas", "meas", "stage"] and max(max(r["pc"]["rows"]) for r in rb) > 2)
    check("real blocks: mean cycle = draft + commit + dense + rows x experts; tok/s = kept / cycle",
          bool(R) and near(R["cycle_mean"], COST["draft"] + COST["commit"] + x["dense"] + R["rows"] * x["e_row"], 1e-5)
          and near(R["tps"], 1e3 * R["kept"] / R["cycle_mean"], 1e-3), str(R))
    check("real blocks: FLAT + rows x slope + commit difference predicts the cycle", bool(R) and near(R.get("predicted", -1), R["cycle_mean"], 1e-5)
          and near(R["commit"], COST["commit"]), str(R))
    check("verdict PASS on the pass cost set", S["verdict"] == "PASS" and S["valid"], f"{S['verdict']} at FLAT {S['flat']:.2f}")
    check("the report ends on the verdict line", any(l.startswith("FLAT-2 VERDICT: PASS") for l in lines))
    for tag, d, want_v in (("between", dict(COST, draft=8.0), "BETWEEN"),
                           ("kill", dict(COST, draft=9.0, gdn=0.45, att=0.55, hc=0.08), "KILL")):
        rc, J2, _, _ = scenario(tag, d, gen, info, passes=3)
        S2, x2 = J2["summary"], expect(d)
        check(f"verdict {want_v} on the {tag} cost set", S2["verdict"] == want_v and near(S2["flat"], x2["flat"]),
              f"{S2['verdict']} at FLAT {S2['flat']:.2f} (expected {x2['flat']:.2f})")
    rc, J3, _, _ = scenario("noprewarm", COST, gen, info, prewarm=False, passes=3)
    S3 = J3["summary"]
    check("without the pool prewarm the passes miss and the verdict is INVALID", S3["verdict"] == "INVALID" and S3["n"] < flat2_core.MIN_CYCLES,
          f"{S3['verdict']}, {S3['n']} zero-miss cycles")
    snap = os.path.expanduser("~/BigMoE/data/r57/snap/r57_screen_pf.py")   # today's pre-flighted text (6adad93), if still there
    if os.path.exists(snap) and open(snap).read() != open(SCREEN).read():
        try:
            g0, i0 = flat2_patch.build(open(snap).read())
            p0 = os.path.join(OUT, "dryrun_generated_snap.py")
            open(p0, "w").write(g0)
            rc, J0, free0, _ = scenario("snap", COST, g0, i0, passes=3, path=p0)
            S0 = J0["summary"]
            check("the same on the pre-flight's frozen screen (commit lands in the next draft there)",
                  rc == 0 and not free0 and near(S0["cycle"], x["cycle"]) and near(S0["flat"], x["flat"]) and S0["verdict"] == "PASS"
                  and near(S0["tail"], 0.0) and near(S0["draft"], COST["draft"] + COST["commit"]), f"{S0['verdict']} {S0['flat']:.2f} tail {S0['tail']:.2f}")
            check("on the HEAD the QOM counters' read moves the commit into the loop's tail (cycle unchanged)",
                  near(S["tail"], COST["commit"]) and near(S["draft"], COST["draft"]) if "qcnt" in gen else True, f"tail {S['tail']:.2f} draft {S['draft']:.2f}")
        except flat2_patch.PatchError as e:
            check("the pre-flight's frozen screen still patches", False, str(e))
    return S


def part_c():
    print("C. run_flat2.sh, dry mode")
    sh = os.path.join(HERE, "run_flat2.sh")
    if not os.path.exists(sh):
        check("run_flat2.sh exists", False)
        return
    r = subprocess.run(["/bin/bash", "-n", sh], capture_output=True, text=True)
    check("bash syntax", r.returncode == 0, r.stderr)
    for avail, c_want in (("60", "224"), ("43", "208"), ("41", "192"), ("37", "160"), ("30", "")):
        r = subprocess.run(["/bin/bash", sh], capture_output=True, text=True, env=dict(os.environ, FLAT2_DRY="1", FLAT2_AVAIL=avail))
        o = r.stdout + r.stderr
        if c_want:
            ok = r.returncode == 0 and f"C {c_want}" in o and o.count("WOULD RUN") == 2 and "peer lock" not in o.replace("would take", "")
        else:
            ok = "nothing fits" in o and "WOULD RUN" not in o
        check(f"NO-SWAP {avail} GB -> {'C ' + c_want + ', two arms' if c_want else 'nothing runs'}", ok, o[-600:])
    r = subprocess.run(["/bin/bash", sh], capture_output=True, text=True, env=dict(os.environ, FLAT2_DRY="1", FLAT2_AVAIL="43"))
    o = r.stdout
    check("the fri arm carries Friday's all-in levers and the split", all(k in o for k in (
        "R57_PRESET=45", "BIGMOE_EXT_BUILD=build_qom", "R57_FAST=QOM", "R57_D17=1", "R57_D7B=code", "R57_HC8=1", "REF2_HC8_FAST=1",
        "R57_DNR=1", "R57_HCB=1", "REF2_SMALL_NATIVE=1", "R57_D7BNAX=1", "REF2_OOO_1C=1", "R57_QOM_SPLIT=0.9")), o[-900:])
    check("the all arm adds the default-off exact levers", "R57_DKV=1" in o and "R57_GDNB=1" in o)


if __name__ == "__main__":
    gen, info = part_a()
    if gen:
        part_b(gen, info)
    part_c()
    print(f"\nDRY-RUN {'FAILED: ' + ', '.join(FAILS) if FAILS else 'OK'} ({'CPU only; the stub checks plumbing, not the engine'})")
    sys.exit(1 if FAILS else 0)
