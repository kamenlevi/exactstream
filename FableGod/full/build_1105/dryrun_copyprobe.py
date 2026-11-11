"""dryrun_copyprobe.py (Fable FULL, LOSSY350 T-A1): CPU dry-run of copyprobe.py + copyprobe_report.py. No mlx, no model.
What it checks, on the engine's own 12 HumanEval prompts (EvalSuite's items) and their stored REF2-T answers:
  1. the block wrapper's math (modes s and t) against an independent numpy reference, and that every other call is
     passed through untouched;
  2. the hook inside a fake engine loop (13 tokens kept per check, as the walk): probe points found, pass order, every
     pass rolled back (cache offsets, FG and the wrapper's switch back to their state), the run's tokens untouched;
  3. REAL DATA: how many copied rows the 125B's own stored answer confirms at each width (the PASS line's 0.90);
  4. an engine that fails inside the experimental t pass: the part-built pass is undone, t is dropped, the run goes on;
  5. the report on the dry-run's JSONL, with ms from a SYNTHETIC cost model (to exercise the table and the verdict only).
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I dryrun_copyprobe.py   -> out/dry_copyprobe.jsonl, out/dry_copyprobe_report.txt"""
import contextlib, importlib.util, json, os, sys, types
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.expanduser("~")
OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)


def load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


CPM = load("copyprobe"); REP = load("copyprobe_report")
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
ENG = ["68", "16", "91", "6", "79", "19", "161", "17", "56", "141", "101", "63"]          # BigMoE scripts/fri_night.sh
ITEMS = {r["key"]: r for r in map(json.loads, open(f"{H}/EvalSuite/data/items/dec_humaneval.jsonl"))}
TRUTH = {r["key"]: r["completion_ids"] for r in map(json.loads, open(f"{H}/EvalSuite/results/ref2t/decode.jsonl"))}


# ---------------------------------------------------------------- 1. the block wrapper on a numpy stand-in for mx
class NPX:
    arange = staticmethod(np.arange); where = staticmethod(np.where); zeros_like = staticmethod(np.zeros_like)
    take_along_axis = staticmethod(np.take_along_axis); argsort = staticmethod(np.argsort); array = staticmethod(np.array)

    @staticmethod
    def argpartition(a, kth, axis=-1):
        return np.argpartition(a, kth, axis=axis)

    @staticmethod
    def softmax(x, axis=-1, precise=False):
        e = np.exp(x - x.max(axis=axis, keepdims=True)); return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def test_block():
    rng = np.random.default_rng(5); D, E, Kt = 12, 40, 10
    Wg = rng.normal(size=(D, E)); Ws = rng.normal(size=(D, D)); wsg = rng.normal(size=(D, 1))
    expert = lambda x, ids: np.tanh(x[..., None, :] * (1 + ids[..., None] / 7.0))       # [..., K, D]: one output per rank

    class Switch:
        _bm = {"L": 0}
        def __call__(self, x, ids):
            self.last = ids
            return expert(x, ids)

    prev_calls = []

    class Block:
        top_k = Kt; norm_topk_prob = True
        def __init__(self):
            self.switch_mlp = Switch()
        gate = staticmethod(lambda x: x @ Wg)
        shared_expert = staticmethod(lambda x: x @ Ws)
        shared_expert_gate = staticmethod(lambda x: x @ wsg)
        def __call__(self, x):
            prev_calls.append(x.shape); return "PREV"

    Q = types.SimpleNamespace(SparseMoeBlock=Block)
    G = dict(mx=NPX, np=np, _QX=Q, EM=types.SimpleNamespace(STATE=types.SimpleNamespace(sorted=True)))
    CPM.install(G)
    b = Block()
    for rows in (4, 16):                                             # <= 16 rows: the screen's block, whatever the mode
        CPM.CP.update(mode="s", sure=np.array([True] * (rows - 1) + [False]))
        assert Block.__call__(b, rng.normal(size=(1, rows, D))) == "PREV"
    CPM.CP.update(mode=None, sure=None)
    assert Block.__call__(b, rng.normal(size=(1, 32, D))) == "PREV"  # no probe pass being built: the screen's block
    res = {}
    for rows in (32, 64):
        x = rng.normal(size=(1, rows, D)); sure = np.array([True] * (rows - 1) + [False])
        g = NPX.softmax(x @ Wg)[0]                                   # the independent reference, row by row
        top = np.argsort(-g, axis=-1)[:, :Kt]; sc = np.take_along_axis(g, top, -1); sc = sc / sc.sum(-1, keepdims=True)
        loads = set()
        for r in range(rows):
            loads.update(top[r, :2].tolist() if sure[r] else top[r].tolist())
        shared = NPX.sigmoid(x @ wsg) * (x @ Ws)
        ref_s = np.zeros((rows, D)); ref_t = np.zeros((rows, D)); kept = 0
        for r in range(rows):
            for j in range(Kt):
                y = np.tanh(x[0, r] * (1 + top[r, j] / 7.0))
                if not sure[r] or j < 2 or int(top[r, j]) in loads:
                    ref_s[r] += sc[r, j] * y; kept += 1
                if j < 2:
                    ref_t[r] += sc[r, j] * y
        for mode, ref in (("s", ref_s), ("t", ref_t)):
            CPM.CP.update(mode=mode, sure=sure)
            y = Block.__call__(b, x)
            assert not isinstance(y, str) and np.allclose(y[0], ref + shared[0], atol=1e-9), (mode, rows)
            if mode == "t":
                assert b.switch_mlp.last.shape == (1, rows, 2), b.switch_mlp.last.shape     # a narrow call: rows x 2 ids
            else:
                ids = b.switch_mlp.last[0]                           # route mode: a dropped rank reads the row's rank 0
                assert all(int(ids[r, j]) in loads or int(ids[r, j]) == int(top[r, 0]) for r in range(rows) for j in range(Kt))
        res[rows] = (kept, rows * Kt, len(loads))
    CPM.CP.update(mode=None, sure=None)
    print("1. block wrapper: modes s and t equal the independent reference at 32 and 64 rows; calls <= 16 rows or outside a "
          "probe pass reach the screen's block. Random routing over 40 experts, kept pairs / all pairs, experts loaded: "
          + ", ".join(f"W {r}: {k} / {a}, {l}" for r, (k, a, l) in res.items()))
    Block.__call__ = lambda self, x: "PREV"                         # leave the class as found for the next tests
    CPM.CP["installed"] = False


# ---------------------------------------------------------------- 2-4. the hook in a fake engine loop
class KV:
    def __init__(self, n): self.offset = n
    def trim(self, n):
        assert 0 <= n <= self.offset, (n, self.offset); self.offset -= n


class Rec:
    def __init__(self): self.state = 0


class FakeCS:
    @staticmethod
    def snapshot_untrimmable_cache_lazy(cache):
        return [c.state for c in cache if isinstance(c, Rec)]

    @staticmethod
    def restore_cache(cache, snap):
        for c, s in zip([c for c in cache if isinstance(c, Rec)], snap):
            c.state = s

    @staticmethod
    def rollback_after_verify(cache, snap, n):
        for c in cache:
            if isinstance(c, KV):
                c.trim(n)
        FakeCS.restore_cache(cache, snap)


class FakeArr:
    def __init__(self, v): self.v = list(v)
    def tolist(self): return list(self.v)


class FakeLogits:
    def __init__(self, am): self.am = am
    def __getitem__(self, i): return self


class Clock:                                                         # the module's time, advanced by the fake engine
    t = 0.0
    def perf_counter(self): return self.t


class FMX(NPX):
    @staticmethod
    def argmax(lg, axis=-1): return FakeArr(lg.am)
    @staticmethod
    def eval(*a): pass


def engine(fail_t=False):
    clock = Clock(); CPM.time = clock
    st = np.zeros(6); warm = set()
    FG = dict(on=False, keep=None, sure=None, sure_np=None, t3=False, cold=False, warm=None, cap=None, dump=None, chk=0,
              qom=False, qcf=None)
    E = dict(full=None, log=[], run=None)

    def model(inp, cache=None):
        toks = [int(t) for t in np.array(inp)[0]]; rows = len(toks); pos0 = cache[0].offset
        mode = CPM.CP["mode"] if rows > 16 else None
        if fail_t and mode == "t" and rows == 64:                    # a build that stops part way: half the cache advanced
            for c in cache[:len(cache) // 2]:
                if isinstance(c, KV): c.offset += rows
                else: c.state += rows
            raise RuntimeError("fake tier call: K 2 not supported at 64 rows")
        am, ok = [], True
        for i, t in enumerate(toks):
            ok = ok and E["full"][pos0 + i] == t
            am.append(E["full"][pos0 + i + 1] if ok and pos0 + i + 1 < len(E["full"]) else -1)
        for c in cache:
            if isinstance(c, KV): c.offset += rows
            else: c.state += rows
        if rows <= 16:
            assert FG["on"] and FG["sure"] is not None and len(FG["sure"]) == rows and FG["qom"] and FG["t3"], "the gate pass"
            assert FG["qcf"] is not None and float(FG["qcf"].min()) == 1.0 and not bool(FG["sure"][-1]), "split cf / last row"
        else:
            assert not FG["on"], "a wide pass must not carry the 16-row FAST gate"
        cold = [p for p in range(pos0, pos0 + rows) if (E["run"], p) not in warm]   # first touch of a row in this run
        warm.update((E["run"], p) for p in range(pos0, pos0 + rows))
        strict = rows <= 16 or mode in ("s", "t")
        # SYNTHETIC cost model (ms), only to exercise the report: fast path 52 + 0.62 per strict row; the wide slot path 60 +
        # 5.0 per exact row, 2.4 per strict row (route mode); the narrow gate call 55 + 0.7; a first-touch row +1.1
        ms = ((52 + 0.62 * rows) if rows <= 16 else (55 + 0.7 * rows) if mode == "t" else (60 + (2.4 if mode == "s" else 5.0) * rows))
        ms += 1.1 * len(cold)
        clock.t += ms / 1e3
        st[0] += 3 * len(cold); st[1] += 0.3e-3 * len(cold); st[2] += 0.2e-3 * len(cold); st[3] += bool(cold)
        st[4] += rows * (38 if strict else 131); st[5] += bool(cold)
        if mode in ("s", "t"):
            CPM.CP["calls"] += 48                                    # the wrapper would build 48 MoE blocks
        E["log"].append((rows, mode, len(cold)))
        return FakeLogits(am)

    @contextlib.contextmanager
    def scope(m):
        yield

    G = dict(mx=FMX, np=np, CS=FakeCS, ref2=types.SimpleNamespace(verify_scope=scope), FG=FG, model=model,
             stats=lambda: st.copy(), states=lambda c: [], _TKZ=TOK, QSPLIT=0.9, FAST="QOM", D17=True, OUT=OUTD, TAG="dry")
    return G, E, FG


def loop(variants, fail_t=False, tag="dry"):
    out_path = os.path.join(OUTD, f"{tag}_copyprobe.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)
    CPM._OUT[0] = out_path; CPM.STATE.clear()
    CPM.configure({"R57_COPYPROBE": variants, "R57_COPYPROBE_MAX": "3"})
    CPM.CP.update(installed=True, mode=None, sure=None)
    G, E, FG = engine(fail_t)
    fz = dict(thr=0.0, copy=False, rules=(), loop=None, q1u=True, t3=True, cold=False, qom=True)
    npts = []
    for n in ENG:
        key = f"humaneval/HumanEval/{n}"; ids = list(ITEMS[key]["prompt_ids"]); truth = TRUTH[key][:256]
        E["full"] = ids + list(TRUTH[key]); E["run"] = key
        cache = [KV(len(ids)), Rec(), KV(len(ids)), Rec()]
        out = [truth[0]]; name = f"{key.replace('/', '_')}_gate+t_0"
        while len(out) < len(truth):
            before = ([c.offset if isinstance(c, KV) else c.state for c in cache], list(out))
            assert cache[0].offset == len(ids) + len(out) - 1
            dt = CPM.maybe(G, dict(mtp=False, name=name, ids=ids, out=out, tok=out[-1], cache=cache, fz=fz))
            assert dt >= 0 and ([c.offset if isinstance(c, KV) else c.state for c in cache], list(out)) == before, "not rolled back"
            assert not FG["on"] and FG["sure"] is None and CPM.CP["mode"] is None, "the gate or the wrapper left on"
            new = truth[len(out):len(out) + 13]                      # the real check: 13 tokens kept, as the walk's copy checks
            for c in cache:
                if isinstance(c, KV): c.offset += len(new)
                else: c.state += len(new)
            out += new
        assert out == truth
        npts.append(CPM.STATE[name]["n"])
    return out_path, npts, E


def main():
    test_block()
    path, npts, E = loop("g+x+s+t")
    recs = [json.loads(l) for l in open(path)]
    pts = [r for r in recs if r["kind"] == "point"]; ps = [r for r in recs if r["kind"] == "pass"]
    assert not [r for r in recs if r["kind"] == "error"] and not [r for r in ps if r["err"]], "a pass failed in the clean run"
    firsts = [p["order"][0] for p in pts]
    print(f"2. hook: {len(pts)} probe points on the 12 prompts ({npts} per prompt, cap 3), {len(ps)} passes, every one rolled "
          f"back, tokens untouched; first (cold) pass by turn: {sorted(set(firsts))}; copied tokens left at the points: "
          f"min {min(p['left'] for p in pts)}, median {int(np.median([p['left'] for p in pts]))}")
    print("3. REAL DATA (the 125B's stored REF2-T answers): copied rows confirmed, per pass shape: " + "; ".join(
        f"{v}{rows} {sum(r['agree'] for r in ps if (r['v'], r['rows']) == (v, rows)) / sum(r['n'] for r in ps if (r['v'], r['rows']) == (v, rows)):.3f}"
        f" (all rows on {100 * np.mean([r['acc'] == r['n'] for r in ps if (r['v'], r['rows']) == (v, rows)]):.0f} % of passes)"
        for v, rows in (("g", 16), ("s", 32), ("s", 64))))
    rep = REP.report(path)
    open(os.path.join(OUTD, "dry_copyprobe_report.txt"), "w").write(
        "SYNTHETIC ms (dryrun_copyprobe.py's cost model): this exercises the report only, it is not a measurement\n" + rep + "\n")
    path2, npts2, E2 = loop("g+x+s+t", fail_t=True, tag="dryfail")
    recs2 = [json.loads(l) for l in open(path2)]
    errs = [r for r in recs2 if r["kind"] == "pass" and r["err"]]
    t_after = [r for r in recs2 if r["kind"] == "pass" and r["v"] == "t"]
    assert len(errs) == 1 and errs[0]["v"] == "t" and errs[0]["rows"] == 64 and "rollback" not in errs[0]["err"], errs
    assert "t" in CPM.CFG["off"] and sum(1 for r in recs2 if r["kind"] == "point") == len(pts), "the run must go on"
    print(f"4. failing engine (t64's build stops part way): 1 failed pass recorded ({errs[0]['err'][:60]}...), the part-built "
          f"pass undone, t dropped for the process ({len(t_after)} t passes in all), probe points unchanged ({len(pts)})")
    print("5. report on the dry-run JSONL (SYNTHETIC ms):")
    print(rep)


if __name__ == "__main__":
    main()
