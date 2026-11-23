"""cc3_probe.py (Leap, Sat 14 Nov 2026 18:4x; WEEK98 route 3 "the cheap check", screen 3, the orchestrator 18:2x):
what a check costs at 1 / 16 / 32 rows, split into dense / experts / misses, in the shipped engine, without changing
the decode loop. Loaded by the loop's own probe hook (R57_COPYPROBE=1 R57_COPYPROBE_PY=<this file>): at every warm-split
check (R57_WARMSPLIT=N, which already times the 16-row check cold and then warm at 1 / 4 / 8 / 16 rows, each rolled
back), this module adds rolled-back passes, all with nothing committed:
  d1   dense only, 1 row ([tok])                 the routed-expert call stubbed to zeros (no gate post, no loads)
  d16  dense only, the check's own rows ([tok] + drafts)
  d32  dense only, 32 rows ([tok] + the next 31 reference tokens)
  s32  the real 32-row verify, once per prompt   above 16 rows today's engine leaves the sync-free gate for the
                                                  slot path (gated.py: a routing row carries at most 185 ids)
The stub keeps the loop's mid-loop async_eval (SB), so the dense passes overlap CPU build and GPU as a live check does.
The reference continuation is the shipped REF2 row (harness/results/ref2/decode.jsonl), used only while the run's
tokens still equal it (checked at each probe; else d32 / s32 are skipped). The probe's time is returned to the loop,
which takes it out of the cycle and decode clocks. Every pass is rolled back with the loop's own snapshot + rollback,
the same calls its warm split uses; the real check runs after the probe as always, and the run's tokens vs REF2 show
that nothing leaked. Output: one JSON line per probe check to CC3_OUT.
Env: CC3_OUT (jsonl path, required), CC3_PASSES (default "d1,d16,d32,s32"), CC3_S32 (real 32-row passes per prompt,
default 1). A pass that raises is written to CC3_OUT with its error and re-raised: the run stops rather than go on
with a cache that a half-built pass may have left inconsistent (the runner then retries with fewer passes).
v2 (21:1x, after screen 3's lossy-24 runs died at start-up: the 24 GB loop asserts R57_ROWCAP and R57_WARMSPLIT exclusive):
  CC3_EVERY=N     with no warm split (WARMSPLIT 0), probe every Nth check itself (the loop's own rule: len(kept) >= 2 and
                  len(kept) % N == 0). d16 is then the check's OWN rows (<= 13 at 24 GB). The stub passes load nothing, so
                  the live check right after the probe is as cold as any: the report takes experts + misses = that live
                  check (checks.json index ci, with its misses and SSD wait) - d16. The loop resets its fast gate (FG, QOM)
                  after every check, so s32 here runs with FG off: no QOM, exact expert reads (labelled so).
  CC3_REFLOOSE=1  if the run has left the REF2 row (lossy modes), still take 32 rows of REF2's continuation from the same
                  position (realistic text, not this run's): the record says loose=True. ref_ok stays the strict test.
  Every record carries trig ("wsp" / "own") and ci (the check's index = len(kept), its slot in checks.json)."""
import json
import os
import time

S = {"saved": {}, "s32": {}, "lines": 0}


def install(g):
    S["out"] = os.path.expanduser(os.environ["CC3_OUT"])
    S["n_s32"] = int(os.environ.get("CC3_S32", "1"))
    S["passes"] = set(os.environ.get("CC3_PASSES", "d1,d16,d32,s32").split(","))
    S["keymap"] = None                                   # REFROWS is defined after the loop installs the probe: built at first use
    S["every"] = int(os.environ.get("CC3_EVERY", "0") or 0)   # v2: own trigger when the loop runs no warm split
    S["loose"] = os.environ.get("CC3_REFLOOSE") == "1"        # v2: REF2's continuation even after the run left REF2
    os.makedirs(os.path.dirname(S["out"]) or ".", exist_ok=True)


def _trig(g, l):
    if l.get("mtp"):
        return None
    if l.get("wsp"):
        return "wsp"
    ev = S["every"]
    if ev and not g.get("WARMSPLIT"):
        k = len(l.get("kept") or ())
        if k >= 2 and k % ev == 0:
            return "own"
    return None


def _key(name, g):
    if not name:
        return None
    if S["keymap"] is None:
        S["keymap"] = {k.replace("/", "_").replace(".json", ""): k for k in g["REFROWS"]}
    base = name.rsplit("_", 2)[0]                        # run(): f"{key_}_{arm}_{n}"; arms carry no "_"
    return S["keymap"].get(base)


def _stub_on(g):
    mx = g["mx"]
    for sw in g["SW_"]:
        c = type(sw)
        if c in S["saved"]:
            continue
        S["saved"][c] = c.__dict__.get("__call__")

        def stub(self, x, indices, *a, **kw):            # the routed experts' output as zeros: combined rows [..., D]
            bm = self._bm                                # (the block adds the shared expert; y.ndim == inds.ndim: no
            ae = bm.get("async_every", 0)                # second combine). SB's hand-off kept as in stream.py.
            if ae and bm["L"] % ae == ae - 1:
                mx.async_eval(x)
            return mx.zeros(x.shape, dtype=x.dtype)
        c.__call__ = stub


def _stub_off():
    for c, f in S["saved"].items():
        if f is not None:
            c.__call__ = f
        else:
            del c.__call__
    S["saved"].clear()


def _pass(g, l, rows, label, stub=False):
    mx, CS, ref2, model, stats, states = g["mx"], g["CS"], g["ref2"], g["model"], g["stats"], g["states"]
    cache = l["cache"]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    s0 = stats()
    t0, c0 = time.perf_counter(), time.thread_time()
    try:
        if stub:
            _stub_on(g)
        with ref2.verify_scope(model):
            lg = model(mx.array([rows]), cache=cache)
        tb, cb = time.perf_counter(), time.thread_time()
        mx.eval(mx.argmax(lg[0], axis=-1))
        te, ce = time.perf_counter(), time.thread_time()
    finally:
        if stub:
            _stub_off()
        CS.rollback_after_verify(cache, snap, len(rows))
        mx.eval(states(cache))
    d = stats() - s0
    return dict(p=label, rows=len(rows), ms=round(1e3 * (te - t0), 2), bld=round(1e3 * (tb - t0), 2),
                wt=round(1e3 * (te - tb), 2), cpu_bld=round(1e3 * (cb - c0), 2), cpu_wt=round(1e3 * (ce - cb), 2),
                miss=int(d[0]), io=round(1e3 * d[1], 2), svc=round(1e3 * d[2], 2))


def maybe(g, l):
    trig = _trig(g, l)
    if trig is None:
        return 0.0
    key = _key(l.get("name"), g)
    if key is None:
        return 0.0
    t_start = time.perf_counter()
    tok, dr, out = int(l["tok"]), [int(x) for x in l["dr"]], [int(x) for x in l["out"]]
    want = [int(x) for x in g["REFROWS"].get(key, {}).get("completion_ids", [])]
    have = len(want) >= len(out) + 31
    ref_ok = have and want[:len(out)] == out
    loose = have and not ref_ok and S["loose"]
    rec = dict(key=key, trig=trig, ci=len(l.get("kept") or ()), n=len(out), rows=len(dr) + 1, ref_ok=ref_ok, loose=loose,
               passes=[], err=None)
    P = S["passes"]
    try:
        if "d1" in P:
            rec["passes"].append(_pass(g, l, [tok], "d1", stub=True))
        if "d16" in P:
            rec["passes"].append(_pass(g, l, [tok] + dr, "d16", stub=True))
        if ref_ok or loose:
            r32 = [tok] + want[len(out):len(out) + 31]
            if "d32" in P:
                rec["passes"].append(_pass(g, l, r32, "d32", stub=True))
            if "s32" in P and S["s32"].get(key, 0) < S["n_s32"]:
                S["s32"][key] = S["s32"].get(key, 0) + 1
                rec["passes"].append(_pass(g, l, r32, "s32"))
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"
        _write(rec, t_start)
        raise
    _write(rec, t_start)
    return time.perf_counter() - t_start


def _write(rec, t_start):
    rec["probe_s"] = round(time.perf_counter() - t_start, 3)
    with open(S["out"], "a") as f:
        f.write(json.dumps(rec) + "\n")
    S["lines"] += 1
