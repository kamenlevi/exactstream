"""The stack's process-wide validity checks (PREREG_lossygate amendment 6, section 1), over a tier-1 run's kept answers:
each key's latest uncut record in the runs.json files that results/<run>/engage_runs.txt lists, for the keys in its
decode.jsonl. The expert rewrite (R57_XRW) acts on every forward in a process, IFEval's included (BigMoE 11:47 Wed), and
runner 69's CODE-ONLY engagement rule VOIDs a term > 0 on a chat key, so these are checked here, over every run. CPU only.
  .venv/bin/python tools/xrw_check.py <run> --mode tgl16eu --xknob <path>/xknob.py:<sha256> [--code-sum field:key ...]
  .venv/bin/python tools/xrw_check.py selftest
- every kept record carries an xrw block whose mode is --mode;
- summed over every kept record: xrw.orig == 0 and xrw.fallback_not_p11 == 0 (in knob_v2's xknob.py a rewritten call
  counts cand only, so orig == 0 means the rewrite ran every call);
- summed over the kept code records (keys outside ifeval/): xrw.cand > 0, and each --code-sum field:key > 0 (e.g.
  tree:sib_accepted, which may run on IFEval too);
- every process's modules manifest (the runs.json's sibling *_modules.json) loads xknob.py from exactly that path, with
  that sha256 (a manifest keeps the first 12 hex digits).
Last line: `XRW CHECK <run>: VALID (...)` or `XRW CHECK <run>: VOID (...)`; exit 0 VALID, 1 VOID, 2 bad input."""
import json, os, shutil, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def records(rdir):
    """(latest uncut record per key in decode.jsonl, the runs.json files) for a run dir"""
    led = os.path.join(rdir, "engage_runs.txt")
    files = [p for p in open(led).read().split() if p] if os.path.exists(led) else []
    rows = {json.loads(l)["key"] for l in open(os.path.join(rdir, "decode.jsonl"))}
    rec = {}
    for p in files:                                     # later processes win, as runner 69's runterms
        for r in json.load(open(p)):
            if not r.get("cut"):
                rec[r["key"]] = r
    return {k: r for k, r in rec.items() if k in rows}, files


def check(rdir, mode, xpath, xsha, sums):
    rec, files = records(rdir)
    bad = []
    if not files or not rec:
        return [f"no kept records ({len(files)} runs.json files)"], {}
    nox = [k for k, r in rec.items() if not isinstance(r.get("xrw"), dict)]
    wrong = [k for k, r in rec.items() if isinstance(r.get("xrw"), dict) and r["xrw"].get("mode") != mode]
    tot = lambda f, k, ks: sum((rec[x].get(f) or {}).get(k) or 0 for x in ks if isinstance(rec[x].get(f), dict))
    code = [k for k in rec if not k.startswith("ifeval/")]
    S = {"orig": tot("xrw", "orig", rec), "fallback_not_p11": tot("xrw", "fallback_not_p11", rec),
         "cand (code)": tot("xrw", "cand", code), "cand (chat)": tot("xrw", "cand", [k for k in rec if k not in code])}
    if nox:
        bad.append(f"{len(nox)} records without an xrw block (e.g. {nox[:3]})")
    if wrong:
        bad.append(f"{len(wrong)} records with xrw.mode != {mode} (e.g. {wrong[0]}: {rec[wrong[0]]['xrw'].get('mode')!r})")
    if S["orig"] != 0 or S["fallback_not_p11"] != 0:
        bad.append(f"xrw.orig {S['orig']}, fallback_not_p11 {S['fallback_not_p11']} (both must be 0)")
    if not S["cand (code)"] > 0:
        bad.append("xrw.cand is 0 over the code records")
    for t in sums:
        f, _, k = t.partition(":")
        S[f"{t} (code)"] = tot(f, k, code)
        if not S[f"{t} (code)"] > 0:
            bad.append(f"{t} is 0 over the code records")
    mans = sorted({p[:-len("_runs.json")] + "_modules.json" for p in files})
    for m in mans:
        if not os.path.exists(m):
            bad.append(f"no modules manifest {os.path.basename(m)}"); continue
        xs = [(n, v) for n, v in json.load(open(m))["modules"].items() if os.path.basename(v[0]) == "xknob.py"]
        if len(xs) != 1 or xs[0][1][0] != xpath or not xsha.startswith(xs[0][1][1]) or len(xs[0][1][1]) < 12:
            bad.append(f"{os.path.basename(m)}: xknob loaded as {xs or 'nothing'} (need {xpath} at {xsha[:12]})")
    S.update({"records": len(rec), "code records": len(code), "processes": len(mans)})
    return bad, S


def selftest():
    d = tempfile.mkdtemp()
    try:
        xp, xs = "/k/knob_v2/xknob.py", "6474eab8d8f57921aae8d868c067fa4f2091e348717bcf16c37f81e7fbd46c56"
        def mk(name, recs, xmod=(xp, xs[:12])):
            run = os.path.join(d, name); os.makedirs(run)
            open(os.path.join(run, "decode.jsonl"), "w").write("".join(json.dumps({"key": r["key"]}) + "\n" for r in recs))
            p = os.path.join(d, f"{name}_p1_runs.json"); json.dump(recs, open(p, "w"))
            json.dump({"modules": {"xknob": list(xmod), "os": ["/x/os.py", "aaaaaaaaaaaa"]}}, open(p[:-10] + "_modules.json", "w"))
            open(os.path.join(run, "engage_runs.txt"), "w").write(p + "\n")
            return run
        x = lambda c, o=0, f=0, m="tgl16eu": {"mode": m, "cand": c, "orig": o, "fallback_not_p11": f}
        good = [{"key": "humaneval/HumanEval/1", "xrw": x(5), "tree": {"sib_accepted": 2}},
                {"key": "ifeval/1", "xrw": x(3), "tree": {"sib_accepted": 0}}]
        cases = [("good", good, None, True), ("orig", [dict(good[0], xrw=x(5, o=1)), good[1]], None, False),
                 ("fallback", [dict(good[0], xrw=x(5, f=1)), good[1]], None, False),
                 ("mode", [dict(good[0], xrw=x(5, m="tgl16e")), good[1]], None, False),
                 ("noblock", [{"key": "mbpp/2", "tree": {"sib_accepted": 1}}, good[1]], None, False),
                 ("cand0", [dict(good[0], xrw=x(0)), good[1]], None, False),
                 ("sib0", [dict(good[0], tree={"sib_accepted": 0}), good[1]], None, False),
                 ("module", good, ("/k/knob_v1/xknob.py", xs[:12]), False),
                 ("modsha", good, (xp, "000000000000"), False)]
        for name, recs, xmod, want in cases:
            run = mk(name, recs, *([xmod] if xmod else []))
            bad, _ = check(run, "tgl16eu", xp, xs, ["tree:sib_accepted"])
            assert (not bad) == want, (name, bad)
        print(f"XRW CHECK selftest: PASS ({len(cases)} cases: a valid run, then orig, fallback, mode, a missing block, "
              f"cand 0, sib 0, the module's path and its sha each VOID it)")
        return 0
    finally:
        shutil.rmtree(d)


def main(argv):
    if argv[:1] == ["selftest"]:
        return selftest()
    try:
        run = argv[0]; mode = argv[argv.index("--mode") + 1]
        xpath, _, xsha = argv[argv.index("--xknob") + 1].rpartition(":")
        sums = [argv[i + 1] for i, a in enumerate(argv) if a == "--code-sum"]
        assert xpath and len(xsha) == 64
    except (IndexError, ValueError, AssertionError):
        print(__doc__); return 2
    bad, S = check(os.path.join(ROOT, "results", run), mode, xpath, xsha, sums)
    print(f"XRW CHECK {run}: {'VALID' if not bad else 'VOID'} ({'; '.join(bad) if bad else ''}{'; ' if bad else ''}"
          + ", ".join(f"{k} {v}" for k, v in S.items()) + ")")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
