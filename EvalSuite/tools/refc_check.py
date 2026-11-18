"""The contract's per-run checks (docs/PREREG_refc_2026-11-18.md, step 0 and Part A) over a runner-70 run's kept answers:
each key's latest uncut record in the runs.json files that results/<run>/engage_runs.txt lists, for the keys in its
decode.jsonl. BigMoE's r57 screen writes runs.json 'refc' per run: the forms taken {c16, c32, s4, s4x2, s4g}, 'shapes'
(the dense forms taken per shape) and 'ref2_off_moves'. CPU only.
  .venv/bin/python tools/refc_check.py <run> --on [--shapes 7]     # the contract: engaged on every run
  .venv/bin/python tools/refc_check.py <run> --off                 # the control: never engaged
  .venv/bin/python tools/refc_check.py exact <chain run> <AR run>  # exactness: the loop == its own AR
  .venv/bin/python tools/refc_check.py selftest
- --on: on every kept run, 'shapes' holds --shapes entries, each > 0; s4 + s4g > 0; ref2_off_moves == 0 (Leap's drafter
  guarantee: refc's counters do not move while ref2 is off, i.e. during the drafts).
- --off: on every kept run, every form taken is 0 (or no refc block).
- exact: on every key of the chain run, its completion ids equal the AR run's, and every kept row's logits hash
  ('lhr': [[answer index, sha1], ...], R57_LOGHASH_ROWS=1) equals the AR's hash at the same answer index. A key whose
  record lacks 'lhr' FAILs.
Last line `REFC CHECK <run>: ENGAGED | NOT ENGAGED (...)`, `... OFF | ON (...)`, or `REFC EXACT <chain> vs <AR>: PASS | FAIL
(...)`; exit 0 on ENGAGED / OFF / PASS, 1 otherwise, 2 on bad input."""
import json, os, shutil, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def records(rdir):
    """{key: latest uncut record} for the keys in the run's decode.jsonl, from its ledger's runs.json files"""
    led = os.path.join(rdir, "engage_runs.txt")
    files = [p for p in open(led).read().split() if p] if os.path.exists(led) else []
    rows = {json.loads(l)["key"] for l in open(os.path.join(rdir, "decode.jsonl"))}
    rec = {}
    for p in files:                                     # later processes win, as runner 69's runterms
        for r in json.load(open(p)):
            if not r.get("cut"):
                rec[r["key"]] = r
    return {k: r for k, r in rec.items() if k in rows}, rows


def forms(r):
    f = (r.get("refc") or {})
    return {k: f.get(k) or 0 for k in ("c16", "c32", "s4", "s4x2", "s4g")}


def check_on(rdir, nshapes):
    rec, rows = records(rdir)
    bad = [] if rec and set(rec) == rows else [f"{len(rows - set(rec))} of {len(rows)} keys have no record"]
    for k, r in sorted(rec.items()):
        c, sh = r.get("refc") or {}, (r.get("refc") or {}).get("shapes") or {}
        f = forms(r)
        why = ([f"{len(sh)} shapes"] if len(sh) != nshapes else []) + [f"shape {s} at 0" for s, v in sh.items() if not v > 0] \
            + (["s4 + s4g 0"] if not f["s4"] + f["s4g"] > 0 else []) \
            + ([f"ref2_off_moves {c.get('ref2_off_moves')}"] if c.get("ref2_off_moves") != 0 else [])
        if why:
            bad.append(f"{k}: {', '.join(why)}")
    return bad, len(rec)


def check_off(rdir):
    rec, rows = records(rdir)
    bad = [] if rec and set(rec) == rows else [f"{len(rows - set(rec))} of {len(rows)} keys have no record"]
    bad += [f"{k}: forms {forms(r)}" for k, r in sorted(rec.items()) if any(forms(r).values())]
    return bad, len(rec)


def check_exact(cdir, adir):
    crec, crows = records(cdir)
    ids = lambda d: {(r := json.loads(l))["key"]: r["completion_ids"] for l in open(os.path.join(d, "decode.jsonl"))}
    cid, aid = ids(cdir), ids(adir)
    arec, _ = records(adir)
    bad, nrows = [], 0
    for k in sorted(crows):
        if k not in aid:
            bad.append(f"{k}: no AR row"); continue
        if cid[k] != aid[k]:
            n = next((i for i, (x, y) in enumerate(zip(cid[k], aid[k])) if x != y), min(len(cid[k]), len(aid[k])))
            bad.append(f"{k}: tokens differ from answer index {n}"); continue
        cl, al = (crec.get(k) or {}).get("lhr"), (arec.get(k) or {}).get("lhr")
        if not cl or not al:
            bad.append(f"{k}: no kept-row hashes ({'chain' if not cl else 'AR'})"); continue
        ah = {int(i): h for i, h in al}
        diff = [int(i) for i, h in cl if ah.get(int(i)) != h]
        nrows += len(cl)
        if diff:
            bad.append(f"{k}: {len(diff)} of {len(cl)} kept rows' hashes differ (first at answer index {min(diff)})")
    return bad, len(crows), nrows


def selftest():
    d = tempfile.mkdtemp()
    try:
        def mk(name, recs, ids=None):
            run = os.path.join(d, name); os.makedirs(run)
            open(os.path.join(run, "decode.jsonl"), "w").write("".join(
                json.dumps({"key": r["key"], "completion_ids": (ids or {}).get(r["key"], [1, 2, 3])}) + "\n" for r in recs))
            p = os.path.join(d, f"{name}_runs.json"); json.dump(recs, open(p, "w"))
            open(os.path.join(run, "engage_runs.txt"), "w").write(p + "\n")
            return run
        sh7 = {f"s{i}": 3 for i in range(7)}
        on = lambda **x: {"c16": 4, "c32": 0, "s4": 2, "s4x2": 0, "s4g": 1, "shapes": dict(sh7), "ref2_off_moves": 0, **x}
        good = [{"key": "a", "refc": on()}, {"key": "b", "refc": on()}]
        assert not check_on(mk("on", good), 7)[0]
        assert check_on(mk("on6", [{"key": "a", "refc": on(shapes={f"s{i}": 3 for i in range(6)})}]), 7)[0]
        assert check_on(mk("on0", [{"key": "a", "refc": on(shapes=dict(sh7, s3=0))}]), 7)[0]
        assert check_on(mk("ons4", [{"key": "a", "refc": on(s4=0, s4g=0)}]), 7)[0]
        assert check_on(mk("ondraft", [{"key": "a", "refc": on(ref2_off_moves=2)}]), 7)[0]
        assert not check_off(mk("off", [{"key": "a", "refc": {"c16": 0, "s4": 0}}, {"key": "b"}]))[0]
        assert check_off(mk("offon", [{"key": "a", "refc": {"c16": 1}}]))[0]
        lhr = [[0, "x0"], [1, "x1"], [5, "x5"]]
        a = mk("ar", [{"key": "a", "lhr": [[0, "x0"], [1, "x1"], [2, "x2"], [5, "x5"]]}])
        assert not check_exact(mk("ch", [{"key": "a", "lhr": lhr}]), a)[0]
        assert check_exact(mk("chh", [{"key": "a", "lhr": [[0, "x0"], [1, "y1"]]}]), a)[0]
        assert check_exact(mk("cht", [{"key": "a", "lhr": lhr}], {"a": [1, 2, 4]}), a)[0]
        assert check_exact(mk("chn", [{"key": "a"}]), a)[0]
        print("REFC CHECK selftest: PASS (on: 7 shapes > 0, s4 + s4g, the drafter guarantee; off; exact: tokens, row hashes, "
              "a missing hash list each FAIL)")
        return 0
    finally:
        shutil.rmtree(d)


def main(argv):
    if argv[:1] == ["selftest"]:
        return selftest()
    try:
        if argv[0] == "exact":
            bad, n, nr = check_exact(os.path.join(ROOT, "results", argv[1]), os.path.join(ROOT, "results", argv[2]))
            print(f"REFC EXACT {argv[1]} vs {argv[2]}: {'PASS' if not bad else 'FAIL'} ({n} keys, {nr} kept rows"
                  + (f"; {len(bad)} differ: {'; '.join(bad[:5])}" if bad else ": every token and kept row's hash equal") + ")")
            return 0 if not bad else 1
        run = argv[0]
        if "--on" in argv:
            ns = int(argv[argv.index("--shapes") + 1]) if "--shapes" in argv else 7
            bad, n = check_on(os.path.join(ROOT, "results", run), ns)
            print(f"REFC CHECK {run}: {'ENGAGED' if not bad else 'NOT ENGAGED'} ({n} runs"
                  + (f"; {len(bad)} fail: {'; '.join(bad[:5])}" if bad else f": {ns} shapes > 0, s4 + s4g > 0, ref2_off_moves 0") + ")")
        elif "--off" in argv:
            bad, n = check_off(os.path.join(ROOT, "results", run))
            print(f"REFC CHECK {run}: {'OFF' if not bad else 'ON'} ({n} runs" + (f"; {'; '.join(bad[:5])}" if bad else ": every form 0") + ")")
        else:
            raise ValueError
    except (IndexError, ValueError):
        print(__doc__); return 2
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
